#!/usr/bin/env python3
"""GPU follow-up: shared mappings, matched timing, channel-feed and window controls.

Uses the existing GPU replay libraries. Writes manifests and count checks alongside
results; never treats these saturated memory replays as token-time predictions.
"""
import argparse
import copy
import gzip
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
import gpu_ds3_study as D
import gpu_replay as G
import remap as R
import memsys_study as MS
import dramtech_cfg as C


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def window(prefix, start, count):
    """Read only the selected records, without expanding a whole multi-GB stream."""
    p = Path(str(prefix) + '.dram')
    if p.exists():
        with p.open('rb') as f:
            f.seek(start * 8)
            raw = f.read(count * 8)
    else:
        with gzip.open(str(p) + '.gz', 'rb') as f:
            f.seek(start * 8)
            raw = f.read(count * 8)
    assert len(raw) == count * 8, (prefix, start, count, len(raw))
    return np.frombuffer(raw, dtype='<u8').copy()


def stream_count(prefix):
    p = Path(str(prefix) + '.dram')
    if p.exists():
        size = p.stat().st_size
    else:
        size = 0
        with gzip.open(str(p) + '.gz', 'rb') as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
                size += len(block)
    assert size % 8 == 0
    return size // 8


def layout(gpu, ib):
    sd = copy.deepcopy(G.SYS[gpu])
    sd['ib'] = ib
    key = sd['layout'] + '_followup'
    R.LAYOUTS[key] = dict(G.LAY[sd['layout']], s=6 + ib)
    return sd, R.Layout(key)


def train(L, intra, ch, sd):
    bus, miss = C.cost_params(sd)
    R.cost.__defaults__ = (bus, miss)
    MS.cost3.__defaults__ = (bus, miss)
    candidates = R.sources(L, False)
    bank = R.search(L, R.sample(intra), candidates, 3)
    chan, sources, selected, _, _ = MS.chan_search(L, intra, ch, sd['nch'], 3, 'chan')
    return dict(bank=bank, bank_sources=candidates, chan=chan,
                chan_sources=sources, chan_selected=selected)


def transform(L, intra, ch, mapping, scheme):
    if scheme == 'xor':
        return R.apply_matrix(L, intra, mapping['bank'], mapping['bank_sources']), ch
    if scheme == 'chan':
        return intra, ch ^ MS.hash_bits(L, intra, mapping['chan'], mapping['chan_sources']).astype(ch.dtype)
    assert scheme == 'identity'
    return intra, ch


def resolved(sd):
    import ramulator as rm
    org, timing = getattr(rm.dram, sd['std'])(org_preset=sd['org'], timing_preset=sd['timing'], **sd['over']).resolve()
    return dict(org), dict(timing)


def matched_config(path, gpu, sd, mode):
    D.ds3_config(str(path), gpu, 'OPEN_PAGE')
    org, timing = resolved(sd)
    changes = {}
    # Match shared JEDEC timing fields to Ramulator's resolved preset. Different
    # command schedulers, HBM PC coupling and queue implementations remain explicit.
    if mode == 'matched':
        changes['tCK'] = sd['tck_ps'] / 1000.0
        aliases = dict(CL='nCL', CWL='nCWL', tRCDRD='nRCDRD', tRCDWR='nRCDWR',
                       tRP='nRP', tRAS='nRAS', tRFC='nRFC', tREFI='nREFI',
                       tRRD_S='nRRDS', tRRD_L='nRRDL', tWTR_S='nWTRS',
                       tWTR_L='nWTRL', tFAW='nFAW', tWR='nWR',
                       tCCD_S='nCCDS', tCCD_L='nCCDL', tRTP_S='nRTP', tRTP_L='nRTP')
        if 'nRTPL' in timing:
            aliases['tRTP_L'] = 'nRTPL'
        if 'nRTPS' in timing:
            aliases['tRTP_S'] = 'nRTPS'
        if gpu == 'a16':
            changes['bankgroup_enable'] = 'true'
        if 'nRFCab' in timing:
            aliases['tRFC'] = 'nRFCab'
        for key, source in aliases.items():
            if source in timing and timing[source] > 0:
                changes[key] = timing[source]
    # A unified transaction queue prevents the known split-write-queue deadlock.
    changes['unified_queue'] = 'True'
    s = path.read_text()
    applied = {}
    for key, value in changes.items():
        s, n = re.subn(r'(?m)^' + key + r'\s*=.*$', '{} = {}'.format(key, value), s)
        if n:
            applied[key] = value
        elif key == 'unified_queue':
            s = s.replace('[system]', '[system]\nunified_queue = True')
            applied[key] = value
    path.write_text(s)
    return dict(ram_org=org, ram_timing=timing, ds3_overrides=applied,
                ds3_config=s, remaining_differences=[
                    'Ramulator controller vs DRAMsim3 controller and queue implementation',
                    'HBM2: Ramulator couples two PCs in one controller; DS3 uses independent PC instances',
                    'Only timing fields available in both models are aligned',
                    'Global Ramulator feed vs independently fed DS3 channels unless feed=independent'])


def aggregate_ram(rows, sd, scheme):
    reads = sum(r['reads'] for r in rows)
    writes = sum(r['writes'] for r in rows)
    cycles = max(r['cycles'] for r in rows)
    tck = sd['tck_ps'] / 1000.
    return dict(sim='ram', scheme=scheme, policy='Open', reads=reads, writes=writes,
                cycles=cycles, time_us=cycles*tck/1000,
                GBps=(reads+writes)*32/(cycles*tck),
                hit_rate=sum(r['hit_rate']*(r['reads']+r['writes']) for r in rows)/max(reads+writes, 1),
                avg_read_latency_ns=sum(r['avg_read_latency_ns']*r['reads'] for r in rows)/max(reads, 1))


def experiment(arr, gpu, ib, mapping, out, procs=8, timing='matched', feed='independent', schemes=('identity','xor','chan'), page_map=None):
    sd, L = layout(gpu, ib)
    st = (arr & np.uint64(1)).astype(bool)
    byte = (arr >> np.uint64(1)) << np.uint64(5)
    if page_map is None:
        phys, pages = G.pack_pages(byte)
    else:
        unique, inv = np.unique(byte >> np.uint64(21), return_inverse=True)
        ranks = np.array([page_map[int(pg)] for pg in unique], dtype=np.uint64)
        phys = (ranks[inv] << np.uint64(21)) | (byte & np.uint64((1<<21)-1))
        pages = len(unique)
    intra, ch = L.to_intra(phys)
    if mapping is None:
        mapping = train(L, intra, ch, sd)
    result = dict(gpu=gpu, ib=ib, requests=len(arr), reads=int((~st).sum()), writes=int(st.sum()),
                  stream_sha256=hashlib.sha256(arr.tobytes()).hexdigest(), pages=pages,
                  timing=timing, feed=feed, mapping=mapping, results=[], checks=[])
    with tempfile.TemporaryDirectory(prefix='gpufollow_', dir=os.environ.get('TMPDIR', '/tmp')) as work:
        work = Path(work)
        cfg = work/'ds3.ini'
        result['configuration'] = matched_config(cfg, gpu, sd, timing)
        tck = sd['tck_ps']/1000. if timing == 'matched' else D.DS3[gpu]['tck']
        with mp.Pool(procs) as pool:
            for scheme in schemes:
                new, channels = transform(L, intra, ch, mapping, scheme)
                # Decoder roundtrip: channel/bank/row/column must survive reassembly.
                encoded = L.from_intra(new, channels)
                back, back_ch = L.to_intra(encoded)
                assert np.array_equal(back, new) and np.array_equal(back_ch, channels)
                if feed == 'global':
                    p = work/'ram.trc'
                    G.write_trace(str(p), st, encoded)
                    rr = pool.map(G.replay, [(sd, str(p), scheme, 'Open')])
                else:
                    tasks = []
                    for channel in range(sd['nch']):
                        selected = channels == channel
                        if not selected.any():
                            continue
                        p = work/('ram_{}.trc'.format(channel))
                        G.write_trace(str(p), st[selected], new[selected])
                        one = copy.deepcopy(sd)
                        one.update(nch=1, issue=1)
                        tasks.append((one, str(p), scheme, 'Open'))
                    rr = pool.map(G.replay, tasks)
                ram = aggregate_ram(rr, sd, scheme)
                paths, counts = D.ds3_files(gpu, L, new, channels, st, str(work), scheme, sd['nch'])
                # Invert the actual DS3 instance/address representation as well.
                all_bank, all_row, all_col = L.fields(new)
                for path in paths:
                    instance = int(Path(path).stem.rsplit('_i', 1)[1])
                    selected = (channels*2+(all_bank&1) == instance) if gpu == 'v100' else (channels == instance)
                    rec = np.fromfile(path, dtype=np.uint64)
                    addr = rec >> np.uint64(1)
                    bankbits, colbits = 4, (5 if gpu == 'v100' else 7)
                    b = ((addr >> np.uint64(5+colbits)) & np.uint64(15)).astype(np.int64)
                    if gpu == 'v100': b = b*2 + instance%2
                    row = (addr >> np.uint64(5+colbits+bankbits)).astype(np.int64)
                    col = ((addr >> np.uint64(5)) & np.uint64((1<<colbits)-1)).astype(np.int64)
                    eb, er, ec = all_bank[selected], all_row[selected], all_col[selected]
                    assert np.array_equal(b, eb) and np.array_equal(row, er) and np.array_equal(col, ec)
                    assert np.array_equal((rec&np.uint64(1)).astype(bool), st[selected])
                    expected = np.column_stack((eb//2 if gpu == 'v100' else eb, er, ec)).astype('<u4')
                    ep = str(path)+'.expected'
                    expected.tofile(ep)
                    decoder = Path.home()/'amcas-pro/tracing/bin/gpu_decode_check'
                    checked = subprocess.run([str(decoder),str(cfg),path,ep],capture_output=True,text=True,check=True)
                    print(checked.stdout.strip(),flush=True)
                    Path(ep).unlink()
                tasks = [(str(cfg), str(work/('o_'+str(i))), p) for i,p in enumerate(paths)]
                ds = pool.map(D.ds3_run, tasks)
                cycles = max(x['cycles'] for x in ds)
                accepted = sum(x['reqs'] for x in ds)
                ds3 = dict(sim='ds3', scheme=scheme, policy='OPEN_PAGE', cycles=cycles,
                           reads=sum(x['reads_done'] for x in ds), writes=sum(x['writes_done'] for x in ds),
                           accepted=accepted, time_us=cycles*tck/1000, GBps=accepted*32/(cycles*tck),
                           hit_rate=sum(x['hits'] for x in ds)/max(accepted,1),
                           missing_callbacks=sum(x['missing'] for x in ds), unissued=sum(x['unissued'] for x in ds),
                           instances=ds, instance_counts=counts.tolist())
                check = dict(scheme=scheme, decoder_roundtrip=True,
                             ram_accepted=ram['reads']+ram['writes'], ds3_accepted=accepted,
                             expected=len(arr), ds3_unissued=ds3['unissued'],
                             ds3_missing_callbacks=ds3['missing_callbacks'])
                result['results'].extend([ram,ds3]); result['checks'].append(check)
                save(out, result)
                assert check['ram_accepted'] == len(arr), check
                assert accepted == len(arr) and ds3['unissued'] == 0, check
                if not st.any():
                    assert ds3['missing_callbacks'] == 0 and ds3['reads'] == len(arr), check
                for p in paths: Path(p).unlink()
    result['validated'] = True
    save(out, result)
    return mapping


def page_metadata(prefix):
    p=Path(str(prefix)+'.dram')
    opener=open if p.exists() else gzip.open
    source=str(p) if p.exists() else str(p)+'.gz'
    total=0; mapping={}
    with opener(source,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            assert len(block)%8==0
            arr=np.frombuffer(block,dtype='<u8')
            # encoded record = (32-byte sector << 1) | write
            pages=arr >> np.uint64(17)
            unique,first=np.unique(pages,return_index=True)
            for pg in unique[np.argsort(first)]:
                k=int(pg)
                if k not in mapping: mapping[k]=len(mapping)
            total+=len(arr)
    return total,mapping


def locate(tag):
    for root in os.environ.get('GPU_STREAM_DIRS', '/tmp/'+os.environ['USER']+'/gt:/scratch/'+os.environ['USER']+'/gt').split(':'):
        p = Path(root)/tag
        if Path(str(p)+'.dram.gz').exists() or Path(str(p)+'.dram').exists(): return p
    raise FileNotFoundError(tag)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage', choices=['audit','compare','windows','moe-replay'])
    ap.add_argument('--root', required=True)
    ap.add_argument('--procs', type=int, default=8)
    ap.add_argument('--smoke', action='store_true')
    a=ap.parse_args(); root=Path(a.root); root.mkdir(parents=True,exist_ok=True)
    cells=[(gpu,fa) for gpu in ('v100','a16') for fa in ('auto','off')]
    if a.stage=='audit':
        for gpu in ('v100','a16'):
            sd,L=layout(gpu,2)
            sequences={
                'rowhits': np.zeros(256,dtype=np.uint64),
                'rowconflicts': np.arange(256,dtype=np.uint64)*np.uint64(1<<L.row_lo),
                'channels': np.arange(1024,dtype=np.uint64)*np.uint64(256),
                'mixed': np.arange(512,dtype=np.uint64)*np.uint64(32)}
            for name,addr in sequences.items():
                stream=(addr>>np.uint64(5))<<np.uint64(1)
                if name=='mixed': stream[::7]|=np.uint64(1)
                experiment(stream,gpu,2,None,root/('audit_'+gpu+'_'+name+'.json'),a.procs,schemes=('identity',))
            if not a.smoke:
                p=locate('q7_'+gpu+'_short_faauto_x13')
                experiment(window(p,0,100000),gpu,2,None,root/('audit_'+gpu+'_captured.json'),a.procs,schemes=('identity',))
        save(root/'audit.done.json',dict(validated=True,smoke=a.smoke))
    elif a.stage=='compare':
        assert json.loads((root/'audit.done.json').read_text())['validated']
        for gpu,fa in cells:
            p=locate('q7_'+gpu+'_32k_fa'+fa+'_x13')
            arr=window(p,0,8000000)
            for ib in (2,5):
                mapping=None
                for timing,feed in [('legacy','global'),('matched','global'),('matched','independent')]:
                    mapping=experiment(arr,gpu,ib,mapping,root/('compare_{}_{}_ib{}_{}_{}.json'.format(gpu,fa,ib,timing,feed)),a.procs,timing,feed)
                save(root/('mapping_{}_{}_ib{}.json'.format(gpu,fa,ib)),mapping)
        save(root/'compare.done.json',dict(validated=True))
    elif a.stage=='windows':
        assert json.loads((root/'compare.done.json').read_text())['validated']
        for gpu,fa in cells:
            p=locate('q7_'+gpu+'_32k_fa'+fa+'_x13')
            total,page_map=page_metadata(p); n=min(8000000,total)
            save(root/('placement_{}_{}.json'.format(gpu,fa)),dict(stream_requests=total,page_map=page_map))
            for ib in (2,5):
                mapping=json.loads((root/('mapping_{}_{}_ib{}.json'.format(gpu,fa,ib))).read_text())
                for label,start in [('begin',0),('middle',(total-n)//2),('end',total-n)]:
                    out=root/('window_{}_{}_ib{}_{}.json'.format(gpu,fa,ib,label))
                    experiment(window(p,start,n),gpu,ib,mapping,out,a.procs,page_map=page_map)
                    d=json.loads(out.read_text()); d.update(window_start=start,stream_requests=total,mapping_trained_on='beginning 8M'); save(out,d)
        save(root/'windows.done.json',dict(validated=True))
    else:
        gate=json.loads((root/'moe-smoke.done.json').read_text()); assert gate['validated']
        gate=json.loads((root/'compare.done.json').read_text()); assert gate['validated']
        import gpu_moe_followup as moe
        capture=moe.capture(root,full=True)
        for device,prefix in enumerate(capture['prefixes']):
            total=stream_count(prefix); n=min(total,8000000)
            for ib in (2,5):
                experiment(window(prefix,0,n),'v100',ib,None,root/('moe_dev{}_ib{}.json'.format(device,ib)),a.procs)
        save(root/'moe-replay.done.json',dict(validated=True,capture=capture,
             note='Separate device memories replayed independently; no inter-GPU transfer or token-time model'))


if __name__=='__main__': main()
