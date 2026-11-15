#!/usr/bin/env python3
"""Two-V100 smoke and full-step capture; separate per-context caches."""
import argparse,csv,gzip,hashlib,json,os,re,shutil,subprocess
from pathlib import Path
from gpu_followup import save,stream_count
W=Path.home()/'amcas-pro/tracing'
G=Path.home()/'amcas-pro/gpu'
E=W/'mamba/envs/gem5'
MODEL=W/'models/Qwen3-30B-A3B-Q4_K_M.gguf'
def launches(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        m=re.search(r'CTX (0x[0-9a-f]+).*Kernel name (.*?) - grid launch id (\d+)',line)
        if m: rows.append((int(m[3]),m[1],m[2]))
    rows.sort()
    assert len({r[0] for r in rows})==len(rows),'launch IDs not globally unique'
    return rows
def environment(prefix,start,end,traced=True):
    env=os.environ.copy()
    env.update(PATH=str(W/'cudatools')+':'+env['PATH'],
        LD_LIBRARY_PATH=':'.join(map(str,[W/'cudalibs',E/'lib'])),
        GGML_CUDA_DISABLE_GRAPHS='1',CUDA_MODULE_LOADING='EAGER',NOBANNER='1',
        AMCAS_TENSORS=str(prefix)+'.tensors.tsv')
    if traced:
        env.update(LD_PRELOAD=str(W/'nvbit/v1.7.7.3/nvbit_release_x86_64/tools/gputrace2-sm70/gputrace2-sm70.so')+':'+str(W/'lib/libgpuattr.so'),
            GPUTRACE_OUT=str(prefix),GPUTRACE_START=str(start),GPUTRACE_END=str(end))
    else: env.pop('LD_PRELOAD',None)
    return env
def command():
    return [str(W/'llama.cpp/build-cuda/bin/llama-completion'),'-m',str(MODEL),
        '-ngl','99','-n','3','--temp','0','--seed','1','--no-warmup','-no-cnv',
        '-t','4','-c','4096','-fa','auto','-p','The capital of France is']
def run(prefix,start,end,traced=True):
    with open(str(prefix)+'.stdout','wb') as out,open(str(prefix)+'.stderr','wb') as err:
        subprocess.run(command(),env=environment(prefix,start,end,traced),
            stdout=out,stderr=err,check=True,timeout=3600)
    return Path(str(prefix)+'.stdout').read_bytes()
def capture(root,full=False):
    root=Path(root)
    if full: assert json.loads((root/'moe-smoke.done.json').read_text())['validated']
    subprocess.run(['nvidia-smi','-L'],check=True)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES')
    assert visible is not None and len(visible.split(','))==2,('need --gres=gpu:2',visible)
    scratch=Path('/scratch')/os.environ['USER']/'gpu_followup_20261115'
    scratch.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(scratch).free>12*1024**3,'need 12 GiB scratch headroom'
    mode='full' if full else 'smoke'
    native=run(root/(mode+'_native'),0,0,False)
    census=root/(mode+'_census')
    assert native.strip() in run(census,999999999,999999999),'native/census output tokens differ'
    rows=launches(str(census)+'.launch.txt')
    start,end=map(int,subprocess.check_output([str(E/'bin/python'),str(G/'find_window.py'),str(census)+'.launch.txt'],text=True).split())
    chosen=[r for r in rows if start<=r[0]<end]
    contexts=list(dict.fromkeys(r[1] for r in rows))
    assert len(contexts)==2 and set(r[1] for r in chosen)==set(contexts),'decode window must cover both GPUs'
    gf=scratch/'gpu_filter2'
    subprocess.run([str(E/'bin/gcc'),'-O2','-o',str(gf),str(G/'gpu_filter2.c')],check=True)
    results=[]; prefixes=[]
    windows=[(start,end,None)] if full else [(next(r[0] for r in chosen if r[1]==ctx),next(r[0] for r in chosen if r[1]==ctx)+4,dev) for dev,ctx in enumerate(contexts)]
    for index,(lo,hi,target) in enumerate(windows):
        pre=scratch/('moe_'+mode+'_'+str(index))
        fifo=Path(str(pre)+'.bin')
        if fifo.exists(): fifo.unlink()
        os.mkfifo(fifo)
        active=list(range(2)) if full else [target]
        procs=[]; handles=[]; filters=[]
        try:
            inputs=[]
            for dev in active:
                inp=scratch/('input_'+mode+'_'+str(index)+'_dev'+str(dev))
                out=scratch/('moe_'+mode+'_dev'+str(dev))
                # Stable census file lets the filter parse device membership
                # before it blocks reading live records from its private FIFO.
                shutil.copyfile(str(census)+'.launch.txt',str(inp)+'.launch.txt')
                fi=Path(str(inp)+'.bin')
                if fi.exists(): fi.unlink()
                os.mkfifo(fi); inputs.append(fi)
                for ext in ('.dram','.dramlaunch'):
                    f=Path(str(out)+ext)
                    if f.exists(): f.unlink()
                    os.mkfifo(f)
                    sink=open(str(f)+'.gz','wb'); handles.append(sink)
                    procs.append(subprocess.Popen(['bash','-c','exec gzip -1 < "$1"','gzip',str(f)],stdout=sink))
                log=open(str(out)+'.filter.log','wb'); handles.append(log)
                procs.append(subprocess.Popen([str(gf),str(inp),str(out),'--var','x13:hash=113,dev='+str(dev),'--emit','x13'],stdout=log,stderr=log))
                filters.append((dev,inp,out))
            log=open(str(pre)+'.tee.log','wb'); handles.append(log)
            procs.append(subprocess.Popen(['bash','-o','pipefail','-c',
                'input=$1; shift; cat "$input" | tee "$@" > /dev/null',
                'stream',str(fifo)]+list(map(str,inputs)),stderr=log))
            assert native.strip() in run(pre,lo,hi),'native/traced output tokens differ'
            for proc in procs:
                assert proc.wait(timeout=300)==0,'stream/filter/compressor failed'
            actual=launches(str(pre)+'.launch.txt')
            assert [(r[0],r[2]) for r in actual]==[(r[0],r[2]) for r in rows],'launch census changed'
            actual_ctx=list(dict.fromkeys(r[1] for r in actual))
            assert len(actual_ctx)==2
            for dev,inp,out in filters:
                expected=[r[0] for r in rows if r[1]==contexts[dev]]
                assert [r[0] for r in actual if r[1]==actual_ctx[dev]]==expected,'device partition changed'
                with open(str(out)+'.perk.tsv') as f: perf=list(csv.DictReader(f,delimiter='\t'))
                assert perf,'no memory records from selected device'
                assert all(lo<=int(r['launch'])<hi and int(r['launch']) in expected for r in perf)
                reads=sum(int(r['x13_miss']) for r in perf)
                writes=sum(int(r['x13_wr']) for r in perf)
                n=stream_count(out)
                assert reads>0 and n==reads+writes,'stream/per-launch count mismatch'
                with gzip.open(str(out)+'.dramlaunch.gz','rb') as f:
                    size=sum(len(b) for b in iter(lambda:f.read(1024*1024),b''))
                assert size==n*4,'missing launch records'
                for ext in ('.perk.tsv','.filter.log'):
                    shutil.copyfile(str(out)+ext,root/(out.name+ext))
                results.append(dict(device=dev,prefix=str(out),requests=n,reads=reads,writes=writes,launch_start=lo,launch_end=hi))
                prefixes.append(str(out))
                Path(str(inp)+'.bin').unlink(missing_ok=True)
                for ext in ('.dram','.dramlaunch'): Path(str(out)+ext).unlink(missing_ok=True)
        finally:
            for proc in procs:
                if proc.poll() is None: proc.kill(); proc.wait()
            for h in handles: h.close()
            fifo.unlink(missing_ok=True)
        for ext in ('.launch.txt','.stdout','.stderr'):
            shutil.copyfile(str(pre)+ext,root/(pre.name+ext))
    manifest=dict(validated=True,full_step=full,visible_devices=visible,prefixes=prefixes,streams=results,
        native_stdout_sha256=hashlib.sha256(native).hexdigest(),decode_start=start,decode_end=end,
        limitations=['Device ordinal inferred from stable CUDA-context launch partition',
            'Private L1/CTA placement and hashed L2 remain models',
            'Peer transfers, local/shared traffic and physical placement not measured'])
    save(root/('moe-full-capture.json' if full else 'moe-smoke.done.json'),manifest)
    return manifest
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    capture(ap.parse_args().root)
