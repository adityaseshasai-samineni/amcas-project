#!/usr/bin/env python3
"""Keep model-internal consistency separate from hardware byte agreement."""
import pathlib,json,sys,time,tempfile,os
R=pathlib.Path.home()/'amcas-pro/results/gpu/campaign_20261120';name=sys.argv[1];O=R/'runs'/name

def load(p):
 if not p.is_file():return None
 for _ in range(10):
  try:return json.load(open(p))
  except json.JSONDecodeError:time.sleep(.05)
 return None
sim=load(O/'simulation.status.json');hw=load(O/'hardware.status.json');cap=load(O/'capture.status.json')
out=dict(case=name,capture_output_validated=bool(cap and cap['passed']),model_export_validated=bool(sim and sim['passed']),hardware_reference_validated=bool(hw and hw['passed']),scope='Model counter/export consistency does not establish exact hardware cache or timing behavior')
if sim and sim['passed']:
 out['model_read_bytes']=sum(c['conversion']['read_bytes'] for c in sim['contexts']);out['model_write_bytes']=sum(c['conversion']['write_bytes'] for c in sim['contexts']);out['model_kernels']=sim['kernels_total']
if hw and hw['passed']:
 out.update(hardware_read_bytes=int(hw['read_bytes']),hardware_write_bytes=int(hw['write_bytes']),hardware_kernels=hw['kernels'])
if out['model_export_validated'] and out['hardware_reference_validated']:
 assert out['model_kernels']==out['hardware_kernels'];out['read_relative_error']=out['model_read_bytes']/out['hardware_read_bytes']-1 if out['hardware_read_bytes'] else None;out['write_relative_error']=out['model_write_bytes']/out['hardware_write_bytes']-1 if out['hardware_write_bytes'] else None
 out['hardware_agreement']='Byte comparison available; report read/write deviations separately. Not an exact physical DRAM trace or native token-time validation.'
with tempfile.NamedTemporaryFile(mode='w',dir=O,prefix='validation.',suffix='.tmp',delete=False) as f:json.dump(out,f,indent=2);tmp=f.name
os.replace(tmp,O/'validation.json');print(json.dumps(out))
