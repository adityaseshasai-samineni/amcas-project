import sys; sys.path.insert(0,'/home/adityaseshasai.s/amcas-pro/tools')
import dramtech_cfg as C
for k,v in C.SYSTEMS.items():
    print(k, C.cost_params(v))
import ramulator
for k in ('lpddr6x32','hbm3x16','hbm4x32'):
    sd=C.SYSTEMS[k]; org,t=getattr(ramulator.dram,sd['std'])(org_preset=sd['org'],timing_preset=sd['timing'],**sd['over']).resolve()
    print(k,{a:t[a] for a in t if a in('nBL','nBL_min','nRAS','nRP','nRC','nRCDRD','nRCDr','nCL','nRL','nCCDS','nCCDL','tCK_ps','nRFC','nREFI')})
