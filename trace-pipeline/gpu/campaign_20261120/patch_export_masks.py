#!/usr/bin/env python3
import pathlib,sys
p=pathlib.Path(sys.argv[1]);s=p.read_text()
if 'AMCAS_DRAM_EXPORT_MASKS' in s:raise SystemExit(0)
s=s.replace('static bool amcas_initialized = false;', 'static bool amcas_initialized = false;\n  static bool amcas_masks = false;')
s=s.replace('const char *path = getenv("AMCAS_DRAM_EXPORT");','const char *path = getenv("AMCAS_DRAM_EXPORT");\n    amcas_masks = getenv("AMCAS_DRAM_EXPORT_MASKS") != nullptr;')
old='fprintf(amcas_out,"sequence,core_cycle,partition,address_hex,write,bytes,access_type\\n");'
new='fprintf(amcas_out, amcas_masks ? "sequence,core_cycle,partition,address_hex,write,bytes,access_type,sector_mask\\n" : "sequence,core_cycle,partition,address_hex,write,bytes,access_type\\n");'
assert s.count(old)==1;s=s.replace(old,new)
old='''  if (amcas_out) fprintf(amcas_out,"%llu,%llu,%u,0x%llx,%u,%u,%u\\n",
      amcas_seq++, (unsigned long long)(m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle),
      id, (unsigned long long)data->get_addr(), data->get_is_write() ? 1 : 0,
      data->get_data_size(), (unsigned)data->get_access_type());'''
new='''  if (amcas_out) {
    fprintf(amcas_out,"%llu,%llu,%u,0x%llx,%u,%u,%u",
      amcas_seq++, (unsigned long long)(m_gpu->gpu_sim_cycle + m_gpu->gpu_tot_sim_cycle),
      id, (unsigned long long)data->get_addr(), data->get_is_write() ? 1 : 0,
      data->get_data_size(), (unsigned)data->get_access_type());
    if (amcas_masks) fprintf(amcas_out,",%lu", data->get_access_sector_mask().to_ulong());
    fprintf(amcas_out,"\\n");
  }'''
assert s.count(old)==1;s=s.replace(old,new);p.write_text(s)
