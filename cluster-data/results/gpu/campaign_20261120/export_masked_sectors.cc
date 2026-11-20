// Convert masked AMCAS controller packets to model32B sectors.
// Sector positions come from mem_fetch masks; ascending within-packet order is explicit.
#include <zlib.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <inttypes.h>
#include <unordered_map>
#include <string>
int main(int argc,char**argv){
 if(argc!=3){fprintf(stderr,"usage: export_sectors INPUT.csv.gz OUTPUT.dram\n");return 2;}
 gzFile f=gzopen(argv[1],"rb");FILE*out=fopen(argv[2],"wb");if(!f||!out)return 3;
 FILE*packed=fopen((std::string(argv[2])+".packed").c_str(),"wb"); if(!packed)return 12;
 std::unordered_map<uint64_t,uint64_t> pages;
 gzbuffer(f,1<<20);char line[1024];if(!gzgets(f,line,sizeof line))return 4;
 if(strcmp(line,"sequence,core_cycle,partition,address_hex,write,bytes,access_type,sector_mask\n")){fprintf(stderr,"bad schema\n");return 5;}
 uint64_t n=0,reads=0,writes=0,rb=0,wb=0,seq,cyc,addr,lastcyc=0;unsigned part,w,b,type,mask;
 while(gzgets(f,line,sizeof line)){
  if(sscanf(line,"%" SCNu64 ",%" SCNu64 ",%u,%" SCNx64 ",%u,%u,%u,%u",&seq,&cyc,&part,&addr,&w,&b,&type,&mask)!=8){fprintf(stderr,"bad row at %" PRIu64 "\n",n);return 6;}
  if(seq!=n||cyc<lastcyc||w>1||b<32||b>128||b%32||addr%32||((addr&127)+b>128)){fprintf(stderr,"invariant seq=%" PRIu64 " expected=%" PRIu64 " addr=%" PRIx64 " bytes=%u\n",seq,n,addr,b);return 7;}
  lastcyc=cyc;n++; if(w){writes++;wb+=b;}else{reads++;rb+=b;}
  if(mask>15||__builtin_popcount(mask)!=int(b/32)){fprintf(stderr,"sector mask/bytes mismatch seq=%" PRIu64 " mask=%u bytes=%u\n",seq,mask,b);return 11;}
  for(unsigned j=0;j<4;j++)if(mask&(1u<<j)){uint64_t sector=(addr&~uint64_t(127))+j*32;uint64_t v=((sector>>5)<<1)|w;if(fwrite(&v,8,1,out)!=1)return 8; auto key=sector>>21;auto it=pages.find(key);if(it==pages.end())it=pages.emplace(key,pages.size()).first;uint64_t pa=(it->second<<21)|(sector&((uint64_t(1)<<21)-1));if(fwrite(&pa,8,1,packed)!=1)return 13;}
 }
 if(!gzeof(f)){fprintf(stderr,"gzip read failed\n");return 9;}gzclose(f);if(fclose(out)||fclose(packed))return 10;
 printf("{\"controller_packets\":%" PRIu64 ",\"read_packets\":%" PRIu64 ",\"write_packets\":%" PRIu64 ",\"read_bytes\":%" PRIu64 ",\"write_bytes\":%" PRIu64 ",\"sectors\":%" PRIu64 ",\"last_core_cycle\":%" PRIu64 ",\"write_offsets\":\"exported_dirty_sector_mask_ascending_within_packet\",\"timestamps\":\"validated_but_not_in_binary_saturated_replay\"}\n",n,reads,writes,rb,wb,(rb+wb)/32,lastcyc);
}
