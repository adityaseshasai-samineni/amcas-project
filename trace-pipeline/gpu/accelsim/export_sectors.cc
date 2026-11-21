// Convert the AMCAS dram_t::push CSV to CANONICAL 32B burst events.
// IMPORTANT: missing dirty-sector masks mean offsets of variable-size writes
// are a convention, not reconstructed hardware dirty-byte addresses.
#include <zlib.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <inttypes.h>
int main(int argc,char**argv){
 if(argc!=3){fprintf(stderr,"usage: export_sectors INPUT.csv.gz OUTPUT.dram\n");return 2;}
 gzFile f=gzopen(argv[1],"rb");FILE*out=fopen(argv[2],"wb");if(!f||!out)return 3;
 gzbuffer(f,1<<20);char line[1024];if(!gzgets(f,line,sizeof line))return 4;
 if(strcmp(line,"sequence,core_cycle,partition,address_hex,write,bytes,access_type\n")){fprintf(stderr,"bad schema\n");return 5;}
 uint64_t n=0,reads=0,writes=0,rb=0,wb=0,seq,cyc,addr,lastcyc=0;unsigned part,w,b,type;
 while(gzgets(f,line,sizeof line)){
  if(sscanf(line,"%" SCNu64 ",%" SCNu64 ",%u,%" SCNx64 ",%u,%u,%u",&seq,&cyc,&part,&addr,&w,&b,&type)!=7){fprintf(stderr,"bad row at %" PRIu64 "\n",n);return 6;}
  if(seq!=n||cyc<lastcyc||w>1||b<32||b>128||b%32||addr%32||((addr&127)+b>128)){fprintf(stderr,"invariant seq=%" PRIu64 " expected=%" PRIu64 " addr=%" PRIx64 " bytes=%u\n",seq,n,addr,b);return 7;}
  lastcyc=cyc;n++; if(w){writes++;wb+=b;}else{reads++;rb+=b;}
  for(unsigned j=0;j<b;j+=32){uint64_t v=(((addr+j)>>5)<<1)|w;if(fwrite(&v,8,1,out)!=1)return 8;}
 }
 if(!gzeof(f)){fprintf(stderr,"gzip read failed\n");return 9;}gzclose(f);if(fclose(out))return 10;
 printf("{\"controller_packets\":%" PRIu64 ",\"read_packets\":%" PRIu64 ",\"write_packets\":%" PRIu64 ",\"read_bytes\":%" PRIu64 ",\"write_bytes\":%" PRIu64 ",\"canonical_sectors\":%" PRIu64 ",\"last_core_cycle\":%" PRIu64 ",\"write_offsets\":\"canonical_contiguous_missing_dirty_masks\",\"timestamps\":\"validated_but_not_in_binary_saturated_replay\"}\n",n,reads,writes,rb,wb,(rb+wb)/32,lastcyc);
}
