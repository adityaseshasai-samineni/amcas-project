#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
int main(int argc,char**argv){
 if(argc!=3)return 2;FILE*f=fopen(argv[1],"rb");if(!f){perror("input");return 2;}setvbuf(f,0,_IOFBF,1<<23);
 uint64_t inst[2]={0},lanes[2]={0},sectors[2]={0},bytes[2]={0},consumed=0,markers=0;uint8_t h[24];uint64_t a[32];
 for(;;){size_t z=fread(h,1,24,f);if(!z)break;if(z!=24){fprintf(stderr,"partial header\n");return 3;}consumed+=24;uint16_t op;uint32_t mask;memcpy(&op,h+18,2);memcpy(&mask,h+20,4);
  if(op==0xffff){markers++;if(mask){fprintf(stderr,"marker mask\n");return 3;}continue;}
  if(!(op&0x4000)){fprintf(stderr,"legacy record requires opcode table; refusing ambiguous parse\n");return 3;}
  unsigned n=__builtin_popcount(mask),w=op&255,st=(op>>8)&1;if(!w||w>16){fprintf(stderr,"bad width\n");return 3;}
  if(fread(a,8,n,f)!=n){fprintf(stderr,"partial addresses\n");return 3;}consumed+=8*n;inst[st]++;lanes[st]+=n;bytes[st]+=(uint64_t)n*w;
  uint64_t s[128];unsigned k=0;for(unsigned i=0;i<n;i++)for(uint64_t q=a[i]>>5;q<=(a[i]+w-1)>>5;q++){unsigned j=0;for(;j<k&&s[j]!=q;j++);if(j==k){if(k==128)return 3;s[k++]=q;}}sectors[st]+=k;
 }
 struct stat fs;if(fstat(fileno(f),&fs)||consumed!=(uint64_t)fs.st_size){fprintf(stderr,"size mismatch\n");return 3;}fclose(f);
 FILE*g=fopen(argv[2],"w");if(!g)return 2;
 fprintf(g,"{\n \"warp_load_instructions\":%llu,\n \"warp_store_instructions\":%llu,\n \"active_lane_load_references\":%llu,\n \"active_lane_store_references\":%llu,\n \"coalesced_load_32B_sectors\":%llu,\n \"coalesced_store_32B_sectors\":%llu,\n \"lane_load_requested_bytes\":%llu,\n \"lane_store_requested_bytes\":%llu,\n \"input_bytes\":%llu,\n \"launch_markers\":%llu\n}\n",(unsigned long long)inst[0],(unsigned long long)inst[1],(unsigned long long)lanes[0],(unsigned long long)lanes[1],(unsigned long long)sectors[0],(unsigned long long)sectors[1],(unsigned long long)bytes[0],(unsigned long long)bytes[1],(unsigned long long)consumed,(unsigned long long)markers);fclose(g);return 0;
}
