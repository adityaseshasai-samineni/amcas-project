
#include <cstdio>
#include <cstdint>
#include "configuration.h"
int main(int argc,char**argv){
 if(argc!=4)return 2;
 dramsim3::Config c(argv[1],"/tmp");
 FILE*f=fopen(argv[2],"rb"),*e=fopen(argv[3],"rb");
 if(!f||!e)return 3;
 uint64_t rec,n=0;uint32_t ex[3];
 while(fread(&rec,8,1,f)==1){
  if(fread(ex,4,3,e)!=3)return 4;
  auto a=c.AddressMapping(rec>>1);
  if(a.channel||a.rank||uint32_t(a.bank*c.bankgroups+a.bankgroup)!=ex[0]||
     uint32_t(a.row)!=ex[1]||uint32_t(a.column)!=ex[2]){
   fprintf(stderr,"decoder mismatch at %llu: got %d,%d,%d expected %u,%u,%u\n",
    (unsigned long long)n,a.bank*c.bankgroups+a.bankgroup,a.row,a.column,ex[0],ex[1],ex[2]);return 5;
  }n++;
 }
 if(fread(ex,4,1,e))return 6;
 printf("DECODE_VALIDATED %llu shift=%d columns=%d banks=%d ranks=%d\n",
   (unsigned long long)n,c.shift_bits,c.columns,c.banks,c.ranks);
 return 0;
}
