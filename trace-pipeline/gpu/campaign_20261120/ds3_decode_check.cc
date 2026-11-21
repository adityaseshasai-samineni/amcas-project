#include <cstdio>
#include <cstdint>
#include "configuration.h"
int main(int argc,char**argv){
 if(argc!=4)return 2;dramsim3::Config c(argv[1],"/tmp");FILE*f=fopen(argv[2],"rb"),*e=fopen(argv[3],"rb");if(!f||!e)return 3;
 uint64_t rec,n=0;uint32_t ex[4];
 while(fread(&rec,8,1,f)==1){if(fread(ex,4,4,e)!=4)return 4;auto a=c.AddressMapping(rec>>1);
 if(uint32_t(a.channel)!=ex[0]||a.rank||uint32_t(a.bank*c.bankgroups+a.bankgroup)!=ex[1]||uint32_t(a.row)!=ex[2]||uint32_t(a.column)!=ex[3]){fprintf(stderr,"decoder mismatch at %llu\n",(unsigned long long)n);return 5;}n++;}
 if(fread(ex,4,1,e))return 6;printf("DECODE_VALIDATED %llu channels=%d tx_shift=%d\n",(unsigned long long)n,c.channels,c.shift_bits);
}
