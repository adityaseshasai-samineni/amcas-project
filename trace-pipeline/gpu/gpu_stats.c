// Parse gputrace .bin (see gputrace_patch.py): per-warp-instruction coalescing into 32 B sectors.
// usage: gpu_stats PREFIX      (reads PREFIX.bin, PREFIX.opcodes.txt)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
static int cmp(const void*a,const void*b){uint64_t x=*(uint64_t*)a,y=*(uint64_t*)b;return x<y?-1:x>y;}
int main(int argc,char**argv){
  char p[512]; snprintf(p,512,"%s.opcodes.txt",argv[1]); FILE*fo=fopen(p,"r");
  int width[4096]={0}, isst[4096]={0}; char nm[256]; int id;
  while(fscanf(fo,"%d %255s",&id,nm)==2){ isst[id]=(nm[0]=='S'||nm[0]=='R'||strncmp(nm,"ATOM",4)==0);
    int w=4; if(strstr(nm,".128"))w=16; else if(strstr(nm,".64"))w=8; else if(strstr(nm,".U8")||strstr(nm,".S8"))w=1; else if(strstr(nm,".U16")||strstr(nm,".S16"))w=2; width[id]=w; }
  snprintf(p,512,"%s.bin",argv[1]); FILE*f=fopen(p,"rb");
  size_t cap=1<<26,nl=0,ns=0; uint64_t*L=malloc(cap*8),*S=malloc(cap*8);
  uint64_t wl=0,ws=0,secl=0,secs=0,thr=0; static uint64_t pl[1<<20],ps[1<<20]; static uint64_t pwl[1<<20]; uint64_t maxl=0; uint8_t h[24]; uint64_t a[32];
  while(fread(h,1,20,f)==20){ uint16_t op=*(uint16_t*)(h+18); uint32_t m=*(uint32_t*)(h+16);
    // layout: u64 launch | u32 cx | u16 cy | u16 cz | u16 warp | u16 op | u32 mask  = 24 bytes
    (void)op;(void)m; break; }
  rewind(f);
  for(;;){ uint8_t hd[24]; if(fread(hd,1,24,f)!=24)break; uint16_t o=*(uint16_t*)(hd+18); uint32_t mask=*(uint32_t*)(hd+20);
    int n=__builtin_popcount(mask); if(fread(a,8,n,f)!=(size_t)n)break; int w=width[o]; int st=isst[o]; thr+=n;
    uint64_t sec[128]; int k=0;
    for(int i=0;i<n;i++){ uint64_t s0=a[i]>>5,s1=(a[i]+w-1)>>5; for(uint64_t s=s0;s<=s1;s++){int d=0;for(int j=0;j<k;j++)if(sec[j]==s){d=1;break;} if(!d&&k<128)sec[k++]=s;} }
    { uint64_t li=*(uint64_t*)hd; if(li<(1<<20)){ if(li>maxl)maxl=li; if(st)ps[li]+=k; else {pl[li]+=k;pwl[li]++;} } }
    uint64_t**T=st?&S:&L; size_t*c=st?&ns:&nl; if(st){ws++;secs+=k;}else{wl++;secl+=k;}
    if(*c+k>cap){ cap*=2; L=realloc(L,cap*8); S=realloc(S,cap*8);} for(int j=0;j<k;j++)(*T)[(*c)++]=sec[j]; }
  qsort(L,nl,8,cmp); qsort(S,ns,8,cmp); size_t ul=0,us=0; for(size_t i=0;i<nl;i++)if(!i||L[i]!=L[i-1])ul++; for(size_t i=0;i<ns;i++)if(!i||S[i]!=S[i-1])us++;
  printf("warp load instrs %lu (sectors %lu, %.2f/instr), warp store instrs %lu (sectors %lu)\n",wl,secl,wl?(double)secl/wl:0,ws,secs);
  printf("distinct load sectors %zu = %.1f MB; distinct store sectors %zu = %.1f MB; total sector requests load %.1f MB\n",ul,ul*32/1e6,us,us*32/1e6,secl*32/1e6);
  if(argc>2){FILE*g=fopen(argv[2],"w"); for(uint64_t i=0;i<=maxl;i++) if(pl[i]||ps[i]) fprintf(g,"%lu %lu %lu %lu\n",i,pwl[i],pl[i],ps[i]); fclose(g);}
  return 0; }
