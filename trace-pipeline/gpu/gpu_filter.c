// Sector-level cache filter for gputrace output: per-SM L1 + shared L2 -> DRAM request stream.
// usage: gpu_filter PREFIX OUT_PREFIX [--sm N] [--l1kb K] [--l1ways W] [--l2mb M] [--l2ways W]
//   reads PREFIX.bin, PREFIX.opcodes.txt, PREFIX.launch.txt
//   writes OUT.dram (u64 per request: (byte_addr>>5)<<1 | is_write) and OUT.dramlaunch (u32 launch id per request)
// Model (documented assumptions, see context.md): warp instruction -> distinct 32 B sectors.
//  L1: per SM, 128 B lines with 4 sector-valid bits, LRU, write-through / no write-allocate.
//      CTA -> SM = (linear CTA id) % nSM (the hardware schedule is not recorded).
//  L2: shared, 128 B lines with sector valid+dirty bits, LRU, write-back, write-validate (a store allocates the
//      sector without a DRAM read), read miss fetches only the missing 32 B sector. Cold at window start.
//  Dirty sectors are written to DRAM on eviction; at the end the remaining dirty sectors are NOT flushed.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
typedef struct { uint64_t tag; uint8_t valid, dirty; uint32_t lru; } Line;
typedef struct { Line* l; uint32_t sets, ways; uint64_t clk; } Cache;
static void cinit(Cache*c, uint64_t bytes, uint32_t ways){ c->ways=ways; c->sets=bytes/128/ways; c->l=calloc((size_t)c->sets*ways,sizeof(Line)); for(size_t i=0;i<(size_t)c->sets*ways;i++)c->l[i].tag=~0ull; c->clk=0; }
static Line* cfind(Cache*c,uint64_t line){ Line*s=&c->l[(size_t)(line%c->sets)*c->ways]; for(uint32_t w=0;w<c->ways;w++) if(s[w].tag==line){ s[w].lru=++c->clk; return &s[w]; } return NULL; }
static Line* cvictim(Cache*c,uint64_t line){ Line*s=&c->l[(size_t)(line%c->sets)*c->ways]; Line*v=&s[0]; for(uint32_t w=0;w<c->ways;w++){ if(s[w].tag==~0ull){v=&s[w];break;} if(s[w].lru<v->lru)v=&s[w]; } return v; }
static FILE *fd,*fl; static uint32_t curlaunch, cursm; static uint64_t nrd,nwr;
static void emit(uint64_t sector,int w){ uint64_t r=(sector<<1)|w; fwrite(&r,8,1,fd); {uint32_t v=curlaunch|(cursm<<20); fwrite(&v,4,1,fl);} if(w)nwr++; else nrd++; }
static void evict(Line*v){ if(v->tag==~0ull)return; for(int s=0;s<4;s++) if(v->dirty>>s&1) emit(v->tag*4+s,1); }
static Cache L2; static Cache* L1; static uint64_t l2rd,l2wr,l1rd;
static int fetch64=0;
static void l2access(uint64_t sec,int st){
  uint64_t line=sec>>2; int s=sec&3; Line*x=cfind(&L2,line);
  if(!x){ x=cvictim(&L2,line); evict(x); x->tag=line; x->valid=0; x->dirty=0; x->lru=++L2.clk; }
  if(st){ x->valid|=1<<s; x->dirty|=1<<s; l2wr++; }
  else { l2rd++; if(!(x->valid>>s&1)){ x->valid|=1<<s; emit(sec,0); if(fetch64){ int o=s^1; if(!(x->valid>>o&1)){ x->valid|=1<<o; emit(sec^1,0);} } } }
}
static void l1access(int sm,uint64_t sec,int st){
  Cache*c=&L1[sm]; uint64_t line=sec>>2; int s=sec&3; Line*x=cfind(c,line);
  if(st){ if(x) x->valid|=1<<s; l2access(sec,1); return; }   // write-through, update if present
  l1rd++;
  if(x && (x->valid>>s&1)) return;                              // L1 hit
  if(!x){ x=cvictim(c,line); x->tag=line; x->valid=0; x->lru=++c->clk; }
  x->valid|=1<<s; l2access(sec,0);
}
#define MAXL (1<<20)
static uint32_t gx[MAXL],gy[MAXL];
int main(int argc,char**argv){
  int nsm=80,l1kb=128,l1w=4,l2w=16; double l2mb=6;
  for(int i=3;i<argc;i+=2){ if(!strcmp(argv[i-1],"--sm"))nsm=atoi(argv[i]); else if(!strcmp(argv[i-1],"--l1kb"))l1kb=atoi(argv[i]); else if(!strcmp(argv[i-1],"--l1ways"))l1w=atoi(argv[i]); else if(!strcmp(argv[i-1],"--l2mb"))l2mb=atof(argv[i]); else if(!strcmp(argv[i-1],"--l2ways"))l2w=atoi(argv[i]); else if(!strcmp(argv[i-1],"--fetch"))fetch64=(atoi(argv[i])==64); }
  char p[512],nm[256]; snprintf(p,512,"%s.opcodes.txt",argv[1]); FILE*fo=fopen(p,"r"); int width[4096]={0},isst[4096]={0},id;
  while(fo&&fscanf(fo,"%d %255s",&id,nm)==2){ isst[id]=(nm[0]=='S'||nm[0]=='R'||strncmp(nm,"ATOM",4)==0);
    int w=4; if(strstr(nm,".128"))w=16; else if(strstr(nm,".64"))w=8; else if(strstr(nm,".U8")||strstr(nm,".S8"))w=1; else if(strstr(nm,".U16")||strstr(nm,".S16"))w=2; width[id]=w; }
  snprintf(p,512,"%s.launch.txt",argv[1]); FILE*fg=fopen(p,"r"); char *ln=malloc(1<<16);
  while(fg&&fgets(ln,1<<16,fg)){ char*a=strstr(ln,"grid launch id "); char*b=strstr(ln,"grid size "); if(!a||!b)continue; long l=atol(a+15); unsigned x,y,z; if(sscanf(b+10,"%u,%u,%u",&x,&y,&z)==3&&l<MAXL){gx[l]=x;gy[l]=y;} }
  cinit(&L2,(uint64_t)(l2mb*1048576),l2w); L1=malloc(sizeof(Cache)*nsm); for(int i=0;i<nsm;i++)cinit(&L1[i],(uint64_t)l1kb*1024,l1w);
  snprintf(p,512,"%s.bin",argv[1]); FILE*f=fopen(p,"rb"); setvbuf(f,0,_IOFBF,1<<24);
  snprintf(p,512,"%s.dram",argv[2]); fd=fopen(p,"wb"); snprintf(p,512,"%s.dramlaunch",argv[2]); fl=fopen(p,"wb");
  uint64_t a[32]; uint64_t wi=0;
  for(;;){ uint8_t hd[24]; if(fread(hd,1,24,f)!=24)break; uint64_t la=*(uint64_t*)hd; uint32_t cx=*(uint32_t*)(hd+8); uint16_t cy=*(uint16_t*)(hd+12),cz=*(uint16_t*)(hd+14);
    uint16_t o=*(uint16_t*)(hd+18); uint32_t mask=*(uint32_t*)(hd+20); int n=__builtin_popcount(mask);
    if(o==0xFFFF){ if(la<MAXL){gx[la]=cx;gy[la]=cy;} continue; } if(fread(a,8,n,f)!=(size_t)n)break;
    curlaunch=(uint32_t)la; uint64_t gxx=la<MAXL&&gx[la]?gx[la]:1, gyy=la<MAXL&&gy[la]?gy[la]:1; uint64_t lin=cx+gxx*(cy+gyy*(uint64_t)cz); int sm=lin%nsm; cursm=sm;
    int w,st; if(o&0x4000){w=o&0xFF;st=(o>>8)&1;} else {w=width[o];st=isst[o];} uint64_t sec[128]; int k=0;
    for(int i=0;i<n;i++){ uint64_t s0=a[i]>>5,s1=(a[i]+w-1)>>5; for(uint64_t s=s0;s<=s1;s++){int d=0;for(int j=0;j<k;j++)if(sec[j]==s){d=1;break;} if(!d&&k<128)sec[k++]=s;} }
    for(int j=0;j<k;j++) l1access(sm,sec[j],st); wi++; }
  fclose(fd); fclose(fl);
  fprintf(stderr,"warp instrs %lu; L1 load sectors %lu; L2 reads %lu writes %lu; DRAM reads %lu (%.1f MB) writes %lu (%.1f MB)\n",wi,l1rd,l2rd,l2wr,nrd,nrd*32/1e6,nwr,nwr*32/1e6);
  return 0; }
