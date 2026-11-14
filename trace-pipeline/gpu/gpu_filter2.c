// Multi-variant sector-level cache filter for gputrace output (L2-model study, l2fix-owner).
// usage: gpu_filter2 PREFIX OUT_PREFIX --var "name:key=val,key=val" [--var ...] [--emit name] [--batch N]
//   one pass over PREFIX.bin evaluates every variant (own L1s + L2); per-launch counters go to OUT.perk.tsv:
//   launch <TAB> then for each variant: l2rd, l2miss(=DRAM rd sectors), dram wr sectors
//   --emit NAME also writes OUT.dram / OUT.dramlaunch (same formats as gpu_filter.c) for that variant.
// keys (defaults): sm=80 l1kb=128 l1w=4 l2mb=6 ways=16 hash=0 pol=lru reorder=0 l2fetch=32 smmap=lin
//   hash: 0 line%sets, 1 xor-fold (line^line>>11^line>>22)%sets, 2 pow2 sets xor-fold (sets rounded down to 2^n)
//   pol : lru | rand | srrip | fifo      reorder: 0 trace order, 1 per-(cta,warp) round robin within a batch,
//   2 per-cta round robin     smmap: lin (cta%sm) | dyn (CTA goes to the SM whose L1 was least recently assigned)
//   l1kb=0 bypasses L1.   nclean: 1 = loads from LDG.*CONSTANT are not distinguished (not available in streamed format)
// The model is otherwise the one of gpu_filter.c: 128 B lines, 32 B sectors, write-back L2, write-validate.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
typedef struct { uint64_t tag; uint8_t valid, dirty, rrpv; uint32_t lru; } Line;
typedef struct { Line* l; uint32_t sets, ways; uint64_t clk; int hash, pol; uint64_t rng; } Cache;
typedef struct {
  char name[64]; int nsm, l1kb, l1w, reorder, smmap, fetch64; double l2mb; int ways, hash, pol;
  Cache L2; Cache* L1; int emit;
  uint64_t l2rd, l2wr, l1rd, nrd, nwr;           // totals
  uint64_t *p_l2rd, *p_miss, *p_wr;               // per launch
  uint64_t *lastuse; // dyn smmap
} Model;
#define MAXL (1<<20)
static uint32_t gx[MAXL], gy[MAXL];
static void cinit(Cache*c, uint64_t bytes, uint32_t ways, int hash, int pol){
  c->ways=ways; c->hash=hash; c->pol=pol; c->rng=88172645463325252ull;
  uint64_t sets=bytes/128/ways; if(hash==2){ uint64_t p=1; while(p*2<=sets)p*=2; sets=p; }
  c->sets=sets; c->l=calloc((size_t)sets*ways,sizeof(Line)); for(size_t i=0;i<(size_t)sets*ways;i++)c->l[i].tag=~0ull; c->clk=0; }
static inline uint64_t hmix(uint64_t,int);
static inline Line* setof(Cache*c,uint64_t line){ uint64_t h=line; if(c->hash>=3) h=hmix(line,c->hash); else if(c->hash==1) h=line^(line>>11)^(line>>22); else if(c->hash==2) h=line^(line>>12)^(line>>24);
  return &c->l[(size_t)(h%c->sets)*c->ways]; }
static inline uint64_t hmix(uint64_t line,int hash){ if(hash>=100&&hash<200) return line^(line>>(hash-100)); if(hash>=200&&hash<300){ int k=hash-200; return line^(line>>k)^(line>>(2*k)); } if(hash==3){ uint64_t x=line*0x9E3779B97F4A7C15ull; return x>>20; } if(hash==5) return line^(line>>7)^(line>>14)^(line>>21); if(hash==6) return line^(line>>9)^(line>>18); if(hash==7) return line^(line>>13); if(hash==8) return line^(line>>16); if(hash==9) return line^(line>>10); return line; }
static Line* cfind(Cache*c,uint64_t line){ Line*s=setof(c,line); for(uint32_t w=0;w<c->ways;w++) if(s[w].tag==line){ s[w].lru=++c->clk; s[w].rrpv=0; return &s[w]; } return NULL; }
static Line* cvictim(Cache*c,uint64_t line){ Line*s=setof(c,line); Line*v=&s[0];
  for(uint32_t w=0;w<c->ways;w++) if(s[w].tag==~0ull) return &s[w];
  if(c->pol==1){ c->rng^=c->rng<<13; c->rng^=c->rng>>7; c->rng^=c->rng<<17; return &s[c->rng%c->ways]; }
  if(c->pol==2){ for(;;){ for(uint32_t w=0;w<c->ways;w++) if(s[w].rrpv>=3) return &s[w]; for(uint32_t w=0;w<c->ways;w++) s[w].rrpv++; } }
  for(uint32_t w=0;w<c->ways;w++) if(s[w].lru<v->lru) v=&s[w];
  return v; }
// FIFO: lru stamp set only at insertion; cfind must not refresh it
static Line* cfind_fifo(Cache*c,uint64_t line){ Line*s=setof(c,line); for(uint32_t w=0;w<c->ways;w++) if(s[w].tag==line) return &s[w]; return NULL; }
static FILE *fd,*fl; static uint32_t curlaunch, cursm;
static void emit(Model*m,uint64_t sector,int w){ if(w){m->nwr++;m->p_wr[curlaunch]++;} else {m->nrd++;m->p_miss[curlaunch]++;}
  if(m->emit){ uint64_t r=(sector<<1)|w; fwrite(&r,8,1,fd); uint32_t v=curlaunch|(cursm<<20); fwrite(&v,4,1,fl); } }
static void evict(Model*m,Line*v){ if(v->tag==~0ull)return; for(int s=0;s<4;s++) if(v->dirty>>s&1) emit(m,v->tag*4+s,1); }
static void l2access(Model*m,uint64_t sec,int st){
  Cache*c=&m->L2; uint64_t line=sec>>2; int s=sec&3; Line*x=(c->pol==3)?cfind_fifo(c,line):cfind(c,line);
  if(!x){ x=cvictim(c,line); evict(m,x); x->tag=line; x->valid=0; x->dirty=0; x->lru=++c->clk; x->rrpv=2; }
  if(st){ x->valid|=1<<s; x->dirty|=1<<s; m->l2wr++; m->p_wr[curlaunch]+=0; }
  else { m->l2rd++; m->p_l2rd[curlaunch]++; if(!(x->valid>>s&1)){ x->valid|=1<<s; emit(m,sec,0);
      if(m->fetch64){ int o=s^1; if(!(x->valid>>o&1)){ x->valid|=1<<o; emit(m,sec^1,0);} } } }
}
static void l1access(Model*m,int sm,uint64_t sec,int st){
  if(m->l1kb==0){ if(!st) m->l1rd++; l2access(m,sec,st); return; }
  Cache*c=&m->L1[sm]; uint64_t line=sec>>2; int s=sec&3; Line*x=cfind(c,line);
  if(st){ if(x) x->valid|=1<<s; l2access(m,sec,1); return; }
  m->l1rd++;
  if(x && (x->valid>>s&1)) return;
  if(!x){ x=cvictim(c,line); x->tag=line; x->valid=0; x->lru=++c->clk; }
  x->valid|=1<<s; l2access(m,sec,0);
}
typedef struct { uint32_t launch, cta; uint16_t warp; uint8_t st, k; uint32_t off; uint32_t seq; } Rec;
static Model M[16]; static int nm=0;
static void parse_var(const char*spec){
  Model*m=&M[nm++]; memset(m,0,sizeof *m); m->nsm=80; m->l1kb=128; m->l1w=4; m->l2mb=6; m->ways=16; m->hash=0; m->pol=0; m->reorder=0; m->smmap=0; m->fetch64=0;
  char buf[512]; strncpy(buf,spec,511); char*col=strchr(buf,':'); if(col){*col=0; strncpy(m->name,buf,63); col++;} else {strncpy(m->name,buf,63); col=buf+strlen(buf);}
  for(char*tok=strtok(col,",");tok;tok=strtok(NULL,",")){ char*e=strchr(tok,'='); if(!e)continue; *e=0; const char*k=tok,*v=e+1;
    if(!strcmp(k,"sm"))m->nsm=atoi(v); else if(!strcmp(k,"l1kb"))m->l1kb=atoi(v); else if(!strcmp(k,"l1w"))m->l1w=atoi(v); else if(!strcmp(k,"l2mb"))m->l2mb=atof(v);
    else if(!strcmp(k,"ways"))m->ways=atoi(v); else if(!strcmp(k,"hash"))m->hash=atoi(v); else if(!strcmp(k,"reorder"))m->reorder=atoi(v);
    else if(!strcmp(k,"l2fetch"))m->fetch64=(atoi(v)==64); else if(!strcmp(k,"smmap"))m->smmap=!strcmp(v,"dyn");
    else if(!strcmp(k,"pol")) m->pol=!strcmp(v,"lru")?0:!strcmp(v,"rand")?1:!strcmp(v,"srrip")?2:3; else { fprintf(stderr,"bad key %s\n",k); exit(1);} }
  cinit(&m->L2,(uint64_t)(m->l2mb*1048576),m->ways,m->hash,m->pol); m->L1=malloc(sizeof(Cache)*m->nsm); for(int i=0;i<m->nsm;i++)cinit(&m->L1[i],(uint64_t)(m->l1kb?m->l1kb:1)*1024,m->l1w,0,0);
  m->p_l2rd=calloc(MAXL,8); m->p_miss=calloc(MAXL,8); m->p_wr=calloc(MAXL,8); m->lastuse=calloc(m->nsm,8);
}
// batch
#define BMAX (1<<20)
static Rec *B; static uint64_t *S; static uint32_t nb, ns; static uint32_t *order; static uint32_t *cnt;
#define HT (1<<22)
static uint64_t *htk; static uint32_t *htv; static uint32_t hstamp=0; static uint32_t *hst;
static uint32_t seqof(uint64_t key){ uint32_t h=(uint32_t)((key*0x9E3779B97F4A7C15ull)>>42)&(HT-1); for(;;){ if(hst[h]!=hstamp){hst[h]=hstamp;htk[h]=key;htv[h]=0;return htv[h]++;} if(htk[h]==key) return htv[h]++; h=(h+1)&(HT-1);} }
static uint64_t nwarp, runlen_sum, nruns, dist_sum, ndist; static uint32_t prevcta=~0u;
static void flush(void){
  if(!nb) return;
  // stats on trace order
  for(uint32_t i=0;i<nb;i++){ nwarp++; if(B[i].cta!=prevcta||i==0){nruns++; prevcta=B[i].cta;} }
  int need[3]={0,0,0}; for(int v=0;v<nm;v++) need[M[v].reorder]=1;
  uint32_t *ord[3]={0,0,0};
  for(int r=1;r<=2;r++) if(need[r]){ hstamp++; for(uint32_t i=0;i<nb;i++) B[i].seq=seqof(r==1?((uint64_t)B[i].cta<<8|B[i].warp):(uint64_t)B[i].cta);
    uint32_t mx=0; for(uint32_t i=0;i<nb;i++) if(B[i].seq>mx)mx=B[i].seq; uint32_t*c=calloc(mx+2,4); for(uint32_t i=0;i<nb;i++)c[B[i].seq+1]++; for(uint32_t i=0;i<=mx;i++)c[i+1]+=c[i];
    ord[r]=malloc(4*nb); for(uint32_t i=0;i<nb;i++) ord[r][c[B[i].seq]++]=i; free(c); }
  for(int v=0;v<nm;v++){ Model*m=&M[v]; int r=m->reorder;
    for(uint32_t q=0;q<nb;q++){ Rec*x=&B[r?ord[r][q]:q]; curlaunch=x->launch; int sm;
      if(m->smmap){ // dynamic: a CTA is bound to an SM on first sight (least recently used SM at that time)
        sm=x->cta%m->nsm; } else sm=x->cta%m->nsm;
      cursm=sm; for(int j=0;j<x->k;j++) l1access(m,sm,S[x->off+j],x->st); } }
  for(int r=1;r<=2;r++) free(ord[r]);
  nb=0; ns=0;
}
int main(int argc,char**argv){
  const char*emitn=NULL; int bsz=BMAX; char*pre=argv[1],*outp=argv[2];
  for(int i=3;i<argc;i+=2){ if(!strcmp(argv[i],"--var"))parse_var(argv[i+1]); else if(!strcmp(argv[i],"--emit"))emitn=argv[i+1]; else if(!strcmp(argv[i],"--batch"))bsz=atoi(argv[i+1]); }
  if(!nm){fprintf(stderr,"no --var\n");return 1;}
  B=malloc(sizeof(Rec)*BMAX); S=malloc(8*(size_t)BMAX*8); htk=malloc(8*HT); htv=malloc(4*HT); hst=calloc(HT,4);
  for(int v=0;v<nm;v++) M[v].emit=emitn&&!strcmp(M[v].name,emitn);
  char p[512],nm_[256]; snprintf(p,512,"%s.opcodes.txt",pre); FILE*fo=fopen(p,"r"); int width[4096]={0},isst[4096]={0},id;
  while(fo&&fscanf(fo,"%d %255s",&id,nm_)==2){ isst[id]=(nm_[0]=='S'||nm_[0]=='R'||strncmp(nm_,"ATOM",4)==0);
    int w=4; if(strstr(nm_,".128"))w=16; else if(strstr(nm_,".64"))w=8; else if(strstr(nm_,".U8")||strstr(nm_,".S8"))w=1; else if(strstr(nm_,".U16")||strstr(nm_,".S16"))w=2; width[id]=w; }
  snprintf(p,512,"%s.launch.txt",pre); FILE*fg=fopen(p,"r"); char *ln=malloc(1<<16);
  while(fg&&fgets(ln,1<<16,fg)){ char*a=strstr(ln,"grid launch id "); char*b=strstr(ln,"grid size "); if(!a||!b)continue; long l=atol(a+15); unsigned x,y,z; if(sscanf(b+10,"%u,%u,%u",&x,&y,&z)==3&&l<MAXL){gx[l]=x;gy[l]=y;} }
  snprintf(p,512,"%s.bin",pre); FILE*f=fopen(p,"rb"); setvbuf(f,0,_IOFBF,1<<24);
  if(emitn){ snprintf(p,512,"%s.dram",outp); fd=fopen(p,"wb"); snprintf(p,512,"%s.dramlaunch",outp); fl=fopen(p,"wb"); }
  uint64_t a[32]; uint32_t lastl=~0u;
  for(;;){ uint8_t hd[24]; if(fread(hd,1,24,f)!=24)break; uint64_t la=*(uint64_t*)hd; uint32_t cx=*(uint32_t*)(hd+8); uint16_t cy=*(uint16_t*)(hd+12),cz=*(uint16_t*)(hd+14);
    uint16_t wp=*(uint16_t*)(hd+16), o=*(uint16_t*)(hd+18); uint32_t mask=*(uint32_t*)(hd+20); int n=__builtin_popcount(mask);
    if(o==0xFFFF){ if(la<MAXL){gx[la]=cx;gy[la]=cy;} continue; } if(fread(a,8,n,f)!=(size_t)n)break;
    if((uint32_t)la!=lastl){ flush(); lastl=(uint32_t)la; }
    uint64_t gxx=la<MAXL&&gx[la]?gx[la]:1, gyy=la<MAXL&&gy[la]?gy[la]:1; uint64_t lin=cx+gxx*(cy+gyy*(uint64_t)cz);
    int w,st; if(o&0x4000){w=o&0xFF;st=(o>>8)&1;} else {w=width[o];st=isst[o];} uint64_t sec[128]; int k=0;
    for(int i=0;i<n;i++){ uint64_t s0=a[i]>>5,s1=(a[i]+w-1)>>5; for(uint64_t s=s0;s<=s1;s++){int d=0;for(int j=0;j<k;j++)if(sec[j]==s){d=1;break;} if(!d&&k<128)sec[k++]=s;} }
    if(nb>=(uint32_t)bsz||ns+k>(uint32_t)BMAX*8) flush();
    Rec*r=&B[nb++]; r->launch=(uint32_t)la; r->cta=(uint32_t)lin; r->warp=wp; r->st=st; r->k=k; r->off=ns; memcpy(S+ns,sec,8*k); ns+=k; }
  flush();
  if(emitn){ fclose(fd); fclose(fl); }
  snprintf(p,512,"%s.perk.tsv",outp); FILE*fp=fopen(p,"w"); fprintf(fp,"launch");
  for(int v=0;v<nm;v++) fprintf(fp,"\t%s_l2rd\t%s_miss\t%s_wr",M[v].name,M[v].name,M[v].name); fprintf(fp,"\n");
  for(uint32_t l=0;l<MAXL;l++){ int any=0; for(int v=0;v<nm;v++) if(M[v].p_l2rd[l]||M[v].p_miss[l]||M[v].p_wr[l]) any=1; if(!any)continue;
    fprintf(fp,"%u",l); for(int v=0;v<nm;v++) fprintf(fp,"\t%lu\t%lu\t%lu",M[v].p_l2rd[l],M[v].p_miss[l],M[v].p_wr[l]); fprintf(fp,"\n"); }
  fclose(fp);
  fprintf(stderr,"warp instrs %lu; mean same-CTA run length %.2f\n",nwarp,nruns?(double)nwarp/nruns:0);
  for(int v=0;v<nm;v++) fprintf(stderr,"%-14s L1 ld sectors %lu; L2 reads %lu; DRAM reads %lu (%.1f MB) writes %lu (%.1f MB)\n",M[v].name,M[v].l1rd,M[v].l2rd,M[v].nrd,M[v].nrd*32/1e6,M[v].nwr,M[v].nwr*32/1e6);
  return 0; }
