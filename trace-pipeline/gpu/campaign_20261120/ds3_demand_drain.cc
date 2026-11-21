// Saturated per-PC DRAMsim3 driver. Write callbacks are early acknowledgments;
// completion additionally requires actual demand queues/maps empty +128cycles.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include "memory_system.h"
int main(int argc,char**argv){
 if(argc<4||argc>5)return 1;const int issue=argc==5?atoi(argv[4]):1;if(issue<1)return 1; uint64_t done=0,wdone=0,added=0,nw=0,clk=0,idle=0,stamp=0,lastprogress=0;
 dramsim3::MemorySystem ms(argv[1],argv[2],[&](uint64_t){done++;},[&](uint64_t){done++;wdone++;});
 FILE*f=fopen(argv[3],"rb");if(!f)return 2;std::vector<uint64_t>buf(1<<16);size_t pos=0,n=0;bool eof=false,have=false,wr=false;uint64_t addr=0;
 while(true){
  ms.ClockTick();clk++;
  for(int slot=0;slot<issue;slot++){
  if(!have&&!eof){if(pos==n){n=fread(buf.data(),8,buf.size(),f);pos=0;if(!n)eof=true;}if(!eof){auto v=buf[pos++];addr=v>>1;wr=v&1;have=true;}}
  if(have&&ms.WillAcceptTransaction(addr,wr)){if(!ms.AddTransaction(addr,wr))return 3;added++;nw+=wr;have=false;}else break;
  }
  if(eof&&!have&&done==added&&!ms.HasPendingTransactions())idle++;else idle=0;
  if(idle>=128)break;
  auto progress=done+added;
  if(progress!=lastprogress||idle){stamp=clk;lastprogress=progress;}
  if(clk-stamp>200000){fprintf(stderr,"Demand drain stalled: accepted=%llu callbacks=%llu pending=%d\n",(unsigned long long)added,(unsigned long long)done,ms.HasPendingTransactions());return 4;}
 }
 fclose(f);ms.PrintStats();printf("DRAIN_CYCLES %llu REQS %llu WRITES %llu MISSING 0 UNISSUED 0 DEMAND_EMPTY 1 GUARD 128 CALLBACKS %llu WR_CALLBACKS %llu\n",(unsigned long long)clk,(unsigned long long)added,(unsigned long long)nw,(unsigned long long)done,(unsigned long long)wdone);
}
