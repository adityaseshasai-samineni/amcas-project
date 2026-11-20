#include "common.h"
extern "C" __device__ __noinline__ void probe_before(int pred,uint64_t ptr){
 if(!pred||threadIdx.x||threadIdx.y||threadIdx.z||blockIdx.x||blockIdx.y||blockIdx.z)return;
 Buffer*b=(Buffer*)ptr; if(b->count>=CAP){b->armed=0;return;}b->armed=1;
 asm volatile("membar.gl;":::"memory");unsigned long long t;asm volatile("mov.u64 %0,%%clock64;":"=l"(t)::"memory");b->start=t;
}
extern "C" __device__ __noinline__ void probe_after(int pred,uint64_t ptr,uint32_t value,uint32_t pc){
 // Passing destination register value creates a true dependency before entering this call.
 unsigned long long end;asm volatile("{.reg .pred p;setp.eq.u32 p,%1,0;@p mov.u64 %0,%%clock64;@!p mov.u64 %0,%%clock64;}":"=l"(end):"r"(value):"memory");
 if(!pred||threadIdx.x||threadIdx.y||threadIdx.z||blockIdx.x||blockIdx.y||blockIdx.z)return;
 Buffer*b=(Buffer*)ptr;if(!b->armed)return;unsigned sm;asm volatile("mov.u32 %0,%%smid;":"=r"(sm));uint32_t n=b->count++;b->rows[n]={end-b->start,pc,sm,value};b->armed=0;
}
