#include <cuda_runtime.h>
#include <stdio.h>
__global__ void k(int*p){int i=threadIdx.x;p[i]=i+1;}
int main(){int*p;cudaMalloc(&p,128);k<<<1,32>>>(p);int h[32];cudaMemcpy(h,p,128,cudaMemcpyDeviceToHost);for(int i=0;i<32;i++)if(h[i]!=i+1)return 2;puts("PASS native data");cudaFree(p);return cudaDeviceSynchronize()!=cudaSuccess;}
