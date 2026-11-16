#include <cuda_runtime.h>
#include <stdio.h>
__global__ void k(volatile int*p){int i=threadIdx.x;for(int j=0;j<4;j++)p[i+j*8192]=p[i+j*8192]+1;}
int main(){int*p;cudaMalloc(&p,131072);cudaMemset(p,0,131072);k<<<1,32>>>(p);int h[32];cudaMemcpy(h,p,128,cudaMemcpyDeviceToHost);for(int i=0;i<32;i++)if(h[i]!=1)return 2;puts("PASS native write test");cudaFree(p);return cudaDeviceSynchronize()!=cudaSuccess;}

