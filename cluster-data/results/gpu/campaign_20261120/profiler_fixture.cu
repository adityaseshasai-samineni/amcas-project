#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cstdio>
__global__ void outside(int*p){p[threadIdx.x]=1;}
__global__ void inside(int*p){p[threadIdx.x]+=2;}
int main(){int*p;cudaMalloc(&p,128*sizeof(int));outside<<<1,128>>>(p);cudaDeviceSynchronize();cudaProfilerStart();inside<<<1,128>>>(p);cudaDeviceSynchronize();cudaProfilerStop();outside<<<1,128>>>(p);cudaDeviceSynchronize();int v;cudaMemcpy(&v,p,sizeof v,cudaMemcpyDeviceToHost);cudaFree(p);if(v!=1)return 1;puts("native_output_pass");}
