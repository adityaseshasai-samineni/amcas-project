#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include "nvbit_tool.h"
#include "nvbit.h"
#include "common.h"
__managed__ Buffer buffer;
static std::unordered_set<CUfunction> seen;static bool skip=false,enabled=false;static unsigned kernel=0;static FILE*out=nullptr;
void nvbit_at_init(){setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC","1",1);}
void instrument(CUcontext ctx,CUfunction f){auto fs=nvbit_get_related_functions(ctx,f);fs.push_back(f);for(auto fun:fs){if(!seen.insert(fun).second)continue;for(auto i:nvbit_get_instrs(ctx,fun)){
 std::string op=i->getOpcode();if(op.rfind("LDG",0)!=0||op.find("LDGSTS")!=std::string::npos)continue;
 const auto*d=i->getOperand(0);if(d->type!=InstrType::OperandType::REG)continue;
 fprintf(stderr,"PROBE_PC %s %u %s reg%d\n",nvbit_get_func_name(ctx,fun),i->getOffset(),i->getSass(),d->u.reg.num);
 nvbit_insert_call(i,"probe_before",IPOINT_BEFORE);nvbit_add_call_arg_guard_pred_val(i);nvbit_add_call_arg_const_val64(i,(uint64_t)&buffer);
 nvbit_insert_call(i,"probe_after",IPOINT_AFTER);nvbit_add_call_arg_guard_pred_val(i);nvbit_add_call_arg_const_val64(i,(uint64_t)&buffer);nvbit_add_call_arg_reg_val(i,d->u.reg.num);nvbit_add_call_arg_const_val32(i,i->getOffset());
}}}
void nvbit_at_cuda_event(CUcontext ctx,int exit,nvbit_api_cuda_t id,const char*,void*params,CUresult*){
 if(skip)return;if(id!=API_CUDA_cuLaunchKernel&&id!=API_CUDA_cuLaunchKernel_ptsz)return;
 auto*p=(cuLaunchKernel_params*)params;
 if(!exit){std::string name=nvbit_get_func_name(ctx,p->f);const char*filter=getenv("PROBE_KERNEL");enabled=!filter||name.find(filter)!=std::string::npos;if(enabled){skip=true;cudaDeviceSynchronize();buffer.count=0;buffer.armed=0;skip=false;instrument(ctx,p->f);}nvbit_enable_instrumented(ctx,p->f,enabled);}
 else{if(enabled){skip=true;auto err=cudaDeviceSynchronize();skip=false;if(err!=cudaSuccess){fprintf(stderr,"PROBE_ERROR %s\n",cudaGetErrorString(err));std::abort();}if(!out){const char*path=getenv("PROBE_OUT");out=fopen(path?path:"observer.csv","w");fprintf(out,"kernel,name,sample,cycles,pc,sm,value\n");}for(unsigned n=0;n<buffer.count;n++){auto&r=buffer.rows[n];fprintf(out,"%u,%s,%u,%llu,%u,%u,%u\n",kernel,nvbit_get_func_name(ctx,p->f),n,(unsigned long long)r.cycles,r.pc,r.sm,r.value);}fflush(out);}kernel++;}
}
