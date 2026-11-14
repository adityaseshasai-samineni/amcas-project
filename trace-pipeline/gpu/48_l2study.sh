#!/bin/bash
# L2-model study cell. usage: 48_l2study.sh MODE(trace|ncu) GPU(v100|a16) CTXNAME FA(auto|off) "VARIANT ARGS" [MODEL]
# trace: lo pass -> window -> streamed NVBit -> gpu_filter2 (per-launch tsv). ncu: per-kernel ncu csv of the same window.
# Env: EMIT=name keeps the gz DRAM stream of that variant (results stay in $SCR). TAGX suffix for outputs.
MODE=$1; GPU=$2; CN=$3; FA=$4; VARS=$5
W=$HOME/amcas-pro/tracing; G=$HOME/amcas-pro/gpu; E=$W/mamba/envs/gem5; R=$HOME/amcas-pro/results/gpu; mkdir -p $R
M=${6:-$W/models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf}; MT=${MODELTAG:-q7}
if [ $GPU = v100 ]; then A=70; SCR=/scratch/$USER/gt; FARGS=""; NCU=/opt/nvidia/nsight-compute/2024.3.2/ncu
else A=86; SCR=$HOME/gt; NCU=/usr/local/cuda-12.6/bin/ncu; fi
mkdir -p $SCR
case $CN in short) CTX=4096; PC=0;; 8k) CTX=8192; PC=30000;; 32k) CTX=32768; PC=116000;; esac
TAG=${MT}_${GPU}_${CN}_fa${FA}; XT=${TAGX:-}
if [ $PC = 0 ]; then unset PFILE; else PF=$SCR/${TAG}_$MODE.prompt.txt; head -c $PC $HOME/amcas-pro/results/sessions/corpus.txt > $PF; export PFILE=$PF; fi
export CTX EXTRA="-fa $FA" PATH=$W/cudatools:$PATH
bash $G/36_gputrace_run.sh $SCR/${TAG}_${MODE}_lo $A $M 3 999999999 999999999 | grep -e wall
read S EN < <($E/bin/python $G/find_window.py $SCR/${TAG}_${MODE}_lo.launch.txt) || exit 1
echo "$TAG window $S $EN"; rm -f $SCR/${TAG}_${MODE}_lo.bin
if [ $MODE = trace ]; then
  bash $G/46_stream2.sh $SCR/$TAG$XT $A $M 3 $S $EN "$VARS"
  cp $SCR/$TAG$XT.perk.tsv $R/${TAG}${XT}_perk.tsv; cp $SCR/$TAG$XT.tensors.tsv $R/${TAG}${XT}_tensors.tsv 2>/dev/null
  cp $SCR/${TAG}_${MODE}_lo.launch.txt $R/${TAG}_launch.txt
else
  export NCU METRICS=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,lts__t_sectors_srcunit_tex_op_write.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
  echo "ncu window $S $((EN-S))" > $R/${TAG}_ncuperk.window
  bash $G/38_ncu_step.sh $A $M 3 $S $((EN-S)) $R/${TAG}_ncuperk 2>&1 | tail -n 12
fi
rm -f $PF
