#!/bin/bash
# Probe: Qwen3-30B-A3B Q4_K_M across 2 GPUs (layer split): (A) plain llama run, (B) NVBit launch census to see both devices' contexts.
# usage: 67_moe_probe.sh ARCH(70|86) SCRDIR [CTX]
A=$1; SCR=$2; CTX=${3:-4096}
W=$HOME/amcas-pro/tracing; G=$HOME/amcas-pro/gpu; E=$W/mamba/envs/gem5; M=$W/models/Qwen3-30B-A3B-Q4_K_M.gguf
mkdir -p $SCR; nvidia-smi -L; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
export PATH=$W/cudatools:$PATH LD_LIBRARY_PATH=$W/cudalibs:$E/lib GGML_CUDA_DISABLE_GRAPHS=1
echo "=== A: plain run"
timeout 900 $W/llama.cpp/build-cuda/bin/llama-completion -m $M -ngl 99 -n 8 --temp 0 --seed 1 --no-warmup -no-cnv -t 4 -c $CTX -p "The capital of France is" 2>&1 | grep -i -e "buffer" -e "offload" -e "eval time" -e "^The" -e "error" -e "out of memory" -e "split" | cut -c1-220
nvidia-smi --query-gpu=index,memory.used --format=csv
echo "=== B: NVBit launch census (no instrumentation window)"
export CTX EXTRA="" 
bash $G/36_gputrace_run.sh $SCR/moe_probe $A $M 3 999999999 999999999 2>&1 | grep -v "^ *search" | tail -n 12
python3 - <<PY
import re, collections
c = collections.Counter(); first = {}
for l in open("$SCR/moe_probe.launch.txt"):
    m = re.search(r"CTX (0x[0-9a-f]+) .* grid launch id (\d+)", l)
    if m: c[m.group(1)] += 1
print("contexts and launches:", dict(c))
PY
