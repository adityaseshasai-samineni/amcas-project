#!/bin/bash
#SBATCH -J pcache-test
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --exclude=node04,node05
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/pcache_%j.log
# Can a prompt evaluated natively be restored inside gem5, so that a long context
# costs no simulated prompt evaluation? Expect the same tokens as a native run.
set -uo pipefail
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib
OUT=$HOME/amcas-pro/results/traces/pcache_test; mkdir -p $OUT
PROMPT="Once upon a time there was a little girl named Lily. She loved to play outside in the sunshine with her friends."
M=$W/models/stories15M.gguf; C=$OUT/session.bin; rm -f $C
ARGS="-t 1 -c 512 -b 512 -n 9 --ignore-eos -no-cnv -lm none --no-warmup --temp 0 --seed 1"
grep -n -E '"--prompt-cache' $W/llama.cpp/common/arg.cpp | cut -c1-90
$W/llama.cpp/build-native/bin/llama-completion -m $M $ARGS --prompt-cache $C -p "$PROMPT" >$OUT/native_save.out 2>$OUT/native_save.err
ls -la $C; echo "NATIVE SAVE: $(tail -c 70 $OUT/native_save.out)"
$W/llama.cpp/build-native/bin/llama-completion -m $M $ARGS --prompt-cache $C --prompt-cache-ro -p "$PROMPT" >$OUT/native_load.out 2>$OUT/native_load.err
echo "NATIVE LOAD: $(tail -c 70 $OUT/native_load.out)"; grep -iE 'session|prompt eval' $OUT/native_load.err | cut -c1-120 | tail -4
CMD="$W/llama.cpp/build-gem5/bin/llama-completion -m $M $ARGS --prompt-cache $C --prompt-cache-ro -p '$PROMPT'"
time $W/gem5/build/X86/gem5.fast --outdir=$OUT --redirect-stdout --redirect-stderr \
  $HOME/amcas-pro/configs/llm_trace.py --cmd "$CMD" --cwd $OUT --l3 2MiB --l3-assoc 16 --prefetch stride
echo "gem5 exit=$?"
echo "GEM5: $(grep -a 'One day' $OUT/simout.txt | tail -c 70)"
cat $OUT/phases.tsv | head -24
awk '/^simInsts/{a[++n]=$2} END{print "load insts=" a[1]}' $OUT/stats.txt
