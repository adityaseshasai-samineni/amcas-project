#!/bin/bash
#SBATCH -J bincmp
#SBATCH -n 1
#SBATCH -c 36
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/bincmp_%j.log
# Same 31k-token prompt state, 4-thread decode, three binaries: llama-completion (build-fast), llama-drattr (Release rebuild),
# llama-drattr.debug-build (empty CMAKE_BUILD_TYPE = no -O flags, the binary used for the drattr c32k run). Explains 2.4 s vs 1.15 s/token.
set -uo pipefail
MODEL=qwen2.5-7b-instruct-q4_k_m-00001-of-00002
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5; export LD_LIBRARY_PATH=$E/lib
F=$W/models/$MODEL.gguf; B=$W/llama.cpp/build-fast/bin/llama-completion
R=$HOME/amcas-pro/results/fullscale/native/bincmp; mkdir -p $R
S=/scratch/$USER/bincmp; rm -rf $S; mkdir -p $S; trap 'rm -rf $S' EXIT
head -c ${PROMPT_CHARS:-116000} $HOME/amcas-pro/results/sessions/corpus.txt > $S/prompt.txt
C=(-m $F -c ${CTX:-32768} -f $S/prompt.txt -no-cnv --temp 0 --seed 1 --no-warmup --ignore-eos --prompt-cache $S/s.session)
uptime > $R/node.txt; hostname >> $R/node.txt
$B "${C[@]}" -b 2048 -ub 512 -t 32 -n 1 > $S/save.out 2> $R/save.err; ls -l $S/s.session
echo "== llama-completion -t 4"; $B "${C[@]}" --prompt-cache-ro -t 4 -n 9 > $S/o 2> $R/completion.err
grep -E "eval time" $R/completion.err | tail -1
for v in llama-drattr llama-drattr.debug-build; do
  echo "== $v -t 4"
  AMCAS_SKIP_STEPS=999999 AMCAS_STEPS=$R/$v.steps $W/drburst-attr-build/$v "${C[@]}" --prompt-cache-ro -t 4 -tb 4 -n 9 > $S/o 2> $R/$v.err
  tail -3 $R/$v.err | cut -c1-200
  awk '$1=="B"{b[$2]=$3} $1=="E"{printf "step %s %.0f ms\n",$2,($3-b[$2])/1000}' $R/$v.steps
done
bash $HOME/amcas-pro/tools/publish.sh "bincmp $SLURM_JOB_ID" || true
