#!/bin/bash
# Re-filter a kept raw trace with new variants. usage: 49_offline.sh TAG OUTTAG "variants"   (V100 scratch)
SCR=/scratch/$USER/gt; R=$HOME/amcas-pro/results/gpu; E=$HOME/amcas-pro/tracing/mamba/envs/gem5
PATH=$E/bin:$PATH gcc -O2 -o $SCR/$1.gf3 $HOME/amcas-pro/gpu/gpu_filter2.c || exit 1
eval "$SCR/$1.gf3 $SCR/$1.raw $SCR/$2 $3"
cp $SCR/$2.perk.tsv $R/$2_perk.tsv; rm -f $SCR/$1.gf3
