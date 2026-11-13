#!/bin/bash
export OPENBLAS_NUM_THREADS=1 LD_LIBRARY_PATH=$HOME/amcas-pro/tracing/mamba/envs/gem5/lib
E=$HOME/amcas-pro/tracing/mamba/envs/gem5; PY=$E/bin/python; export PATH=$E/bin:$PATH
BIN=$HOME/amcas-pro/tracing/llama.cpp/build-gem5/bin/llama-completion
cd $HOME/amcas-pro/results/traces
echo "NATIVE: $(grep -a 'Once upon' ../native/stories15M.n9.out | tail -c 80)"
for r in s15_l3-22m_stride s15_l3-2m_stride s15_l3-2m_nopf; do
  echo "GEM5 $r: $(grep -a 'One day' $r/simout.txt | tail -c 80)"
done
r=s15_l3-2m_stride
awk '/^simInsts/{a[++n]=$2} END{print "simInsts first dump (load, atomic)=" a[1] "  last=" a[n]}' $r/stats.txt
awk '/^simInsts/{a[++n]=$2} END{print "22m: first=" a[1] "  last=" a[n]}' s15_l3-22m_stride/stats.txt
echo "=== characterise $r"; $PY $HOME/amcas-pro/scripts/_chk.py $r $BIN 2>&1 | cut -c1-130
echo "=== nopf timing"; $PY $HOME/amcas-pro/scripts/_chk.py s15_l3-2m_nopf $BIN 2>&1 | grep 'phase ms' | cut -c1-130
echo "=== 22m timing"; $PY $HOME/amcas-pro/scripts/_chk.py s15_l3-22m_stride $BIN 2>&1 | grep -E 'phase ms' | cut -c1-130
echo "=== offline with real scatter"
for r in s15_l3-2m_stride s15_l3-2m_nopf; do $PY $HOME/amcas-pro/tools/analyze.py $r --no-export 2>&1 | sed -n '/offline/,$p' | cut -c1-90; done
