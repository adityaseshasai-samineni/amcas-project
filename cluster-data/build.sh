#!/bin/bash
#SBATCH -J gem5-build
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --exclude=node04,node05
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/build_%j.log
W=~/amcas-pro/tracing; cd $W
export APPTAINER_CACHEDIR=$W/.apptainer APPTAINER_TMPDIR=$W/.apptainer/tmp; mkdir -p $APPTAINER_TMPDIR
rm -rf .apptainer/cache .apptainer/tmp/*
python3 ~/amcas-pro/fetch.py || exit 1
(cd gem5-oci && tar cf ../gem5-oci.tar oci-layout index.json blobs)
apptainer build --force gem5-deps.sif oci-archive:gem5-oci.tar 2>&1 | tail -2
ls -la gem5-deps.sif || exit 1
rm -rf gem5-oci.tar gem5-oci
echo "=== llama.cpp SSE static"
cat > inner_llama.sh <<'IN'
cd llama.cpp; rm -rf build-sse
cmake -B build-sse -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF -DGGML_SSE42=ON -DGGML_OPENMP=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_SERVER=OFF -DCMAKE_EXE_LINKER_FLAGS=-static 2>&1 | tail -2
cmake --build build-sse -j16 --target llama-completion 2>&1 | tail -2
file build-sse/bin/llama-completion
echo AVXcount: $(objdump -d build-sse/bin/llama-completion | grep -cE '\s(vmov|vfmadd|vpadd|vbroadcast)')
IN
apptainer exec gem5-deps.sif bash inner_llama.sh
echo "=== native baseline"
cd models; BIN=$W/llama.cpp/build-sse/bin/llama-completion
for n in 1 11; do perf stat -x, -e instructions:u,cache-misses:u -- $BIN -m qwen25-05b-q4_0.gguf -t 1 -p "Once upon a time" -n $n --ignore-eos -no-cnv -lm none --seed 1 2>&1 | grep -E 'instructions|cache-misses|eval time|error'; done
echo "=== gem5"
cd $W/gem5; git log -1 --format='%h %ad' --date=short
apptainer exec ../gem5-deps.sif bash -c "yes | scons build/X86/gem5.opt -j16 --ignore-style" 2>&1 | tail -5 | cut -c1-200
ls -la build/X86/gem5.opt
