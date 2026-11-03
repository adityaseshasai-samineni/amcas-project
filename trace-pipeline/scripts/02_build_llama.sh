#!/bin/bash
#SBATCH -J llama-static
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --exclude=node04,node05
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/llamastatic_%j.log
# Build libm5, the trace decoder, and two static SSE2-only llama-completion
# binaries from the same source: one with work item markers for gem5, one
# without for validating against native execution.
#
# SSE2-only because gem5's X86 decoder treats pmaddubsw, phadd* and psign* as
# warn-and-skip, so an SSSE3 binary would silently compute wrong numbers.
set -uo pipefail
W=$HOME/amcas-pro/tracing
E=$W/mamba/envs/gem5
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib
cd $W
python3 $HOME/amcas-pro/patches/apply_patches.py $W || exit 1

(cd gem5/util/m5 && scons -j16 build/x86/out/libm5.a 2>&1 | tail -2 | cut -c1-160)
LIBM5=$W/gem5/util/m5/build/x86/out/libm5.a
ls -la $LIBM5 || exit 1

gcc -O2 -o $W/bin/pkt2bin $HOME/amcas-pro/tools/pkt2bin.c -I$E/include -L$E/lib -Wl,-rpath,$E/lib -lz || exit 1

COMMON=(-DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF
  -DGGML_SSE42=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF -DGGML_F16C=OFF
  -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF -DGGML_OPENMP=OFF -DGGML_LLAMAFILE=OFF
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++
  "-DCMAKE_EXE_LINKER_FLAGS=-static -Wl,--whole-archive -lpthread -Wl,--no-whole-archive")

cd llama.cpp
rm -rf build-gem5 build-native
cmake -B build-native "${COMMON[@]}" 2>&1 | tail -1
cmake --build build-native -j16 --target llama-completion 2>&1 | tail -1 | cut -c1-160
cmake -B build-gem5 "${COMMON[@]}" \
  "-DCMAKE_CXX_FLAGS=-DAMCAS_M5 -I$W/gem5/include" \
  "-DCMAKE_CXX_STANDARD_LIBRARIES=$LIBM5" 2>&1 | tail -1
cmake --build build-gem5 -j16 --target llama-completion 2>&1 | tail -1 | cut -c1-160

for b in native gem5; do
  B=build-$b/bin/llama-completion
  file $B | cut -c1-120
  echo "$b: AVX instructions=$(objdump -d $B | grep -cE '\sv(mov|fmadd|padd|broadcast|xorp|zeroupper)') SSSE3+ instructions=$(objdump -d $B | grep -cE '\s(pshufb|pmaddubsw|phadd[dw]|psign[bwd]|pabs[bwd]|pmulld|ptest|pmovzx)')"
done
