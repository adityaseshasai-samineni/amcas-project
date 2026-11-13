#!/bin/bash
E=$HOME/amcas-pro/tracing/mamba/envs/gem5; export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib
cd $HOME/amcas-pro/tracing/llama.cpp; cmake --build build-gem5 -j8 --target llama-completion 2>&1 | tail -3 | cut -c1-180
ls -la build-gem5/bin/llama-completion
