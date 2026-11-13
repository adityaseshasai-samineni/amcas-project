echo "##### $(hostname)"
echo "max_user_ns=$(cat /proc/sys/user/max_user_namespaces 2>&1) perf_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>&1) kptr=$(cat /proc/sys/kernel/kptr_restrict 2>&1) thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled 2>&1)"
ls -la /dev/kvm 2>&1
lscpu | grep -E 'Model name|L2 cache|L3 cache|^Socket|NUMA node\(s\)'
echo "isa: $(grep -m1 -o -E 'avx512f|avx2' /proc/cpuinfo | head -1)"
unshare -U true 2>&1 && echo "unshare-U OK" || echo "unshare-U FAIL"
cd ~/amcas-pro/tracing && apptainer exec gem5-deps.sif bash -c 'echo CONTAINER_OK; gcc --version | head -1; python3 --version; scons --version 2>/dev/null | sed -n 2p; protoc --version; ls /usr/lib/x86_64-linux-gnu/libc.a; which aarch64-linux-gnu-gcc' 2>&1 | grep -v 'singularity/ exists' | head -12
