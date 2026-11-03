#!/bin/bash
#SBATCH -J dram-sims
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=node04,node05
#SBATCH -o /home/adityaseshasai.s/amcas-pro/logs/dramsims_%j.log
# Build DRAMsim3 and Ramulator2 and replay one captured trace through each, for a
# small grid of mappings and page policies.
set -uo pipefail
W=$HOME/amcas-pro/tracing; E=$W/mamba/envs/gem5
export PATH=$E/bin:$PATH LD_LIBRARY_PATH=$E/lib CC=gcc CXX=g++ OPENBLAS_NUM_THREADS=1
T=$HOME/amcas-pro/results/traces/${1:-s15_l3-2m_stride}
cd $W
[ -d DRAMsim3 ] || git clone -q --depth 1 https://github.com/umd-memsys/DRAMsim3
[ -d ramulator2 ] || git clone -q --depth 1 https://github.com/CMU-SAFARI/ramulator2

echo "=== DRAMsim3"
(cd DRAMsim3 && cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 2>&1 | tail -1 | cut -c1-120 \
  && cmake --build build -j8 2>&1 | tail -1 | cut -c1-120)
LAST=$(tail -1 $T/dram.dramsim3.trc | cut -d' ' -f3)
echo "trace: $(awk '{c[$2]++} END{for(k in c)printf "%s=%d ",k,c[k]}' $T/dram.dramsim3.trc)"
for map in rochrababgco rocorababgch; do
  for pol in OPEN_PAGE CLOSE_PAGE; do
    O=$T/dramsim3/${map}_$pol; mkdir -p $O
    sed -e "s/^address_mapping = .*/address_mapping = $map/" -e "s/^row_buf_policy = .*/row_buf_policy = $pol/" \
      DRAMsim3/configs/DDR4_8Gb_x8_2400.ini > $O/config.ini
    (cd DRAMsim3 && ./build/dramsim3main $O/config.ini -c $((LAST * 2 + 200000)) -t $T/dram.dramsim3.trc -o $O >$O/run.log 2>&1) &
  done
done
wait
$E/bin/python - <<P
import glob,json,os
print(f"{'mapping':<15}{'policy':<12}{'reads':>10}{'writes':>9}{'hit':>7}{'rd_lat':>8}{'act':>10}")
for f in sorted(glob.glob("$T/dramsim3/*/dramsim3.json")):
    s=json.load(open(f))["0"]
    r=s["num_read_row_hits"]+s["num_write_row_hits"]; n=s["num_reads_done"]+s["num_writes_done"]
    m,p=os.path.basename(os.path.dirname(f)).split("_",1)
    print(f"{m:<15}{p:<12}{s['num_reads_done']:>10}{s['num_writes_done']:>9}{r/max(n,1):>7.3f}{s['average_read_latency']:>8.1f}{s['num_act_cmds']:>10}")
P

echo "=== Ramulator2"
(cd ramulator2 && cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 2>&1 | tail -1 | cut -c1-120 \
  && cmake --build build -j8 2>&1 | tail -1 | cut -c1-140)
LD_LIBRARY_PATH=$W/ramulator2:$E/lib PYTHONPATH=$W/ramulator2/python \
  $E/bin/python $HOME/amcas-pro/tools/ramulator_replay.py $T/dram.ramulator.trc $T/ramulator.json
echo "ramulator exit=$?"
