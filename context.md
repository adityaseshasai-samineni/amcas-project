# AMCAS project context

Last updated: 2026-11-03, about 08:30 IST. Authors: Aditya Samineni, Prasanna N R.

## Problem statement

**Address-Mapping and Row-Buffer Policy Optimization for Transformer Attention Access Patterns**

Attention and MoE (mixture-of-experts) workloads produce access patterns very different from classical HPC (irregular gather over KV-cache, expert-routing indirection). Using DRAMSim/Ramulator, characterize row-buffer hit rates for representative traces and propose an address-interleaving or row-buffer management policy tuned for this access class.

**Deliverable:** trace-driven simulation study, proposed mapping policy, hit-rate/latency improvement numbers.

## Aim

1. **Current milestone:** obtain representative memory traces between the SoC and DRAM for small LLM and MoE models.
2. Replay them in Ramulator/DRAMSim and characterize row-buffer behaviour under baseline mappings and page policies.
3. Propose a workload-aware mapping or row-buffer policy and measure the hit-rate and latency change against baselines.

The literature survey is in `problem_stmt_and_literature_survey.tex`.

## Capture method

gem5 and llama.cpp both run on the RCE cluster's CPU nodes.

- **Trace point:** a `CommMonitor` between the memory bus and the DRAM controller. It records LLC misses, writebacks and prefetches with physical address, tick, PC and (our patch) virtual address.
- **Mode:** gem5 SE mode, X86, one core, one thread. Atomic fast-forward with warm caches, then O3 from a chosen phase (`--detail-from`).
- **Phases:** each llama.cpp graph compute is one phase, marked with `m5_work_begin/end`. Phase 0 is the prompt; each later phase is one generated token. Tensor addresses are dumped to `amcas_tensors.tsv`.
- **GPU not used:** NVBit gives pre-cache GPU virtual addresses, not DRAM traffic. `perf`/PEBS cannot give physical addresses here (`perf_event_paranoid=2`, no root).
- **Restored prompts:** with `SESSION=`, phase 0 only replays the last prompt token, and it runs atomic as cache warm-up under `--detail-from 1`. `CTX` must match `BASE.ctx`.

## Build details

Scripts: `01_env_and_gem5.sh`, `02_build_llama.sh`, `05_build_fast.sh`. Patches: `patches/apply_patches.py` (idempotent).

**Toolchain:** micromamba environment `gem5` from conda-forge: Python 3.12, GCC 13, `libprotobuf=3.21`, `sysroot_linux-64=2.28`, plus `gperftools` for `gem5.fast`.

**gem5 builds**
- `gem5.opt` keeps debug flags. `gem5.fast` with tcmalloc is for long runs.
- Patch: adds a `vaddr` field to the packet trace (`packet.proto`, `probe/mem.hh`, `probes/mem_trace.cc`), so physical accesses can be attributed to tensors. It is only written when `with_pc` is on.
- `tools/pkt2bin.c` decodes this format by hand, because gem5's Python decoder cannot handle hundreds of millions of records.

**llama.cpp builds** (same source for both; only `build-gem5` sets `-DAMCAS_M5`)
- Static, built only as the `llama-completion` target. There is no `llama-tokenize`.
- SIMD above SSE2 is off (`GGML_NATIVE`, `SSE42`, `AVX`, `AVX2`, `FMA`, `F16C`, `BMI2`, `AVX_VNNI`, `AVX512`). 02 prints an instruction count from `objdump` as a check.
- `GGML_OPENMP=OFF`, `GGML_LLAMAFILE=OFF`. Linked with `-static` and whole-archive `libpthread`.
- Patch, phase markers: each graph compute is wrapped in `m5_work_begin/end`, and tensor addresses are written to `amcas_tensors.tsv`.
- Patch, logger: SE mode with one CPU has no spare thread context, so `clone()` fails. The logger's worker thread is the only one created with `-t 1`, so `build-gem5` prints inline instead.

**Capture config (`configs/llm_trace.py`)**
- 3 GHz core; L1I and L1D 32 KiB; L2 1 MiB; L3 set by `--l3`; 64-byte lines; 8 GiB DDR4-2400.
- O3 window sized to Cascade Lake: 224 ROB, 72 LQ, 56 SQ. Stride prefetcher at L2, degree 8.
- `HOME` must be set for the process. Otherwise llama.cpp calls `getpwuid()`, and glibc in the static binary loads the host's NSS libraries, whose TLS does not work in SE mode.

## Status

### Done

- gem5 26.0 (`gem5.opt`, and `gem5.fast` with tcmalloc) built without containers, using micromamba.
- Static SSE2-only llama.cpp built in two variants: `build-gem5` (phase markers) and `build-native` (validation).
- Native baseline measured for all three models.
- Three stories15M traces captured and validated.
- Exported to DRAMsim3 and Ramulator text formats. Both simulators replay the `s15_l3-2m_stride` export with every request accounted for (114852).
- Prompt restore works inside gem5 (114833): a session saved natively gives text identical to native on stories15M.
- Long-context stories15M traces (1024 and 4096 tokens) captured, validated and replayed in both DRAM simulators.
- Qwen at short context and at 1024 tokens, and MoE at 1024 tokens, captured and validated (2 detailed tokens each).
- **Found:** `tinymoe4` has only 2 experts and routes to both on every token, so it has no routing sparsity. A real MoE model is needed.

### Cluster jobs: last known state

Jobs keep running without the VPN, and the state below may be out of date. Last checked 2026-11-03 at about 08:10 IST.

| Job | Last known state |
|---|---|
| `qwen_l3-22m_stride` (114797) | completed in 14h 49m and validated |
| `moe_l3-22m_stride` (114798) | still fast-forwarding the prompt at 08:06 on 2026-11-03, after 15.5 hours |
| `long-ctx-native` (114857) | completed; sessions for all three models are in `results/sessions/` |
| `s15_c1024_l3-2m` (114858) | completed and validated |
| `s15_c4096_l3-2m` (114859), `s15_c4096_l3-22m` (114860) | completed and validated |
| `qwen_c1024_l3-22m` (114897), `moe_c1024_l3-22m` (114898) | completed in 6h 58m and 11h 25m, and validated |

Check with:

```bash
squeue -u $USER
sacct -X --starttime=2026-11-02T15:00 --format=JobID,JobName%20,State,ExitCode,Elapsed
cat ~/amcas-pro/logs/longctx_114857.log
tail ~/amcas-pro/results/traces/{qwen,moe}_l3-22m_stride/phases.tsv
```

## Results

### stories15M

Run: 28-token prompt plus 8 generated tokens, one DDR4-2400 channel.

**Validation**
- Generated text is identical to native in all three runs ("One day, Lily and her friends decided").
- Total instructions: about 2,127.8M in gem5 against 2,128.1M native.
- Trace read and write counts equal gem5's controller counters in every phase.
- LLC misses only bracket the native figure. Per token, gem5 shows about 0.00 MPKI with a 22 MiB LLC and 3.3 with 2 MiB; native shows about 0.64. Native ran on node06 (18 MiB L3, possibly shared), and `cache-misses:u` is not an identical measure.

**Per generated token**

| LLC / prefetcher | Time per token | DRAM reads | DRAM writes | gem5 row-hit rate |
|---|---|---|---|---|
| 22 MiB / stride | 14.3 ms | about 20 | about 20 | not meaningful (too few requests) |
| 2 MiB / stride | 16.1 ms | about 215k | about 12k | 0.918 |
| 2 MiB / none | 27.1 ms | about 215k | about 12k | 0.908 |

**Observations (one small model; do not generalize yet)**
- With a 22 MiB LLC, stories15M fits in cache, so it produces almost no DRAM traffic per token. Use a scaled-down LLC for this model.
- About 89% of reads go to weights. On this model most of those come from `ggml_vec_dot_q8_0_q8_0`, probably the output projection (not yet confirmed).
- **KV cache is only about 1.6% of reads** at this 36-token context. Studying KV gather needs long contexts.
- About 7% of reads and 79% of writes are not in graph tensors. By function and address these look like sampler and logits buffers on the heap, with some work-buffer traffic. This is only partly confirmed.
- Streams are strongly sequential: 62% +1-line strides with the prefetcher, 81% without.
- Baseline row-hit rate is already about 0.92, so there may be limited room for hit-rate gains on this model.

**Offline row-hit estimate (2 MiB LLC, stride)**

This is an in-order open-page model with no timing or scheduling. Use it only to compare mappings.

| Placement | RoRaBaCoCh | RoCoRaBaCh | XOR-bank |
|---|---|---|---|
| SE first-touch | 0.941 | 0.800 | 0.979 |
| 2 MiB pages shuffled | 0.941 | 0.800 | 0.980 |
| 4 KiB pages shuffled | 0.965 | 0.401 | 0.967 |

Random 4 KiB placement raising the RoRaBaCoCh figure is unexplained. The working hypothesis is that it acts like a bank hash. Check it in a real DRAM simulator before relying on it.

**DRAM simulator replay (`s15_l3-2m_stride`, 2 ranks DDR4-2400)**

DRAMsim3 replays at the trace's timestamps:

| Mapping | Policy | Row-hit rate | Read latency (cycles) | ACTs |
|---|---|---|---|---|
| `rochrababgco` | open | 0.949 | 43.6 | 113,851 |
| `rocorababgch` | open | 0.632 | 44.3 | 813,384 |
| `rocorababgch` | close | 0 | 55.1 | 2,213,163 |

`rochrababgco` with close page is excluded: it saturated (1389-cycle read latency) and did not finish the trace (2,021,105 of 2,071,299 reads) within the cycle budget.

Ramulator 2.1 `LoadStoreTrace` has no timestamps, so it replays saturated, one request per cycle when accepted. Its latency is not comparable with DRAMsim3:

| Mapper | Policy | Row-hit rate | Conflict rate | Read latency (cycles) | Total Mcycles |
|---|---|---|---|---|---|
| RoBaRaCoCh | open | 0.970 | 0.017 | 238.9 | 14.1 |
| RoBaRaCoCh | ClosedCAP | 0.784 | 0.013 | 369.1 | 22.4 |
| MOP4CLXOR | open | 0.799 | 0.162 | 177.8 | 9.5 |
| MOP4CLXOR | ClosedCAP | 0.641 | 0.136 | 193.5 | 9.5 |
| ChRaBaRoCo | open | 0.966 | 0.033 | 290.6 | 17.5 |
| ChRaBaRoCo | ClosedCAP | 0.780 | 0.029 | 536.1 | 33.2 |

- **Hit rate alone can mislead.** When saturated, MOP4CLXOR has the lower hit rate but finishes in about 33% fewer cycles than RoBaRaCoCh with open page, because it spreads sequential streams across banks. Report latency and completion time alongside hit rate.
- **At the trace's real load**, mapping barely changes read latency (43.6 against 44.3 cycles) but changes ACT count about sevenfold, so activation energy may matter more than latency on this model. This one trace is low-bandwidth and neither result is general yet.
- MOP4CLXOR is not the offline `XOR-bank` model above; the two figures should not be compared.

**Long context: `s15_c1024_l3-2m` (1023-token restored prompt, 2 MiB LLC)**

Validated: text matches native, instructions per token match (147.8M against 148.2M), and trace counts equal gem5's counters in all detailed phases.

| Measure | 36-token context | 1024-token context |
|---|---|---|
| KV share of reads | 1.6% | 32.9% |
| Weights share of reads | about 89% | 60.9% |
| DRAM reads per token | about 215k | about 326k |
| gem5 row-hit rate | 0.918 | 0.902 |
| +1-line strides | 62% | 51% |

- KV traffic is now substantial, so the KV gather question can be studied. This is still stories15M only.
- Offline estimate at this context: RoRaBaCoCh 0.942 first-touch and 0.937 4 KiB shuffled, RoCoRaBaCh 0.597 and 0.286, XOR-bank 0.964 and 0.937. The earlier rise under 4 KiB shuffling does not appear here.
- `analyze.py` totals and exports now cover detailed phases only, from the first through the last, including requests between phases. This run's export has 1,405,509 requests.

**Long context: 4096-token restored prompt (`s15_c4096_l3-2m`, `s15_c4096_l3-22m`)**

Both validated: text matches native, and trace counts equal gem5's counters in all detailed phases.

| Measure | 1024, 2 MiB | 4096, 2 MiB | 4096, 22 MiB |
|---|---|---|---|
| KV share of reads | 32.9% | 65.5% | 66.6% |
| Weights share of reads | 60.9% | 30.1% | 30.9% |
| DRAM reads per token | about 326k | about 670k | about 653k |
| gem5 row-hit rate | 0.902 | 0.882 | 0.894 |
| +1-line strides | 51% | 43% | 44% |
| Offline RoRaBaCoCh, first-touch | 0.942 | 0.935 | 0.927 |
| Offline RoCoRaBaCh, first-touch | 0.597 | 0.326 | 0.333 |
| Offline XOR-bank, first-touch | 0.964 | 0.931 | 0.931 |

- KV becomes the largest read source at 4096 tokens, with a 27 MiB footprint, and row-hit rate falls as context grows.
- A 22 MiB LLC no longer filters traffic: reads per token are close to the 2 MiB figure, against about 20 at short context. KV plus weights is about 40 MiB. Cyclic scans larger than the cache evicting themselves would explain this, but that is not confirmed.
- XOR-bank's offline advantage over RoRaBaCoCh disappears at 4096 tokens.

**DRAM simulator replay across context length (stories15M, open page unless noted)**

All runs finished the trace with every request accounted for. Short context is `s15_l3-2m_stride`; the others are the long-context traces above.

| Measure | Short, 2 MiB | 1024, 2 MiB | 4096, 2 MiB | 4096, 22 MiB |
|---|---|---|---|---|
| DRAMsim3 `rochrababgco` hit rate | 0.949 | 0.930 | 0.902 | 0.919 |
| DRAMsim3 `rochrababgco` read latency | 43.6 | 43.2 | 37.4 | 37.0 |
| DRAMsim3 `rocorababgch` hit rate | 0.632 | 0.457 | 0.241 | 0.248 |
| DRAMsim3 `rocorababgch` read latency | 44.3 | 49.8 | 54.4 | 54.1 |
| DRAMsim3 `rochrababgco` close-page read latency | did not finish | 959.9 | 491.4 | 498.3 |
| DRAMsim3 `rocorababgch` close-page read latency | 55.1 | 54.5 | 51.7 | 51.6 |
| Ramulator RoBaRaCoCh hit rate | 0.970 | 0.957 | 0.945 | 0.957 |
| Ramulator RoBaRaCoCh Mcycles | 14.1 | 8.0 | 13.9 | 13.5 |
| Ramulator MOP4CLXOR hit rate | 0.799 | 0.658 | 0.470 | 0.485 |
| Ramulator MOP4CLXOR Mcycles | 9.5 | 6.0 | 12.0 | 11.7 |

- Mapping now affects latency at the trace's real load: at 4096 tokens, line interleaving is about 45% slower than `rochrababgco` (54.4 against 37.4 cycles), with about 8 times the ACTs. At short context the two were nearly equal.
- Under saturation, MOP4CLXOR's completion-time advantage over RoBaRaCoCh shrinks with context: 33%, 25%, then 14%. Its conflict rate reaches 42%, and its read latency is now slightly worse (194.9 against 186.0).
- Row-interleaved close page is poor for this workload in every run. At 4096 tokens, close page with line interleaving (51.7) slightly beats open page with line interleaving (54.4), but not open page with row interleaving (37.4).
- These stories15M traces do not yet show a mapping that beats `rochrababgco` at real load. MOP4CLXOR has not been replayed with trace timing, because Ramulator's frontend has no timestamps.

### Qwen 0.5B and tinymoe4

All with a 22 MiB LLC, stride prefetcher, 2 detailed tokens.

**Validation:** in all three runs the gem5 text is a prefix of the native text (gem5 stops after 2 tokens), and trace counts equal gem5's counters in every detailed phase. Instructions per token match native: Qwen at 1024 tokens 3,799M against 3,801M, MoE 6,353M against 6,353M.

| Measure | Qwen, short | Qwen, 1024 | MoE, 1024 |
|---|---|---|---|
| Run | `qwen_l3-22m_stride` | `qwen_c1024_l3-22m` | `moe_c1024_l3-22m` |
| Minsts per token | 2,831 | 3,799 | 6,353 |
| DRAM reads per token | about 5.43M | about 5.63M | about 30.3M |
| MPKI | 1.94 | 1.50 | 4.79 |
| gem5 row-hit rate | 0.923 | 0.921 | 0.933 |
| KV share of reads | 0.1% | 3.5% | 1.2% |
| Weights share of reads | 99.0% | 95.6% | 15.0% dense + 83.3% experts |
| Footprint (MiB) | weights 330, KV 0.3 | weights 330, KV 12.1 | experts 1,543, dense 278, KV 22.1 |
| +1-line strides | 41% | 41% | 94% |
| Offline RoRaBaCoCh, first-touch | 0.983 | 0.983 | 0.988 |
| Offline RoCoRaBaCh, first-touch | 0.964 | 0.937 | 0.974 |
| Offline XOR-bank, first-touch | 0.989 | 0.989 | 0.990 |

- **Weights dominate real models.** On Qwen, KV is only 3.5% of reads at 1024 tokens, against 32.9% on stories15M at the same context. Qwen streams its 330 MiB of weights every token, while its KV stays small (12 MiB at 1024 tokens). The large KV shares on stories15M come from its tiny 12.6 MiB of weights, and do not carry over. KV becomes significant on Qwen only at much longer contexts or batch sizes, which is not yet measured.
- **`tinymoe4` is not a useful MoE workload.** Its router outputs 2 logits per token and top-k keeps 2 (`ffn_moe_logits` and `ffn_moe_topk` are 8 bytes per token; `ffn_gate_inp` is 2048 x 2). Both experts run every token, so it reads almost the whole 1.98 GB model per token, with 94% +1-line strides and no routing irregularity. The expert-routing part of the study needs a model with many experts and a small top-k. Candidates to check for size and llama.cpp support: OLMoE-1B-7B (64 experts, top-8), granite-3.0-1b-a400m (32 experts, top-8), Qwen1.5-MoE-A2.7B (60 experts, top-4).
- **Row-buffer locality is already high** on all three, at 0.92–0.93 in gem5 and 0.98–0.99 offline, because the traffic is mostly sequential weight scans. The room for a better mapping on these workloads looks small. Bank parallelism and ACT energy are more promising angles than hit rate.
- **Restored prompts save most of the run:** `qwen_l3-22m_stride` spent about 14 hours fast-forwarding a 24-token prompt (about 58G instructions). The restored 1024-token run finished in under 7 hours.
- DRAM simulator replay of these traces is not done yet. Their exports have 11.0M (Qwen short), 11.4M (Qwen 1024) and 60.9M (MoE) requests.

**Native cost per generated token with a restored prompt** (`(insts[n=5] - insts[n=1]) / 4`)

| Model | Prompt tokens | Minsts/token | Session size |
|---|---|---|---|
| stories15M | 256 | 88 | 1.8 MB |
| stories15M | 1024 | 148 | 7.1 MB |
| stories15M | 4096 | 390 | 28 MB |
| Qwen 0.5B | 256 | 3,059 | 3.1 MB |
| Qwen 0.5B | 1024 | 3,801 | 12.6 MB |
| tinymoe4 | 256 | 4,799 | 5.7 MB |
| tinymoe4 | 1024 | 6,353 | 23.1 MB |

O3 ran at about 0.26 MIPS in the prompt-cache test, so a Qwen token takes roughly 3.3 hours of detailed simulation at this context.

## Limitations

- **SSE2-only binary:** gem5's decoder lists `pmaddubsw`, `phadd*` and `psign*` as unimplemented, so SSSE3 could silently compute wrong values. Instruction mix and weight layout (no AVX2 repack) therefore differ from a normal build.
- **SE-mode page placement:** first-touch and contiguous, not a real Linux allocator. Syscalls and kernel work are invisible; weights are loaded by emulated `read()`.
- **One core, one thread, one channel.** Multi-thread interleaving is not captured.
- **Stride prefetcher** is an approximation of real hardware prefetchers.
- **Bandwidth regime:** replayed traces use only 1–1.8% of one DDR4 channel, so mapping effects at real timing are understated (`ANALYSIS.md`, F5).
- **Short contexts** under-represent KV traffic. Even at 1024 tokens, KV is small on Qwen, and longer contexts are not yet captured for real models.
- **`tinymoe4` has 2 experts, both used every token**, so no trace yet has MoE routing behaviour.
- **Trace timing** reflects gem5's DDR4 model, so replay in another DRAM configuration is not closed-loop.

## Next steps

The complete analysis of all experiments so far is in `ANALYSIS.md`, with generated tables and figures in `analysis/` (`tools/report.py`). The direction is now a full-scale study of application-grade models (Qwen2.5-7B dense, a Qwen MoE with real routing), aimed at publication. The plan and its reasoning are in `ANALYSIS.md`, section "Plan for the full-scale study". In order:

1. Choose the MoE model (Qwen3-30B-A3B or Qwen1.5-MoE-A2.7B); check llama.cpp support, memory and native speed on the cluster.
2. Confirm DynamoRIO `drmemtrace` works on the cluster CPUs with an AVX2 multi-threaded llama.cpp build.
3. Build the trace route: cache-model filter, physical placement model, closed-loop Ramulator CPU-trace replay. Validate it against the gem5 Qwen2.5-0.5B trace.
4. Capture full-scale scenarios: contexts 1k, 8k and 32k; batch 1, 4 and 16; prefill separately; 8-channel DDR5-4800 as the primary memory system.
5. Characterize the baselines, including an application-specific XOR mapping in the style of Rotaru et al.
6. Implement and evaluate a region-aware mapping and page policy.
7. Search 2023–2026 related work before claiming novelty.

Housekeeping: cancel `moe_l3-22m_stride` (114798); revoke the trace-upload token.

## Environment

- **Cluster:** `ssh adityaseshasai.s@rce.iiit.ac.in`. Slurm; existing jobs exclude GPU nodes `node04,node05`. Run numpy on compute nodes (it fails on login) with `OPENBLAS_NUM_THREADS=1`.
- **Containers do not work:** `max_user_namespaces=0` on the nodes checked and login, so `apptainer exec` fails.
- **VPN:** NetworkManager profile `iiit`. As a full tunnel it made Claude unreachable. On 2026-11-02 it was changed to split tunnel (never-default, route `10.4.0.0/16`, DNS search `~iiit.ac.in`), and this was confirmed working the same day: cluster traffic goes via `tun0`, other traffic does not, and SSH connects.

To undo it:

```bash
nmcli connection modify iiit ipv4.never-default no ipv6.never-default no ipv4.routes "" ipv4.dns-search ""
```

## Where things are

**Backup:** private GitHub repo `adityaseshasai-samineni/amcas-project` holds the code, this file, logs and small results (`cluster-data/`). Raw traces are in release `traces-2026-11-03`, one `<run>.dram.trc.gz` per finished run plus `SHA256SUMS`; `moe_l3-22m_stride` is not included because it was still running. They were uploaded from the cluster by `scripts/09_upload_traces.sh`, which reads a repo-scoped token from stdin. Git ignores `*.trc.gz`. Not backed up: `tracing/` (rebuildable), `dram.bin` and exports (regenerated from the traces).

```bash
gh release download traces-2026-11-03 -R adityaseshasai-samineni/amcas-project -D cluster-data/release
cd cluster-data/release && sha256sum -c SHA256SUMS
```

**Local:** `trace-pipeline/`

| Path | Purpose |
|---|---|
| `configs/llm_trace.py` | gem5 capture config |
| `patches/apply_patches.py` | gem5 and llama.cpp patches |
| `tools/pkt2bin.c` | trace decoder |
| `tools/analyze.py` | validation, attribution, offline mappings, export |
| `tools/ramulator_replay.py` | Ramulator 2.1 replay over mappers and row policies |
| `scripts/09_upload_traces.sh` | upload raw traces from the cluster to a GitHub release |
| `scripts/01`–`08` | environment build, llama build, native baseline, capture (`SESSION=` restores a prompt), `gem5.fast`, DRAM simulators, prompt-cache test, long-context sessions and native cost |

**Cluster:** `~/amcas-pro/`

| Path | Purpose |
|---|---|
| `configs/`, `patches/`, `tools/`, `scripts/` | same as local (`scripts/` also has a few throwaway helpers) |
| `tracing/` | gem5, llama.cpp, models, micromamba environment |
| `results/native/` | native baseline output |
| `results/traces/<run>/` | `dram.trc.gz`, `dram.bin`, `phases.tsv`, `amcas_tensors.tsv`, `stats.txt`, `summary.json`, exports |
| `logs/` | job logs |

To capture a run:

```bash
sbatch --export=ALL,GEM5=gem5.fast -J NAME -o logs/trace/NAME_%j.log scripts/04_trace.sh \
  NAME MODEL CTX N_PREDICT --detail-from 1 --max-work-items 2 --l3 22MiB --l3-assoc 11 --prefetch stride
```

## Earlier cluster work

- NVBit GPU trace of `vectoradd` (`traces/vectoradd_memtrace.log`): not DRAM traffic.
- Models in `tracing/models/`: `stories15M`, `qwen25-05b-q4_0`, `tinymoe4` (2 experts, top-2, 1.98 GB).
- `gem5-deps.sif` and `build.sh` are obsolete, because containers cannot run.

## Explanation sessions and decisions (2026-11-03)

This section records the walkthrough given in chat on 2026-11-03 and the decisions taken at the end of it. Nothing above was changed; the full analysis is in `ANALYSIS.md`.

### Part 1: the problem statement

- DRAM is organized as banks, each a grid of rows. To read data, the controller opens a whole row (8 KiB here) into that bank's row buffer, then reads the 64-byte line it needs.
- A **row hit** is a request to the row that is already open: fast. A **row conflict** is a request to a different row in the same bank: the open row must be closed and the new one opened, which is slow and costs activation energy.
- **Address mapping** decides which address bits select rank, bank, row and column. **Row-buffer policy** decides whether a row stays open after an access (open page) or closes at once (close page).
- The problem statement claims attention (KV-cache gathers) and MoE (expert routing) access memory differently from classic HPC. The task is to trace LLM memory traffic, replay it in DRAMsim3 and Ramulator, measure row-buffer behaviour, propose a better mapping or policy, and show the improvement.

### Address mapping, worked through

Our simulated DDR4 has 2 ranks x 16 banks (4 bank groups x 4 banks), 65,536 rows per bank and 8 KiB rows, so a physical address uses 34 bits: 6 line offset, 7 column, 4 bank group and bank, 1 rank, 16 row.

- **Row-interleaved** (`rochrababgco`): `[row 16][rank 1][bank 2][bank group 2][column 7][offset 6]`. Consecutive lines stay in one row: good locality, but a sequential stream keeps one bank busy at a time.
- **Line-interleaved** (`rocorababgch`): `[row 16][column 7][rank 1][bank 2][bank group 2][offset 6]`. Consecutive lines rotate across all 32 banks: good parallelism, but each bank sees one line per visit and loses its hit if anything else touches it.
- **XOR hashing**: bank bits XOR some row bits. Keeps sequential runs together, but scatters arrays that align on power-of-two boundaries, which would otherwise collide in one bank with different rows (row ping-pong). Ramulator's `MOP4CLXOR` also switches bank every 4 lines.
- The core trade-off is **locality against parallelism**.

Worked example (bank shown as one number 0-31 combining rank, bank and bank group):

| Addresses | Row-interleaved | Line-interleaved | XOR (bank ^ low row bits) |
|---|---|---|---|
| Bytes 1,000,000 to 1,000,010 | all bank 26, row 3, column 9 | all bank 9, row 3, column 104 | all bank 25 |
| Lines 1,000,000 + 64k, k = 0..10 | bank 26, row 3, columns 9 to 19 | banks 9 to 19, row 3, column 104 | bank 25 |
| 1,000,000 + 262,144k, k = 0..3 | bank 26, rows 3, 4, 5, 6 (conflicts) | not shown | banks 25, 30, 31, 28 |

- Bytes inside one 64-byte line never differ to DRAM: the cache fetches the whole line.
- Consecutive lines show the trade-off: row-interleaved keeps one row open in one bank, line-interleaved spreads across banks.
- Same column in successive rows shows XOR's purpose: four rows that would ping-pong in bank 26 spread over four banks.

### Part 2: how the traces were captured

- A trace lists every request reaching DRAM, with time, address and read or write. It is captured by a `CommMonitor` between the memory bus and the DRAM controller, so it holds exactly what the controller sees: LLC misses, writebacks and prefetches. Our gem5 patch adds the virtual address, which lets each request be attributed to weights, KV cache or other data.
- The real cluster cannot provide this: NVBit on the GPU sees pre-cache accesses, and `perf`/PEBS needs root for physical addresses. So gem5 simulates a whole computer running real llama.cpp.
- llama.cpp changes: SSE2-only build (gem5's x86 model gets some newer vector instructions wrong), static binary (SE mode cannot load shared libraries reliably), and phase markers around each token's computation.
- Speed: gem5's O3 CPU runs at about 0.26 MIPS, about 4 hours per Qwen 0.5B token. Model loading is fast-forwarded on a simple CPU that warms the caches; long prompts are evaluated natively, saved as a session and restored inside gem5.
- Validation: every run generates the same text as native, and trace counts equal the controller's counters.

### Part 3: what the traces showed

- **Each token reads the whole model.** Qwen 0.5B at 1,024 context reads about 360 MB per token against 330 MB of weights; the 22 MiB LLC saves almost nothing.
- **KV share depends on model size.** stories15M: 1.6% short, 33% at 1,024, 66% at 4,096 (its weights are 12.6 MB). Qwen 0.5B at 1,024: 3.5%. KV catches up with weights at batch x context of about W/k: about 28k tokens for Qwen 0.5B, about 77k for Qwen 7B (derived estimates). KV gather matters for long-context or multi-user serving.
- **`tinymoe4` is not a real MoE workload** (2 experts, both used every token).
- Asked which mapping gives the higher hit rate on sequential weight scans: **row-interleaved**, correctly (Qwen 0.983 against 0.937 offline; stories15M at 4,096 in DRAMsim3 0.902 against 0.241).
- **The DRAM was 1-1.8% busy** in timed replay, because one SSE2 thread issues requests slowly. At that load mapping barely changes latency (43.6 against 44.3 cycles on the short trace).
- **Under saturation the "worse" mapping wins:** XOR hashing (`MOP4CLXOR`) finishes 33%, 24% and 13% faster than row-interleaved on short, 1,024 and 4,096 traces, despite hit rates of 0.80, 0.66 and 0.47. Keeping all banks busy matters more than avoiding row opens when memory is the bottleneck. The gap shrinks as KV grows, hinting that weights and KV want different mappings: the seed of a region-aware mapping.

### How many tokens were simulated

42 token steps in detail across the analysed runs: 3 short stories15M runs x 8, 3 long-context stories15M runs x 4, and 3 Qwen and MoE runs x 2. Each restored-prompt run also generated one atomic warm-up token, excluded from analysis; the prompt-cache test simulated 9 more for validation only. Decode is very repetitive (Qwen's two tokens differed by 26 reads in 5.6M), so this is enough for dense decode, but a real MoE varies per token with expert choice and needs tens to hundreds of tokens per run, and a paper should report variance across tokens and prompts.

### Part 4: why gem5 cannot scale, and the plan

- Three walls: speed (Qwen 7B at roughly 50G instructions per token would take about 2 days per token), instructions (gem5 x86 lacks AVX, so traces have the right addresses but the wrong pace), and one core with one channel (real servers run 16-64 threads on 8-12 channels).
- Plan: run llama.cpp natively (AVX2, multi-threaded), record memory accesses with DynamoRIO (no root), filter through a model of the server's caches, assign physical addresses with a huge-page placement model, and replay closed-loop in Ramulator.
- Open-loop replays are wrong in opposite directions: timestamped replay cannot show a speedup, saturated replay ignores that loads depend on earlier loads. Closed-loop replay stalls each core on outstanding misses, so the headline result becomes tokens per second under each mapping.
- Validation of the new route: run Qwen 0.5B SSE2 single-threaded through it and reproduce the gem5 trace (reads per token, region shares, strides).

### Decisions taken

- **Models:** Qwen3-30B-A3B (MoE, 128 experts, 8 active), Qwen2.5-7B (dense), Qwen2.5-0.5B (cross-validation with gem5).
- **Automation:** every full-scale script publishes its results to GitHub when it finishes, and regenerates and publishes the analysis.
- **This file:** add to it; do not delete existing content.

### Full-scale phase: setup and first jobs (2026-11-03, about 22:15 IST)

Cluster facts found:
- **node01**: 2x Xeon Gold 6226R (Cascade Lake), 32 cores, 384 GB RAM, AVX-512 VNNI, 22 MiB L3 per socket. node03: 4x Gold 6226 (48 cores). node06: 2x Gold 5317 (Ice Lake, 24 cores). All have AVX-512 VNNI.
- `/scratch` is local to each node: 650 GB free on node01, 724 GB on node03, 75 GB on node06. Home has a **100 GB quota** (28 GB used before the new models).
- The llama.cpp checkout (`0ec4027`, 2026-10-30) supports `LLM_ARCH_QWEN3MOE`.
- Model files: `Qwen3-30B-A3B-Q4_K_M.gguf` (18.56 GB) from `Qwen/Qwen3-30B-A3B-GGUF`; `qwen2.5-7b-instruct-q4_k_m-0000{1,2}-of-00002.gguf` (3.99 + 0.69 GB) from `Qwen/Qwen2.5-7B-Instruct-GGUF`. Qwen2.5-0.5B reuses `qwen25-05b-q4_0.gguf`.
- `moe_l3-22m_stride` (114798) completed after 26h 58m; not yet analysed.
- A separate micromamba environment `analysis` (Python 3.12, numpy, matplotlib) runs `report.py` on the cluster, leaving the gem5 environment untouched. numpy on the login node needs `OPENBLAS_NUM_THREADS=1`.

New scripts (all end by calling `tools/publish.sh`):

| Script | Job | What it does |
|---|---|---|
| `10_fetch_models.sh` | 115475 | downloads the two new models with resume, checks SHA-256 against Hugging Face |
| `11_build_llama_fast.sh` | 115476 | builds `build-fast` on node01: native SIMD (AVX-512), OpenMP; `llama-completion`, `llama-bench`, `llama-batched-bench` |
| `13_dynamorio_check.sh` | 115477, after 115476 | installs DynamoRIO, traces 2 tokens of Qwen 0.5B on 4 threads into node01 `/scratch`, reports slowdown, trace size and a cache simulation |
| `12_native_fullscale.sh` | 115478, after all three | node01 exclusive: decode speed against threads (1, 8, 16, 32) and context depth (0, 1k, 8k, 32k); batched decode with 1, 4, 16 sequences at 1k and 8k prompts; JSON in `results/fullscale/native/` |

`tools/publish.sh` (automatic GitHub upload):
- Copies results and logs under 50 MB into a clone at `~/amcas-repo/cluster-data/`, regenerates `analysis/` with `report.py`, commits and pushes; large traces named as arguments go to release `traces-2026-11-03`.
- Scripts and tools are not copied back from the cluster, so newer laptop edits are never overwritten. Sync scripts laptop to cluster before submitting.
- One publisher at a time (lock directory `~/.config/amcas/publish.lock`).
- Needs a fine-grained token, this repository only, Contents read and write, at `~/.config/amcas/gh_token` mode 600. Without it, jobs skip publishing and results stay on the cluster. The token reaches git through a credential helper and curl through stdin, never a command line.

### Overnight run, 2026-11-03 to 2026-11-04 (handoff written about 22:25 IST)

Everything below runs on the cluster and publishes itself; the laptop can be off. GitHub publishing is set up: a fine-grained token (this repo, Contents read and write) is at `~/.config/amcas/gh_token` on the cluster, checked to have push access, and a test publish pushed `8cbbc5c`. A clone used for publishing lives at `~/amcas-repo` on the cluster. `publish.sh` now clears a lock older than 2 hours.

| Job | What | Node | Waits for | Publishes |
|---|---|---|---|---|
| 115475 `fetch-models` | Qwen2.5-7B Q4_K_M (done and verified by 22:18), then Qwen3-30B-A3B Q4_K_M (18.6 GB) | node06 | - | yes |
| 115482 `dynamorio-check` | installs DynamoRIO, traces Qwen 0.5B on 4 threads, reports slowdown, trace size, cache simulation | node01 | - | yes |
| 115478 `native-fullscale` | all three models, node01 exclusive: threads, depth, batch | node01 | fetch and build | yes |
| 115490 `an_moe_l3` | `analyze.py` on `moe_l3-22m_stride` | not node01 | - | yes |
| 115495, 115496, 115497 | DRAM sweep (`06_dram_sims.sh`) of `qwen_l3-22m_stride`, `qwen_c1024_l3-22m`, `moe_c1024_l3-22m` | not node01 | - | yes |
| 115498 | DRAM sweep of `moe_l3-22m_stride` | not node01 | 115490 | yes |

- `115477`, the first DynamoRIO attempt, failed because GitHub's API rate limit (60 an hour per address, shared behind the cluster's NAT) refused the release list. `13_dynamorio_check.sh` now retries for up to an hour.
- `06_dram_sims.sh` now builds DRAMsim3 and Ramulator only when they are missing, so parallel sweeps do not rebuild the same tree; it has 64 GB and 12 hours, and publishes at the end. The first submission (115491-115494) was cancelled before starting and resubmitted with this version.
- The sweeps run on traces captured at about 1-2% DRAM utilization (`ANALYSIS.md`, F5), so they extend the characterization to Qwen and the 2-expert MoE; they are not the full-scale bandwidth-bound results.
- `report.py` now includes every analysed run and every run with DRAM replays in its tables; figures stay on stories15M.
- Home quota: 34 GB of 100 GB used at 22:20, before the 18.6 GB MoE model and the sweep exports (a few GB). Watch it.

In the morning:

```bash
git pull                                   # laptop: brings in everything the jobs published
ssh adityaseshasai.s@rce.iiit.ac.in 'squeue -u $USER; sacct -X --starttime=2026-11-03T22:00 --format=JobID,JobName%22,State,ExitCode,Elapsed'
```

Then read `cluster-data/logs/drcheck_115482.log` first: whether DynamoRIO works decides the full-scale trace route. Next, `cluster-data/results/fullscale/native/` for tokens per second and the bandwidth real inference needs, and `analysis/tables.md` for the Qwen and MoE DRAM sweeps.

### Overnight results (read 2026-11-05, about 11:30 IST)

All jobs published themselves (`2995d47` to `042c79e`). Home quota: 54 GB of 100 GB used.

| Job | Result |
|---|---|
| 115475 `fetch-models` | done in 1h 41m; both models verified against Hugging Face SHA-256 |
| 115478 `native-fullscale` | done in 57m; depth sweep skipped because this `llama-bench` has no `--n-depth`; batched runs cover context instead |
| 115490 `an_moe_l3` | `moe_l3-22m_stride` validated (counts and text match) |
| 115495-115498 | all four DRAM sweeps finished; every DRAMsim3 run completed its trace |
| 115482 `dynamorio-check` | **failed**: GitHub's API refused the release list for the whole hour of retries (shared rate limit) |

**Native full-scale decode on node01** (2x Xeon Gold 6226R, 32 cores, 4 memory controllers, 16 unbuffered DDR4 DIMMs; speed not readable without root). Bandwidth is weight bytes per token times tokens per second; for Qwen3-30B-A3B it uses the active parameters (3.3B of 30.5B), an estimate.

| Model (size) | 1 thread | 8 | 16 | 32 | Weight traffic at 32 threads |
|---|---|---|---|---|---|
| Qwen2.5-0.5B Q4_0 (0.42 GB) | 20.95 tok/s | 100.5 | 117.9 | 119.3 | about 50 GB/s |
| Qwen2.5-7B Q4_K_M (4.68 GB) | 1.77 | 9.18 | 12.44 | 13.55 | about 63 GB/s |
| Qwen3-30B-A3B Q4_K_M (18.55 GB) | 3.84 | 15.78 | 19.79 | 22.01 | about 44 GB/s (estimate) |

- Speed flattens from 16 to 32 threads on all three: decode is memory-bound on this node.
- **Real inference drives about 260 times the DRAM bandwidth of our gem5 traces** (63 GB/s against 0.19-0.24 GB/s for Qwen), and even one native thread drives about 35 times more (8.3 GB/s). This quantifies `ANALYSIS.md` F5's limitation and confirms the full-scale trace route is needed.
- 63 GB/s is likely well below node01's peak (two sockets of six-channel DDR4). The model is probably placed on one NUMA node by first touch; a run with `--numa distribute` or `numactl --interleave=all` would check that. Not yet done.

**Batched decode, 32 threads** (total decode tokens per second across sequences, and per sequence):

| Model | 1k x 1 | 1k x 4 | 1k x 16 | 8k x 1 | 8k x 4 | 8k x 16 |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 82.4 | 236.8 (59.2) | 479.6 (30.0) | 63.8 | 85.7 (21.4) | 127.3 (8.0) |
| Qwen2.5-7B | 11.9 | 39.0 (9.7) | 91.1 (5.7) | 10.5 | 15.1 (3.8) | 17.1 (1.1) |
| Qwen3-30B-A3B | 16.4 | 29.8 (7.5) | 61.4 (3.8) | 11.8 | 10.0 (2.5) | 9.1 (0.6) |

Prefill: about 150-200 tokens per second for 7B and 30B-A3B, 1,150-2,000 for 0.5B.

- **Batching pays at short context, not at long.** At 1k, 16 sequences give Qwen 7B 7.7 times the throughput of one, because they share each weight read. At 8k, 16 sequences give only 1.6 times: KV traffic (131k tokens x 56 KiB, about 7.4 GB per step, more than the 4.7 GB of weights) takes over. Batch x context there is 131k, above the predicted crossover of about 77k (`ANALYSIS.md`, F2), consistent with the crossover model; not yet a direct measurement of KV traffic.
- **The MoE batches much worse than the dense model.** At 1k, 16 sequences give Qwen3-30B-A3B only 3.8 times one sequence (dense 7B: 7.7 times). Different sequences route to different experts, so a batched step reads the union of their experts. At 8k, adding sequences lowers total throughput (11.8 to 9.1). This is the expert-routing effect the problem statement is about, visible at full scale for the first time; it needs traces to measure directly.

**DRAM sweeps of the Qwen and 2-expert MoE gem5 traces** (`analysis/tables.md`):

| Trace | Utilization | Row-interleaved open: hit, read latency | Line-interleaved open: hit, read latency | Row-interleaved close: read latency | MOP4CLXOR saturated cycles vs RoBaRaCoCh |
|---|---|---|---|---|---|
| Qwen, short | 1.2% | 0.971, 44.7 | 0.572, 45.2 | 933.4 | 0.67 |
| Qwen, 1024 | 1.0% | 0.968, 44.4 | 0.553, 45.7 | 898.2 | 0.67 |
| tinymoe4, 1024 | 4.5% | 0.986, 40.8 | 0.888, 38.7 | 1756.8 | 0.67 |
| tinymoe4, short | 7.7% | 0.988, 40.8 | 0.897, 38.5 | 1770.8 | 0.67 |

- Weight-dominated traces keep row-interleaved hit rates near 0.97-0.99. At their low load, line interleaving costs almost no latency despite a much lower hit rate (on the MoE traces it is even slightly faster), but needs 11 to 14 times the ACTs on Qwen and 8 times on tinymoe4.
- Row-interleaved close page serializes badly on every trace (900-1,770 cycles).
- Under saturation, the XOR mapping finishes in 0.67 of RoBaRaCoCh's cycles on all four traces. On weight streams the advantage does not shrink as it did with stories15M's KV-heavy traces (0.87 at 4096 tokens), supporting different mappings for weights and KV.

**DynamoRIO, not yet installed.** The latest GitHub release is a nightly build that needs glibc 2.34; the cluster has glibc 2.28. The newest stable release, `release_11.3.0-1` (`DynamoRIO-Linux-11.3.0.tar.gz`), is the next to try. Installing it and test-tracing `/bin/ls` was blocked by the session's permission check (downloading and running a third-party binary on the cluster), pending the user's decision. Building from source with the conda toolchain (sysroot 2.28) is the fallback if the stable release also needs a newer glibc.

### DynamoRIO installed (2026-11-05)

- The user approved installing the stable release `release_11.3.0-1` (`DynamoRIO-Linux-11.3.0.tar.gz`) into `~/amcas-pro/tracing/dynamorio`. It runs under the cluster's glibc 2.28. The cronbuilds need 2.34 and do not run.
- On the login node, `drrun` aborts with "Out of memory", because the login node limits virtual memory (`ulimit -v` 8192000 KB). On node01 (no limit), both `drrun -- /bin/ls` and `-t drcachesim` work. Run DynamoRIO only on compute nodes.
- `13_dynamorio_check.sh` was resubmitted as job 116659. It skips the install because `bin64/drrun` now exists, and publishes when it finishes.
- `CLAUDE.md` was added at the project root, to guide future Claude Code sessions.

### DynamoRIO check results and rerun (2026-11-05, about 18:00 IST)

Job 116659 (node01, 18 min, published `ab89944`):
- Tracing works. The traced run's text is identical to the native run's. 0.97 s native against 30.2 s traced (about 31× slowdown); max RSS 1.09 to 1.12 GB.
- Raw trace size: 4.4 GB for the prompt plus 2 tokens of Qwen 0.5B on 4 threads. Scaled up, one 7B token at full detail would be a few hundred GB, so the full-scale route needs sampled windows or an online cache filter.
- The offline cache simulation failed during conversion: `raw2trace failed: Failed to process file for thread 2484037: Failed to close prior component`, after 18 minutes.

Rerun, job 116729 (`13_dynamorio_check.sh` revised):
- Traces `llama-completion` directly, without the `bash -c` wrapper.
- Runs `tools/bin64/drraw2trace -jobs 8 -verbose 1` as a separate step. Its log goes to `results/fullscale/drcheck/`.
- Then runs drcachesim on the converted trace.
- If conversion fails, it retraces once with `--no-mmap`.
- Time limit raised to 4 h.
- In the 11.3.0 release tarball, the tools are in `tools/bin64/`, not `clients/bin64/`.

### DynamoRIO rerun results and third attempt (2026-11-05, about 22:50 IST)

Job 116729 (node01, 19 min, published `d41235f`):
- Tracing `llama-completion` directly: same text as native; 0.77 s native against 28.8 s traced; 4.4 GB raw again.
- Conversion still failed, after 1,094 s: `Worker 2 hit error Failed to process file for thread 2823677: Failed to close prior component`. The failure came right after chunk #215 of the main thread (10M instructions per chunk). Writing the default zip output, it fails while closing a component, which points to zip's 4 GB limit. The `bash` wrapper was not the cause.
- The `--no-mmap` fallback was invalid. This llama.cpp rejects the flag (`error: invalid argument: --no-mmap`), so that "trace" covered only argument parsing (15M instructions), and its cache-simulation numbers are meaningless. Here mmap is turned off with `-lm none` (`--load-mode`).

Third attempt, job 116898 (`REUSE=1`):
- Reuses the 4.4 GB raw trace still on node01 `/scratch` and only redoes conversion, with `-compress gzip`, then the cache simulation.
- The fallback is now `-lm none`. An attempt stops if its text differs from native.

### DynamoRIO route works end to end (job 116898, read 2026-11-06)

Job 116898 (node01, 22:48 to 23:52, published `d2de8f4`):
- Conversion succeeded with `-compress gzip`: 6 thread files in 1,221 s, 3.76G instructions, 3.5 GB. This confirms the zip output (probably its 4 GB limit) caused the earlier failures.
- Per-thread trace sizes: the main thread 3.16 GB, the three compute workers about 165 MB each, two other threads tiny. Most of the trace is the main thread, probably model load plus the weight repacking for AVX-512.
- drcachesim, 4 cores, 22 MB 11-way LLC (simulation took 2,632 s, about 1.4M instructions per second):

  | | value |
  |---|---|
  | LL misses | 20.4M (1.31 GB at 64 B) |
  | LL compulsory misses | 15.9M (1.02 GB) |
  | LL local miss rate | 57.7% |
  | LL total miss rate | 0.34% |

  For comparison, the model file is 429 MB. A footprint of about 1.0 GB is consistent with the file plus a repacked copy plus the KV cache and buffers, but load and inference are not separated, so the figures cannot be attributed yet.
- The drcachesim log was cut to its last 40 lines, so cores 0 and 1 are not in it.

Conclusion: the DynamoRIO route works on optimized multithreaded llama.cpp, with the right output and about 30× slowdown. Next: separate inference from load (start tracing after load), then add per-token boundaries and get physical addresses for the DRAM replay.

### Splitting load from inference: job 117023 (submitted 2026-11-06, about 12:00 IST)

New script `scripts/14_drmem_phases.sh [MODEL] [N_PREDICT] [THREADS]` (node01, 6 h):
- Traces with `-record_function 'llama_decode|2'` and `--no-warmup`. `build-fast` links `libllama.so` dynamically and exports `llama_decode`, so DynamoRIO can find it.
- Converts with `-compress gzip`, then runs `basic_counts` per thread.
- Finds each `llama_decode` call on the main thread with the view tool. View's marker format, checked on `/bin/ls` with `malloc`: `<marker: function #ID>` opens both a call and a return. A call is the one followed by `function return address`; a return by `function return value`. The step's start time is the last `<marker: timestamp N>` before it.
- Runs drcachesim from each step's start to the end (`-skip_to_timestamp`), and over the whole trace. Consecutive differences give each step's LLC misses; "whole minus step 0" gives the cost of loading.
- Writes `results/fullscale/drphases/<model>/`: `basic_counts.txt`, `markers.txt`, `step_ts.txt`, `cachesim_*.txt`.
- Caveat: each run from a step starts with a cold cache. With a 22 MB LLC against about 430 MB of weights, that should add little.

### Phase split, second design: job 117023 timed out; new job (2026-11-06, about 18:10 IST)

- Job 117023 hit its 6 h limit inside the view step. The view tool (it disassembles every instruction) did not get through the main thread's 2.56G instructions in about 5.5 h. Before that: native 0.64 s, traced 19.9 s, same text; raw 2.8 GB (4.4 GB with warmup); raw2trace 953 s.
- Instruction counts from basic_counts (Qwen 0.5B, 4 threads, `--no-warmup`, prompt plus 2 tokens), 3.37G in total:

  | thread | instructions |
  |---|---|
  | main | 2.56G |
  | each of 3 workers | 0.26–0.30G |

  If the main thread does about one worker's share of compute, loading is about 2.3G (about 68%) and inference about 1.1G.
- This llama.cpp's completion tool does not call `llama_decode`. It calls `llama_process(ctx, type, llama_batch_ext *)`, so `-record_function 'llama_decode|2'` could never have marked anything.
- `drcachesim -skip_to_timestamp` fails on these traces: "Missing as-traced istream", and with `-cpu_schedule_file` "Failed to read initial input contents". The traces' `cpu_schedule.bin.zip` is empty (22 bytes).
- What works: `drrun -t drcachesim -indir T -simulator_type record_filter -outdir O -trim_before_timestamp A -trim_after_timestamp B`, tested on `/bin/ls`. The 11.3.0 release ships `record_filter_launcher.debug` only as debug info, so it must be run this way.
- DynamoRIO timestamps are wall-clock microseconds since 1601-01-01 UTC (`unix_us + 11644473600e6`). Verified: a trace timestamp decoded to the time of the run.
- New `tools/decode_stamp.c`, built to `tracing/lib/libdecode_stamp.so`: an LD_PRELOAD wrapper that logs `B|E type dr_us` around each `llama_process` (and `llama_decode`). Tested natively on node01: Qwen 0.5B `-n 2` gives 2 steps, the prompt (about 60 ms) and one token (about 20 ms).
- `14_drmem_phases.sh` now:
  - logs steps with the wrapper;
  - cuts windows `load` (before the first step) and `stepK` (each step's B to E) with record_filter, in parallel;
  - runs basic_counts and drcachesim on each window (cold cache) and on the whole trace;
  - writes `windows_summary.txt`.

### Phase split works; DynamoRIO matches gem5 (job 117350, read 2026-11-06, about 20:50 IST)

Job 117350 ran on node03, moved from node01 while pending. node01 was full, and node03 is a Xeon Gold 6226, Cascade Lake with the same AVX-512 flags. It took 1 h 46 min and was published as `f9ebe3e`.
- The LD_PRELOAD wrapper logged both steps under DynamoRIO. Native: prompt 44 ms, token 12 ms. Traced: 0.62 s and 0.25 s, of a 22.5 s traced run.
- Windows, cut with record_filter. Each was cache-simulated cold, 4 cores, 22 MB 11-way LLC. DRAM reads are LL demand misses plus LL prefetch misses, at 64 B:

  | window | instructions | LL misses | LL prefetch misses | DRAM reads | MB |
  |---|---|---|---|---|---|
  | load | 2.200G | 8,076,073 | 7,661,276 | 15.74M | 1,007 |
  | step0 (prompt, about 11 tokens) | 0.835G | 2,837,874 | 2,658,985 | 5.50M | 352 |
  | step1 (one token) | 0.254G | 2,737,966 | 2,735,175 | 5.47M | 350 |
  | whole trace | 3.370G | 14,318,117 | 13,496,253 | 27.81M | 1,780 |

- **Cross-validation:** gem5 `qwen_l3-22m_stride` measured 5,434,710 DRAM reads per token. DynamoRIO plus drcachesim gives 5,473,141 for step1, **0.7% apart**. This is despite a different ISA path (SSE2 single thread against AVX-512 on 4 threads), a different prefetcher, and gem5's physical addresses against DynamoRIO's virtual ones. Instructions per token differ about 11× (2,831M in gem5 against 254M), because of AVX-512.
- Loading is 65% of the instructions and 57% of the DRAM reads. The prompt step reads the same bytes as one token (the weights are read once per batch) with 3.3× the instructions.
- Rate for the full-scale estimate: the step1 simulation ran at about 2.2M instructions per second. A 7B token is roughly 14× the 0.5B, so about 3.5G instructions and about 30 min of cache simulation per token, which is feasible.
- Note for analysis: in drcachesim's LL stats, "Compulsory misses" includes first touches served by prefetch, so it can exceed "Misses". Count DRAM reads as Misses plus Prefetch misses.

### DRAM request streams from DynamoRIO windows: job 117563 (submitted 2026-11-06, about 22:30 IST)

- `drcachesim -LL_miss_file F.gz` writes one line `tid,pc,byte address` per **demand** LLC miss only; prefetch fills are not written. Test on `/bin/ls`: 4,799 lines equal LL "Misses", while "Prefetch misses" was 2,466. With `-data_prefetcher none`, every line reaching DRAM is a demand miss: 6,986 lines, against 7,265 misses plus prefetch misses with it on. The streams therefore use the prefetcher off. drcachesim's only data prefetcher is `nextline`; gem5 used a stride prefetcher at L2.
- drcachesim does not model dirty lines, so there are no writebacks, and store misses count as reads. gem5 Qwen 0.5B writes were about 1% of reads per token.
- New `tools/drmiss_export.py MISS OUT [--placement]`:
  - imports `analyze.py`'s mappings, `open_page_stats` and stride categories, so DynamoRIO and gem5 numbers use the same definitions;
  - reports stride structure merged and per thread;
  - computes offline row-hit rates for three placements: VA-order packed (closest to gem5 SE, whose model pages were first touched in file order), 4 KiB shuffled, and THP 2 MiB shuffled, all in 8 GiB;
  - writes `dram.ramulator.trc` for the chosen placement (default VA-order packed). Tested locally on synthetic data.
- New `scripts/15_drmem_dram_trace.sh [MODEL] [WINDOWS]` (node03, where job 117350's windows are on `/scratch`): for each window, in parallel, drcachesim without prefetch plus a miss file, then `drmiss_export.py`, then `ramulator_replay.py`. Output in `results/fullscale/drdram/<model>/<window>/`. Job 117563 runs step0 and step1 of Qwen 0.5B.

### Burst tracing: trace only chosen steps (2026-11-06, about 23:00 IST)

Model loading was 65% of a short Qwen 0.5B trace and grows with the model, so tracing the whole process would not scale to 7B or 30B. New `tools/drburst/`:
- `drburst_main.cpp` builds `llama-drburst`: llama-completion's `llama_completion(argc, argv)` entry point, with drmemtrace linked statically. It calls `dr_app_setup()` at start. Its own exported `llama_process` (found with `-rdynamic` and `RTLD_NEXT`) calls `dr_app_start()` at the start of step `AMCAS_SKIP_STEPS` and `dr_app_stop_and_cleanup()` after `AMCAS_TRACE_STEPS` steps. Step times go to `AMCAS_STEPS`, in the same format as `decode_stamp.c`. Tracer options are given as `DYNAMORIO_OPTIONS="-stderr_mask 0xc -client_lib ';;-offline -outdir DIR'"`.
- `CMakeLists.txt`, built to `tracing/drburst-build/`:

  ```bash
  cmake -S tools/drburst -B $W/drburst-build -DDynamoRIO_DIR=$W/dynamorio/cmake -DLLAMA=$W/llama.cpp -DDRLIBS=$W/mamba/envs/drlibs
  ```

  Build pitfalls, all handled in the file:
  - drmemtrace_static is not an imported target: link it from `tools/lib64/release`, plus the `*_static` extensions, including `drpttracer_static`.
  - It needs lz4, snappy and zlib. These are in the new micromamba env `drlibs`, installed from conda-forge (`lz4-c snappy zlib`).
  - The release's imported targets hard-code `/usr/lib/x86_64-linux-gnu/libpthread.so` and `libz`; these are rewritten.
  - The conda linker needs every llama and ggml library listed, with `-rpath-link`. `BUILD_RPATH` did not take effect, so an explicit `-Wl,-rpath` is used.
- Test (node03, Qwen 0.5B, 4 threads, `-n 2`, `AMCAS_SKIP_STEPS=1 AMCAS_TRACE_STEPS=1`, so the one token step only):

  | | full trace, cut afterwards (job 117350) | burst |
  |---|---|---|
  | traced run | 22.5 s | 1.7 s |
  | raw trace | 2.8 GB | 531 MB |
  | token-step instructions | 254M | 270M (4 threads, about 68M each) |
  | token DRAM reads (LL misses plus prefetch misses) | 5,473,141 | 5,491,736 |

  The text was the same as an untraced run. gem5 measured 5,434,710 DRAM reads per token, so burst is 1.0% from gem5.
- Throwaway test script: `~/amcas-pro/scratchtests/burst_test.sh` (cluster only).

### Burst pipeline jobs: 117679 (0.5B) and 117680 (7B) (2026-11-07, about 01:20 IST)

- Job 117563 was still pending after about 2.5 h and was cancelled. node03 had idle CPUs, but 190.6 of its 192.3 GB of memory was allocated, so the 32 GB request could not fit. When a job is stuck on `Priority` with idle CPUs, check `scontrol show node N | grep AllocMem`.
- New `scripts/16_drburst.sh MODEL SKIP NTRACE [THREADS]` (env: `PROMPT_FILE`, `CTX`, `TAG`; default node01, 24 GB, 8 h). It needs nothing from `/scratch`, so it runs on any AVX-512 node. It does:
  1. an untraced reference run with the same binary;
  2. the burst trace, raw2trace with gzip, and basic_counts;
  3. drcachesim with prefetch (DRAM reads = LL misses plus prefetch misses) and without prefetch with `-LL_miss_file`, in parallel;
  4. `drmiss_export.py`, then `ramulator_replay.py`.

  Results go to `results/fullscale/drburst/<MODEL>_s<SKIP>n<NTRACE>t<THREADS><TAG>/`; the Ramulator trace stays on `/scratch`.
- `drmiss_export.py` now reads the miss file in chunks of 4M lines into numpy. A 7B token is about 75M lines, too many for Python lists.
- Submitted:
  - 117679: Qwen 0.5B, step 1 (the first token after the prompt), 4 threads. It replaces 117563 and checks the pipeline against gem5.
  - 117680: Qwen2.5-7B Q4_K_M (`qwen2.5-7b-instruct-q4_k_m-00001-of-00002`, which loads both splits), step 1, 4 threads, 48 GB. It runs after 117679 (`afterany`).

### First full-scale results: Qwen 0.5B and Qwen 7B burst traces (read 2026-11-07, about 09:30 IST)

Job 117679 (Qwen 0.5B, step 1, 4 threads, node01, 4.5 min, published `c8f6181`):
- Traced run 1.48 s, same text. Raw 526 MB; raw2trace 17 s. 268.6M instructions in 4 balanced threads (62–71M each).
- DRAM reads: with prefetch, 2,748,646 + 2,743,785 = 5,492,431; without prefetch, 5,485,339 (351 MB). gem5 measured 5,434,710 (+0.9 to +1.1%).
- Stride (merged stream): +1 line 0.336, within 16 lines 0.001, beyond 0.663. gem5's single core gave 0.41 / 0.51 / 0.02. The difference is mostly four threads interleaved by drcachesim's scheduler. The miss file's first field is the pid, not the thread, so the stream cannot be split by thread.
- Offline open-page row-hit rate. This matches gem5 Qwen 0.5B closely (0.983 / 0.964 / 0.989):

  | placement | RoRaBaCoCh | RoCoRaBaCh | XOR |
  |---|---|---|---|
  | VA-order packed | 0.961 | 0.058 | 0.958 |
  | 4 KiB shuffled | 0.955 | 0.028 | 0.954 |
  | THP 2 MiB | 0.961 | 0.057 | 0.958 |

  The exception is RoCoRaBaCh, which collapses here because four interleaved streams defeat line interleaving in the in-order model.
- Ramulator, saturated:

  | mapper | policy | hit | conflict | rd_lat | Mcycles |
  |---|---|---|---|---|---|
  | RoBaRaCoCh | Open | 0.984 | 0.005 | 186.5 | 28.4 |
  | RoBaRaCoCh | ClosedCAP | 0.790 | 0.002 | 222.7 | 34.1 |
  | MOP4CLXOR | Open | 0.630 | 0.277 | 181.4 | 23.5 |
  | MOP4CLXOR | ClosedCAP | 0.600 | 0.154 | 183.6 | 23.5 |
  | ChRaBaRoCo | Open | 0.966 | 0.033 | 262.8 | 41.5 |
  | ChRaBaRoCo | ClosedCAP | 0.793 | 0.011 | 469.2 | 76.4 |

  MOP4CLXOR finishes in 0.83 of RoBaRaCoCh's cycles; on the gem5 traces it was 0.67.

Job 117680 (Qwen2.5-7B Q4_K_M, step 1, 4 threads, node01, still in Ramulator at 09:20):
- Untraced 13.8 s; traced run 5.25 s (loading was not traced), same text. Native token step 189 ms on 4 threads.
- Raw 1.4 GB; raw2trace 170 s. **2.81G instructions per token** (10.5× the 0.5B), four threads of 698–707M.
- **DRAM reads per token: 68.7M lines, 4.40 GB.** That is 94% of the 4.68 GB model file, so 7B decode streams essentially the whole model each token, as gem5 showed for 0.5B. 4.40 GB in 0.189 s is about 23 GB/s on 4 threads.
- Cache simulation took 910 s with prefetch and 982 s without.
- Stride: +1 line 0.213, -1 line 0.037, within 16 lines 0.148, beyond 0.601.
- Offline row-hit rate:

  | placement | RoRaBaCoCh | RoCoRaBaCh | XOR |
  |---|---|---|---|
  | VA-order packed | 0.964 | 0.030 | 0.964 |
  | 4 KiB shuffled | 0.955 | 0.014 | 0.956 |
  | THP 2 MiB | 0.963 | 0.029 | 0.963 |

- Glitch: in the 7B traced run, the step-1 `B` stamp written right after `dr_app_start` came out as the literal format string `%c %ld %llu`. `drburst_main.cpp` now stamps before attaching, and has been rebuilt; the next jobs use it.

### Qwen 7B Ramulator result (job 117680 finished 09:36, published `ebd6cbe`; read 2026-11-07, about 21:10 IST)

Qwen2.5-7B, one decode token, 68,679,913 reads, saturated replay on one channel of DDR4-2400 (2 ranks):

| mapper | policy | hit | conflict | rd_lat | Mcycles |
|---|---|---|---|---|---|
| RoBaRaCoCh | Open | 0.985 | 0.005 | 183.9 | 350.4 |
| RoBaRaCoCh | ClosedCAP | 0.789 | 0.002 | 213.4 | 406.6 |
| MOP4CLXOR | Open | 0.608 | 0.293 | 182.3 | 295.0 |
| MOP4CLXOR | ClosedCAP | 0.587 | 0.169 | 183.7 | 294.8 |
| ChRaBaRoCo | Open | 0.963 | 0.032 | 220.9 | 429.8 |
| ChRaBaRoCo | ClosedCAP | 0.787 | 0.008 | 364.6 | 732.2 |

- MOP4CLXOR finishes in 0.84 of RoBaRaCoCh/Open's cycles, the same as on 0.5B (0.83), despite a 0.61 hit rate. The pattern from the gem5 traces holds at 7B: bank parallelism beats row hits for a weight stream under saturation.
- RoBaRaCoCh/Open drains at 5.1 cycles per 64 B line (about 15 GB/s, 78% of this one channel's 19.2 GB/s peak). The replay models one channel; node01 measured about 63 GB/s natively over many channels. The full-scale DRAM model needs multiple channels (the planned 8-channel DDR5) before absolute numbers mean anything.
- Close page (ClosedCAP) loses with every mapper except XOR, where it ties.

### Next four items: thread streams, MoE, long context, DDR5 (submitted 2026-11-07, about 21:45 IST)

The user approved all four.

**Multi-channel DDR5 in Ramulator.** `tools/ramulator_replay.py` gained `--dram ddr4|ddr5 --channels N --interleave-bits B --issue K --procs P`. The defaults are unchanged: one DDR4-2400 channel, issue 1, 6 processes.
- ddr5 is DDR5-4800B 16Gb x8 with one rank per 32-bit subchannel, and one controller per subchannel. 16 subchannels = 8 DIMM channels, 307 GB/s peak.
- `CacheLineInterleave(interleave_bits=B)` takes the channel bits just above the 64 B line offset plus B (B=2 gives a 256 B interleave).
- Several controllers report `stats["memory_system"]["controller"]` as a list, one per channel, each with `total_throughput_MBps`. The script sums them, takes the maximum cycles, and weights latency by reads served.
- LoadStoreTrace offers `clock_ratio` requests per memory cycle. At 1 it caps throughput at about 150 GB/s (64 B per 416 ps); at 4, a sequential test reached 260–282 GB/s. Full-scale runs use `--issue 4`.

**Per-thread streams.** drcachesim's miss file gives the pid, so `16_drburst.sh` now also runs drcachesim without prefetch for each compute thread alone (`-only_thread T -cores 1`, threads with more than 1M instructions), with `drmiss_export.py --stride-only`. Results go to `thread_<tid>/`.

**`16_drburst.sh`, other changes:**
- `--exclude=node02,node04,node05,node07,node08` instead of `-w node01`. node06 is a Xeon Gold 5317, Ice Lake with AVX-512 VNNI, so it also runs build-fast. Its `/scratch` is 97% full, with 71 GB free.
- New environment variables: `PROMPT_CHARS` (prompt = first N characters of `results/sessions/corpus.txt`, the llama.cpp docs, about 4.2 characters per Qwen token), `TB` (prompt threads, `-tb`) and `RAMU_PROCS`.
- Both Ramulator configurations are run: one DDR4 channel, and DDR5 with 16 subchannels at 256 B interleave, issue 4.

**Cluster:** node01 had only about 18 GB of memory unallocated, and node03 about 28 GB.

**Jobs:**

| job | model | context | memory |
|---|---|---|---|
| 118386 | Qwen 0.5B | short | 16G |
| 118387 | Qwen 7B | short | 40G |
| 118388 | Qwen3-30B-A3B | short | 64G |
| 118389 | Qwen 7B | `PROMPT_CHARS=32000 CTX=8192` (about 7.6k tokens), `-c 16 TB=16` | 40G |
| 118390 | Qwen 7B | `PROMPT_CHARS=120000 CTX=32768` (about 28.6k tokens), `-c 16 TB=16` | 48G |

All trace step 1 (the first token after the prompt) on 4 threads.

### Results of jobs 118386–118390, and the long-context failure (read 2026-11-08, about 09:30 IST)

**Finished:** 118386 (Qwen 0.5B, node01, 7 min), 118387 (Qwen 7B, node06, 1 h 26 min), 118388 (Qwen3-30B-A3B, node06, 37 min). All were step 1 on 4 threads with the same text as untraced.

| model | instructions/token | DRAM reads/token | share of model file |
|---|---|---|---|
| Qwen 0.5B | 0.261G | 5.48M lines, 351 MB | 82% |
| Qwen 7B | 2.81G | 68.7M lines, 4.40 GB | 94% |
| Qwen3-30B-A3B | 1.35G | 30.3M lines, 1.94 GB | **10.5%** of 18.56 GB |

For the MoE, the 1.94 GB fits about 3.3B active parameters at Q4_K_M.

**Per-thread streams** (`-only_thread`, no prefetch). The per-thread read counts add up to the merged count (0.5B 5.50M against 5.48M; 7B 68.74M against 68.69M).
- 0.5B, each thread: +1 line 0.977–0.991, beyond 0.008–0.021. gem5's single core: +1 0.41, within 16 lines 0.51, beyond 0.02. The share of long jumps agrees. The +1 against within-16 split differs because gem5's L2 stride prefetcher (degree 8) reorders lines, while this stream has no prefetch.
- 7B and the MoE, each thread: +1 line about 0.63, -1 line about 0.04, within 16 lines about 0.31, beyond 0.004–0.011.
- So each thread sweeps its slice of the weights almost sequentially. The merged stream's "beyond" share of about 0.6 comes entirely from interleaving four such sweeps.

**Offline row-hit rate (RoRaBaCoCh):** 0.955–0.969 for every model and placement. Line interleave (RoCoRaBaCh) collapses to 0.015–0.079 in the in-order model.

**Ramulator, saturated:**

| model | DDR4 1 ch: RoBaRaCoCh/Open GB/s | DDR4: MOP4CLXOR/Open GB/s | XOR time ratio | DDR5 16 subch: RoBaRaCoCh/Open GB/s | DDR5: MOP4CLXOR/Open GB/s | XOR time ratio |
|---|---|---|---|---|---|---|
| 0.5B | 14.6 | 17.9 | 0.82 | 272.8 | 276.1 | 1.00 (3.1 against 3.1 Mcycles) |
| 7B | 15.0 | 17.9 | 0.84 | 269.6 | 276.0 | 0.98 |
| 30B-A3B | 14.7 | 17.9 | 0.82 | 269.4 | 275.2 | 0.98 |

- On 16 DDR5-4800 subchannels (307 GB/s peak), the XOR advantage shrinks from 16–18% to about 2%. With 16 × 32 banks there is enough bank parallelism either way, and both reach about 88–90% of peak.
- Close page (ClosedCAP) costs row-interleaved about 24% (205 against 270 GB/s). ChRaBaRoCo is worst (about 135 GB/s open, about 76 GB/s closed).
- **Implication for the proposal:** for a single-stream weight sweep on a many-channel system, mapping hardly matters. The gains must come where the stream is not a clean sweep: KV at long context, expert gathers, and batched or concurrent requests. Saturated open-loop replay with 4 requests per cycle also overstates how many requests a CPU keeps in flight, so closed-loop replay is still needed for absolute numbers.

**Failed:**
- 118390 (7B, about 32k context, `-tb 16 -t 4`, node06): `Fatal error: failed to write trace for T3614340 window -1: wrote 10824 of 32784`. node06's `/scratch` was full: 100%, with 71 GB of it our leftover traces. Prompt: 120,000 characters = 32,143 tokens (3.73 characters per token, not 4.2), native prompt 890 s at 36 tok/s, token 1.35 s.
- 118389 (7B, about 8k context, node01): prompt 7,363 tokens in 149 s, native token 318 ms. The traced run took 13,071 s, never printed `detached`, and its text differed. Its traced.err is on node01's `/scratch`, and node01 has been drained since 22:50 (`Kill task failed`), so it cannot be read.
- Likely common cause: `-tb 16` creates 16 OpenMP threads while the token uses 4, so after `dr_app_start` DynamoRIO also traces the 12 idle, spinning threads. This is not yet confirmed.

**Fixes in `16_drburst.sh`:**
- `TB` must equal `THREADS` (documented; the resubmissions do not set it).
- `-max_trace_size 17179869184` (16 GiB per thread).
- Refuse to trace with under 100 GB free on `/scratch`.
- `trap` on exit copies `*.err` and `raw2trace.log` to the results folder, then deletes the run's `/scratch` folder.
- Our leftovers were deleted on node06 (back to 71 GB free) and node03 (`burst`, `drphases`, `drcheck`, `fmt`). node01's cannot be reached while it is drained; clean it when it returns.

**Resubmitted on node03** (717 GB free): 118727 (8k: `PROMPT_CHARS=32000 CTX=8192`) and 118728 (32k: `PROMPT_CHARS=116000 CTX=32768`, 14 h). Both use 4 threads for the prompt too.

### Long context: step numbering bug and resubmission (2026-11-08, about 10:20 IST)

- **Bug:** llama.cpp evaluates a prompt in batches of 2048 tokens, one `llama_process` call each. A 7,363-token prompt is 4 steps; a 31,001-token prompt is 16 steps (17 with the one decode). Counting from 0 as in short runs, "step 1" was the **second prompt batch**, not a decode step. That explains the 65 GB raw trace of job 118727 (8k) and the 3.6 h traced run of 118389. 118727 was cancelled while converting; its `/scratch` folder was gone afterwards.
- 118728 (32k, node03, 36 min) failed differently: after attaching mid-prompt at call 1, llama.cpp reported `init: sequence 0 positions are decreasing (not allowed)` / `decode: failed to initialize batch` / `failed to eval`. 118727 also attached mid-prompt and kept the right text, so this is not systematic. It is an open question whether attaching can corrupt state. Every attach at a decode step has worked so far.
- Native, node03, 4 threads, Qwen 7B:

  | prompt | prompt eval | next token |
  |---|---|---|
  | 7,363 tokens | 261 s (28 tok/s) | 295 ms |
  | 31,001 tokens | 2,072 s (15 tok/s) | 1,239 ms |

  At 32k the token costs 4.2× the short-context token (189 ms on node01). This is the KV-cache cost appearing.
- **Fix in `16_drburst.sh`:** `SKIP` now counts decode steps (1 = first step after the prompt). `PROMPT_STEPS` = (number of `B` lines in `steps_native.txt`) − (N − 1), and the traced run attaches at call `PROMPT_STEPS + SKIP − 1`. For short prompts this is call 1, as before.
- Resubmitted on node03: 118735 (8k: `PROMPT_CHARS=32000 CTX=8192`) and 118736 (32k: `PROMPT_CHARS=116000 CTX=32768`).

### Long-context results: 8k done, 32k almost (read 2026-11-08, about 14:10 IST)

Qwen2.5-7B, the first decode step after the prompt, 4 threads, node03. Both with the same text as untraced. The fixed step selection attached at call 4 (8k) and call 16 (32k), and the earlier "positions are decreasing" failure did not recur.

| | short | 8k (7,363 tokens) | 32k (31,001 tokens) |
|---|---|---|---|
| native token (4 threads) | 189 ms (node01) | 279 ms | 1,245 ms |
| instructions | 2.81G | 4.11G | 8.26G |
| raw / converted trace | 1.4 GB / 1.4 GB | 4.2 / 1.9 GB | 14 / 3.3 GB |
| DRAM reads | 68.7M (4.40 GB) | 75.7M (4.85 GB) | 152.7M (9.77 GB) |
| weights + one KV read (expected) | 4.40 GB | 4.82 GB | 6.18 GB |
| measured / expected | 1.00 | 1.01 | **1.58** |
| per-thread stride: beyond | 0.005 | 0.027–0.032 | 0.075–0.080 |
| offline row hit, RoRaBaCoCh (packed / 4K / THP) | 0.966 / 0.955 / 0.964 | 0.966 / 0.958 / 0.946 | 0.958 / 0.931 / **0.841** |
| offline row hit, XOR (packed / 4K / THP) | 0.962 / 0.955 / 0.965 | 0.969 / 0.958 / 0.964 | 0.953 / 0.931 / 0.928 |

The expected size uses KV f16 = 28 layers × 4 KV heads × 128 × 2 (K, V) × 2 B = 57,344 B per token: 422 MB at 7,363 tokens and 1.78 GB at 31,001.

**Finding: KV traffic is amplified once a layer's KV exceeds the LLC.** At 8k, the measured reads equal the weights plus one pass over KV. At 32k they are 3.6 GB more than that, so KV is read about 3× instead of once. Per layer, K+V is 15 MB at 7,363 tokens, which fits the 22 MB LLC, and 63.5 MB at 31,001 (K alone 31.7 MB), which does not.
- Hypothesis: with GQA, each KV head serves 7 query heads. While a layer's KV fits the LLC, the 7 reuses hit; once it does not, some of them go to DRAM.
- To verify: an LLC-size sweep on the 32k trace (the excess should vanish at about 64 MB or more), or attribution of misses to the KV buffer.
- This is also the first workload where placement matters: THP drops RoRaBaCoCh to 0.841.

**Ramulator, one DDR4 channel:**
- 8k: RoBaRaCoCh/Open 404.0 Mcycles, MOP4CLXOR 324.8 (0.80).
- 32k: RoBaRaCoCh/Open 0.969 hit, 740.9 Mcycles; MOP4CLXOR 0.377 hit, 0.516 conflict, 665.1 Mcycles (0.90). XOR's advantage halves.

**Ramulator, 16 DDR5 subchannels, 8k:** RoBaRaCoCh/Open 231.2 GB/s, MOP4CLXOR/Open 235.9 GB/s (0.98), ClosedCAP row-interleaved 165.9, ChRaBaRoCo 122.0. Lower than the short-context 270 GB/s: the KV part of the stream drains less efficiently. The 32k DDR5 replay was still running.

**Native bandwidth falls with context:** 4.40 GB / 0.189 s ≈ 23 GB/s (short), 4.85 / 0.279 ≈ 17 GB/s (8k), 9.77 / 1.245 ≈ 7.9 GB/s (32k). At long context on 4 threads the token is limited by attention compute, not DRAM bandwidth.

Note: the trace is deleted on exit, so the LLC sweep needs a new job, or a `KEEP=1` option.

### dev agent: tensor attribution for the DynamoRIO route, and literature baseline mappings (2026-11-08)

**Task 1: region attribution (built, validated on Qwen 0.5B; Qwen3-30B-A3B job 119133 queued, see end of section).**
- `tools/drburst/drattr_main.cpp` = `drburst_main.cpp` plus an interposed `ggml_backend_sched_graph_compute_async` (libllama.so imports exactly that one of the three; `nm -D`). After the real call (the scheduler assigns data pointers inside it) and only while tracing, it walks `ggml_graph_node(g,i)` and `->src[k]`, resolves views to `view_src`, and appends one TSV line per distinct root tensor: `step name data nbytes type op ne0..ne3`, to `$AMCAS_TENSORS`. Public API only. `llama-drburst` was not touched.
- Build (separate dir, new name): `cmake -S tools/drburst -B tracing/drattr-build -DATTR=ON -DDynamoRIO_DIR=... -DLLAMA=... -DDRLIBS=...` gives `tracing/drattr-build/llama-drattr`. (A second `add_executable` in one CMakeLists fails with DynamoRIO's `configure_DynamoRIO_global` error, hence the `ATTR` option; default build is unchanged.)
- `scripts/17_drattr.sh MODEL SKIP NTRACE [THREADS]` (short prompt; results in `results/fullscale/drattr/<run>/`): untraced run, traced run with the tensor dump, text compare, raw2trace, drcachesim without prefetch (22M/11-way LLC), then `drmiss_export.py ... --tensors`. `KEEP_MISS=1` keeps `miss.gz`. Cleans its `/scratch` folder on exit.
- `tools/drmiss_export.py` gained `--tensors TSV` and `--no-trace`. Reads are labelled by virtual address into expert weights (`*_exps*`), KV cache (`cache_k*`, `cache_v*`), dense weights (op NONE, `blk.*`, `token_embd`, `output*`), activations (everything else in the graph), unattributed. It writes `attribution.json`: reads, share, per-class stride, tensors touched, in-order open-page row hit per class (shared bank state, per placement, RoRaBaCoCh and XOR-bank), and for 3-D `*_exps` tensors the per-expert read counts (slice = nbytes/ne2) summed over layers, plus experts touched and "mostly read" (at least half the slice's lines) per tensor.
- Evidence, Qwen 0.5B (`qwen25-05b-q4_0`, decode step 1, 4 threads, node03): text identical to untraced; 357 tensors in the dump, including `cache_k_l0`, `cache_v_l0`. Attribution of the 5,487,082 reads: dense weights 98.6%, KV 0.10% (5,447 lines), activations 0.05%, unattributed 1.2% (output of graph-external buffers such as the embedding row, logits and runtime/stack traffic). Row hit (packed, RoRaBaCoCh): dense 0.973, KV 0.939, activations 0.942; THP placement drops KV to 0.524 and activations to 0.609 while weights stay 0.973 (XOR recovers KV to 0.963).
- Limits: addresses are virtual and a step's activations share one arena (names of overlapping tensors are arbitrary, the class is right). Prompt steps are not handled specially (every traced step's graph is dumped; `step` column).

**Task 2: baseline mappings as offline transforms.** `tools/remap.py TRACE OUT --scheme {identity,bitrev,xor_gf2} [--dram ddr4|ddr5] [--matrix-out F|--matrix F] [--max-fanin K] [--allow-col]` rewrites a Ramulator `LD/ST 0xADDR` trace so that the unchanged RoBaRaCoCh mapper decodes the intended bank/row. Bit layout (read from `ro_ba_ra_co_ch.cpp`, `cache_line_interleave.cpp`): the mapper is, LSB to MSB, `line[5:0] | column | rank | bankgroup | bank | row` (rank is lowest of the bank region despite the name); channel bits are cut out at bit `6+interleave_bits`. DDR4 (2 ranks, 1 channel): col [12:6], bank region [17:13], row [33:18]. DDR5 (16 subchannels, 256 B, 1 rank): channel = addr[11:8], intra-channel col [11:6], bank region (3 BG + 2 bank bits) [16:12], row [32:17]; the DDR5 org numbers are JEDEC, not read from Ramulator's tables.
- `bitrev` (Shao and Davis 2005, our reading): reverse the bit order of everything above the column field within the trace's active width. `xor_gf2` (Rotaru style): bank bit j ^= XOR of up to K candidate row (and optionally column) bits; greedy search on 8 sampled 100k-request chunks with a cheap cost = max(bus time, busiest-bank occupancy) + 0.05 sum occupancy + a window-conflict term (same bank, different row within 3 requests). Row and column are unchanged, so it is a bijection; the tool asserts this. Matrix saved as JSON.
- Evidence: the vectorised in-order hit count equals `analyze.open_page_stats` (0.0 vs 0.0 on the synthetic; real stream 0.982 reported by both paths). Synthetic trace (4 interleaved sweeps at 256 MiB offsets, 200k reads, one DDR4 channel, Ramulator RoBaRaCoCh/Open): identity hit 0.930, 1.6 Mcycles, 9.6 GB/s; xor_gf2 (found bank bits ^= row10/row11) 0.990, 0.8 Mcycles, 18.1 GB/s; bitrev 0.987, 0.9 Mcycles, 18.0 GB/s. Real Qwen 0.5B step (first 1.5M reads, packed placement, DDR4 1 channel, RoBaRaCoCh/Open): identity 0.987, 8.3 Mcycles; xor_gf2 0.987, 8.2 Mcycles (search found only row4/row9 taps, in-order hit 0.982 to 0.981); bitrev 0.976, 9.1 Mcycles (uses 10 of 32 banks, offline hit 0.866). So on a clean weight sweep these mappings add nothing and bit reversal hurts; they only help where aligned streams collide. This agrees with the DDR5 result above (mapping hardly matters for the sweep).
- HAPPY-style adaptive page closure: not feasible with the existing Python bindings, which only select among compiled row policies (Open, ClosedCAP, ...). It would need a new C++ `IRowPolicy` implementation in ramulator2 (per-bank predictor deciding precharge-after-access from recent hit/miss history) and a rebuild of the shared tree, which was not done. Cheapest route: build a copy of ramulator2 in a separate directory with the policy added.

#### dev agent addendum: Qwen3-30B-A3B attribution (job 119168, node03, 2026-11-08)
Decode step 1, 4 threads, same text as untraced, 701 distinct tensors in the dump, including `blk.N.ffn_{gate,up,down}_exps.weight` (144 tensors = 48 layers x 3) and `cache_k_lN`/`cache_v_lN`. Results in `results/fullscale/drattr/Qwen3-30B-A3B-Q4_K_M_s1n1t4_attr/` (`tensors.tsv`, `attr/attribution.json`, `export.txt`); not published to GitHub (small, can be copied).

| class | reads | share | MB | stride +1 / within16 / beyond (class stream) |
|---|---|---|---|---|
| expert weights | 17,141,770 | 0.565 | 1097 | 0.38 / 0.19 / 0.40 |
| dense weights | 12,853,303 | 0.423 | 823 | 0.47 / 0.23 / 0.24 |
| KV cache | 60,564 | 0.002 | 3.9 | 0.75 / 0.00 / 0.25 |
| activations | 105,718 | 0.003 | 6.8 | |
| unattributed | 195,577 | 0.006 | 12.5 | |

- Experts: on average **8.0 of 128 experts are touched per expert tensor**, and all 8 are read at least half-way (top-8 routing); the per-expert read counts over layers are in `attribution.json` (`experts.per_expert_reads_all_layers`). So the 1.94 GB per token is 1.1 GB expert slices plus 0.82 GB attention/shared/router/embedding weights.
- Offline row hit, RoRaBaCoCh packed: experts 0.971, dense 0.973, KV 0.952. With THP 2 MiB placement the small KV class drops to 0.736 (XOR: 0.884); the weight classes do not move.
- Note: the first 30B submission died in DynamoRIO (`heap.c` debug check failure at `dr_app_start`, 262 fragments); the rerun at the same settings (only `--mem` differed) was fine. Treat as a flaky attach; resubmit if seen.
- Queue note: node03 had about 10 GB unallocated memory; jobs asking 48-64 GB sat pending for 30+ min, 32 GB started at once.

### LL-size sweep, 8k control (job 119097; recorded 2026-11-08 evening, rce-runner)

- Added optional `LL_SWEEP="size:assoc ..."` to `16_drburst.sh` (runs with-prefetch drcachesim per LL size on the same trace, writes `cachesim_pf_LL<size>.txt`).
- Qwen 7B 8k (run `..._s1n1t4_c8k_llsweep`), DRAM reads = LL Misses + Prefetch misses: 22M/11-way 75.64M (4.84 GB); 32M/16 75.50M; 64M/16 75.20M; 128M/16 75.04M (4.80 GB). Flat (-0.8%), as expected when a layer's KV already fits the LLC.
- 32k sweep (job 119096) still running. Queued on node03: 119347 (Qwen 0.5B 32k, `_c32k`), 119348 (Qwen3-30B-A3B 8k, `_c8k`).

## 2026-11-08 Full-scale report and draft analysis (report-dev)

- `trace-pipeline/tools/fullscale_report.py RESULTS_DIR OUT_DIR` (numpy, matplotlib, json only; tolerates missing runs) writes `analysis/fullscale/tables.md` and `fig_reads_per_token.png`, `fig_llc_sweep.png`, `fig_region_time.png`. `publish.sh` already calls it. Do not hand-edit its output.
- `ANALYSIS_FULLSCALE.md` is a draft of the full-scale write-up (method, findings, limits, what remains). It cites the generated tables; pending items are marked.
- The 7B 32k runs differ: burst 152.7M lines (9.78 GB), its LL-sweep twin 161.2M (10.32 GB), the drattr run 179.5M (11.5 GB). Unexplained.

### dev agent round 2: DDR5 layout check, region-aware address mapping (2026-11-08)

**DDR5 layout check (remap.py): correct, no change.** Read from `ramulator2`: `python/ramulator/dram/ddr5.py` has DDR5_16Gb_x8 = 8 bank groups x 4 banks, 2^16 rows, 1024 columns, rank 1, internal prefetch 16. `addr_mapper_base.cpp` takes the level sizes in order rank, bankgroup, bank, row, column (channel removed), subtracts log2(prefetch) = 4 from the column bits (10 - 4 = 6) and a 64 B transaction offset; `ro_ba_ra_co_ch.cpp` slices column first, then rank (0 bits), bankgroup, bank, row. `cache_line_interleave.cpp` takes channel bits at `6 + interleave_bits` and closes up the rest. So with 16 subchannels and `interleave_bits=2`: channel = addr[11:8], intra col [11:6], BG [14:12], bank [16:15], row [32:17], exactly what `remap.py` assumes. Identity, xor_gf2 and bitrev were replayed on DDR5 16 subchannels on real streams (below).

**Pipeline additions.**
- `scripts/17_drattr.sh`: `PROMPT_CHARS`, `TAG`, `KEEP_MISS` (miss.gz stays on the cluster, >50 MB so not published), and a `publish.sh` call at the end. Run names are `<model>_s1n1t4_attr<TAG>`. The drattr results of job 119168 had been published by that job's commit already. Jobs: 119178 (30B, short, `_attr_miss`, 15 min), 119179 (7B 8k, `_attr_c8k`, 66 min), 119180 (7B 32k, `_attr_c32k`, about 4.8 h on node06 with three of our jobs sharing it; the Python in-order-per-class row-hit loop in `drmiss_export.py --tensors` is slow for 152M reads).
- `tools/region_remap.py` (new), `scripts/18_region.sh RUN [WINDOW]`, `tools/proposal_report.py` (writes `analysis/proposal_results.md` from `cluster-data/results/fullscale/region/*/`), `ramulator_replay.py --mappers`.
- Evaluated on a contiguous 12M-request window starting at 30% of the token's miss stream (a few layers; the stream is periodic over layers), replayed with RoBaRaCoCh, Open and ClosedCAP, DDR4 1 channel and DDR5 16 subchannels (256 B, issue 4), placements VA-order packed and THP 2 MiB shuffled (4 KiB shuffled not run).

**The scheme.** Each 2 MiB physical granule carries a 2-bit class tag W (dense + expert weights), K (KV cache) or O (activations + unattributed), by majority of the reads it receives. W granules keep the default row-interleave (identity). K and O granules get their own bank-index XOR (bank bit j ^= up to 3 row bits), searched with the same greedy method as `xor_gf2`, but on that class's sub-stream of the whole token. Bijection: the XOR changes only the bank field, as a function of row bits; bank, column and channel bits lie below physical bit 21 in both layouts (DDR4 bank [17:13]; DDR5 intra bank [16:12] = physical [20:16]), so every granule maps onto itself (bank, row, col -> bank ^ f(row), row, col), the granule index is unchanged, and the whole map is a bijection for any tagging. `region_remap.py` asserts granule, row and column are preserved. Hardware: a granule tag table plus XOR trees. OS: needs class-pure 2 MiB granules (natural with THP; with 4 KiB pages mixed granules just use the majority class's transform).

**Results** (full tables with latency and GB/s in `analysis/proposal_results.md`). Mcycles, and region's cycles relative to the best of identity/bitrev/xor_global ("vs best", below 1 = gain):

| stream | placement | DRAM / policy | identity | bitrev | xor_global | region | region vs identity | vs best baseline |
|---|---|---|---|---|---|---|---|---|
| Qwen3-30B-A3B short (KV 0.2%) | THP | DDR5 Open | 6.80 | 7.41 | 6.80 | 6.80 | 1.000 | 1.001 |
| Qwen3-30B-A3B short | THP | DDR4 Open | 61.25 | 64.13 | 60.93 | 61.24 | 1.000 | 1.005 |
| 7B 8k (KV 8.8% of reads) | packed | DDR5 Open | 7.87 | 11.32 | 7.85 | 7.88 | 1.001 | 1.004 |
| 7B 8k | THP | DDR5 Open | 8.44 | 8.41 | 7.88 | 7.90 | 0.936 | 1.003 |
| 7B 8k | THP | DDR5 ClosedCAP | 11.61 | 12.26 | 9.89 | 10.02 | 0.863 | 1.013 |
| 7B 32k (KV 61% of 179M reads (the earlier burst run counted 152M; this run is a separate trace)) | packed | DDR5 Open | 13.27 | 22.58 | 13.30 | 13.27 | 1.000 | 1.000 |
| 7B 32k | THP | DDR4 Open | 61.77 | 60.03 | 57.05 | 56.75 | 0.919 | 0.995 |
| 7B 32k | THP | DDR5 Open | 16.09 | 13.98 | 13.40 | 13.38 | 0.831 | 0.998 |
| 7B 32k | THP | DDR5 ClosedCAP | 21.36 | 17.83 | 15.14 | 15.22 | 0.713 | 1.006 |

- THP, DDR5, 7B 32k, Open: 114.7 GB/s (row hit 0.847, read latency 249 cycles) with the default mapping; region 138.0 GB/s (0.951, 198 cycles), 17% fewer cycles. With ClosedCAP the saving is 29%.
- **Region-aware does not beat a global XOR hash.** In every condition region is within 0.5% of the better of xor_global (sometimes ahead by under 1%, sometimes behind by up to 2.5%: packed 8k DDR5 ClosedCAP, where the global search found a small gain on weights that region leaves at identity). The whole gain over the default comes from hashing the KV and activation granules, and the global XOR obtains it as well while leaving the weights essentially untouched (the search finds no improvement for them). Bit reversal is never better and often much worse (up to 2.4x cycles on packed 32k DDR5 ClosedCAP, because it breaks the weight stream's bank spread); it only helps in the THP case where the default is bad.
- Where any hashing helps: only THP placement (random 2 MiB frames) once the KV stream is significant (8k about 6% fewer cycles on DDR5, 32k 17%; KV-heavy streams lose row locality to row conflicts between the K and V streams of different heads). With VA-order packed placement the default is already within 1% of any alternative at every context length. For the 30B MoE at short context nothing helps (weights are 99% of traffic; KV 0.2%): identity, region and xor_global are the same to 0.5%, and DDR5 sustains about 271 GB/s of 307 GB/s peak with the default mapping.
- Offline: the K granules' in-order row hit rises from 0.50 to 0.83-0.93 under region in the THP cases (30B too), but at 0.2% of the traffic this does not show in cycles.

**Caveats.** (1) One decode token, one thread layout (4 threads), saturated open-loop replay with 4 requests per cycle on DDR5 overstates memory-level parallelism; absolute GB/s are upper bounds. (2) The XOR matrices are searched on the same token they are evaluated on (in sample, though on a different part for the region search: the whole token vs a window), so the gains are optimistic; a held-out token (different position) is not tested. KV addresses move with the position, and weights do not. (3) Only a 12M-request window per condition. (4) THP frames are random uniform over 8 GiB (an aged allocator); a real THP allocator may place KV more contiguously. (5) The class tag is exact (from the tensor dump), a real system would need allocator support. (6) Channel hashing and 4 KiB-shuffled placement were not evaluated; the cost model in the search is a heuristic. (7) No batched/concurrent requests were tested, which is where a region-aware mapping might separate from a global hash.

**Takeaway for the proposal:** the data supports "an XOR bank hash on the KV (and activation) pages" as the useful mechanism (up to 17% on 7B 32k THP DDR5, 29% with closed page), not a per-class choice of mapping; the per-class variant is better only in being smaller in scope (weights provably untouched, no risk of a global hash hurting weight streams) and equal in speed. Whether the per-class choice matters when weights are not a clean sweep (batched decode, MoE with many experts per token at high batch) is untested.

Files: `results/fullscale/region/<run>/` (manifest.json, replay_*.json, make.txt), `results/fullscale/drattr/*_attr_miss`, `*_attr_c8k`, `*_attr_c32k` (miss.gz kept on the cluster only, 115 MB / 281 MB / 619 MB).

### LL sweep at 32k, and KV on 0.5B and 30B-A3B (jobs 119096, 119347, 119348; 2026-11-09, rce-runner)

**Hypothesis confirmed.** Qwen 7B 32k (31,001 tokens), DRAM reads = LL Misses + Prefetch misses, with prefetch:

| LL | reads | GB | vs expected 6.18 GB |
|---|---|---|---|
| 22M/11 | 161.3M | 10.32 | 1.67 |
| 32M/16 | 109.2M | 6.99 | 1.13 |
| 64M/16 | 97.1M | 6.21 | 1.00 |
| 128M/16 | 96.3M | 6.16 | 1.00 |

The 8k control is flat (75.6M to 75.0M). So the 3.6+ GB excess at 32k is KV re-reads that miss a 22 MB LLC; with an LLC at least as large as one layer's K+V (63.5 MB) the traffic is weights + one KV pass. (The 22M figure here, 161.3M, counts Misses+Prefetch misses of the with-prefetch run; the earlier 152.7M in the table above used the other counting, so compare ratios within one method.) Ramulator (7B 32k): DDR4 time ratio XOR/RoBaRaCoCh = 702.8/782.2 = 0.90; DDR5x16 179.9/179.7 = 1.00.

**Second models** (first decode step, 4 threads, node03). Expected = active weights (short-prompt reads) + one KV pass; KV dims taken from model configs (not re-read from logs): 0.5B 24 layers x 2 KV heads x 64 = 12,288 B/token; 30B-A3B 48 x 4 x 128 = 98,304 B/token.

| model | context | DRAM reads | expected | measured/expected | Ramulator DDR4 XOR ratio | DDR5 XOR ratio |
|---|---|---|---|---|---|---|
| Qwen 0.5B | 31,001 tok | 11.66M lines (746 MB) | 351 MB + 381 MB = 732 MB | 1.02 | 50.0/65.6 = 0.76 | 6.5/6.6 = 0.98 |
| Qwen3-30B-A3B | 7,356 tok | 42.1M lines (2.70 GB) | 1.94 GB + 0.72 GB = 2.66 GB | 1.01 | 181.2/241.7 = 0.75 | 35.4/36.1 = 0.98 |

Both match weights + one KV pass: 0.5B KV per layer is 12 MB at 31k, and 30B-A3B per layer 15 MB at 7.4k, so both fit the 22 MB LLC. No amplification, consistent with the hypothesis (amplification needs per-layer KV above the LLC). Native: 0.5B token 130 ms at 31k.

Submitted: 119402 (30B-A3B s1n4 `_n4`), 119403 (7B s1n4 `_n4`).

### MoE across tokens (jobs 119402, 119403; 2026-11-09, rce-runner)

Decode steps 1 to 4, 4 threads, short prompt, node03. DRAM reads = LL misses without prefetch (the replayed stream). Footprint = distinct 4 KB pages touched (`summary.json footprint_pages`, so an upper bound on distinct lines).

| model | DRAM reads, 4 tokens | per token | 1-token run | 4 x 1-token | footprint (4 KB pages) | footprint / 4 x per-token reads |
|---|---|---|---|---|---|---|
| Qwen3-30B-A3B | 121.4M lines (7.77 GB) | 30.4M (1.94 GB) | 30.3M | 121.2M | 738,558 = 3.02 GB | 0.39 |
| Qwen 7B (dense) | 274.7M (17.58 GB) | 68.7M (4.40 GB) | 68.7M | 274.8M | 1,069,147 = 4.38 GB | 0.25 |

- Per-token reads are the same as for one token in both models: the 22 MB LLC keeps nothing between tokens, so every token re-reads its whole weight set from DRAM.
- Dense: the footprint equals one token's reads, so all 4 tokens read the same bytes (reuse factor 4).
- MoE: the 4 tokens together touch 3.02 GB, 1.55x one token's 1.94 GB; the experts chosen by consecutive tokens overlap strongly (union of 4 tokens = 39% of 4 x reads, against 25% if fully shared and 100% if disjoint). About 61% of the 4-token reads are re-reads of data an earlier token already fetched. Implication: expert-weight reuse across consecutive tokens is large but LLC-sized caches cannot capture it (union 3 GB).
- Ramulator, 1 DDR4 channel (Mcycles XOR/RoBaRaCoCh): MoE 523/598 about 0.87; DDR5x16 gives the usual near 1.0 (RoBaRaCoCh 261.0 vs MOP4CLXOR 286.3 GB/s in the first column listing; read ramulator_ddr5x16.txt for time ratios).
- /scratch: no leftover directories on node03 listing at the time of check.

### batch-dev: batched-decode DRAM trace (drbatch), 2026-11-09 (partial: 7B B=1,4 and 30B B=1 done; 30B B=4,16 and 7B B=16 queued)

**Tool.** `tools/drburst/drbatch_main.cpp` (CMake `-DBATCH=ON`, build `tracing/drbatch-build`, needs `-DCMAKE_BUILD_TYPE=Release`; the unoptimised build crashed under DynamoRIO) is its own `main` on libllama.so, not llama-completion. B sequences, each with its own prompt (P=512 tokens from corpus slice k, offset k*30000 chars), decode together (kv_unified=0, per-sequence KV). `scripts/18_drbatch.sh MODEL B [THREADS]` runs: untraced reference, traced decode step 1 (one `llama_process` with B tokens, after all prompts and a batched last-prompt-token step 0), tensor dump (drattr hook is included), raw2trace gzip, basic_counts, drcachesim with/without prefetch, `drmiss_export.py --tensors`, Ramulator DDR4x1 and DDR5x16 (`--procs 3`). Results in `results/fullscale/drbatch/<MODEL>_B<B>_P512t4/`.
- **Prompt cache.** The first P-1 tokens of each sequence are stored with `llama_state_seq_save_file` in `results/sessions/drbatch/` and restored by every later run (B=1 and B=4 share the first sequences). Step numbering is by decode step, so no prompt-step counting is needed. Text equality (greedy token ids, 4 steps) holds between the fresh-evaluated untraced run and the restored traced run: yes for all finished runs.
- **DynamoRIO crash and fix.** Attaching in `llama_process` crashed with B=16 (SIGSEGV in `unordered_map<bitset<256>>::operator[]` inside `llama_batch_allocr::init`, reproducible on Qwen 0.5B). The driver now attaches inside the interposed `ggml_backend_sched_graph_compute_async` (just before the graph compute). The B=1 and B=4 7B and the 30B B=1 runs used the earlier attach point (batch setup included, negligible).
- **drmiss_export.py** got `AMCAS_CAPACITY_GB` (default 8). With 30B B>=4 the touched 2 MiB pages exceed 8 GiB and the THP placement failed (no attribution, no Ramulator); `18_drbatch.sh` now exports 32. Row-hit numbers of the shuffled placements in new runs are therefore not exactly comparable with the earlier 8 GiB runs.

**Results (Qwen2.5-7B Q4_K_M, 512-token prompts, 4 threads, node03; DRAM reads = LL misses + prefetch misses, 22 MB LLC):**

| B | DRAM reads | GB | GB per sequence | native step | instructions | DDR4 RoBaRaCoCh/Open hit | DDR5x16 GB/s (RoBaRaCoCh / MOP4CLXOR) |
|---|---|---|---|---|---|---|---|
| 1 | 69.17M | 4.43 | 4.43 | 174 ms | 2.9G | 0.979 | 270.9 / 273.0 |
| 4 | 71.32M | 4.56 | 1.14 | 283 ms | 6.3G | 0.963 | 250.9 / 262.8 |

Dense model: B=4 reads 1.03x of B=1 (weights are read once per step, KV reads rise from 29 MB to 118 MB), so DRAM traffic per sequence falls 3.9x; native step time 1.6x (throughput 2.5x).

**Qwen3-30B-A3B Q4_K_M:** B=1: 31.16M reads = 1.99 GB, 104 ms, 8.0 of 128 experts touched in every expert tensor (matches the drattr run). B=4: 66.78M reads = 4.27 GB (nopf stream; step 276 ms, 5.0G instr), i.e. 2.1x of B=1 for 4 sequences (1.07 GB per sequence), against 4x for no sharing and 1.0x for a dense model. Native step time 2.65x (throughput gain 1.5x). The expert union per layer at B=4 and the B=16 numbers are pending (first B=4 attempt hit the 8 GiB placement limit; rerun queued).

**Queued (node03 is saturated by another user's array; priority 1):** 119987 (30B B=4), 119988 (30B B=16), 119989 (7B B=16), dependent chain, each publishes itself. Per-run table: a small script is not in the repo; read `cachesim_pf.txt`/`cachesim_nopf.txt` (DRAM reads), `steps_native.txt` (step time), `attribution.json` -> `experts.per_layer[*].experts_touched`.

### The 9.8 / 10.3 / 11.5 GB discrepancy at 32k explained (2026-11-09)

The three numbers are DRAM reads (no-prefetch LL misses × 64 B) for one Qwen 7B decode token at 31,003 tokens of context, from three separate traces. All three had the same prompt (116,000 characters), 4 threads, the same 22 MB 11-way LLC flags (`-cores 4 -LL_size 22M -LL_assoc 11`) and nearly the same instructions (8.26G against 8.30G; the drattr run has no basic_counts):

| run | script / binary | no-prefetch misses | GB | ÷ 6.18 GB expected |
|---|---|---|---|---|
| 118736 `_c32k` | 16_drburst / llama-drburst | 152.7M | 9.77 | 1.58 |
| 119096 `_c32k_llsweep` | 16_drburst / llama-drburst | 161.2M | 10.31 | 1.67 |
| 119180 `_attr_c32k` | 17_drattr / llama-drattr | 179.5M | 11.49 | 1.86 |

- Cause: run-to-run variation in how the four threads' work interleaves, in the one regime where the shared LLC is thrashing. ggml hands out matmul and attention chunks dynamically through an atomic counter, so which thread reads which K/V block, and when, differs between runs. drcachesim interleaves the four traces in a shared LLC. When a layer's KV (63.5 MB) is about 3× the LLC, whether the reuses across the 7 query heads that share a KV head hit depends strongly on that interleaving. The drattr run also walks the graph inside the traced step, which disturbs the cache a little more.
- Evidence that this regime-specific sensitivity is the cause: where nothing thrashes, the same comparison is stable. Short context gives 68.69M against 68.68M lines (0.01%) between two runs. At 8k the LLC sweep is flat, and at ≥64 MB the 32k run gives 6.16–6.21 GB in the sweep run.
- How to report it: at 32k with a 22 MB LLC, KV amplification is **1.6–1.9× the expected traffic (one pass over the weights plus one over the KV), depending on thread scheduling**; it vanishes at 64 MB and above. Absolute 32k numbers should be quoted as a range. The direction of every finding is unchanged.

### policy-dev: row-policy plugins and replay harness (2026-11-09, results pending)

- **Plugins** (`trace-pipeline/ramulator-ext/rowpolicy/`, built by `ramulator-ext/install.sh` into a private copy `~/amcas-pro/tracing/ramulator2-amcas`; the shared `ramulator2` tree is untouched; bindings regenerated with `python -m ramulator codegen`): `Timeout` (precharge after `timeout` idle cycles, default 100, evaluated at 64), `Region` (ranges file `START END CLASS`; classes in `close_classes` use ClosedCAP-style auto-precharge after `cap` accesses, others open), `HAPPY` (address-hash indexed 2-bit predictor of row reuse, `mode=binary`; `mode=count` variant; simplified: no global/per-thread history combination, no tournament, no queue lookahead; paper not re-read in detail).
- **Harness:** `tools/policy_traces.py` (window of the attributed miss stream, placements packed/THP, DDR4 1ch / DDR5 16ch, mappings identity/xor_global, writes saturated trace, per-thread `bubbles addr` traces, granule class ranges), `tools/policy_replay.py` (saturated LoadStoreTrace and closed-loop SimpleO3, 4 cores, 16 MSHRs/core, constant bubbles = instructions per read), `scripts/19_policy.sh RUN BUBBLES`, `tools/policy_report.py` (writes `analysis/policy_results.md`).
- Smoke test (synthetic 60k-read DDR4): all policies run in both modes; ClosedCAP costs about 30% cycles there, Timeout equals Open (saturated queues never idle), HAPPY about 4% worse, count mode much worse (rows are not revisited, so hash entries do not generalise).
- Jobs 120091 (30B short, bubbles 44), 120092 (7B 8k, 54), 120093 (7B 32k, 52) submitted; still pending (priority) at time of writing.

### finisher status, 2026-11-10 (results mostly pending; no headline numbers yet)

- Finished and published: 119987 (30B B=4 drbatch), bmiss4 120085, bmiss16 120086 (batch miss streams in `cluster-data/results/fullscale/drbatch_miss/`). Running: 119988 (30B B=16). Still pending on node03 (other user's array): 119989 (7B B=16), policy 120091-120093.
- **Failures.** (a) moe_short 120087: `moe_remap.py` assertion "not injective" for the `expert_xor` variants. Cause: the bank XOR constant was applied only to expert reads, but non-expert reads in the same DRAM row were left unchanged, so two cells could collide. Fix: the constant now depends on the row only (applied to every access in a row that has a winning expert). Resubmitted as 120933. (b) attr_t2 120081 (7B 32k, SKIP=2) and moe_n4 120084 (30B, 4 steps): both ended with "same text: NO" in `17_drattr.sh` (tail 300 B of plain and traced stdout differ); scratch was removed by the trap so the cause is unknown. `17_drattr.sh` now prints the diff and the last stderr lines on mismatch. Resubmitted once as 120934 (attr_t2) and 120835 (moe_n4, tag `_n4m`). The held-out XOR test (`19_mapstudy.sh heldout`) needs attr_t2's miss stream and has not run.
- Not done: policy results (`analysis/policy_results.md`), expert-aware MoE numbers, held-out result, batched union numbers for B=16, headline table.

### status, 2026-11-11 (batched decode complete; drattr failures diagnosed)

**Batched decode, all six runs done** (119987-119989; tokens of untraced and traced runs identical in all six). DRAM reads = LL Misses + Prefetch misses (with prefetch), 22 MB LLC, 512-token prompts, 4 threads, node03; step = native decode step 1; Ramulator ratios are Mcycles MOP4CLXOR / RoBaRaCoCh, Open page (below 1 = XOR faster):

| model | B | DRAM reads | GB | GB per seq | native step | mean experts touched / expert tensor (of 128) | DDR4 hit (default) | DDR4 XOR ratio | DDR5x16 XOR ratio |
|---|---|---|---|---|---|---|---|---|---|
| Qwen 7B | 1 | 69.2M | 4.43 | 4.43 | 174 ms | - | 0.979 | 0.881 | 0.992 |
| Qwen 7B | 4 | 71.3M | 4.56 | 1.14 | 283 ms | - | 0.963 | 0.914 | 0.954 |
| Qwen 7B | 16 | 79.6M | 5.10 | 0.32 | 855 ms | - | 0.899 | 0.968 | 0.976 |
| Qwen3-30B-A3B | 1 | 31.2M | 1.99 | 1.99 | 104 ms | 8.0 | 0.977 | 0.850 | 0.978 |
| Qwen3-30B-A3B | 4 | 67.0M | 4.29 | 1.07 | 291 ms | 23.2 | 0.975 | 0.836 | 0.958 |
| Qwen3-30B-A3B | 16 | 153.9M | 9.85 | 0.62 | 779 ms | 57.25 | 0.974 | 0.805 | 0.973 |

- Dense: B=16 reads 1.15x of B=1 (weights once per step, KV and activations grow); per-sequence traffic falls 14x. MoE: B=16 reads 4.9x of B=1, because the expert union grows (8 -> 23 -> 57 of 128 experts per layer, against 32 / 128 if the 4 / 16 sequences chose disjoint experts: about 72% / 45% of the disjoint union). MoE batching amortises far less than dense (0.62 vs 0.32 GB per sequence at B=16 although the 30B's active weights are less than half the 7B's).
- XOR gains on DDR5x16 stay at 2-5% across batch sizes; on DDR4 1 channel the MoE gain grows with B (15% -> 20%), the dense gain shrinks (12% -> 3%). (30B B=4 step 291 ms here vs 276 ms quoted earlier: a different native measurement.)

**MoE expert-aware mapping (120933, Q30B_short, B=1):** identity, xor_global, region, expert_hash, expert_lin, region_expert are all within about 1% (DDR4 Open 30.3-30.6 Mcycles, DDR5 3.4) in both placements. Negative at B=1: with 8 experts read as whole contiguous slices the default mapping already streams them at about 0.96 row hit. The batched streams (bmiss4/bmiss16, `drbatch_miss/`) are where expert-aware mapping could matter; not yet run through `moe_remap.py`.

**drattr failures (120081, 120084, 120835, 120934) diagnosed.** Not a text divergence: the traced process aborts in DynamoRIO, so its stdout is truncated. traced.err: `Internal Error: DynamoRIO debug check failure: core/heap.c:1987 ... vmh->num_free_blocks == vmh->num_blocks - unfreed_blocks` at `dr_app_stop_and_cleanup` (no "detached" line). Cause: `tracing/drburst-attr-build` had an empty `CMAKE_BUILD_TYPE`, so `configure_DynamoRIO_static` linked DynamoRIO's **debug** library (asserts on); the Release-built llama-drburst never hit it (its 4-step 119402/119403 detached fine), and single-step drattr runs (119178-119180) happened to pass the check. Fix: `tools/drburst/CMakeLists.txt` now defaults to Release; the old binary is kept as `llama-drattr.debug-build`. Rebuild is srun job 121990 (node03, waiting on priority); chain: 121991 smoke (0.5B, SKIP=2, mid-run detach, tag `_smoke`) -> 121992 attr_t2 and 121993 moe_n4 (same args as 120934/120835). 121990 is an srun held by an ssh session from the laptop; if that session dies, cancel the chain and resubmit with an sbatch build. Policy jobs 120091-120093 still pending (priority; node01 down).

### writeup: analysis update (2026-11-11)

- `ANALYSIS_FULLSCALE.md` updated: batched decode B=1/4/16 (7B, 30B), MoE across tokens, expert-aware mapping at B=1 (negative), 32k KV amplification as a 1.6-1.9x range with the thread-scheduling explanation, LL sweep, region vs global XOR (no gain over global), held-out/policy/batched-mapping/32k-30B-0.5B/writes/DDR5-4800/prefill marked Pending. Numbers cite context.md sections or `tables.md`.
- `tools/fullscale_report.py` now also writes a "Batched decode" table (GB per sequence, vs B=1, native step, experts touched vs the disjoint bound) and `analysis/fullscale/fig_batched.png` from `results/fullscale/drbatch/*_B*_P*t*` (skips `_test` dirs; missing runs tolerated). Regenerated locally with system python3.
- gem5 `moe_l3-22m_stride` (114798) had in fact already been analysed on 2026-11-05 (jobs 115490, 115498), and was already in `report.py` RUNS and `analysis/tables.md`; no job was needed. Added the run to the ANALYSIS.md run table, changed "all 9 runs" to 10, and added subsection F3a (reads the whole 1.92 GB model per token, expert weights 84.3%, same mapping ratios as the 1024-token run; only one LLC size/prefetcher exists for tinymoe4, so no cache-size or prefetch comparison). The `report.py` LABEL needed no change (REPLAYED unchanged).
- Job IDs 121997/121998 for the 30B/0.5B 32k runs were read from squeue names, not from a context.md note.

### scenarios: 1k context and prefill (2026-11-11) - SUBMITTED, results pending

Submitted via `scripts/17_drattr.sh` (unmodified), node03, `--dependency=afterok:121991` (attr_smoke, the Release-build check), all with KEEP_MISS=1:
- 122008 `sc7k1`: Qwen 7B, 1k-context decode (TAG=_c1k, PROMPT_CHARS=4000, CTX=2048, SKIP=1 NTRACE=1).
- 122009 `sc30k1`: same for Qwen3-30B-A3B (60G mem).
- 122010 `sc7pf32`, 122011 `sc30pf32`: PREFILL window, SKIP=0 NTRACE=1 (traces llama_process call 0 = the prompt), TAG=_pf32, PROMPT_CHARS=120 (about 30 tokens), CTX=512.
- Why prefill is only ~30 tokens, not 512/2048: instruction count scales with tokens (decode of 7B is ~2.9G instructions; 512 tokens would be ~1.5T, a multi-TB DynamoRIO trace, and the 16 GB `-max_trace_size` cap would truncate it). Weight reuse across tokens already appears at 30 tokens; larger prompts need a native-counter approach instead. If the 30-token runs fit, 128 tokens is the next step.
- Not yet done: validation (same text, reads vs weights+KV), 18_region.sh replays (identity vs xor_global, DDR4x1/DDR5x16, Open/ClosedCAP; run `sbatch scripts/18_region.sh <RUN>` with RUN = `<model>_s1n1t4_attr_c1k` and `<model>_s0n1t4_attr_pf32`), tables. Caveat: if 17_drattr's PROMPT_STEPS arithmetic miscounts for the prompt call (N=1), the traced step may be wrong; check `steps_traced.txt` and "same text".

### longctx: drattr Release fix, 30B and 0.5B at 32k (2026-11-11, partial: blocked on node03 memory)

- **Release build done.** Job 121990 (interactive srun, dead client) was still pending, so the rebuild was submitted as sbatch: 121996 (16 GB, starved) and then 122019 (3 GB, 4 cpus), which ran in 5 s and COMPLETED. `tracing/drburst-attr-build` is now `-DCMAKE_BUILD_TYPE=Release` (link.txt uses `dynamorio/lib64/release/*` and `-DNDEBUG`); 121991 was repointed to `afterok:122019` and 121996 cancelled. Not yet verified: the smoke log lines "drburst: detached after step" and "same text: yes".
- **Blocker.** node03 has AllocMem 188 of 192 GB, held by a pawan array (121288, 32 GB per task, 318 tasks still pending, about 1.3 h each, three running at a time) plus two other 32 and 56 GB jobs. Our pending jobs (priority 1, later job IDs) never get memory: smoke 121991 (16 GB), attr_t2 121992, moe_n4 121993, and the policy jobs 120091-120093 (pending since 11-09). Node01 is down, and the other nodes lack AVX-512 and are excluded by the scripts.
- **Queued behind 121991 (afterok), all on node03:** 121997 `q30_32k` (Qwen3-30B-A3B, `PROMPT_CHARS=116000 CTX=32768 TAG=_c32k KEEP_MISS=1`, 1 1 4 threads, 64 GB), 121998 `q05_32k` (qwen25-05b-q4_0, same env, 32 GB), then 121999 (`18_region.sh Qwen3-30B-A3B-Q4_K_M_s1n1t4_attr_c32k`, afterok:121997) and 122000 (`18_region.sh qwen25-05b-q4_0_s1n1t4_attr_c32k`, afterok:121998). Results for items (a)-(c) are therefore pending; the expected-reads check (30B: 1.94 GB weights + 98,304 B/token x 32k about 3.2 GB KV, per-layer KV 67 MB > 22 MB LLC; 0.5B: 746 MB) will be made when they run.
- **Suggestion.** Lowering the smoke job to about 3.5 GB (it was denied here as an action on the user's job; needs the user's go-ahead) would run it now; the 30B job needs 64 GB, which only fits when the pawan array drains or node01 returns.

- scenarios update (2026-11-11): smoke 122041 passed (dependency met). Jobs 122008-122011 (sc7k1, sc30k1, sc7pf32, sc30pf32) are on user hold under main-session queue control, with 8 CPUs and limits 2h/1h/4h/4h; the main session releases them. Not resubmitted or released by the scenarios agent. Risk: the 1h limit on sc30k1 (or the 2h on sc7k1) may be too short for DynamoRIO trace + drraw2trace + cachesim; check for TIMEOUT.

### longctx update (2026-11-11)

Smoke 122041 (0.5B, SKIP=2, Release build 122019) PASSED: "detached after step 2", "same text: yes". It replaces 121991 and 121990 (cancelled). The main session now runs the queue 4 at a time: 121992 attr_t2 and 121997 q30_32k running; 121993 moe_n4 (48 GB), 121998 q05_32k and region jobs 121999/122000 held by the user until released. The 30B and 0.5B 32k tables are still pending; this section supersedes the job references to 121991 above.

### robustness-owner: memory system, page size, 1k context, prefill (2026-11-11), part 1: 7B 32k memory-system study (job 122014)

Job 122014 (`21_memsys.sh`, run `..._s1n1t4_attr_c32k`, tag `q7_32k_w12M`, 12M-request window, 71 min on node03, published; results in `results/fullscale/memsys/q7_32k_w12M/`). Smoke test first: 1M-request replay takes about 12 s per trace (2 policies), generation about 2 min. Replay: Ramulator DDR5-4800, RoBaRaCoCh, saturated, 8 channels (issue 2/cycle) or 16 (issue 4). Mcycles Open / ClosedCAP; every one of the 72 replays reads exactly 12,000,000 requests, 0 writes.

| placement | DRAM | identity | chan | xor_global | xor_global_chan | region | region_chan |
|---|---|---|---|---|---|---|---|
| packed | x8 | 26.5 / 29.0 | 17.4 / 21.2 | 26.6 / 29.4 | 17.8 / 21.7 | 26.7 / 29.4 | 20.4 / 24.0 |
| packed | x16 | 13.3 / 14.6 | 8.9 / 10.9 | 13.3 / 14.8 | 8.7 / 10.8 | 13.3 / 14.6 | 9.1 / 10.8 |
| THP | x8 | 31.8 / 43.1 | 21.0 / 28.0 | 27.0 / 31.5 | 19.4 / 23.4 | 26.8 / 30.9 | 20.1 / 23.9 |
| THP | x16 | 16.1 / 21.4 | 10.5 / 13.9 | 13.4 / 15.1 | 9.7 / 11.7 | 13.4 / 15.2 | 9.4 / 11.5 |
| 4 KiB | x8 | 26.6 / 26.7 | 14.6 / 14.8 | 26.6 / 26.8 | 14.6 / 14.8 | 26.6 / 26.7 | 14.6 / 14.8 |
| 4 KiB | x16 | 13.4 / 13.3 | 7.1 / 7.1 | 13.4 / 13.3 | 7.1 / 7.1 | 13.4 / 13.3 | 7.1 / 7.1 |

Findings:
- **The KV bank-XOR gain holds on 8-channel DDR5-4800**, THP only: xor_global is -15% Open / -27% ClosedCAP vs identity (x16: -17% / -29%); region equals xor_global. With packed or 4 KiB placement the bank XOR gives nothing (default already 0.95 row hit packed; 4 KiB has 0.75 hit that no bank hash changes).
- **A channel-bit hash is larger than the bank XOR**: -35% packed, -47% 4 KiB, -35% THP (x16 Open); on x8 -34% / -45% / -34%. Combined bank+channel is best for THP (x16 Open 9.7 vs 16.1, -40%).
- Cheaper than the bank search, but see the check below for whether it is an artefact of the saturated replay.

**Channel-hash verification** (requested by main): (1) the study asserts per scheme that row and column are unchanged and the number of unique physical addresses is unchanged (injective), and the run passed all asserts; (2) reads equal in all replays (12,000,000); (3) static channel load over the 12M window, max/mean: identity 1.057, chan 1.001 (packed) / 1.042 (THP), so identity is balanced on average; (4) hashed bits: the channel bit j is XORed with listed row bits of the channel-removed address, found by greedy search on the window. Examples (x8): packed chan bit0 ^= row0,row2,row4; bit1 ^= row1; bit2 ^= row1. THP chan: bit0 ^= row5,row14; bit1 ^= row5; bit2 ^= row13. 4 KiB x16: bit0 ^= row6,row10; bit1 ^= row9; bit2 ^= row4; bit3 ^= row1 (all in `manifest.json`/`make.txt`).
- **Temporal imbalance** (2M-request window, 8 channels, channel = addr[10:8]; `tools/chan_temporal.py`, job 122340; mean over consecutive windows of the busiest channel's count / mean count; random uniform would be about 1.8 at W=32): identity W=32 2.43 (6.4 of 8 channels used per window), W=128 2.04 (7.6/8); chan W=32 1.86 (7.5/8), W=128 1.44 (8.0/8), for both packed and THP. So the default mapping concentrates consecutive requests on fewer channels (the four threads' streams are phase-aligned in the channel bits), while the hashed one looks like uniform random. The gain is real in the model (burst-level channel imbalance under limited controller queues), not a miscount, but: the replay is saturated open-loop with the miss stream in drcachesim interleave order, no timestamps, so how much burst imbalance appears depends on that order; a closed-loop run would show smaller gains. Treat the magnitude (-35 to -47%) as an upper bound.

### policy-owner: row policies (closed loop), channel hash (2026-11-11), part 1

Harness fixes: 122135/120093 failures were (a) node05 `/scratch` 100% full (ENOSPC; node05 now excluded), (b) the miss-stream window not interleaved over 4 recorded threads (only `t0` traces written); `policy_traces.py` now falls back to 4 contiguous pseudo-threads, and `19_policy.sh` exits nonzero on failure. Smoke 122141 passed. Full runs: 122152 (7B 32k, 82 min), 120092 (7B 8k, 19 min), 120091 (30B, 20 min); table in `analysis/policy_results.md` (4M-read window, bubbles 52/54/44, closed-loop = SimpleO3 4 cores x 16 MSHR).

**Row policies.** Open is best or within 0.5% of best in nearly every condition. Saturated: ClosedCAP +12-38%, Timeout +2-4%, Region +2-33%, HAPPY +0-5%. Closed-loop (core-bound, about 13 GB/s demand, 47-90 cycle read latency): HAPPY median 1.001-1.003 of best baseline, best cell 0.968 (7B 32k THP DDR5 identity, one cell); Region median 1.000-1.022, best 0.976. No policy gain above 3.2% in any cell, so **no row-buffer policy worth proposing**; the mapping is what moves the row-hit rate. (Timeout shows hit 0.22 in DDR5 closed-loop: queues idle there, so it closes rows needed soon; it is the worst policy.)

**Channel hash, closed loop** (`22_chanloop.sh`, job 122364, 7B 32k, 4M window, tag `c32k_w4M`; `policy_traces.py --schemes` reuses `memsys_study.chan_search/hash_bits` read-only; smoke 122346). Open, Mcycles / avg read latency (cycles). DDR5 x16 = `ddr5`, x8 = `ddr5x8`.

| placement | DRAM | identity | xor_global | chan | xor_global_chan |
|---|---|---|---|---|---|
| packed | x16 | 46.61 / 78 | 46.53 / 77 | 46.65 / 77 | 46.68 / 77 |
| packed | x8 | 46.55 / 76 | 46.39 / 76 | 46.34 / 76 | 46.46 / 76 |
| THP | x16 | 52.35 / 89 | 46.38 / 77 | 49.18 / 82 | 46.25 / 77 |
| THP | x8 | 53.58 / 91 | 46.42 / 76 | 50.59 / 84 | 46.41 / 76 |
| 4 KiB | x16 | 49.36 / 84 | 49.37 / 84 | 49.19 / 84 | 49.20 / 84 |
| 4 KiB | x8 | 48.99 / 82 | 48.94 / 82 | 48.60 / 82 | 48.68 / 82 |

Same traces saturated (Open, Mcycles): packed x16 3.91 / 3.91 / 2.78 / 2.90; THP x16 4.71 / 3.95 / 3.36 / 3.16; 4 KiB x16 3.93 / 3.93 / 2.34 / 2.33 (x8 similar, about 2x larger). So the robustness-owner's saturated gain (-30 to -40%) reproduces, but **closed-loop it almost vanishes**: channel hash alone is 0-1% for packed and 4 KiB, 6% for THP (bank XOR alone already gives 11-13% there), and the combination is no better than xor_global (THP 46.25 vs 46.38). ClosedCAP shows the same pattern. Cause: with 4 cores and 16 MSHRs the demand (about 13 GB/s) is far below channel capacity, so temporal channel clumping does not stall the cores. Magnitude -35% to -47% is therefore a saturated (bandwidth-bound) upper bound; the closed-loop gain is the THP bank XOR only (about 11-13% cycles, 14-15% latency). Caveat: closed-loop is core-bound by construction (constant bubbles, 4 cores); a machine with many more cores/threads would sit between the two.

**Writes, 0.5B and 7B 8k** (`20_drwb.sh`, jobs 122013 and 122338). 0.5B: KV writes 12,288 B/token exactly, activations 51,584 B, writes 0.39% of requests. 7B 8k: 75.3M reads, 204k writes (0.27%), KV append 57,344 B/token (28 layers x 2 x 4 heads x 128 x 2 B, exact), activations 7.4 MB, unattributed 5.6 MB. Writes add 0.3-1.1% saturated cycles. XOR gain holds: 7B 8k THP DDR5 identity -> xor_global 8.45 -> 7.85 Mcycles reads-only (-7.1%), 8.54 -> 7.91 with writes (-7.4%). 7B 32k writes job 122378 pending; result to be appended.

### mapping-owner: held-out, 32k MoE, batched mapping (2026-11-12)

Jobs: 121992 attr_t2 (7B 32k, decode token 2; "detached after step 17", same text yes), 121997 q30_32k (same text yes; LL misses 224.1M = 14.3 GB at 64 B, no prefetch, versus about 5 GB naive weights+KV: strong KV amplification), 121998 q05_32k (same text yes), 121993 moe_n4 (same text yes, detached after step 4; capture only, not replayed), 122001 bmiss7b16 (7B B=16 miss stream). Replays: heldout 122333, region 122334 (30B 32k), 122262 (7B B=16), 122150 (30B B=4 THP; first try 122134 died with ENOSPC on node05 /scratch, 122260/121999 on node01 scratch, both full from other data; rerun after cleanup), 122005 (30B B=16 mapstudy), 122000 (0.5B 32k). Ratios below are Ramulator Mcycles relative to identity (id) and to xor_global (xg), 8M-request window, Open / ClosedCAP; h = row hit.

**1. Held-out test (7B 32k, 122333): the XOR searched on token 1 transfers to token 2.** THP placement (matrices over physical row bits, searched on stream A and applied to stream B; "self" = searched on B itself):

| DRAM, policy | identity Mcycles (hit) | xor_train | xor_self | region_train | region_self |
|---|---|---|---|---|---|
| DDR4x1 Open | 44.02 (0.833) | 37.85 = 0.860 (0.958) | 0.854 | 0.855 | 0.856 |
| DDR4x1 ClosedCAP | 47.48 (0.738) | 41.36 = 0.871 | 0.859 | 0.865 | 0.865 |
| DDR5x16 Open | 10.78 (0.856) | 9.10 = 0.844 (0.959) | 0.840 | 0.847 | 0.840 |
| DDR5x16 ClosedCAP | 14.47 (0.710) | 10.30 = 0.712 (0.767) | 0.698 | 0.750 | 0.713 |

Held-out gain is 14-16% (Open) and 13% / 29% (DDR4 / DDR5 ClosedCAP), within 1-2 points of the in-sample numbers. Packed placement: no gain on either token (identity hit 0.966; xor_train 1.003-1.018). Region_train is not better than the global XOR. Verdict: the XOR gain is a property of the THP page-offset structure of the KV stream, not an overfit to one token's addresses.

**2. Qwen3-30B-A3B at 32k (122334), THP:**

| DRAM, policy | identity Mcycles (hit) | xor_global | region |
|---|---|---|---|
| DDR4x1 Open | 65.77 (0.822) | 0.843 (0.954) | 0.852 |
| DDR4x1 ClosedCAP | 73.43 (0.750) | 0.819 | 0.831 |
| DDR5x16 Open | 21.60 (0.801) | 0.776 (0.953) | 0.780 |
| DDR5x16 ClosedCAP | 29.36 (0.683) | 0.656 (0.764) | 0.670 |

Packed: identity already 0.952-0.954 hit, gain <= 2%. The KV-page XOR gain appears on the MoE model at 32k and is larger than for 7B (22-34%); region is within 1-2% of xor_global (never better).

**3. 0.5B control at 32k (122000), THP:** xor_global 0.880 / 0.811 (DDR4 Open / Closed), 0.883 / 0.815 (DDR5); region 0.871 / 0.807 / 0.878 / 0.800 (region slightly ahead here, 0-2%). Packed: no gain. Same shape as 7B and 30B.

**4. Batched streams, THP** (xor_global / identity; region within 1%):
- 7B B=16 (122262): DDR4 Open 0.968, Closed 0.947; DDR5 Open 0.931, Closed 0.862. Packed: 1.000 (no gain).
- 30B B=4 (122150, THP now complete): DDR4 Open 0.990, Closed 0.985; DDR5 Open 0.980, Closed 0.990. Hit identity 0.968 -> 0.975.
- 30B B=16 (122005): DDR4 Open 0.974, Closed 0.977; DDR5 Open 0.922, Closed 0.857 (identity hit 0.952 -> 0.962 Open). region 0.927 / 0.869, region_expert 0.926 / 0.873, expert_hash / expert_lin 1.000-1.004: expert-aware mappings add nothing over xor_global at B=16 either.
- Packed 30B B=16 gains <= 2%; expert-aware ~ identity.

**Verdict for the proposal.** A page-offset-aware XOR (permute bank bits with row bits, the matrix being searched offline once) removes the THP-induced row conflicts of the KV and large-activation streams: 12-34% fewer cycles at long context (32k) across 0.5B, 7B and 30B-A3B, and it transfers across tokens (held-out). It is a no-op when pages are contiguous (packed). Gains shrink with batching (large weight reads dominate): DDR5 Open 7% (7B B16), 8% (30B B16), 2% (30B B4). Region-specific (K/V/O) matrices and expert-aware mappings never beat the single global XOR by more than 1-2%; the global XOR is the proposal. Caveats: replays are open-loop, 8M-request windows, placement is modelled (random THP), single decode token per stream.

Operational: node05 and node01 /scratch were full (other users' data; node01 freed by the main session); scripts use node-local /scratch, so check `df /scratch` before replays.

### policy-owner: closed-loop calibration and headline gain (2026-11-12), part 2

**Calibration** (`23_calib.sh`, job 122404; `policy_replay.py --window-depth --mshr`). SimpleO3 runs at the memory clock (2.4 GHz); ipc is an integer, so a slower core is modelled by dilating compute by k: bubbles' = k(b+1)-1 and window' = k x window (same MLP, k-times slower compute). MSHRs per core (16 vs 32) change nothing; the instruction window is the effective-MLP knob. Native targets: 7B short (B1 P512, steps 1-3, 4 threads): 69.1M reads x 64 B in ~172 ms = 25.7 GB/s, 42 instr/read; 7B 8k: 7.8 GB/s; 7B 32k (drattr c32k step 16): 179.5M reads in 2.40 s = 4.8 GB/s (about 0.4 IPC per thread: the attention phase is compute-bound in this build). Sweep (1M-read windows, THP DDR5 identity): short proxy (7B 8k stream, weights window 94% W, bubbles 42): window 128/192/256/384/512 -> 14.3/21.9/26.2/35.9/42.6 GB/s; **chosen window 256 (26.2 GB/s, +2% vs native)**. 32k (bubbles 52): k=1/6/8/10/12 at window 256k -> 18.5/7.3/5.6/4.5/3.8 GB/s; **chosen k=10 (bubbles 529, window 2560): 4.5 GB/s, -6% vs 4.8**.

**Rerun, 7B 32k, 2M window, closed loop** (`22_chanloop.sh`, MODES=o3). Mcycles / avg read latency (cycles) / GB/s, Open, identity | xor_global | chan | xor_global_chan.

(A) matched 32k native rate (k=10; job 122422): every cell, all placements, DRAM widths and both policies, is 67.94-67.96 Mcycles at 4.5 GB/s: **no cycle gain from any mapping or policy**, because the decode step at 32k is compute-bound in the native build. Only read latency moves, on THP: 107 (x16) / 110 (x8) identity -> 102 / 101 xor_global, 104 chan, 102 / 101 xor_global_chan (-5 to -8%); packed and 4 KiB flat.

(B) fast-core bound (k=1, window 256, bubbles 52, ~22 GB/s demand, about the short-context native rate; job 122423):

| placement | DRAM | identity | xor_global | chan | xor_global_chan |
|---|---|---|---|---|---|
| packed | x16 | 14.88 / 78 / 21.9 | 14.62 / 78 / 21.7 | 14.19 / 77 / 22.1 | 14.28 / 78 / 21.9 |
| packed | x8 | 14.63 / 78 / 21.5 | 14.50 / 78 / 21.6 | 14.49 / 78 / 21.5 | 14.64 / 78 / 21.6 |
| THP | x16 | 17.33 / 99 / 18.7 | 14.49 / 78 / 21.8 | 15.71 / 85 / 20.4 | 14.23 / 77 / 22.0 |
| THP | x8 | 17.69 / 102 / 18.2 | 14.66 / 78 / 21.5 | 15.85 / 86 / 20.1 | 14.55 / 77 / 21.7 |
| 4 KiB | x16 | 16.13 / 94 / 19.5 | 16.13 / 94 / 19.5 | 16.08 / 94 / 19.5 | 16.08 / 94 / 19.5 |
| 4 KiB | x8 | 15.96 / 89 / 19.7 | 15.94 / 89 / 19.7 | 15.83 / 89 / 19.8 | 15.81 / 89 / 19.8 |

ClosedCAP has the same pattern (about +3% cycles over Open). Headline: with fast cores, **THP bank XOR -16% to -17% cycles (-21 to -22% latency)**, channel hash alone -9%, combined -18% (x16) / -18% (x8) at best, i.e. the channel hash adds at most 1-2% over bank XOR; packed gains <=4% from either, 4 KiB <=1%. The mapping gain therefore needs (i) THP placement (huge pages concentrate KV across few banks; 4 KiB placement already spreads them) and (ii) a workload that is actually memory-bound (short/medium context, fast cores). At native 32k speeds the gain is latency only. The saturated -35..-47% channel-hash figure is unreachable in the closed loop (<=2% beyond bank XOR).

### policy-owner: writes at 32k (2026-11-12), part 3

Job 122378 (`20_drwb.sh`, 7B, `PROMPT_CHARS=116000 CTX=32768 TAG=_c32k`, 2h37 on node01, published to `results/fullscale/drwb/..._wb_c32k/`). Stream: 177.9M DRAM reads, 225,433 writes (0.127% of requests); KV append 57,344 B/token (896 lines, exact), activations 8.55 MB, unattributed 5.8 MB (graph buffers not in the tensor table). Writes are negligible at 32k, as at 8k. Saturated Ramulator replay, Open/ClosedCAP not separated here (two rows per key, Open first-listed lower), Mcycles reads-only -> reads+writes: THP DDR5 identity 16.97 -> 17.00, xor_global 13.59 -> 13.61; packed DDR5 13.62 -> 13.64 / 13.52 -> 13.54; THP DDR4 64.56 -> 64.62 / 55.44 -> 55.50. **The XOR gain is unchanged by writes** (THP DDR5 -19.9% reads-only and -19.9% with writes; THP DDR4 -14.1% / -14.1%; packed DDR5 -0.8% / -0.8%). Adding the write stream costs 0.1-0.4% cycles.

### robustness-owner: memory system, page size, 1k context, prefill (2026-11-12), part 2 and verdict

Jobs: memsys 122014 (7B 32k, 12M window), 122015 (7B 8k), 122016 (30B short), 122017 (7B 32k full 179.5M reads, DDR5x8), 122018 (7B 32k full, DDR5x16); scenario captures 122008 (7B 1k, step 1), 122009 (30B 1k), 122010 (7B prefill, 30 tokens, 3 h 24 min), 122011 (30B prefill, 2 h 29 min); region replays 122375, 122415, 122545, 122555. All captures: "same text: yes", steps_traced.txt shows the intended call (1k: decode step 1; prefill: call 0 = the prompt). 7B 1k: 69.6M reads (4.45 GB), KV 55.7 MB = 1.3% of reads (expected 58 MB for 1k tokens x 57 KB). 7B prefill: 74.7M reads (4.78 GB for 30 tokens: weights read once, 4.39 GB, activations 5.3%, KV 0.06%; the export reports "1 threads"); 30B prefill 125.3M reads (8.0 GB). Node01 scratch was full for a while (stale drburst, cleared by the user) and node05 scratch is full: all jobs ran on node01/03/04.

**Full 7B 32k windows** (179.46M reads, THP, Mcycles Open / ClosedCAP, 12M-window values in part 1):

| DRAM | identity | chan | xor_global | xor_global_chan | region |
|---|---|---|---|---|---|
| DDR5-4800 x8 | 461.1 / 610.7 | 317.6 / 434.3 | 393.1 / 441.7 | 305.6 / 362.7 | 396.9 / 465.5 |
| DDR5 x16 | 235.0 / 310.7 | - | 196.8 / 224.3 | 144.0 / 175.2 | 197.8 / 227.9 |

xor_global saves 14.7% / 27.7% (x8) and 16.3% / 27.8% (x16) over the whole token, matching the 12M window (15% / 27% and 17% / 29%); the 12M window is representative. Adding the channel hash: -34% / -41% (x8), -39% / -44% (x16).

**Other conditions** (Mcycles Open / ClosedCAP, THP, DDR5x16 replay, xor_global vs identity; 12M windows):

| stream | identity | xor_global | saving Open / Closed | 8k-style chan hash (x16 Open) |
|---|---|---|---|---|
| 7B 32k | 16.09 / 21.36 | 13.36 / 15.14 | 17% / 29% | 10.53 (-35%) |
| 7B 8k | 8.44 / 11.61 | 7.90 / 9.96 | 6% / 14% | 7.40 (-12%) |
| 7B 1k | 7.10 / 9.22 | 6.98 / 8.88 | 1.7% / 3.7% | not run |
| 30B 1k | 7.80 / 10.72 | 7.50 / 9.88 | 3.8% / 7.8% | not run |
| 7B prefill (30 tok) | 7.31 / 9.55 | 7.18 / 9.24 | 1.8% / 3.2% | not run |
| 30B prefill | 7.48 / 11.26 | 7.43 / 11.12 | 0.7% / 1.2% | not run |
| 30B short (x8 only) | 13.57 / 17.12 | 13.63 / 17.19 | 0% | 13.57 (0%) |

On DDR4 1 channel the 1k/prefill savings are 0.4-1.3% (7B 1k 61.07 -> 60.68; 30B 1k 62.90 -> 62.37 region) except 7B prefill ClosedCAP 3.9%. 7B 8k, memsys x8 (THP): identity 16.71 / 22.84 -> xor_global 15.67 / 19.31 (6% / 15%), xor_global_chan 14.80 / 18.40.

**Page size.** With 4 KiB pages (shuffled) the default mapping already reaches only 0.59-0.75 row hit and neither bank hash (xor_global, region) changes anything (7B 32k x16 13.4 -> 13.3 to 13.4 Mcycles); packed gains nothing from bank XOR either. THP (random 2 MiB frames) is the only placement where the bank XOR helps, because the default mapping then puts the K and V streams of different heads into the same banks at different rows.

**Channel hash.** At 7B 32k it removes 31-47% of cycles for every placement, including packed (13.3 -> 8.7) and 4 KiB (13.4 -> 7.1, i.e. it fixes the 4 KiB case that no bank hash can). Verified (part 1) that it is a bijection, with equal read counts, and the channel load over time is clumped under the default mapping (busiest channel 2.43x mean over 32 requests, 6.4 of 8 channels used) and near uniform when hashed (1.86, 7.5/8). Weaker at shorter context: 7B 8k packed x16 no change (7.87 -> 7.88), THP -12%, 4 KiB -8%; 30B short none. So it is a KV-stream effect (K/V streams with a power-of-two stride across heads land on few channels). Magnitudes are upper bounds from the saturated open-loop replay, whose channel-clumping depends on the miss-stream interleave.

**Verdict.** The KV-page XOR gain is robust to memory system (8 vs 16 DDR5-4800 channels, 15-17% Open / 28-29% Closed) and to window length (full token = 12M window), but it is conditional: it needs THP-style random 2 MiB placement and a long KV (>= 8k tokens; 6% at 8k, 17% at 32k). It vanishes with packed or 4 KiB pages, shrinks to 2-4% at 1k context (30B up to 4-8%), is about 1% for prefill of 30 tokens, and is zero for the 30B at short context. Region-aware again equals global XOR within 1-2%. A channel-bit hash on top is an additional, larger and more placement-independent win for long KV (at 32k), which the study had not looked at before; it should be reported as a separate mechanism with the clumping caveat. Not covered: closed-loop replay, DDR5 timing variants beyond 4800, prefill longer than 30 tokens, held-out hash search.

### main session: verification of the 2026-11-11/12 agent results (2026-11-12)

Re-derived from the raw replay outputs in `cluster-data/results/fullscale/` and the cluster job logs; all headline numbers in the mapping-owner, policy-owner and robustness-owner sections match.
- Held-out (`heldout/7B32k_t1t2`, THP): DDR5 Open identity 10.8 -> xor_train 9.1 Mcycles (0.84), ClosedCAP 14.5 -> 10.3 (0.71); DDR4 Open 44.0 -> 37.9 (0.86). Matches.
- Full 7B 32k token (`memsys/q7_32k_full_x8|x16`, THP): x8 461.1 -> 393.1 (-14.7%) Open, 610.7 -> 441.7 (-27.7%) ClosedCAP; x16 235.0 -> 196.8 (-16.3%), 310.7 -> 224.3 (-27.8%). Matches.
- Closed loop (`chanloop/`): c32k_w4M THP x16 52.35 -> 46.38 (-11.4%), chan 49.18, combo 46.25; calA_k10 (native-matched) every cell 67.94-67.95 M; calB_k1 THP 17.33 -> 14.49 (-16.4%), chan 15.71 (-9.3%), combo 14.23 (-17.9%). Native 7B 32k decode step 16 = 2.395 s for 179.46M reads = 4.79 GB/s (calibration target confirmed).
- Writes 32k (`drwb/..._wb_c32k`, 12M-read window, 10,693 writes): THP DDR5 Open 17.0 -> 13.6 with or without writes; ClosedCAP 21.9 -> 14.8/14.9 (-32%). The policy-owner note says Open/ClosedCAP were not separated; they are, and ClosedCAP shows the larger gain.
- Policies (`analysis/policy_results.md`, 150 non-Open cells): best non-Open cell 0.968 of Open (HAPPY, 7B 32k THP DDR5 identity, closed loop); next 0.976-0.988 (Region). "No policy better than 3.2%" confirmed.
- 1k / prefill (`region/*_c1k`, `*_pf32`, THP DDR5): 7B 1k 7.1 -> 7.0 / 9.2 -> 8.9; 30B 1k 7.8 -> 7.5 / 10.7 -> 9.9; 7B prefill 7.3 -> 7.2 / 9.6 -> 9.2; 30B prefill 7.5 -> 7.4 / 11.3 -> 11.1. Matches (small gains).
- Capture validation (logs): 121992, 121993, 121997, 121998, 122008, 122009, 122010, 122011, 122378 all "detached after step N" with the intended step and "same text: yes"; 122001 (7B B=16 batch) "same tokens: yes".
- One defect: 122011 (30B prefill) `drmiss_export.py` stage raised in `placements()` (rng.choice over CAPACITY//2MiB: the THP placement needs 7,106 x 2 MiB = 14 GB > the 8 GiB default of `17_drattr.sh`'s export), so that run has no attribution.json. Its region replay (18_region.sh, AMCAS_CAPACITY_GB=32) is unaffected and valid. Fix if the attribution is wanted: export AMCAS_CAPACITY_GB=32 in 17_drattr.sh (as 18_drbatch.sh does) and rerun the export on the kept stream.
- Superseded claim: the robustness-owner verdict's "channel-bit hash ... additional, larger win" (-31 to -47%) is saturated-replay only; closed loop (policy-owner part 1 and calB) gives 0-1% (packed, 4 KiB) and at most 2% beyond the bank XOR on THP. Report the channel hash as a saturated-only effect.
