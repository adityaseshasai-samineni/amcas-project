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
