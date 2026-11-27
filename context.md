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

### 30B prefill attribution rerun (job 122693, 2026-11-12)

`17_drattr.sh` now exports `AMCAS_CAPACITY_GB=${CAP_GB:-32}` (commit 2808a92; the 8 GiB default could not hold 7,106 THP frames for 30B prefill). Re-ran only `drmiss_export.py` on the kept `miss.gz` of `Qwen3-30B-A3B-Q4_K_M_s0n1t4_attr_pf32` (10 min, node01, published 52d1c28). 125.3M reads (8.02 GB): expert weights 83.8% (6.72 GB), dense weights 10.3%, activations 4.7%, unattributed 1.2%, KV 0.1%. Experts touched per expert tensor: 48.9 of 128 for a ~30-token prompt (decode B=1: 8; B=4: 23; B=16: 57). Offline row hit THP / RoRaBaCoCh -> XOR-bank: expert weights 0.984 -> 0.984, KV 0.806 -> 0.924, activations 0.735 -> 0.821; total gain under 1% (as in the region replay). Caveat: the prefill export sees a single thread id ("1 threads"), also for 7B prefill, so prefill streams lack the 4-thread interleave of decode. Future drattr runs use 32 GiB placements, so their shuffled-placement row hits are not directly comparable with earlier 8 GiB runs.

### policy-owner: when is long-context decode memory-bound? (2026-11-12), part 4

Native thread sweep, job 122477 (`24_native_ctx.sh`, node04, not exclusive, node load average 4-7 during the run, no pinning/NUMA policy; `results/fullscale/native/ctx32k/`). Build-fast (AVX-512, OpenMP) 7B Q4_K_M; 32k = prompt of 31,001 tokens evaluated once (449 s at 32 threads) and restored with `--prompt-cache-ro`, 32 decode tokens per thread count. GB/s = DRAM bytes per token x tokens/s with 11.49 GB/token at 32k (179.46M LL misses x 64 B, no-prefetch cachesim of the 4-thread trace; approximate) and 4.43 GB short.

| threads | 32k tok/s | 32k GB/s | short tok/s | short GB/s |
|---|---|---|---|---|
| 4 | 0.87 | 10.0 | 6.17 | 27.3 |
| 8 | 1.74 | 20.0 | 9.71 | 43.0 |
| 16 | 2.80 | 32.2 | 11.67 | 51.7 |
| 24 | 3.59 | 41.2 | 12.35 | 54.7 |
| 32 | 3.78 | 43.4 | 12.63 | 55.9 |
| 48 | 3.63 | 41.7 | 11.60 | 51.4 |

Earlier benchmark (qwen7b.threads.json, short): 1/8/16/32 threads 1.77/9.18/12.44/13.55 tok/s. 30B short: 3.84/15.78/19.79/22.01 tok/s. **Memory-bound point: about 16 threads for short context (about 52-56 GB/s plateau), about 24-32 threads at 32k (about 43 GB/s plateau)**; 4 threads (the traced configuration) are far from it (10 GB/s at 32k, 27 GB/s short). Note: this build decodes 32k at 4 threads in 1.15 s/token, not the 2.40 s of the DynamoRIO-binary plain run used for the part-2 calibration (4.8 GB/s); the part-2 "matched-rate" cell therefore under-states native speed, and part-2 case (B) (about 22 GB/s) is closer to the real 4-thread rate (10 GB/s native at this build is slower than (B) but faster than (A)).

Closed-loop at the plateau (jobs 122705/122706, 2M window, bubbles 52, Open; window depth 512 / 1024 gives 33-35 / 40 GB/s, i.e. up to 93% of the native 43 GB/s plateau). Mcycles at window 1024: packed x16 identity 7.61, xor_global 7.62, chan 7.58, xor_global_chan 7.58; THP x16 7.72 / 7.63 / 7.61 / 7.59, THP x8 7.87 / 7.69 / 7.70 / 7.65; 4 KiB x16 7.71 / 7.71 / 7.63 / 7.63. Window 512: THP x16 9.68 / 8.99 / 9.41 / 8.88, THP x8 10.12 / 9.32 / 9.81 / 9.26; packed and 4 KiB within 1-2%. So at the native plateau bandwidth the expected token-time gain of xor_global is **about 1% (x16) to 2% (x8) at 40 GB/s demand (THP), up to 7% at 33 GB/s window 512 where the sim cores are still MLP-limited**, and 0% for packed/4 KiB; THP read latency still drops 25%. Reason: the plateau (43 GB/s) is only 15% of the node's DDR4 peak (2 sockets x 6 channels DDR4-2933, about 280 GB/s) and 14% of the simulated DDR5 x16 peak (307 GB/s), so the plateau is set by the cores/NUMA/software, not by DRAM row conflicts or channel clumping; the mapping gain appears only when demand approaches the DRAM limit. Caveats: Ramulator DDR5 is not this node's DDR4; window depth is the effective-MLP knob, not a pinned microarchitectural parameter; 2M-read window; native sweep ran on a shared node.

### policy-owner: 4-thread 32k rate reconciled; final verdict (2026-11-12), part 5

**Why 4.8 GB/s (part 2) vs 10 GB/s (part 4).** Job 122716 (`25_binary_cmp.sh`, node03, same 31,001-token prompt state restored with `--prompt-cache-ro`, 4 threads, 9 tokens): llama-completion (build-fast) 1.36 s/token; llama-drattr (Release rebuild) 1.23-1.46 s/step, mean about 1.33 s (the graph-walk hook costs nothing when not tracing); llama-drattr.debug-build (empty CMAKE_BUILD_TYPE, i.e. ggml compiled without -O, the binary used for the drattr c32k untraced step) 1.47-1.92 s. The build type explains about 25%. (a) binary: yes, partly; (b) the 2.395 s step 16 is a true decode step at the same 31k context (steps 0-14 are 2048-token prompt chunks, 15 the last 280-token chunk), so not a context mismatch; (c) bytes: 11.49 GB = 179.46M no-prefetch LL misses x 64 B, the same convention as the 4.43 GB short figure, so the bandwidth convention is consistent (prefetch counts not available for c32k); (d) node: node04 1.15 s/token (0.87 tok/s, part 4), node03 1.36 s/token here (loaded node). Remaining factor of about 1.4 between 2.4 s and 1.7 s is unexplained (probably contention on the shared node during the original untraced run). **Real 4-thread 32k rate: 8.4-10 GB/s (1.15-1.36 s/token).**

**Recalibrated matched rate** (jobs 122730 k=4: bubbles 211, window 1024; 122731 k=5: bubbles 264, window 1280; 2M window, Open): k=4 gives 10.8 GB/s (28.4 Mcycles), k=5 gives 8.6 GB/s (35.6 Mcycles); the native 8.4-10.0 GB/s lies between, so either bounds it (**chosen k=4.5, effectively**). In both, every mapping is identical within 0.3% cycles in all placements and both DDR5 widths (THP x16: k=4 28.43 / 28.37 / 28.39 / 28.37 for identity / xor_global / chan / xor_global_chan; packed and 4 KiB flat); only THP read latency changes: identity 97 (x16) / 100 (x8) -> 85 / 84 with bank XOR (-12% to -16%), chan 89. **The conclusion does not change: at 4 threads the 32k decode is compute/MLP-limited at about 10 GB/s and no mapping gives a cycle gain; the gain appears only above about 20 GB/s demand (about 8-12 threads at 32k), where THP bank XOR is -16% (22 GB/s, window 256) falling to 1-2% at the 40 GB/s plateau (part 4), and nowhere exceeds that range.** Cycle-vs-demand summary, THP x16 identity -> xor_global (Mcycles): 10.8 GB/s 28.43 -> 28.37 (-0.2%); 22 GB/s 17.33 -> 14.49 (-16%); 33 GB/s (window 512) 9.68 -> 8.99 (-7%); 40 GB/s (window 1024) 7.72 -> 7.63 (-1.2%). The non-monotone shape is the sim's: the gain needs both enough demand to stress banks (>=20 GB/s) and a window small enough that core MLP, not the plateau, limits.

**Final verdict.** (1) No row-buffer policy (ClosedCAP, Timeout, HAPPY, Region) beats Open by more than 3% in any closed-loop cell; Open is the right default. (2) The mapping that matters is the THP bank XOR (xor_global, a GF(2) hash of up to 3 row bits per bank bit): -16 to -17% closed-loop cycles and -21 to -22% read latency in the memory-demanding regime on a THP system; no gain for packed or 4 KiB placement. (3) The channel-bit hash gives 0-2% beyond that in the closed loop (saturated -35..-47% is an upper bound). (4) Writes are 0.1-0.4% of requests and do not change any of this. (5) Native decode is core/NUMA-limited well below DRAM peak (43 GB/s at 32k, 52-56 GB/s short, about 15% of the node's DDR4 peak), so the headline gain on this hardware is small (latency-only at 4 threads, 1-2% at the plateau); it would matter on machines or batch sizes that push demand toward the DRAM limit (batched decode, MoE with many experts, higher-core-count parts). Caveats: Ramulator DDR5 stands in for the node's DDR4; window depth is the MLP knob; 2M-read windows; the native sweeps ran on shared non-exclusive nodes.

### validation-owner: real DDR4 memory system and DRAMsim3 cross-check (2026-11-12)

**Machine facts.** node01 (and the other Cascade Lake nodes): 2 sockets, `lscpu` 2x16 cores. EDAC (`/sys/devices/system/edac/mc`) lists 32 GB DIMMs on 16 slots: socket 0 and 1 each have 2 iMCs x 3 channels, 8 DIMMs per socket (two channels at 2 DIMMs per channel, four at 1), 512 GB total; DDR4 speed not readable without root, so DDR4-2933 (the 6226R maximum) is assumed. No pinning or numactl was used by the native runs, so the 4 llama.cpp threads may have spanned sockets; we model one socket (6 channels, 141 GB/s peak) and two (12 channels). Not known: the real Intel address hash (Skylake already XORs bank bits), the 2-DIMM channels.

**Model** (new files `tools/realmem_traces.py`, `realmem_replay.py`, `dramsim3_replay.py`, `dramsim3_drain.cc`, `scripts/25_realmem.sh`; nothing existing edited). Per channel one 32 GB dual-rank 16Gb x8 DDR4-2933Y (21-21-21) DIMM, intra-channel RoBaRaCoCh (col 7, bank+rank 5, row 17 bits), channel = (addr>>8) mod N (N=6/12; Ramulator gets a power-of-two channel field with the spare controllers idle). Same attributed stream, window 3M reads from 30% (7B 32k, `..._s1n1t4_attr_c32k`), same placements (32 GB pool), same greedy bank-XOR search (`region_remap.search`). Closed loop = SimpleO3 (4 cores, ipc 4, MSHR 16, core 2.93 GHz = 2x memory clock) with bubbles/window points: k5 (bubbles 264, window 1280), k1 (52, 256), w512 (52, 512), w1024 (52, 1024). Jobs: 122767 (6ch sat + DRAMsim3), 122783 (6ch closed loop), 122844 (12ch everything; 122766/122817 failed on asserts: granule check invalid for mod-N, region not injective at 12ch, so region is 6ch only), smoke 6ch on node01.

**Results, Ramulator, THP, xor_global cycles / identity (Open | ClosedCAP), hit identity -> xor, region within 1-2% of xor at 6ch:**

| operating point (6ch demand GB/s, identity) | 6ch Open | 6ch ClosedCAP | 12ch Open | 12ch ClosedCAP |
|---|---|---|---|---|
| saturated (77.6 / 95 GB/s) | 0.901 (hit .911->.949) | 0.913 | 0.872 (.935->.965) | 0.825 |
| fast core k1 (22.9 / 24.2) | 0.954 (lat 47.5->45.1) | 0.964 | 0.928 | 0.962 |
| w512 (36.2 / 38.6) | 0.962 | 0.973 | 0.948 | 0.969 |
| w1024 plateau (46.6 / 48.2) | 0.988 (lat 58.6->54.5) | 0.991 | 0.990 | 0.993 |
| k5 native 4-thread 32k (10.5) | 1.000 (lat 49.3->47.8) | 1.000 | 1.000 | 1.000 |

Packed and 4 KiB: 0.99-1.00 everywhere (6ch sat packed 0.992, 4k 1.000; 12ch sat packed 0.935 Open / 0.969, a channel-imbalance artefact, closed loop 1.000). Absolute saturated: 6ch identity THP 3.63 Mcycles (77.6 GB/s of 141 peak), packed 3.17.

**Token-time gain on the real-machine model:** at the corrected native 4-thread rate (10 GB/s) none (0.0%; latency -3%); at the ~43 GB/s plateau 1.2% (6ch) / 1.0% (12ch) of memory-time, and these workloads at plateau are only 33% (6ch) of peak; up to 4-5% (6ch) / 7% (12ch Open) only for cores faster than the machine delivers (23-39 GB/s demand, not reachable natively). So the XOR is a bandwidth-headroom effect: it becomes visible only when demand approaches the channel capability, which the native 4-thread run is far from. Note 12ch has uneven channel loads in this mod-12 map (max/mean 1.30; 1.10 at 6ch): an address-pattern effect of the mod-N interleave, not measured on hardware.

**DRAMsim3 cross-check** (DDR4_8Gb_x8_2933 config edited to 16Gb x8, 2 ranks, rochrababgco, per-channel instances, 1 request/cycle saturated, run to drain; DRAMsim3 has no DDR5). Mcycles / row hit, OPEN_PAGE:

| | packed id | packed xor | THP id | THP xor | 4k id | 4k xor |
|---|---|---|---|---|---|---|
| 6ch DRAMsim3 | 2.680 / .963 | 2.692 / .964 | 2.782 / .943 | 2.708 / .961 (-2.7%) | 2.429 / .730 | 2.427 |
| 6ch Ramulator Open | 3.168 / .956 | 0.992x | 3.628 / .911 | 0.901x / .949 (-9.9%) | 2.912 | 1.000x |
| 12ch DRAMsim3 | 1.593 / .974 | 1.673 | 1.749 / .961 | 1.634 / .975 (-6.6%) | 1.432 | 1.433 |
| 12ch Ramulator Open | 2.626 | 0.935x | 2.955 / .935 | 0.872x / .965 (-12.8%) | | |

DRAMsim3 CLOSE_PAGE (always auto-precharge, hit 0.000, 2.7-4x slower) is a different policy from ClosedCAP and is not comparable; THP xor still -8% (6ch) / -13% (12ch) there. **Verdict: DRAMsim3 agrees in direction and mechanism** (THP only; packed/4k flat; hit 0.94-0.96 -> 0.96-0.975) **but the saturated gain is about a third to a half of Ramulator's** (-2.7% vs -9.9% at 6ch, -6.6% vs -12.8% at 12ch). Its identity baseline is already better (hit 0.943 vs 0.911; per-bank command queues of 8 plus a 32-entry transaction queue reorder more). Both simulators therefore support the effect, with DRAMsim3 suggesting the Ramulator numbers (-10..-17% / -27..-29% on DDR5) are upper-end for a controller with deeper reordering. Latencies in DRAMsim3 (300-500 cycles) are queueing delay of the back-to-back injection and not comparable with Ramulator.

**Overall verdict for the proposal:** on the machine's own memory (DDR4-2933, 6-12 channels) the bank-XOR gain is -9..-13% saturated (Ramulator) / -3..-7% (DRAMsim3), but only ~1% at the native 43 GB/s plateau and 0% at the corrected 10 GB/s 4-thread rate; the real mapping already differs (Intel hash unknown), THP placement is required. Caveats: 3M-read window, model address map and DIMM organisation assumed, closed-loop core model is calibrated for DDR5 (re-checked only by achieved GB/s here), no row-policy/refresh differences evaluated, region not run at 12ch.

### transfer-owner: cross-model transfer, fragmentation, MoE 4-token (2026-11-12)

Tools (new): `tools/transfer_remap.py` (train-set/test-set generalisation of heldout_remap.py, imports mapprep/region_remap/remap), `tools/frag_remap.py` (placement sweep), `tools/moe_experts_touched.py`, `tools/transfer_report.py` (tables), scripts `26_moe4.sh`, `27_transfer.sh`, `28_frag.sh`. Jobs: 122723 (MoE 4-token, 1h18), 122792 (transfer, 1h45; first try 122791 died in 7 s on an unbound `$SMOKE` under `set -u`), 122922 (fragmentation, 1h54). Results in `cluster-data/results/fullscale/{transfer/xfer,frag/7B32k,frag/30B32k,moe/Q30B_n4}` and `region/Qwen3-30B-A3B-Q4_K_M_s1n4t4_attr_n4m`. All emitted traces assert unchanged channel/row/column/2 MiB granule and equal distinct-address count (bijection) per test stream; no assertion failed. Window 8M requests (start frac 0.3), Ramulator RoBaRaCoCh.

**Task 4: one fixed matrix.** Matrices: T7 = searched on 7B 32k; T30 = on 30B 32k; pool = on equal windows (2 x 100k chunks each) of 7B 32k, 30B 32k, 0.5B 32k, 7B 8k; pool_loo = the same mix without the test stream (no leakage); naive = bank bit j ^= row bit j; self = searched on the test window. Cycle ratios vs identity (<1 = XOR helps), "-" = identical to self. Matrices are searched per placement/DRAM on that placement's physical bits.

THP, DDR5 x16, Open:

| test | identity Mcyc | naive | self | T7 | T30 | pool | pool_loo |
|---|---|---|---|---|---|---|---|
| 7B 32k | 10.68 | 0.822 | 0.818 | - | 0.821 | 0.827 | 0.819 |
| 7B 32k token 2 | 10.78 | 0.857 | 0.840 | 0.844 | 0.849 | 0.846 | - |
| 30B 32k | 14.50 | 0.784 | 0.772 | 0.778 | - | 0.781 | 0.779 |
| 0.5B 32k | 5.13 | 0.894 | 0.892 | 0.897 | 0.897 | 0.903 | 0.897 |
| 7B 8k | 5.50 | 0.949 | 0.948 | 0.946 | 0.949 | 0.947 | 0.949 |
| 7B 1k | 4.75 | 0.986 | 0.985 | 0.982 | 0.986 | 0.982 | - |
| 30B 1k | 5.24 | 0.960 | 0.967 | 0.959 | 0.964 | 0.963 | - |
| 30B short | 4.54 | 1.003 | 1.000 | 1.000 | 1.000 | 1.001 | - |

THP, DDR5 x16, ClosedCAP:

| test | identity Mcyc | naive | self | T7 | T30 | pool | pool_loo |
|---|---|---|---|---|---|---|---|
| 7B 32k | 14.45 | 0.702 | 0.700 | - | 0.713 | 0.726 | 0.697 |
| 7B 32k token 2 | 14.47 | 0.776 | 0.698 | 0.712 | 0.730 | 0.728 | - |
| 30B 32k | 19.22 | 0.699 | 0.663 | 0.687 | - | 0.687 | 0.687 |
| 0.5B 32k | 7.60 | 0.824 | 0.821 | 0.830 | 0.832 | 0.835 | 0.827 |
| 7B 8k | 7.54 | 0.859 | 0.860 | 0.860 | 0.866 | 0.858 | 0.862 |
| 7B 1k | 6.19 | 0.970 | 0.967 | 0.965 | 0.970 | 0.962 | - |
| 30B 1k | 7.23 | 0.915 | 0.933 | 0.916 | 0.927 | 0.920 | - |
| 30B short | 5.73 | 1.005 | 0.995 | 0.999 | 0.995 | 1.001 | - |

THP, DDR4 x1 (Open | ClosedCAP, ratios): 
| test | identity Mcyc | naive | self | T7 | T30 | pool | pool_loo |
|---|---|---|---|---|---|---|---|
| 7B 32k | 41.67 \| 46.41 | 0.975 \| 0.968 | 0.911 \| 0.918 | - | 0.917 \| 0.925 | 0.931 \| 0.942 | 0.931 \| 0.942 |
| 30B 32k | 43.52 \| 48.43 | 0.964 \| 0.928 | 0.858 \| 0.830 | 0.875 \| 0.867 | - | 0.871 \| 0.853 | 0.868 \| 0.844 |
| 0.5B 32k | 48.59 \| 68.17 | 0.958 \| 0.869 | 0.885 \| 0.823 | 0.872 \| 0.816 | 0.876 \| 0.824 | 0.879 \| 0.826 | 0.892 \| 0.831 |

Packed (4 KiB-compacted) placement: DDR5 every variant 0.99-1.03 (no gain; T7 on 0.5B Open 1.013). DDR4 packed: self and fixed matrices **hurt** by 1-4% on the 32k streams (7B self 1.020/1.016, 30B self 1.015/1.013, T7 on 30B 1.044/1.048, pool 1.016-1.024), and gain 1-7% only on 0.5B (T7 0.934).

Verdict (task 4). Gap to the self-searched matrix, THP DDR5 x16, over the 32k/8k streams: **Open: all fixed searched matrices within 1.1 points (max pool on 0.5B +1.1, pool on 30B +0.9); ClosedCAP: within 0.3-3.2 points** (T7 on 30B +2.4, pool on 30B +2.4, pool on 7B +2.6, T30 on 7B token 2 +3.2). Cross-model transfer works in both directions (T7 -> 30B 0.778 vs 0.772 self; T30 -> 7B 0.821 vs 0.818), the pooled matrix is not better than the single-model one (pool_loo, which excludes the test stream, is about equal to pool and to T7), and equal-weight pooling over four streams does not buy robustness. The **naive** row-bit-j matrix is already within 0.4-1.2 points of self in Open at 32k (30B 0.784 vs 0.772) and equal at 8k/1k, but ClosedCAP 3.6 points worse on 30B and 7.8 points worse on 7B token 2 (0.776 vs 0.698): the classic permutation-based interleaving captures most of the DDR5 THP gain, and searching adds 1-8 points in ClosedCAP. On DDR4 x1 the naive matrix is far worse (7B 0.975 vs 0.911; 30B 0.964 vs 0.858, 10 points) and fixed searched matrices lose 0.6-3.7 points to self. Weak-gain streams (1k, short) get 0-8% from any matrix and never lose in THP; there the self matrix is sometimes worse than a fixed one (30B 1k), so differences below 1-2 points are in the noise of the 8M-window search. So: a single fixed matrix (searched once, e.g. on 7B 32k, or the naive one for DDR5 Open) is a defensible hardware proposal for DDR5 x16 with 2 MiB pages, within about 1 point (Open) / 3 points (ClosedCAP) of per-workload search; for DDR4 x1 it costs 2-4 points and the naive one is inadequate. Risk: with 4 KiB-compacted pages the XOR is a small loss (up to 4% DDR4), so the XOR should be enabled only with huge pages.

**Task 5: fragmentation.** Model: distinct 2 MiB VA granules cut into runs of L; a fraction c of the runs is placed in physically contiguous aligned L-granule slots (VA order), the rest on random free frames (32 GiB, 16384 frames; 7B 3122 granules). Identity is **identical for every (c, L)** (7B 32k: 10.68 / 14.45 Mcycles, hit 0.853 / 0.705; 30B 14.50 / 19.22): row-buffer behaviour under the default mapping depends on the frame only through row equality, and a 2 MiB granule keeps its in-granule offset layout. What changes is the XOR, because its inputs are row bits = the frame number. Ratios vs identity, DDR5 x16, 7B 32k (c0 = matrix searched once on the fully random placement, applied unchanged; self = searched on that placement):

| placement | Open xor_c0 / self | ClosedCAP xor_c0 / self |
|---|---|---|
| random (c=0) | 0.815 / 0.815 | 0.687 / 0.687 |
| c=0.25, L=8 / 64 / 512 | 0.817 / 0.824 / 0.820 (self 0.814-0.818) | 0.690 / 0.709 / 0.705 (self 0.687-0.691) |
| c=0.5, L=8 / 64 / 512 | 0.818 / 0.874 / 0.824 (self 0.823 / 0.826 / 0.814) | 0.691 / 0.859 / 0.711 (self 0.694 / 0.720 / 0.688) |
| c=0.75, L=8 / 64 / 512 | 0.820 / 0.815 / 0.865 (self 0.819-0.821) | 0.691 / 0.688 / 0.856 (self 0.692 / 0.692 / 0.701) |
| c=0.9, L=8 / 64 / 512 | 0.815 / 0.868 / 0.865 (self 0.822-0.824) | 0.691 / 0.858 / 0.856 (self 0.716 / 0.703 / 0.713) |
| c=1, L=8 / 64 / 512 | 0.858 / 0.870 / 0.865 (self 0.816-0.824) | 0.817 / 0.860 / 0.856 (self 0.695 / 0.707 / 0.719) |
| fully contiguous in VA order, granule level | 0.865 / 0.825 | 0.855 / 0.720 |

30B 32k: xor_c0 stays at 0.769-0.794 Open (identity 14.50) and 0.656-0.739 ClosedCAP over all 17 placements (worst c=1 L=8: 0.794 / 0.739; self 0.769-0.787 / 0.653-0.682).

Findings. (1) The gain does **not** need fragmentation at 2 MiB granularity: with fully contiguous granules the self-searched XOR still gives 0.825 (Open) / 0.720 (ClosedCAP) on 7B and 0.779 / 0.681 on 30B, versus 0.815 / 0.687 and 0.775 / 0.662 for random frames. What the gain needs is **2 MiB pages preserving the VA in-granule offsets** (K/V head strides sharing banks); the earlier 'packed' placement of the study is a 4 KiB-page compaction (untouched pages dropped, offsets re-packed), which is a different thing and is the one with no gain. The 'aged allocator' wording in the task is therefore wrong: frame randomisation is irrelevant for identity, and for the XOR only the matrix's fit to the frame bits matters. (2) A matrix searched on the random placement (c0) and applied unchanged **loses part of the gain on 7B when long contiguous runs or full contiguity occur**: Open 0.865-0.874 instead of 0.815-0.825 (5 points), ClosedCAP 0.856-0.860 instead of 0.69-0.72 (15 points), i.e. most of the ClosedCAP gain is gone for c >= 0.75 with L=512, any c=1, and c=0.5 L=64; the self-searched matrix keeps 0.82 / 0.69-0.72 everywhere. (3) The XOR **never hurts** in this model: every ratio, of either matrix, is <= 0.874 (7B) / 0.794 (30B), none above 1. (4) 30B is insensitive (the c0 matrix is within 2 points of self except a few cases), because its row-bit structure differs. Conclusion: the XOR gain is placement-independent for 2 MiB granules, but the matrix must be designed against the frame-number bit statistics; a matrix tuned on randomly placed frames is fragile for 7B when the allocator hands out contiguous runs. Caveat: granule-level model only; the 4 KiB-compaction case is not a point on this axis; identical identity values across (c, L) are by construction, not a bug (verified: baseline cost 1065605 for every key).

**Task 7: Qwen3-30B-A3B, 4 decode tokens (job 121993 stream, 121.4M reads, replay 122723).** Experts touched (gate tensor, mean over 48 layers; stream cut in 4 equal read-count parts, boundaries approximate): 8.1 / 8.1 / 8.2 / 8.0 per token (top-8), cumulative 8.1 -> 11.0 -> 13.2 -> 15.8 distinct experts per layer over 4 tokens, consecutive-token Jaccard 0.51 / 0.54 / 0.42 (compare: B=4 batch 23 experts, B=16 57, prefill of 30 tokens 49 of 128). Replay (16M window, 8M-trace Mcycles ratios vs identity, from moe_remap.py; region from 18_region.sh 12M window in brackets):

| placement | DRAM | policy | identity Mcyc (hit) | xor_global | region | expert_hash | expert_lin | region_expert |
|---|---|---|---|---|---|---|---|---|
| THP | DDR4 | Open | 78.92 (0.977) | 0.995 [0.996] | 1.000 | 0.996 | 0.996 | 0.995 |
| THP | DDR4 | ClosedCAP | 86.78 (0.780) | 0.993 [0.993] | 0.999 | 0.992 | 0.994 | 0.992 |
| THP | DDR5 | Open | 8.97 (0.966) | 1.000 [0.998] | 1.000 | 1.001 | 1.000 | 1.000 |
| THP | DDR5 | ClosedCAP | 10.31 (0.774) | 1.002 [0.998] | 0.999 | 1.005 | 1.002 | 1.005 |
| packed | DDR4 | Open | 78.51 | 0.998 | 1.000 | 1.001 | 1.002 | 1.001 |
| packed | DDR5 | Open | 8.94 | 1.003 | 1.000 | 1.004 | 1.003 | 1.004 |

Same as the B=1 sample (Q30B_short: all variants 0.99-1.01): expert-aware mappings, region and the global XOR are all within 1% of identity (the best cell is 0.992), because experts are read as long contiguous slices at 0.97 row hit already. Two routing samples (1 and 4 tokens) agree: no case for expert-aware mapping; the bank XOR only matters through the KV stream at long context. (bitrev, for reference, is 1.08-2.23 worse.)

**Proposal answer.** One fixed bank-XOR matrix can be the hardware mapping for DDR5 x16 with 2 MiB pages: searched once on 7B 32k (or even the naive row-bit-j XOR in Open) it is within about 1 point (Open) and 3 points (ClosedCAP) of per-workload search across 0.5B/7B/30B and 1k-32k contexts, never worse than identity with 2 MiB pages, and neutral where there is no gain. Caveats: ClosedCAP and DDR4 x1 lose 2-4 points to self-search and the naive matrix is inadequate there; a matrix tuned on random frames is fragile (up to 15 ClosedCAP points on 7B) under contiguous physical runs, so the matrix should be searched on a mix of placements; with 4 KiB pages the XOR is a 1-4% loss on DDR4; open-loop saturated replays (the closed-loop analysis of the policy-owner says the real token-time gain is smaller: about 1-2% at the native bandwidth plateau).

### serving-owner: batching x long context (2026-11-12), part 1: 7B B=4 P=8192

Tool changes: `18_drbatch.sh`/`19_batchmiss.sh` take `P`, `STRIDE` (corpus offset between sequences, chars; part of the prompt-cache name), `MAXTRACE_GB` (per-thread `-max_trace_size`), `TOL_TOK` (continue when <= n tokens differ). `drbatch_main.cpp` got a prefill-window mode (`AMCAS_PREFILL=1 AMCAS_UBATCH AMCAS_TRACE_COMPUTE[S]`), driven by new `26_prefill.sh`; old binary kept as `tracing/drbatch-build/llama-drbatch.prev`. cmake lives in the gem5 env (`$E/bin/cmake`; `/usr/bin/cmake` is missing on compute nodes).

Token check (job 122732 failed on it): one of 16 tokens differs in traced vs untraced (step 1, seq 2: 698 vs 43488). Untraced fresh and untraced restored runs are identical, and two restored runs are identical, so the prompt-cache restore is fine; the flip is a DynamoRIO-induced near-tie at the traced step (reproduced identically in 122823 and 123015). Hence `TOL_TOK`.

Runs (7B Q4_K_M, B=4 sequences x 8192-token prompts, STRIDE=40000, 4 threads, node01): 122823 (full pipeline, 11.4G instructions, 16 GB raw trace), 123015 (miss stream kept), region replays 123118 (24M-request window of 117.7M). DRAM reads (no prefetch) 117.7M = 7.53 GB (with prefetch 63.6M + 54.2M = 117.8M): weights 68.3M (58%), KV 48.2M (41%, i.e. 3.09 GB against 1.9 GB of raw KV: LLC amplification 1.6x), activations 0.7M. Native step: see steps_native.txt in `results/fullscale/drbatch/..._B4_P8192t4`. Offline row hit (full stream, THP shuffled): RoRaBaCoCh 0.862, XOR-bank 0.945. Full-stream Ramulator (122823): DDR4 RoBaRaCoCh Open hit 0.956, 534 Mcycles; DDR5x16 Open 108.6 Mcycles, 167 GB/s.

Region replay (123118), THP placement, Mcycles of identity (hit) and ratios to identity:

| DRAM, policy | identity | xor_global | region | bitrev |
|---|---|---|---|---|
| DDR4x1 Open | 117.82 (0.884) | 0.989 (0.916) | 0.921 (0.952) | 1.099 |
| DDR4x1 ClosedCAP | 128.60 (0.753) | 0.984 (0.756) | 0.912 (0.764) | 1.181 |
| DDR5x16 Open | 19.92 (0.961) | 0.963 (0.960) | 0.963 (0.960) | 1.079 |
| DDR5x16 ClosedCAP | 22.52 (0.769) | 1.011 (0.769) | 1.035 (0.769) | 1.330 |

Packed: DDR4 xor_global 1.003/1.005, region 1.030/1.025 (Open/Closed, no gain); DDR5 packed identity is worse than THP (21.87 / 29.31) and both XOR and region give 0.86 / 0.77 there (fixing channel/bank hot-spotting of the packed layout, not row conflicts of the KV pages).
In-order hit by class (DDR4 THP, identity -> xor_global -> region): W 0.957 -> 0.958 -> 0.957, K 0.732 -> 0.816 -> 0.938, O 0.748 -> 0.890 -> 0.964. So at B=4 the four sequences' KV streams need distinct matrices per class (the K-specific matrix recovers 0.94) while the single global matrix, searched on weights+KV together, only gets 0.82 for K: the first case where region beats the global XOR (DDR4 Open 0.921 vs 0.989). On DDR5 the identity KV hit is already 0.965 and nothing helps (0.963).
Verdict so far: at B=4, P=8192 the global XOR gain is 1-4% (no gain on DDR5 ClosedCAP); the region-specific matrix gives 8-9% on DDR4. Window caveat: the first 24M requests of the stream (first layers). P=32768 (job 123129) and windowed prefill (123130) pending.

### serving-owner: batching x long context, windowed prefill (2026-11-13), part 2 and verdict

**Token gate changed (coordinator request).** `TOL_TOK` (count tolerance) is removed from `18_drbatch.sh` / `19_batchmiss.sh`. New positional gate: per sequence, every token of a step before the traced step (`SKIP`) must be identical; the first divergence step per sequence is logged (`token_check.txt`) and a divergence before the traced step fails the run. First-divergence steps (traced step = 1): 7B B=4 P=8192 (122823/123015): seq 2 at step 1, others identical; P=32768 (123129 first try, 123253): seq 0 and 1 at step 2, seq 2 and 3 identical. (Earlier message to main said "0/1/2 at 2/2/1"; correct is 0/1 at step 2.) Untraced fresh, untraced restored and restored-vs-restored runs are identical, so the prompt-cache restore is not the cause; the late flips are DynamoRIO near-ties. `26_prefill.sh` keeps strict equality (single sequence).

**Task 6, 7B Q4_K_M, B=4, P=32768 (STRIDE=70000, so the four 32k prompts overlap in corpus text but sit at different offsets), 4 threads, node01.** Jobs: 123129 (died on the old count gate; untraced prompt evaluation 9557 s), 123253 (complete, miss stream kept, 28.1G instructions, raw trace 54 GB, native step 4.31 s), region replay 123538 (123536 OOM at 40 GB; 40M-request window of 893M). LL misses (no prefetch) 893M = 57.2 GB per decode step, against 4.4 GB weights + 7.5 GB KV: about 5x amplification of the KV reads (every KV page is re-read by several query heads/threads and the 22 MB LLC holds none of it). The windowed replay therefore sees the first 4.5% of the step.

Region replay, THP, cycles relative to identity (hit):

| DRAM, policy | identity Mcycles (hit) | xor_global | region |
|---|---|---|---|
| DDR4x1 Open | 203.4 (0.707) | 0.885 (0.934) | 0.872 (0.925) |
| DDR4x1 ClosedCAP | 204.2 (0.685) | 0.897 (0.744) | 0.888 (0.737) |
| DDR5x16 Open | 27.14 (0.956) | 0.938 (0.959) | 1.000 |
| DDR5x16 ClosedCAP | 28.90 (0.772) | 0.979 | 1.000 |

Packed: DDR4 xor_global / region 1.001-1.003 / 1.004-1.010 (no gain, identity hit 0.949); DDR5 packed identity is bad (hit 0.829 Open, K class 0.594: the contiguous 32k KV streams alias onto few banks/channels) and both XOR and region give 0.651-0.675 Open/Closed.
In-order row hit by class (DDR4 THP, identity -> xor_global -> region): W 0.942 -> 0.942 -> 0.942, K 0.736 -> 0.935 -> 0.909, O 0.838 -> 0.854 -> 0.966. DDR5 THP: identity already W 0.955 / K 0.952 / O 0.839; xor_global leaves K 0.957, O 0.845; region fixes O (0.983, only 40 granules) but gives no cycle gain.

**Task 8, 7B prefill window, 512-token prompt, n_ubatch=32 (16 graph computes), only the last (positions 480-511) traced, 4 threads (job 123130, run `..._prefill_P512u32c15n1t4`; ub=64 was not run: estimated 55G instructions / ~80 GB raw, versus 27.0G / 23 GB raw / 17 GB trace for ub=32).** Same tokens as the untraced run: yes. basic_counts shows 4 threads, 6.7G instructions each (the cache simulator's merged miss file carries one thread id, as in all earlier runs). 75.3M DRAM reads (no prefetch) = 4.8 GB: about one pass over the weights per micro-batch plus activations. Replay 123352 (24M-request window of 75M), cycles relative to identity (hit):

| DRAM, policy | identity Mcycles (hit) | xor_global | region |
|---|---|---|---|
| DDR4x1 Open (THP) | 106.5 (0.898) | 0.992 | 0.994 |
| DDR4x1 ClosedCAP | 117.0 (0.734) | 0.996 | 0.995 |
| DDR5x16 Open | 14.26 (0.908) | 1.004 | 1.003 |
| DDR5x16 ClosedCAP | 17.63 (0.740) | 1.000 | 1.002 |

Packed is the same within 1%. In-order hit by class (DDR4 THP): W 0.937, K 0.753 (459k reads only), O 0.755. Prefill is mapping-insensitive: the stream is weight reads (hit 0.94) with few KV reads, so there is nothing for the KV-page XOR to fix; row-hit about 0.90 Open (DDR4/DDR5), 0.73-0.74 under ClosedCAP, and Open vs ClosedCAP is the only lever (10-20%).

**Verdict.** Batched decode with long per-sequence context (7B, B=4), xor_global / identity cycles, THP, DDR4 Open (DDR5 Open) :
| prompt per sequence | B=1 (32k, held-out) | B=4 | note |
|---|---|---|---|
| 512 | - | 0.914 (0.954) offline-ratio table | weights dominate |
| 8192 | - | 0.989 (0.963) | region 0.921 (0.963) |
| 32768 | 0.860 (0.844) | 0.885 (0.938) | region 0.872 (1.000) |
So the KV-page XOR gain does come back with long contexts under batching, but only at 32k: 11-12% on DDR4 (single sequence 14-16%) and 2-6% on DDR5; at 8k per sequence it is 1-4% (the per-sequence KV is 1.9 GB against 4.4 GB of weights, and the 8k window was weight-heavy). Region-specific matrices are not generally better: they win on DDR4 at 8k (8% vs 1%, the global matrix compromises between weight and KV streams) but only by 1% at 32k, and lose to xor_global on DDR5 THP at 32k (1.000 vs 0.938). Single global XOR remains the proposal. Packed placement shows no DDR4 effect but a large DDR5 effect at 32k (35%), which is channel/bank aliasing of contiguous KV, also fixed by the XOR. Caveats: windows are the first 24M/40M requests of 117M/893M-request steps (early layers), open-loop replay, modelled THP placement, one traced step. Jobs: 122732 (failed gate), 122823, 123015, 123118 (P=8192); 123129, 123253, 123536 (OOM), 123538 (P=32768); 123130, 123352 (prefill). Not done: 30B B=4 P=8192 (time), prefill at ub=64.

### main session: check of serving-owner results (2026-11-13)

Re-derived from `cluster-data/results/fullscale/region/7B_B4_P32768/` (40M-request window): THP DDR4 Open identity 203.4 -> xor_global 179.9 (0.884) -> region 177.3 (0.872); THP DDR5 Open 27.1 -> 25.4 (0.937) / region 27.1 (1.000). Matches the serving-owner verdict: the KV XOR gain returns at batch 4 only with 32k per sequence, and region-aware gives no general gain over the global XOR. Two qualifications: (1) the "packed DDR5 -35%" (35.9 -> 23.3 Mcycles, identity hit 0.829) is an artefact of the VA-compacted 4 KiB placement model, which with 4 sequences creates bank aliasing that the XOR then removes; that placement is not a real allocator (see transfer-owner fragmentation section), so this gain should not be claimed. (2) The replay windows cover the first 40M of ~890M requests of the step (~4.5%, from the step start), unlike the earlier runs' windows at 30% of the step; the stream is periodic over layers, but state it. All 8 follow-up experiments (E20-E27 on the tracker) are done; no jobs are queued.

### correction to the 2026-11-13 serving-owner check

The batch-4 32k replay window was not the first 40M requests of the step: `logs/region_123538.log` shows 893,051,885 reads and window [267,915,565, 307,915,565), i.e. 40M requests starting at 30% of the step, the same placement as the earlier region runs. Qualification (2) of the previous section is withdrawn. Class shares: full step W 0.077 / K 0.921 / O 0.003, window 0.051 / 0.946 / 0.002.

### gpu-owner: GPU trace route, feasibility and first traces (2026-11-13)

**Result in one paragraph.** NVBit works on both GPUs; a streamed pipeline (NVBit -> FIFO -> sector coalesce + L1/L2 filter -> gzip DRAM stream, no raw trace) produces per-token DRAM request streams that match ncu (V100 within 1.4% at short context, L2 read misses on A16 within 0.4%), and replays through Ramulator HBM2 (V100) and GDDR6 (A16). On the GPU the bank XOR hash gives about nothing on the V100 (0.99-1.01x bandwidth) and +8..28% on the A16 (FlashAttention on); the large lever is the CHANNEL mapping: at the baseline 256 B channel interleave, the V100 at 32k context reaches only 128 of 819 GB/s, a channel XOR hash gives 1.46x and a 2 KiB interleave 2.3x. Row-hit loss is concurrency (many SMs on different rows), not a layout a static bank hash fixes.

**Hardware/software (checked).** node04: 2 x V100-PCIE-16GB (sm_70, driver 580.65.06, 80 SM, 6 MiB L2, 4096-bit HBM2 877 MHz); node05: 8 x A16 (sm_86, driver 610.43.02, per GPU 10 SM, 2 MiB L2, 128-bit GDDR6 6.251 GHz = 200 GB/s, ECC on). nvcc 12.6 exists on node04 only (/usr/local/cuda-12.6; node05 has nvcc 13.x which dropped Volta); binaries built on node04 (sm_70+sm_86) run on both. Login node has no CUDA; cmake 3.20 is too old for NVBit's CMake, so use the gem5 env's cmake 4.4.5. node05 /scratch is full: A16 runs use the streamed path and write only gzip streams under $HOME.
- NVBit v1.8.1 (sha256 9c383591...618, release states driver <= 595) builds and runs instr_count on both (A16 driver 610 works despite the stated limit), but its mem_trace fails on V100 with CUDA_ERROR_INVALID_SOURCE. NVBit v1.7.7.3 works on both. Our tool is a patch of its mem_trace (`trace-pipeline/gpu/gputrace_patch.py`; NVBit sources are not redistributed): launch-id window, true active-lane mask, global/generic only (local and shared skipped), binary records, self-describing op field, `USE_ASYNC_STREAM` (without it records were silently lost: a first run with ~40% of load sectors missing, caught by comparing per-kernel against ncu).
- llama.cpp `build-cuda` (CUDA 12.6, sm 70;86, Release): 0.5B V100 339 tok/s, A16 165; 7B Q4_K_M V100 120, A16 26. First tokens match the CPU build, then diverge on fp rounding. Graphs are disabled for tracing (`GGML_CUDA_DISABLE_GRAPHS=1`); FlashAttention is `-fa auto` (on) or `-fa off`.
- ncu: counters readable as a normal user (RmProfilingAdminOnly=0). V100 needs `/opt/nvidia/nsight-compute/2024.3.2/ncu` (the 2025.3 one refuses Volta); A16 uses `/usr/local/cuda-12.6/bin/ncu`. Used `--cache-control none` so caches persist between kernels like the real run.

**Pipeline** (all code in `trace-pipeline/gpu/`, results in `results/gpu/` and cluster-data): `30..35` build/test NVBit and the tool; `36_gputrace_run.sh` runs llama with the tool (window = kernel launch ids; `find_window.py` takes the last period of the launch-name sequence = one decode step, 538 kernels for 7B FA on); `42_stream.sh` streams to `gpu_filter.c` (L1 per SM 128 KB 4-way, CTA->SM = linear CTA id mod nSM; L2 6 MiB / 2 MiB 16-way, 128 B lines with 32 B sectors, write-back, write-validate; V100 `--sm 80 --l2mb 6`, A16 `--sm 10 --l2mb 2`; cold at window start; dirty lines not flushed) -> `.dram.gz` (u64 per 32 B request: sector<<1|write) ; `gpu_attr.cpp` (LD_PRELOAD in the same process as NVBit; addresses differ between runs) + `gpu_attrib.py` for tensor classes; `gpu_replay.py` (Ramulator2 HBM2_2Gb x32 channels 1.6 Gb/s, or GDDR6 x16 x8 channels 14 Gb/s, saturated LoadStoreTrace, 2 MiB pages packed in first-touch order, 256 B channel interleave unless `--ib`, schemes identity/xor (searched on the HBM2 or GDDR6 layout via remap.py search, bank region = pc+bg+bank bits)/chan/xor_chan (memsys_study.chan_search), Open and ClosedCAP; first 80 M requests); `gpu_diag.py` (in-order row hit per tensor class, per-SM sub-stream); `43_matrix.sh`/`44_chain.sh` run a whole cell. Validation: the streamed DRAM stream is byte-identical to the stored-trace path (0.5B V100, 87.7 MB). Jobs: 124199 (V100 chain, 4h02), 124179 (A16 chain, 3h28), earlier 124073/124102/124109/124110/124150/124105.

**Model vs ncu** (DRAM read GB per decode token, 7B Q4_K_M): V100 short 4.39 vs 4.39 (arithmetic ~4.4); 8k 4.89 vs 4.81; 8k FA off 5.20 vs 4.80 (+8%); 32k 9.75 vs 7.40 (+32%, model re-reads KV more than the real 6 MiB L2 does); 32k FA off 14.8 vs 13.1 (+13%). A16: ncu is 16-20% above the model at short context (5.24 vs 4.39): L2 read misses match the model (10.99 M vs 10.9 M sectors) but DRAM sectors read are 16% higher; 64 B fetch was tested in the filter (--fetch 64) and changes nothing, so the extra is not from L2 misses; most likely GDDR6 inline ECC traffic (ECC enabled on the A16, not on HBM2); not modelled, treat A16 absolute bytes as x1.16.

**Table** (7B, one decode step, 2 MiB pages, 256 B channel interleave, saturated replay, Open page, first 80 M requests; BW ratio = bandwidth vs identity mapping). 'peak' is the Ramulator system peak (V100 819, A16 224 GB/s).
| GPU | ctx | FA | DRAM GB (model/ncu) | KV share | Open row hit | GB/s | ClosedCAP hit / GB/s | xor | chan | xor+chan |
|---|---|---|---|---|---|---|---|---|---|---|
| V100 | short | on | 4.39/4.39 | 0% | 0.671 | 397 | 0.600/371 | 0.99 | 1.00 | 0.99 |
| V100 | short | off | 4.39/4.39 | 0% | 0.671 | 396 | 0.600/372 | 0.99 | 0.99 | 0.99 |
| V100 | 8k | on | 4.89/4.81 | 10% | 0.692 | 249 | 0.617/209 | 1.00 | 1.17 | 1.11 |
| V100 | 8k | off | 5.20/4.80 | 15% | 0.687 | 220 | 0.623/224 | 1.00 | 1.46 | 1.45 |
| V100 | 32k | on | 9.75/7.40 | 55% | 0.773 | 128 | 0.688/108 | 1.00 | 1.46 | 1.45 |
| V100 | 32k | off | 14.8/13.1 | 69% | 0.725 | 115 | 0.699/128 | 1.01 | 2.11 | 2.19 |
| A16 | short | on | 4.39/5.24 | 0% | 0.907 | 127 | 0.751/79 | 1.28 | 1.00 | 1.28 |
| A16 | short | off | 4.39/5.24 | 0% | 0.905 | 127 | 0.749/80 | 1.28 | 1.00 | 1.28 |
| A16 | 8k | on | 4.79/5.68 | 9% | 0.912 | 124 | 0.752/73 | 1.23 | 1.00 | 1.23 |
| A16 | 8k | off | 6.00/5.93 | 27% | 0.861 | 64 | 0.743/55 | 1.08 | 1.72 | 1.91 |
| A16 | 32k | on | 6.17/7.20 | 29% | 0.931 | 116 | 0.761/59 | 1.14 | 0.99 | 1.12 |
| A16 | 32k | off | 17.1/20.5 | 73% | 0.783 | 54 | 0.738/56 | 1.01 | 1.93 | 2.08 |
(The A16 32k FA-on KV share is 29%, not 55%: the 2 MiB L2 and FA tile kernel re-read less; see attrib JSON.) 0.5B short checks: V100 349.4 MB model vs 354 MB ncu; A16 350 MB vs 400 MB (ECC).

**Diagnosis of the low row hit (cheap, from the diag JSONs).** (1) Per class, in-order row hit on V100 with the merged stream: weights (ffn) 0.76 at every context, KV K/V 0.94/0.79 (8k FA on) and 0.75/0.78 (FA off); the same weight streams considered per SM are 0.87: the loss is concurrency. The V100 sees ~70 distinct SMs in every 1000 consecutive DRAM requests (A16: ~55 of 10 SM-groups... it has only 10 SMs, per-SM hit 0.91) and the merged in-order hit falls from per-SM 0.87-0.93 to 0.76-0.83. (2) The XOR was searched on the HBM2 layout (col 5 bits, bank region 5 = pseudo-channel+bank-group+bank, row 14 bits; sources = intra-channel row bits; fan-in <= 3), e.g. V100 32k FA on: bank bit0 ^= row0, bit2 ^= row1, bit3 ^= row1; matrices in `*_diag.json` and `*_sweep_*.json`. Its in-order gain is <0.1 point, bandwidth 1.00x. (3) The bottleneck at 32k is channel imbalance and bank-group conflicts on the 256 B channel interleave: channel max/mean load 1.29 (FA on, 32k), and a 2 KiB interleave (`--ib 5`) lifts the V100 32k FA-on stream from 128 to 297 GB/s (row hit 0.77 -> 0.80), 4 KiB 277 GB/s; with FA off 115 -> 128 (2 KiB) with the channel hash still adding +1.7x there. On the A16 (8 channels) a coarser interleave does not help FA-on (116/115/101 GB/s) but the bank XOR does (1.14-1.28x). (4) ClosedCAP is worse than Open on every FA-on cell (hit 0.60-0.69 vs 0.67-0.77 V100; bandwidth -7% to -50%); with FA off it is equal or slightly better at 8k/32k on the V100.

**Caveats.** Saturated, open-loop replay of the first 80 M requests (the 32k streams are ~290 M, so a window of the step); CTA->SM mapping approximated; Ramulator has no 1.75 Gb/s HBM2 or 12.5 Gb/s GDDR6 bin (819 vs 900 GB/s, 224 vs 200 GB/s peaks); channel interleave and page mapping (2 MiB, first-touch packing) are assumptions, not measured GPU address maps; 4 KB pages, local/shared memory traffic, the CUDA memcpy of the host-mapped 0.5B embedding and A16 ECC traffic are not modelled. KV share/bytes differ from ncu by up to +32% at V100 32k FA on. Do not extrapolate the XOR/chan numbers beyond these streams.

### dramtech-owner: LPDDR5/6, HBM3/4, GDDR6 (2026-11-13)

**Question.** Does the KV-page bank-XOR gain (DDR5 x16, 7B 32k, THP: -16% Open / -29% ClosedCAP; DDR4-2933 6ch: -10%) hold on other DRAM technologies? Framing: LPDDR5/5X is the memory of NVIDIA Grace, Apple M-series and edge CPUs; HBM3/4 that of Xeon Max-style CPU+HBM; GDDR6 is GPU memory and the CPU stream on it is a what-if. LPDDR6 is the next LPDDR generation.

**Layouts (verified).** Derived from `ramulator2-amcas` (same mapper code as the DDR5 check: `addr_mapper_base.cpp`, `ro_ba_ra_co_ch.cpp`, `cache_line_interleave.cpp`; level order after Channel: [Rank | PseudoChannel, Sid] BankGroup, Bank, Row, Column; tx bytes = prefetch x channel_width/8 or `data_payload_bytes`). All new standards have **32 B transactions**, so a 64 B LLC miss is replayed as two sector requests (addr, addr+32; same channel and row). `tools/dramtech_probe.py` classifies every address bit with synthetic replays (channel from per-channel counts, row from conflicts, col vs bank from a row-bit-paired flip): all 8 systems MATCH. Physical bits (256 B channel interleave, RoBaRaCoCh):

| system | config, peak | off/col | bank region (phys) | row (phys) | channel (phys) |
|---|---|---|---|---|---|
| lpddr5x32 | LPDDR5-6400, 32 x16 ch, 410 GB/s (Grace-like; no 8533 preset) | 5 / 6 (row 2 KiB) | [16:19] 4 bits (BG2+bank2) | [20:34] | [8:12] |
| lpddr5x8 | 8 x16 ch, 2 ranks, 102 GB/s (laptop) | same | [14:18] 5 bits | [19:33] | [8:10] |
| lpddr6x32 | LPDDR6-10667 BL24, 32 x12 subch, 455 GB/s payload | 5 / 6 (2 KiB) | [16:19] | [20:34] | [8:12] |
| lpddr6x8 | 8 subch, 2 ranks, 114 GB/s | same | [14:18] | [19:33] | [8:10] |
| hbm3x16 | one stack, 16 ch x 2 PC, 6.4 Gb/s, 819 GB/s, 32 GiB | 5 / 5 (PC row 1 KiB) | [14:19] 6 bits (PC1+sid1+BG2+bank2) | [20:33] | [8:11] |
| hbm4x32 | one stack, 32 ch x 2 PC, 8 Gb/s, 2.05 TB/s | 5 / 5 (1 KiB) | [15:20] 6 bits (PC,sid,BG1,bank3) | [21:33] | [8:12] |
| gddr6x16 | 14 Gb/s (only preset), 16 x16 ch = 8 x 32-bit, 448 GB/s | 5 / 7 (row 4 KiB in the model) | [16:19] 4 bits | [20:33] | [8:11] |

Caveats on configuration: GDDR6 32 Gb preset resolves no refresh timings, set to the 8 Gb values (210/105 ticks) by override; HBM controllers use the test-suite queue sizes (HBM3 128/64, HBM4 64/64), others the default 32/32; controllers are the standards' own (LPDDR5, LPDDR6, HBM34, GenericDDR), FRFCFS, AllBank refresh. The XOR-search cost model bus/tRC terms are taken from each standard's timing (`dramtech_cfg.cost_params`).

**Method.** `tools/dramtech_study.py` (+ `dramtech_cfg.py`, `dramtech_probe.py`, `dramtech_report.py`, `scripts/29_dramtech.sh`; `remap.py` got a backward-compatible `off` field). Same placements (packed/THP/4k, 32 GB pool), window 8M requests from 30% of the step, schemes identity / xor_global (region_remap.search) / chan / xor_global_chan, all asserted injective with row/col unchanged. Replay saturated (LoadStoreTrace, issue 2x peak request rate) and closed loop with SimpleO3 4 cores (first 2M lines, 32 B LLC lines, 32 MSHRs): `fast` = policy-owner k=1 window 256 bubbles 52 (at 2.4 GHz-equivalent; instructions per line and window scaled by core clock / 2.4 GHz so wall-time compute per request is the same on every technology), `nat4` = k=4.5 window 1152 (native 4-thread-matched). I did not add `--dram` options to `ramulator_replay.py` (the sector expansion and the per-standard controllers live in `dramtech_study.py` instead; ramulator_replay.py is unchanged). Jobs: 124132 and 124133 (7B 32k), 124215 and 124219 (30B 32k), 124230 (7B 8k), 124245 (30B short); smoke 124131. 30B/8k/short runs use sat+fast and xor_global/chan only. Results `results/fullscale/dramtech/<tag>/<system>.json`; full per-condition tables via `tools/dramtech_report.py`.

**Results.** Cycles ratio vs identity (<1 = gain), Open unless stated, "xor" = bank-XOR hash (`xor_global`), "chan" = channel hash.

| stream | system | THP xor sat | sat ClosedCAP | fast | nat4 | packed xor sat | 4k chan sat | THP hit id->xor |
|---|---|---|---|---|---|---|---|---|
| 7B 32k | lpddr5x8 | 0.76 | 0.74 | 0.90 | 0.99 | 1.08 | 0.64 | 0.89->0.96 |
| 7B 32k | lpddr5x32 | 0.64 | 0.60 | 0.93 | 1.00 | 0.95 | 0.61 | 0.89->0.95 |
| 7B 32k | lpddr6x8 | 0.74 | 0.60 | 0.81 | 0.99 | 1.05 | 0.64 | 0.90->0.96 |
| 7B 32k | lpddr6x32 | 0.58 | 0.59 | 0.88 | 1.00 | 0.97 | 0.62 | 0.88->0.95 |
| 7B 32k | gddr6x16 | 0.45 | 0.59 | 0.93 | 1.00 | 1.03 | 0.65 | 0.86->0.96 |
| 7B 32k | hbm3x16 | 0.74 | 0.64 | 0.79 | 1.00 | 1.03 | 0.55 | 0.92->0.94 |
| 7B 32k | hbm4x32 | 0.69 | 0.61 | 0.81 | 1.00 | 1.00 | 0.58 | 0.91->0.94 |
| 30B 32k | lpddr5x8 | 0.74 | 0.72 | 0.87 | - | 0.99 | 0.53 | 0.85->0.96 |
| 30B 32k | lpddr5x32 | 0.62 | 0.59 | 0.91 | - | 0.99 | 0.49 | 0.84->0.95 |
| 30B 32k | lpddr6x8 | 0.72 | 0.57 | 0.80 | - | 0.99 | 0.54 | 0.87->0.96 |
| 30B 32k | lpddr6x32 | 0.55 | 0.57 | 0.88 | - | 0.99 | 0.50 | 0.84->0.95 |
| 30B 32k | gddr6x16 | 0.44 | 0.59 | 0.91 | - | 0.97 | 0.52 | 0.80->0.96 |
| 30B 32k | hbm3x16 | 0.71 | 0.61 | 0.78 | - | 1.02 | 0.44 | 0.90->0.93 |
| 30B 32k | hbm4x32 | 0.63 | 0.56 | 0.77 | - | 0.98 | 0.45 | 0.89->0.93 |
| 7B 8k | lpddr5x32 | 0.91 | 0.86 | 0.98 | - | 1.00 | 0.91 | 0.96->0.97 |
| 7B 8k | lpddr6x8 | 0.91 | 0.82 | 0.94 | - | 1.01 | 0.92 | 0.97->0.98 |
| 7B 8k | gddr6x16 | 0.69 | 0.81 | 0.98 | - | 0.97 | 0.96 | 0.95->0.97 |
| 7B 8k | hbm3x16 | 0.94 | 0.89 | 0.94 | - | 1.00 | 0.88 | 0.96->0.96 |
| 30B short | all four | 1.00 | 0.99-1.01 | 0.97-1.01 | - | - | 1.00-1.06 | unchanged |

Closed-loop read latency (7B 32k, THP, identity -> xor): fast 0.74-0.76x on HBM3/4 and LPDDR6x8, 0.87-0.92x LPDDR5/GDDR6/LPDDR6x32; nat4 0.81-0.95x with cycles flat. Closed-loop demand of the fast point is 12-22 GB/s (a few percent of any of these peaks). DDR5 x16 reference (earlier jobs, policy-owner): sat -16% (12M window), fast -16%, nat4 -0.2%.

**Verdict.**
1. **The mechanism holds on every geometry and is larger when saturated**: THP + KV-heavy context only; packed placement and 4 KiB placement get nothing from the bank XOR (0.95-1.08; the small >1 values are search-cost-model artefacts), the 30B short control and the 8k stream get 0-9% (GDDR6 31% at 8k saturated). Saturated gain on 32k: -24..-26% (x8 LPDDR) / -36..-45% (x32 LPDDR) / -26..-31% (HBM) / -55% (GDDR6), versus -16% on the DDR5 x16 baseline and -10% DDR4. It grows because (a) rows are small (1-2 KiB vs DDR5's 8 KiB per bank-row seen at 64 B lines: the KV streams of different heads inside one 2 MiB page collide in a few rows, identity row hit 0.80-0.92 vs 0.95 packed), so every KV conflict costs a full tRC and the hash restoring 0.93-0.96 hit pays more; (b) many narrow channels leave few banks per channel to absorb conflicts (GDDR6: 16 banks per 16-bit channel, biggest gain; HBM with 64 banks per channel, smallest, 26-31%). Within a technology more channels (x32 vs x8) help the XOR more (LPDDR5 0.76 -> 0.64), because the interleaved channel stream spreads each head's page across more channels.
2. **ClosedCAP gain is as large or larger** (-26..-41%), as on DDR5 (-29%).
3. **Closed loop, the gain depends on demand, as before.** At the fast-core operating point -7..-10% (LPDDR5, GDDR6), -12..-20% (LPDDR6, 30B), **-19..-23% on HBM3/4** (HBM3 0.79 / HBM4 0.81 for 7B, 0.78 / 0.77 for 30B), larger than DDR5's -16%, with 13-26% lower read latency; at the native 4-thread operating point (about 9 GB/s) cycles do not move (0.99-1.00) while latency still drops 5-19% on THP. So the 4-thread native machine would see latency, not time, gains on any technology; time gains need cores that sustain 15-25 GB/s (the Grace/Xeon-Max-class many-core parts do, with 100+ GB/s demand headroom, but we only modelled 4 cores).
4. **Channel hash**: unlike closed loop, the saturated channel-hash gains (-35..-55% for the 4 KiB placement) are again largest; in closed loop (fast) chan is 0.88-0.99 (HBM3/4 0.88) and the combination adds at most 1-2% over xor, consistent with the DDR5 finding that this is a saturated-replay artefact (policy-owner part 1-2).
Caveats: 4 cores only (demand far below HBM/Grace peak); GDDR6 is a what-if; saturated numbers are upper bounds; one window per condition, matrices searched in sample; LPDDR5 uses 6400 not 8533; 30B/8k/short runs have fewer schemes/points; no region scheme and no held-out test here; row size is the model's (GDDR6 4 KiB), address interleave fixed at 256 B (a coarser interleave or a different channel/bank hash in real controllers would change the baseline).

### mop-owner: MOP4CLXOR vs xor_global (2026-11-14)

**What MOP4CLXOR does** (`ramulator2/src/ramulator/controller/addr_mapper/impl/mop4clxor.cpp`; the other built-in mappers are RoBaRaCoCh, ChRaBaRoCo (channel/rank/bank on top, row below: each bank a contiguous region), PassThrough and RIT (a row-swap decorator for RowHammer defences; not a mapping policy)). On the intra-channel address (channel bits already removed by CacheLineInterleave, 256 B), LSB to MSB: [6 line bits] col[1:0] | bank region (rank, bankgroup, bank = 5 bits) | col[hi] | row. So 4 lines (256 B) are consecutive, then the bank region sits just above them, not above the whole column field as in RoBaRaCoCh. Then a fixed hash: each level below Row (rank, bankgroup, bank in turn) is XORed with the next bits of the full column value (effective bank index = raw5 ^ (col & 31), so the 4 lines of a 256 B chunk go to 4 different bank groups and the bank rotates every 4 lines of column), and the row is XORed with col >> 5 (1 bit DDR5, 2 bits DDR4). Row/col stay at the same address positions; only the bank region moves. Spreads a sequential stream over all 32 banks line by line (little row hit, max bank-level parallelism, bank-group alternation).

**Method.** `tools/mop4_traces.py`, `mop4_replay.py` (mapper switch, wraps policy/realmem replay code; nothing existing edited), `mop4_report.py`, `scripts/31_mop4.sh`. Schemes: ident (RoBaRaCoCh), xor_global (searched as before), mop (identity stream through MOP4CLXOR), mopxor = MOP4CLXOR + xor_global (composable: the same greedy search is run on the MOP4CLXOR-decoded (bank, row), and f(row bits) is XORed into the raw bank field of the address, so MOP4CLXOR yields bank_eff ^ f(row); row/col/channel unchanged and injective, asserted), chra (ChRaBaRoCo). DRAM: x16 and x8 DDR5-4800, d4r6 DDR4-2933 6 channels (realmem model), d4 DDR4-2400 1 channel. Windows 8M requests from 30% (saturated) and the first 2M (closed loop: k1 = bubbles 52 window 256 (~22 GB/s demand), w1024 = bubbles 52 window 1024 (~40 GB/s plateau), k4 = bubbles 211 window 1024 (native 4-thread, ~10 GB/s); d4 1-channel only k4). Placements packed/THP/4 KiB (32 GB), Open and ClosedCAP. Jobs: 124413 (7B 32k, `mop4/q7_32k`), 124414 (30B 32k, `q30_32k`), 124444 (7B 8k `q7_8k`, held-out `q7_32k_heldout`; points sat,k1,k4), 124445 (30B short `q30_short`, 7B B=16 `q7_B16`), all ~55-100 min on node01/03. Full per-condition tables (Mcycles, ratio, hit, latency, GB/s, winner, Open and ClosedCAP) in `cluster-data/results/fullscale/mop4/*/replay.json`, rendered by `mop4_report.py`. Ratios below are cycles vs identity/RoBaRaCoCh under the same policy (<1 is better).

**7B 32k, Open, xor_global / mop / mopxor (ident Mcycles in front)**

| point | placement | x16 | x8 | d4r6 (DDR4 6ch) | d4 (1ch) |
|---|---|---|---|---|---|
| sat | packed | 8.7: 1.00/1.01/1.00 | 17.4: 1.00/1.01/1.00 | 8.3: 1.01/**0.90**/0.90 | 37.5: 1.02/0.93/**0.93** |
| sat | THP | 10.7: 0.82/0.83/**0.82** | 21.1: 0.83/0.84/**0.83** | 9.9: 0.89/0.76/**0.75** | 41.7: 0.91/0.84/**0.84** |
| sat | 4 KiB | 8.7: 1.00/1.02/1.02 | 17.4: 1.00/1.00/1.00 | 7.9: 1.00/1.05/**0.98** | 37.1: 1.00/0.94/**0.93** |
| k1 | packed | 14.9: **0.97**/1.21/1.21 | 14.6: **0.99**/1.28/1.29 | 16.4: **0.99**/1.26/1.26 | - |
| k1 | THP | 17.3: **0.83**/1.08/1.04 | 17.7: **0.83**/1.09/1.06 | 17.1: **0.95**/1.21/1.21 | - |
| w1024 | THP | 7.72: **0.99**/1.02/1.01 | 7.87: **0.98**/1.03/1.00 | 8.10: **0.99**/1.06/1.04 | - |
| k4 | THP | 28.4: 1.00/1.00/1.00 | 28.5: 1.00/1.00/1.00 | 28.6: 1.00/1.01/1.01 | 33.0: **0.98**/1.05/1.05 |

30B 32k is the same picture (saturated THP x16 0.77/0.79/0.78, x8 0.81/0.82/0.81, d4r6 0.80/0.71/0.70, d4 0.86/0.82/0.80; k1 THP xor 0.81-0.97 vs mop 1.09-1.25; k4 flat except d4 where xor 0.91 vs mop 1.00). Other streams, Open, saturated THP x16 / d4r6 / d4 (xor_global / mop / mopxor): 7B 8k 0.95/0.94/0.94, 0.98/0.78/0.78, 0.99/0.85/0.85; held-out token 0.84/0.87/0.84, 0.87/0.77/0.76, 0.85/0.80/0.79; 30B short 1.00/0.98/0.98, 1.00/0.77/0.77, 1.00/0.85/0.85; 7B B=16 0.92/0.91/0.91, 0.95/0.82/0.82, 0.96/0.95/0.95. Closed loop k1 on all these: xor_global 0.82-1.01 vs mop 1.12-1.38 (mop never wins), k4 identical (<=1%) except DDR4 1-channel, where xor_global is 0.95-1.00 and mop 1.04-1.10.

**Findings.**
1. **Saturated, DDR5 (x16, x8): equal.** On THP, xor_global, mop and mopxor are within 1-3% of each other (e.g. 7B 32k x16 0.818 / 0.832 / 0.815); packed and 4 KiB nothing moves. MOP4CLXOR reaches the same gain without any search because the bank region is next to the column LSBs and the hash by column bits spreads the streams; row hit drops to ~0.4-0.5 but the DDR5 data bus (16/8 channels) is the bound, so it does not matter.
2. **Saturated, DDR4 (6 channels and 1 channel): MOP4CLXOR/mopxor is better than xor_global**, including for packed placement where xor_global gives nothing: d4r6 packed 0.90 vs 1.01, THP 0.76 vs 0.89 (mopxor 0.75), 4 KiB mopxor 0.98; 1 channel packed 0.93 vs 1.02, THP 0.84 vs 0.91, 4 KiB 0.93 vs 1.00. Mechanism: DDR4 has 4 bank groups with tCCD_L > tCCD_S, so a row-hit-heavy stream on one bank group is bank-group-bound; MOP4CLXOR puts 4 consecutive lines in 4 different groups. Row hit falls to 0.3 and still wins. These DDR4 saturated MOP wins are in the validation-owner's real-node model and the 1-channel model where MOP4CLXOR first showed 0.67-0.91 cycles.
3. **Closed loop (any core demand below the DRAM limit): xor_global wins or ties everywhere; MOP4CLXOR loses.** k1 (22 GB/s): mop +4% to +38% cycles (packed +21-31%, THP +8-25%); w1024: +1-13%; k4: +0-1% on DDR5, +4-10% on DDR4 1 channel. Cause: MOP4CLXOR gives up row hits (hit 0.0-0.2, latency +30-50%, e.g. 7B 32k k1 x16 packed 78 -> 116 cycles) and, with demand well below peak, the lost row locality is paid in latency while bandwidth gain does not matter; xor_global keeps the 0.8-0.96 row hit of RoBaRaCoCh. ClosedCAP shows the same (MOP gets 0.6-0.9 of ClosedCAP-identity cycles saturated only because ClosedCAP+RoBaRaCoCh is a poor combination, not because MOP beats Open+xor_global; MOP's own Open and ClosedCAP are equal).
4. **MOP4CLXOR + xor_global (mopxor)** buys 1-3% over mop in saturated runs (the bank search adds a little on top of MOP's fixed hash, mostly THP and DDR4), and nothing closed loop.
5. ChRaBaRoCo (each bank one contiguous region): +40% to 3x cycles, never best except tie on 4 KiB DDR4 1ch.

**Verdict.** xor_global is **not** universally best. (a) Where the memory is the limit and the DRAM is DDR5 (the target) and placement is THP, xor_global is equal to MOP4CLXOR (within 1-3%) while keeping row locality; with packed or 4 KiB placement neither helps. (b) In every closed-loop regime (the realistic ones: 10-40 GB/s demand) xor_global is better than MOP4CLXOR by 4-38% in cycles and 25-50% in latency, and MOP4CLXOR never beats identity there (it is worse than the default mapping). (c) **MOP4CLXOR wins when the stream is saturated on DDR4** (real-node 6-channel DDR4-2933 and the 1-channel DDR4-2400): 5-25% over xor_global, including packed placement, and mopxor is the best of all there; but the native machine never reaches that regime (its 4-thread decode is at ~10 GB/s, plateau 43 GB/s of 141 GB/s socket peak, where mop is +0-9% worse). So the claim is: xor_global is best for the latency-oriented, not-DRAM-saturated regime on any DRAM and equal-best for saturated DDR5 THP; it cannot be called best for bandwidth-saturated DDR4 workloads, where a bank-group-interleaving mapping (MOP4CLXOR, optionally plus xor_global) is better. A possible stronger proposal for saturated DDR4 is the combination mopxor, or xor_global plus a bank-group-first low-bit placement; not tested here. Caveats: Ramulator only (DRAMsim3 gave smaller DDR4 xor gains in the validation study and has not replayed MOP), the closed-loop core model is the calibrated DDR5 one reused with clock ratios 2 and 3 for DDR4, the xor_global search is on the replayed window (held-out style not repeated for MOP, which is fixed), 8M/2M windows, tests of MOP only through Ramulator's CacheLineInterleave 256 B channel interleave.

### l2fix-owner: GPU L2 model fix and 32k rerun (2026-11-14)

**Verdict.** The V100 32k overstatement was the L2 set index, not capacity, replacement, associativity, L1 or CTA ordering. `gpu_filter.c` indexed sets with `line % 3072`; the real L2 hashes the index. An XOR-folded index `set = (line ^ (line>>13)) % sets` (line = 128 B line number, i.e. XOR of address bit 20+ into the low bits) brings every condition within ~5%: model DRAM read GB vs ncu: short 4.39 vs 4.39, 8k FA on 4.82 vs 4.81, 8k FA off 4.80 vs 4.80, 32k FA on 7.32 vs 7.40 (ncu per-kernel sum 7.56), 32k FA off 12.41 vs 13.14 (-5.5%). Old modulo index: 4.39 / 4.89 / 5.20 / 9.3-9.76 / 14.8. All earlier V100 32k KV-traffic and replay numbers in the gpu-owner section (which used modulo) are superseded by the tables below.

**Task 1, per kernel (V100 7B Q4_K_M, 32k, MB DRAM read; ncu job 124412, `results/gpu/*_ncuperk.raw`, `l2_compare.py`).** L2 read sectors model vs ncu: 17175 vs 17032 MB (+0.8%), so the L1 model and the access stream are right; only the L2 miss count differs.
| kernel | ncu | old modulo | x13 hash |
|---|---|---|---|
| FA on: flash_attn_ext_vec (28) | 3167 (lookup-miss 3402) | 4917 | 2937 |
| FA on: mul_mat_vec_q (169) | 4381 | 4373 | 4373 |
| FA on total | 7563 | 9296 (second run 9755) | 7318 |
| FA off: mul_mat_f (KQ, 28) | 4870 | 6217 | 3762 |
| FA off: mul_mat_vec_f (KQV, 28) | 3796 | 4170 | 4179 |
| FA off: mul_mat_vec_q | 4377 | 4373 | 4373 |
| FA off: soft_max | 95 | 60 | 97 |
| FA off total | 13142 | 14824 | 12415 |
Weights (mul_mat_vec_q) match exactly; the whole error sits in the KV-reading kernels (7 GQA query heads re-read each KV head; the re-reads hit in L2 only if the sets do not alias).

**Task 2, knobs (one NVBit pass, many variants: `gpu_filter2.c`, per-launch counters, raw traces kept on node04 /scratch via `KEEPRAW`; base variant byte-identical to gpu_filter.c on 0.5B).** 32k FA-on total DRAM read GB (ncu 7.4-7.56): modulo 9.3-9.76 (run-to-run trace order varies ~5%); FIFO 9.18; random 9.85; SRRIP 9.34; 8/16/32 ways 9.29/9.30/9.31; L1 128/64/32 KB or L1 off: 9.30 (L1 only changes L2 request count, not misses); round-robin CTA reorder is moot (the NVBit trace is already interleaved, mean same-CTA run length 1.00; our batch reordering destroyed L1 locality, not used). Index hash is the only knob that matters: ideal random hash 6.66 (-10%), XOR folds of the line number by shift s (>>s, 32k on / off GB): s=9..12 6.63-6.65 / 11.36-12.24, s=13 7.32 / 12.41, s=14 8.83 / 12.35, s=15 9.71 / 12.19, s=16 6.93 / 14.28, s=17 6.68 / 14.77, 4 MB pow2 sets 8.17. Short and 8k are insensitive (all hashes within 0.5%). Honest uncertainty: ideal-hash LRU at 6 MB is the lower bound (-10..-14% vs ncu at 32k), modulo the upper bound (+13..+32%); the real hash is unknown, so the shift 13 is a calibrated choice that falls between them, and a single-shift fold is sensitive at 32k (s=14..17 differ by +-20%), so treat 32k miss counts as +-10%. FA-off per-kernel still deviates (mul_mat_f -23%, mul_mat_vec_f +10%, cancelling). No evict-first/nc-load hints were modelled (the streamed format does not carry the cache operator).

**Task 3, rerun of V100 32k with x13 (job 124562; `results/gpu/q7_v100_32k_fa{auto,off}_x13_*`).** Attribution: 32k FA on KV 2932 MB of 7318 = 40% (was 55%); FA off KV 7838 of 12415 = 63% (was 69%). Merged in-order row hit identity: FA on 0.826 (K 0.946, V 0.832, ffn 0.763); FA off 0.771. Replay, saturated, first 80 M requests, HBM2 819 GB/s peak, 256 B channel interleave unless noted; GB/s (Open row hit):
| cell | identity | xor_global | chan | xor_chan | 2 KiB interleave identity / chan | 4 KiB identity / xor |
|---|---|---|---|---|---|---|
| FA on Open | 156.5 (0.751) | 156.2 (1.00x) | 211.3 (1.35x) | 208.5 (1.33x) | 285.3 (0.819) / 293.3 (1.03x) | 273.1 / 273.7 |
| FA on ClosedCAP | 129.3 (0.665) | 129.1 | 172.4 | 168.1 | 257.7 / 264.5 | 248.7 / 249.5 |
| FA off Open | 142.0 (0.723) | 145.3 (1.02x) | 255.7 (1.80x) | 264.2 (1.86x) | 140.4 (0.787) / 224.4 (1.60x) | 133.0 / 143.4 (xor 1.08x) |
| FA off ClosedCAP | 157.3 (0.693) | 161.7 | 279.4 | 290.8 | 141.0 / 216.1 | 134.4 / 144.7 |
Changes against the earlier conclusions: (1) FA-on baseline BW 128 -> 156 GB/s because the stream is 25% smaller and rows more reusable; (2) 2 KiB interleave gain 2.3x -> 1.82x (285 vs 156.5); (3) channel hash 1.46x -> 1.35x with FA on (still real) and 2.11x -> 1.80x with FA off, and combined with a 2 KiB interleave the hash adds only 1-3% (FA on) so the interleave is the simpler fix there; with FA off the 2 KiB interleave alone does nothing (140 GB/s) and the channel hash is what helps (1.6x); (4) bank XOR stays ~1.00-1.02x (1.08x at 4 KiB FA off), ClosedCAP still worse than Open with FA on (-17% to -10%) and about equal with FA off; (5) KV share at 32k FA on is 40% not 55%. Qualitative conclusions stand; magnitudes shrink by 10-25%. 8k results (short, 8k) are unchanged by the fix (model already within 2%).

**Task 4, A16 ECC (node05, driver 610.43.02).** `nvidia-smi -q -d ECC`: ECC mode Enabled (current and pending), 0 errors; usable memory 15356 MiB of 16384 MiB = 15/16, which is the signature of GDDR6 inline (sideband-free) ECC reserving 1/16 of capacity. ncu on the 0.5B short step (462 kernels): dram__sectors_read 12.53 M (400.9 MB) vs L2 read lookup-miss 10.96 M (350.6 MB): DRAM reads are 14.3% above L2 misses (reproduces the +16% of the 7B). The only ECC counter, lts__t_sectors_data_ecc, is 1302 sectors (L2-side, negligible), so the extra is DRAM-side. There is no DRAM-ECC traffic counter and ECC cannot be disabled without admin rights, so the cause is supported (inline ECC reserve, extra DRAM reads with no extra L2 misses, V100 HBM2 has none and matches within 1%) but not directly isolated. Treat A16 absolute DRAM bytes as x1.14-1.16 of the model.

**Jobs and files.** 124411/124412 (32k FA-on trace with 12 variants / ncu per-kernel), 124442/124443 (32k on/off with raw trace kept), 124450 (offline variants), 124471/124472 (8k on/off, short, hash-shift sweep), 124561 (x13 on 8k/short), 124562 (x13 rerun + replays). Code: `gpu_filter2.c`, `46_stream2.sh`, `47_ncu_perk.sh`, `48_l2study.sh`, `49_offline.sh`, `50_rerun.sh`, `l2_compare.py`. `gpu_filter.c` itself still uses the modulo index and is not changed (old tables remain reproducible); use `gpu_filter2 --var x13:hash=113` for new work. Raw traces (about 24 and 38 GB) remain in node04:/scratch/$USER/gt/q7_v100_32k_fa{auto,off}.raw.bin.

### tracker-owner: tracker updates (2026-11-14)

The project tracker page (artifact https://claude.ai/artifact/1Es5tvyhpjedAa56Vc6MRu, source in the session scratchpad, not in the repo) was extended from E1-E27 to E1-E39 and republished (versions 8 to 10). It now covers: the dramtech-owner result (E28-E31, reference table), the gpu-owner route and V100/A16 matrices (E32-E35, GPU table), the mop-owner MOP4CLXOR comparison (E37, decision step 19; xor_global wins or ties in closed loop, MOP4CLXOR wins only on saturated DDR4), the final write-up (E38) and the l2fix-owner result (E36; V100 32k rows replaced by the XOR-hashed L2 set index numbers, channel gains 10-25% smaller). Decision flow has steps 15-19, milestone M6 (DRAM technology and GPU) is closed, all analysis agents are marked done. Left on the page: manual novelty check (user) and housekeeping (token revocation, node05 /scratch). The V100 short and 8k replay columns in the GPU table are from the pre-fix run (their DRAM bytes were already within 2% of ncu).

### novelty-owner: conference program check (2026-11-14)

Updated `related_work_2023_2026.md` (new section 9, corrected BibTeX). Programs read (title-level keyword filter): ISCA 2025, ISCA 2026, MICRO 2025, MICRO 2026, HPCA 2026 (up to Tue 3 Feb). Not checked: ASPLOS 2026 (page 404), DBLP (blocked), IISWC/SC/MEMSYS/CAL (search only). No paper found on KV bank conflicts under huge pages, XOR bank hashing for LLM traffic, page policy for LLM, or DRAM-level llama.cpp/MoE study.
- Citations: Flatfish verified (TCAD 41(11):4758-4770, 2022). AttAcc was two papers: CAL 2023 (Choi et al., DOI 10.1109/LCA.2023.3305386) and ASPLOS 2024 "AttAcc!" (Park et al., DOI 10.1145/3620665.3640422); cite the latter. RH+ = arXiv:2606.05511 (Jung, Anik, Lee, Ryoo; preprint). UMDAM confirmed withdrawn, dropped.
- New must-cites: FACIL (HPCA 2025, per-huge-page DRAM mapping, closest precedent), RoMe (HPCA 2026), PFM/Dual-View Memory (MICRO 2026), REFLEX (DAC 2026), Sudoku (DRAMSec 2025), SGDRC (GPU channel hash). Unread but to watch: DUALign (MICRO 2026, memory mapping for PNM LLM serving; title only).
- Verdicts: claim 1 partly published (idea yes, CPU row-hit measurement new); claim 2 new as finding, ingredients published (XOR hashing, FACIL huge-page mapping, KV row placement on PIM); claim 3 new as negative results; claim 4 partly published (GPU channel hash known, attention-specific bank vs channel comparison and L2 set-hash correction not found).

### extras-sim-owner: DRAMsim3 closed loop and XOR hardware cost (2026-11-14)

**Task A: closed-loop DRAMsim3.** New `tools/dramsim3_loop.cc` (built with the gem5 env g++, the system g++ cannot link libdramsim3.so), `tools/dramsim3_loop_run.py`, `scripts/32_ds3loop.sh`. N per-channel DRAMsim3 instances ticked together (channel decode as in dramsim3_replay), 4 SimpleO3-style cores (ipc 4, core clock = 2x memory clock, in-order retire, window W instructions, `bubbles` non-memory instructions per read, MSHR 16/core, stall when window/MSHR/transaction queue full, loads complete through the DRAMsim3 read callback; no LLC, the stream is post-LLC). Run ends when every core has retired min(len)*(bubbles+1) instructions (as SimpleO3). Modes: `sat` (per-channel in-order, 1 req/cycle: reproduces the old `dramsim3_drain` exactly, e.g. 6ch THP identity 2.782 Mcycles, 4k 2.429, packed 2.680 = validation-owner's numbers) and closed loop. Same streams as validation-owner (7B 32k attr, 3M reads from 30%, 32 GB pool, DDR4-2933 16Gb x8 2 ranks, rochrababgco, OPEN_PAGE). Jobs: smoke 124724, 6ch 124729 (11m), 12ch 124730 (15m), CLOSE_PAGE 124734/124735. Results `cluster-data/results/fullscale/ds3loop/{n6,n12,n6_close,n12_close}/loop.json`.

Calibration vs Ramulator (identity, THP, 6ch, GB/s DRAMsim3 / Ramulator): k5 (264,1280) 10.1 / 10.5; plateau (52,1024) 46.0 / 46.6; w512 39.5 / 36.2; fast k1 (52,256) 27.4 / 22.9. So native and plateau points match within 5%; DRAMsim3 sustains 15-20% more at the fast point (better reordering, row hit 0.77 vs 0.77 but lower latency 49 vs ~53 cycles).

**xor_global / identity cycles (Open), THP; Ramulator DDR4 value in brackets; hit identity -> xor:**

| point | 6ch DRAMsim3 [Ramulator] | 12ch DRAMsim3 [Ramulator] |
|---|---|---|
| saturated (1 req/cyc/ch) | 0.973 [0.90] | 0.934 [0.87] |
| unlimited window, 4 cores | 0.976 | 0.910 |
| fast k1 (52,256) | 0.957 [0.954] hit .773->.810, lat -5% | 0.908 [0.928] hit .813->.865, lat -9% |
| w512 | 0.986 [0.962] | 0.970 [0.948] |
| plateau w1024 | 0.996 [0.988] | 0.990 [0.990] |
| k4 (211,1024) | 1.002 | 0.996 |
| native k5 (264,1280) | 1.002 [1.000] | 0.999 [1.000] |

Packed: 0.985-1.005 at 6ch, 0.987-1.000 at 12ch in the closed-loop points (only the saturated 12ch point shows +5%, identity packed is already the best there). 4 KiB: 0.999-1.001 everywhere. CLOSE_PAGE (DRAMsim3 auto-precharge every access, hit 0, not ClosedCAP), THP xor/id: sat 0.919 / 0.866 (6ch/12ch), fast 1.005 / 0.987, plateau 0.958 / 0.940 (GB/s identity 14 at fast, 26 at plateau: 2x lower than Open).

**Verdict A.** DRAMsim3 closed loop agrees with Ramulator's closed-loop conclusion (on DDR4-2933, where Ramulator's own k1 gain is -4.6% at 6ch / -7.2% at 12ch, smaller than its DDR5 -16%): fast-core THP bank XOR -4.3% (6ch) / -9.2% (12ch) cycles, -5/-9% read latency; ~-0.4..-1% at the plateau; 0 at the native k4/k5 rates; packed and 4 KiB flat. The saturated gap seen before (DRAMsim3 -2.7% vs Ramulator -9.9%) mostly disappears once the loop is closed (the two simulators then differ by <=2 points, DRAMsim3 slightly larger at 12ch fast, Ramulator slightly larger at w512). So the mapping effect is not a Ramulator-controller artefact; the gain exists only in the memory-demanding closed-loop regime on THP. Caveats: 3M-read window, DDR4 only (no DRAMsim3 DDR5 config), reads only, no LLC/MSHR merging, same assumed address map; the core model is the calibrated Ramulator one, not re-calibrated for DRAMsim3 (its fast point sustains 15-20% more bandwidth).

**Task B: XOR hash hardware cost.** `tools/xorcost.py` over 458 searched matrices (335 bank, 123 channel) and the note `docs/xor_hardware_cost.md`. Bank hash: median 5 XOR2 gates (DDR4 0-10, DDR5 x16 1-13, x8 1-12; 11 after sharing at worst), max 3 row bits per output bit so at most 2 XOR levels (about 3-5 FO4); channel hash 3-4 outputs, median 3-4 gates, depth <=2. Programmable form: 16-bit row mask per output (row bits 0-15 appear), ~144 register bits and ~280 gates (~800 GE), depth 4 XOR + AND. Negligible area and off the DRAM-timing critical path; Intel servers already use XOR channel/rank/bank hashes (DRAMA, Pessl et al. 2016; Zhang et al. MICRO 2000 permutation interleaving). ps/mm^2 values are scaled estimates (factor ~2).

### main session: cluster home cleanup (2026-11-14)

Home was at 97.7 of 100 GB (hard limit 102 GB) while the 30B batch-4 32k capture (124714) was still to write its miss stream. With the user's approval, deleted the regenerable gem5 derived files in `results/traces/*/` (`dram.bin`, `dram.ramulator.trc`, `dram.dramsim3.trc`; ~15.6 GB) and kept every run's raw `dram.trc.gz` (11 files; rebuild with `tools/analyze.py`). Home is now 81.9 GB. Big remaining items: `results/sessions/drbatch/` 25 GB (prompt caches; the 30B P32768 ones are in use), `tracing/models/` 24 GB. Note: the laptop copy `cluster-data/release/moe_c1024_l3-22m.dram.trc.gz` is 110 MB against 629 MB on the cluster, so it looks truncated; re-download it from the release.

### dram-measure-owner: measured DRAM on V100, A16 and CPU (2026-11-14)

**What was measured.** Nsight Compute (ncu) hardware counters, per kernel, over exactly one decode step of llama.cpp `build-cuda` (-ngl 99, greedy, graphs off under ncu, `--cache-control none --clock-control none`, so clocks are not locked and L2 state persists between kernels). Windows are the last decode step found by `find_window.py` (538 kernels FA on, 566 FA off for 7B; 462 for 0.5B). Metrics: dram__bytes_read.sum, dram__bytes_write.sum, dram__throughput.avg.pct_of_peak_sustained_elapsed, gpu__time_duration.sum, lts__t_sector_hit_rate.pct, sm__throughput.avg.pct_of_peak_sustained_elapsed, gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed. Token time without profiler: llama.cpp's own `eval time` over 63 decoded tokens with CUDA graphs on (nsys is not installed on the cluster; two runs agree within 1-3%). Jobs 124956 (test), 124957/124958 (first pass, long-context cells failed on a prompt-argument bug), 124977/124978 (rerun of 8k/32k). Code: `trace-pipeline/gpu/50_dram_measure.sh`, `51_measure_all.sh`, `measure_parse.py`; data `results/measure/` (`*.perk.tsv` per kernel, `*.summary.json`, `*.timing.txt`, raw ncu csv).

**Qwen2.5-7B Q4_K_M, one decode step** (read GB/s per token = DRAM read bytes / un-profiled token time; kernel-time GB/s = (read+write) / sum of ncu kernel times; % columns time-weighted over kernels; peak = the counter's own peak: V100 about 900 GB/s, A16 200 GB/s):
| GPU | ctx | FA | DRAM read GB/token | write GB | token ms | read GB/s per token | kernel-time GB/s | DRAM % peak | SM % peak |
|---|---|---|---|---|---|---|---|---|---|
| V100 | short | on | 4.41 | 1.28 | 7.81 | 565 | 545 | 66 | 36 |
| V100 | short | off | 4.41 | 1.35 | 8.47 | 520 | 540 | 64 | 37 |
| V100 | 8k | on | 4.82 | 1.48 | 9.35 | 516 | 532 | 63 | 34 |
| V100 | 8k | off | 4.84 | 0.71 | 10.38 | 467 | 481 | 57 | 36 |
| V100 | 32k | on | 7.41 | 0.59 | 13.96 | 531 | 523 | 61 | 30 |
| V100 | 32k | off | 13.13 | 0.99 | 22.34 | 588 | 576 | 67 | 27 |
| A16 | short | on | 5.25 | 0.16 | 38.11 | 138 | 155 | 78 | 69 |
| A16 | short | off | 5.25 | 0.16 | 40.11 | 131 | 147 | 74 | 69 |
| A16 | 8k | on | 5.69 | 0.18 | 40.51 | 140 | 157 | 79 | 65 |
| A16 | 8k | off | 5.95 | 0.25 | 49.39 | 120 | 137 | 69 | 69 |
| A16 | 32k | on | 7.21 | 0.18 | 48.00 | 150 | 164 | 82 | 56 |
| A16 | 32k | off | 20.50 | 0.55 | 118.91 | 172 | 181 | 90 | 45 |
0.5B (FA on): V100 short 0.35 GB read, 2.22 ms, 159 GB/s, DRAM 20% / SM 10%; V100 8k 0.44 GB, 2.54 ms; A16 short 0.41 GB, 4.12 ms, 99 GB/s, DRAM 52% / SM 42%; A16 8k 0.50 GB, 4.67 ms.

**Per kernel class (7B, % of kernel time; DRAM % / SM % of peak).** V100 short FA on: mul_mat_vec 82% of time, 637 GB/s, 75% / 43%; quantize_q8 6%, 17% / 2%; flash_attn 6%, 183 GB/s, 42% / 7%. V100 32k FA on: mul_mat_vec 49%, 645 GB/s, 76% / 44%; flash_attn_ext_vec 43%, 477 GB/s, 55% / 21%. V100 32k FA off: mul_mat_vec 63%, 568 GB/s, 66% / 33%; the other kernels (KV copy/mat-mul on KV, 115 launches) 28%, 738 GB/s, 84% / 20%. A16 short: mul_mat_vec 96% of time, 160 GB/s, 80% / 71%. A16 32k FA on: mul_mat_vec 75%, 160 GB/s, 80% / 71%; flash_attn 23%, 190 GB/s, 95% / 10%. A16 32k FA off: mul_mat_vec 61%, 178 GB/s, 89% / 50%; other 36%, 191 GB/s, 96% / 39%.

**Findings.**
1. Memory-bound everywhere: DRAM % of peak exceeds SM % of peak in every row and class that moves data. The weight GEMV (mul_mat_vec) reaches 75-76% of V100 peak (637-645 GB/s) and 80% of A16 peak (160 GB/s); KV kernels reach 95-96% of A16 peak at 32k (A16 is bandwidth-saturated) and 55-84% on V100. Whole-step DRAM utilisation is 57-67% on V100 and 69-90% on A16 (rising with context). The 0.5B model on V100 is not bandwidth-limited (20% DRAM, 10% SM; launch/latency bound, 462 small kernels in 2.2 ms), on A16 it is 52% / 42%.
2. Measured DRAM read bytes per token against our pipeline model (V100 4.39 / 4.82 / 7.32 / 12.4 GB for short / 8k / 32k FA on / 32k FA off): measured 4.41 / 4.82 / 7.41 / 13.13 GB, i.e. +0.5%, 0%, +1.2%, +5.9%. A16 model (V100 model x 1.143 for ECC) 5.02 GB at short vs measured 5.25 GB (+4.6%); at 32k FA on measured 7.21 and FA off 20.50 GB. Note the A16 measured value 5.25 is a counter fact (ncu dram__bytes_read), not an inference.
3. Inferred vs measured bandwidth: the earlier V100 figure ~530 GB/s (4.4 GB x 120 tok/s) vs measured 565 GB/s at short context (4.41 GB / 7.81 ms; 7.8 ms/token is 128 tok/s with graphs) and 545 GB/s by kernel-time. A16 earlier ~115 GB/s (4.39 GB x 26 tok/s) vs measured 138 GB/s read (5.25 GB / 38.1 ms), 155 GB/s by kernel time: the inference understated it because the real DRAM bytes on A16 are 5.25 GB, not 4.39 GB. The A16 sits at 69-78% of its 200 GB/s at short context and 82-90% at 32k.
4. FlashAttention off costs 13.13 vs 7.41 GB (V100) and 20.50 vs 7.21 GB (A16) at 32k; the token time follows the bytes (22.3 vs 14.0 ms; 118.9 vs 48.0 ms).
5. Anomaly: V100 `dram__bytes_write` is 0.6-1.5 GB per step, concentrated in mul_mat_vec (1.07 GB at short, FA on) even though those kernels write only megabytes of activations; A16 shows 0.16-0.55 GB. We did not find the cause (HBM2 ECC traffic or counter artefact are untested); it does not affect the read figures but kernel-time GB/s on V100 includes it (reads alone give 545 -> about 420 GB/s kernel-time at short, still dominated by the 565 GB/s read per token figure measured with graphs).
   (Correction for item 5: kernel-time GB/s in the table adds write bytes; read-only kernel-time GB/s on V100 short is 4.41 GB / 10.4 ms = 424 GB/s because ncu kernel times (10.4 ms) exceed the graph-mode token time (7.81 ms). Graph mode removes the gaps and ncu isolation overhead, so the per-token column is the better bandwidth figure.)

**Hardware row-buffer counters: none.** `ncu --query-metrics-mode all` on V100 lists, under dram__ and fbpa__, only bytes, sectors (read/write), cycles (active, elapsed, in frame/region), and throughput (listed in `results/measure/allmet_v100.txt` and `metrics_v100.txt`). No activate, precharge, row-hit or page counters; a real hardware row-hit rate cannot be measured on these GPUs. The A16 query was not run separately (same ncu counter families). The row hit rates in the GPU section therefore stay simulation outputs.

**CPU branch: stopped.** `perf list` shows uncore_imc_0..5 cas_count_read/write, but `/proc/sys/kernel/perf_event_paranoid` is 2 on the login node and on node03, and `perf stat -a -e uncore_imc_0/cas_count_read/,uncore_imc_0/cas_count_write/ -- sleep 1` (also `-a -A`, `-C 0`, and per-process) prints `<not supported>` for every IMC counter as a normal user (perf 4.18; the event is forced to the user-only `/u` modifier). `/dev/cpu/0/msr` is root:root 0600 and `/sys/fs/resctrl` is empty, so no alternative route exists without root. No CPU DRAM bytes could be measured; the DynamoRIO + drcachesim figures (4.40 GB short, 9.8-11.5 GB at 32k) stay as the CPU numbers.

### main session: provenance of the gem5 models (2026-11-14)

The three gem5-era models were downloaded on 2026-10-31 before `10_fetch_models.sh` existed, so their source was never recorded. Identified by exact file size (Hugging Face API / resolve headers) and the GGUF `general.name`:
- `stories15M.gguf` (19,077,344 B, Q4_0, llama, 6 layers): https://huggingface.co/ggml-org/models, file `tinyllamas/stories15M-q4_0.gguf` (Karpathy's llama2.c TinyStories 15M, converted by ggml-org).
- `qwen25-05b-q4_0.gguf` (428,730,208 B, Q4_0, `general.name = qwen2.5-0.5b-instruct`): https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF, file `qwen2.5-0.5b-instruct-q4_0.gguf`.
- `tinymoe4.gguf` (1,979,016,640 B, Q8_0, `general.name = Tinyllama-moe4`, 22 layers, expert_count 2, expert_used_count 2): https://huggingface.co/israellaguan/Tinyllama-moe4-Q8_0-GGUF, file `tinyllama-moe4-q8_0.gguf` (GGUF of https://huggingface.co/aipib/Tinyllama-moe4).
Full-scale models: `Qwen/Qwen2.5-7B-Instruct-GGUF` and `Qwen/Qwen3-30B-A3B-GGUF` (see `scripts/10_fetch_models.sh`).

### extras-capture-owner: 30B batch x long context, prefill ub64 (2026-11-15)

**Task A, Qwen3-30B-A3B Q4_K_M, B=4, one decode step traced, 4 threads, node01, scripts/19_batchmiss.sh (miss.gz kept in `results/fullscale/drbatch_miss/Qwen3-30B-A3B-Q4_K_M_B4_P{8192,32768}t4/`).** Jobs: P=8192 STRIDE=40000: capture 124713 (untraced prompt evaluation 1502 s, 15.1G instructions, 28 GB raw trace, 148.5M LL reads no-prefetch = 9.5 GB); region 124751, moe_remap 124752, experts touched 124860 (24M-request window of 148.5M). P=32768 STRIDE=70000: capture 124714 (prompt evaluation 13,470 s = 3.7 h, 47.4G instructions, raw trace about 100 GB under MAXTRACE_GB=48, 1.68G LL reads = 107 GB, step about 7.9 s native, miss.gz 5.1 GB); first replay attempts 125024-6 OOM-killed (150 GB, run together); rerun one at a time with 330 GB: region 125044 (1 h 53), moe_remap 125045 (2 h 20), experts touched 125046; windows are 40M requests starting at 30% of the step (503.9M to 543.9M). Positional token gate: all four sequences identical in both runs. Cost per 30B run: the 32k sbatch needs --mem >= 100 GB (RSS 44.7 GB for the model plus KV, the replay needs far more), and --time of days.

Cycles relative to identity (identity Mcycles and in-order hit in brackets), THP placement:

| run | DDR4 Open xor_global / region | DDR4 Closed xor_global / region | DDR5 Open xor_global / region | DDR5 Closed xor_global / region |
|---|---|---|---|---|
| 30B B=4 P=8192 | 0.850 / 0.860 (131.4, 0.862 -> 0.961) | 0.843 / 0.854 (144.8) | 0.942 / 0.959 (27.4, 0.955) | 0.992 / 0.992 |
| 30B B=4 P=32768 | 0.872 / 0.862 (206.8, 0.702 -> 0.927) | 0.882 / 0.880 (209.0) | 0.942 / 0.916 (26.2, 0.951) | 0.969 / 0.978 |
| 7B B=4 P=8192 (serving-owner) | 0.989 / 0.921 | | 0.963 / 0.963 | |
| 7B B=4 P=32768 (serving-owner) | 0.885 / 0.872 | | 0.938 / 1.000 | |

Packed placement: DDR4 no gain (0.993-1.005) at both lengths; DDR5 packed identity is bad (hit 0.885 at 8k, 0.851 at 32k, K class 0.587 at 32k) and xor_global / region give 0.849 / 0.853 (8k Open) and 0.740 / 0.725 (32k Open), 0.66 / 0.62 ClosedCAP (the VA-compaction aliasing effect already noted for 7B B=4, not a KV-row conflict effect).

**Expert-aware mapping (moe_remap.py, expert_hash, expert_lin, region_expert): no effect.** expert_hash and expert_lin give 1.000 to 1.002 of identity in every case at both lengths (in-order hit unchanged); region_expert equals region (+-0.001). Reason: weights are already at hit 0.95-0.96 in identity (W class), and at B=4 long context the stream is 96% KV reads (class shares full step W 0.039 / K 0.959 / O 0.002 at 32k), so expert weights are not where row conflicts are. Experts touched per layer (gate tensor, one B=4 decode step): 24.5 of 128 at 8k and 24.5 at 32k (of at most 32 with 4 tokens x 8 experts; B=1 touches 8.0), so batching makes the expert union about 3x larger but weights remain a small share of reads at long context.

**Answer.** For the batched MoE at long context the KV-page XOR gain appears, as for the dense 7B, and is larger at short per-sequence length: 30B B=4 xor_global 0.85 (8k) and 0.87 (32k) on DDR4 Open (vs 7B B=4 0.99 at 8k, 0.885 at 32k), 0.94 on DDR5 Open, 0.97-0.99 ClosedCAP DDR5. The 30B has 98 KB KV per token (7B 57 KB) and 4x as much KV per decode as the 7B at the same P, so its stream is KV-dominated already at 8k. Region-specific matrices tie xor_global (0.86 vs 0.87 at 32k DDR4; 0.916 vs 0.942 DDR5 Open at 32k). Expert-aware mapping does not matter. The single global XOR remains the proposal.

**Task B, 7B Q4_K_M 512-token prefill, n_ubatch=64, last graph compute (compute 7 of 8, positions 448-511) traced, 4 threads, node01, job 124859 (7 h 5 min, scripts/26_prefill.sh with MAXTRACE_GB=40; raw trace 41 GB, 32 GB converted, 49.5G instructions, 12.4G per thread x 4 thread ids in basic_counts; the merged miss file carries a single thread id as in earlier runs).** Same tokens as untraced: yes. No code change was needed: the earlier estimate (55G instructions, about 80 GB raw) was too high; raising the per-thread cap to 40 GB was enough. 82.5M LL reads (no prefetch) = 5.3 GB (ub=32: 75.3M). Region replay 125047 (24M window of 82.5M, window class shares W 0.81 / K 0.006 / O 0.18), cycles relative to identity (identity Mcycles, hit):

| DRAM, policy | THP identity | xor_global | region | packed xor_global / region |
|---|---|---|---|---|
| DDR4x1 Open | 109.1 (0.879) | 0.997 | 0.979 | 0.996 / 0.982 |
| DDR4x1 ClosedCAP | 139.9 (0.724) | 0.968 | 0.963 | 0.967 / 0.970 |
| DDR5x16 Open | 15.6 (0.904) | 1.003 | 1.004 | 1.002 / 1.008 |
| DDR5x16 ClosedCAP | 21.2 (0.743) | 1.000 | 1.001 | 1.001 / 1.004 |

Same picture as ub=32 (0.990-1.004, hit 0.898/0.908 Open, 0.73-0.74 ClosedCAP): prefill with 64-token micro-batches is mapping-insensitive; the only small effect is 2-4% on DDR4 (O class, 18% of the window, in-order hit 0.507 -> region fixes it). 30B prefill with ub=64 was not run (the 7B ub=64 trace alone took 7 h of wall time, mostly raw2trace; a 30B prefill would be longer).

### GPU follow-up cluster update (2026-11-15T22:33:34+05:30)

scheduled audit=125717 compare=125718 windows=125719 moe-smoke=125720 moe-replay=125721. See cluster-data/results/gpu/followup_20261115/ for source snapshots, manifests, validation gates and results. Saturated memory replay results are not measured token-time speedups.

### GPU follow-up cluster update (2026-11-15T22:34:17+05:30)

audit job 125717 exited 0. See cluster-data/results/gpu/followup_20261115/ for source snapshots, manifests, validation gates and results. Saturated memory replay results are not measured token-time speedups.

### GPU follow-up cluster update (2026-11-15T23:23:12+05:30)

compare job 125718 exited 0. See cluster-data/results/gpu/followup_20261115/ for source snapshots, manifests, validation gates and results. Saturated memory replay results are not measured token-time speedups.

### GPU follow-up cluster update (2026-11-15T23:52:45+05:30)

windows job 125719 exited 0. See cluster-data/results/gpu/followup_20261115/ for source snapshots, manifests, validation gates and results. Saturated memory replay results are not measured token-time speedups.

### GPU follow-up cluster update (2026-11-16T01:46:09+05:30)

Nov16 01:45 IST check: audit 125717, compare 125718 (48m48s), windows 125719 (29m24s) completed; MoE smoke 125720 pending Priority, replay 125721 pending Dependency. Nonempty converted CPU 0.5B and 7B short-context traces remain on node01 scratch; no trace.gz found in node03 scratch search. 21 retained miss.gz files cannot substitute for original pre-cache traces when adding L2. L2 runtimes are estimates, not measured: representative short-trace sensitivity about 1-3 hours including configuration/validation, broad recapture campaign several days. No L2 jobs submitted.. See cluster-data/results/gpu/followup_20261115/ for source snapshots, manifests, validation gates and results. Saturated memory replay results are not measured token-time speedups.

### GPU follow-up cluster update (2026-11-16T01:56:09+05:30)

cache audit: node01 private L2 1MiB 16-way, most full CPU traces require recapture; GPU compare/window jobs completed, gains depend on feed and window. See cluster-data/results/gpu/followup_20261115/ for source snapshots, manifests, validation gates and results. Saturated memory replay results are not measured token-time speedups.

### GPU follow-up cluster update (2026-11-16T02:29:27+05:30)

CPU presentation tables: recovered saved response IDs using vocabulary only; exact prompt counts and missing stdout documented; step0 after windowed prefill is decode, not full-prefill timing. See cluster-data/results/gpu/followup_20261115/ for source snapshots, manifests, validation gates and results. Saturated memory replay results are not measured token-time speedups.

### GPU follow-up cluster update (2026-11-16T17:38:01+05:30)

moe-smoke job 125720 exited 1. See cluster-data/results/gpu/followup_20261115/ for source snapshots, manifests, validation gates and results. Saturated memory replay results are not measured token-time speedups.

### GPU follow-up cluster update (2026-11-16T22:45:23+05:30)

Schedule repaired MoE validation: smoke 126467 and replay 126468. See cluster-data/results/gpu/followup_20261116/ for source snapshots, manifests, validation gates and results. Saturated memory replay results are not measured token-time speedups.

### Accel-Sim audit cluster update (2026-11-17T07:55:44+05:30)

Cross-checked V100 hardware writes are about9.75/10.35MB per token; earlier large multi-metric write volumes superseded; full replay126821 running. See cluster-data/results/gpu/cache_audit_20261117/ for source snapshots, manifests, validation gates and results. Saturated memory replay results are not measured token-time speedups.

### GPU follow-up cluster update (2026-11-17T18:07:07+05:30)

moe-smoke job 126467 exited 0. See cluster-data/results/gpu/followup_20261116/ for source snapshots, manifests, validation gates and results. Saturated memory replay results are not measured token-time speedups.

### GPU follow-up cluster update (2026-11-17T18:12:04+05:30)

moe-replay job 126468 exited 1. See cluster-data/results/gpu/followup_20261116/ for source snapshots, manifests, validation gates and results. Saturated memory replay results are not measured token-time speedups.

### Accel-Sim audit cluster update (2026-11-17T19:12:18+05:30)

V100 trace asset publication exit=0; see reconciled simulation summary and trace_asset.json. See cluster-data/results/gpu/cache_audit_20261117/ for source snapshots, manifests, validation gates and results. Saturated memory replay results are not measured token-time speedups.

### Accel-Sim audit cluster update (2026-11-20T22:00:58+05:30)

Nov20 resumed campaign: results/gpu/campaign_20261120 contains20CPU-workload coverage,19 native passes and1 maximum-context cuBLAS-init failure; fresh0.5Bcapture131049 passed462kernels/exacttokens; hardware131067 read349489472B/write3175584B matched462kernels; modelreplay131062 running. Capturearrays131056 dense(10pending/running cases,up to2GPUtasks),131057 MoE9cases automaticallyspawnmodel/export, independenthardwarecounters, 4M-sector fullRamulator mapping/policy grids and maskedrawCSVreleaseupload. Dirty-sectorfixture131051/58, dense400replayfixture131059, expert120fixture131069 passed. Isolated DRAMsim3 true-demand-drain/global32issue64PCfixture131079 passed, publishedb2e5e43; oldwritecallback-onlydrain was early-ack, not physicalcompletion. OriginalcompressedSASSretained; onlyregenerabletracezremovedaftervalidatedCSVassetsdurable. Read/writehardwareaccuracy stillcase-specific; existing7Bpilot writegap27.9percent unresolved. Fullcampaignnotcomplete. SlowSSHrequiresminutes; earlier30stimeoutdiagnosiswaspremature.. See cluster-data/results/gpu/cache_audit_20261117/ for source snapshots, manifests, validation gates and results. Saturated memory replay results are not measured token-time speedups.
### Codex continuation: GPU follow-up scheduling (2026-11-15, IST)

User requested all five GPU follow-ups be prepared and scheduled, then explicitly directed that job files be created on the cluster and results pushed from the cluster using the existing logistics. Read the previous GPU scripts, context handoffs, and live queue before proceeding. Pulled origin/main with rebase/autostash (11f22ba -> 1ee2278), preserving existing local edits.

Five stages: (1) synthetic + captured-stream replay correctness audit; (2) shared-map/timing/feed comparison for V100/A16 32k FA on/off, 256 B/2 KiB; (3) beginning/middle/end windows using a mapping frozen from the beginning; (4) separate-device 30B two-V100 capture smoke; (5) gated full-step 30B capture and independent-device mapping replay. Outputs: ~/amcas-pro/results/gpu/followup_20261115; compressed address streams remain on node-local scratch/tmp. New code is being created in ~/amcas-pro/gpu, with source snapshots under the result directory for reproducibility and GitHub publication. No existing traces or scripts are being overwritten.

Audit of old gpu_ds3_study.py found global ordered Ramulator injection versus independent per-PC/channel DRAMsim3 injection, and simulator-specific XOR searches. New comparisons hold matrices identical between simulators and add independent-channel Ramulator injection. Resolved timing/geometry differences are saved explicitly; HBM coupled PCs versus separate DRAMsim3 PC instances remain a model difference. A one-minute node01 srun preflight passed for both GPU standards on synthetic row-hit, row-conflict, channel-spread and mixed read/write streams (2 CPUs/4 GB); submissions and further decoder checks follow below.

Connectivity: VPN profile iiit appeared active but had lost its tunnel IP/routes. Reconnecting with nmcli down/up restored SSH. SSH uses -F /dev/null because this host rejects /etc/ssh/ssh_config.d/20-systemd-ssh-proxy.conf ownership/permissions. This is a client workaround, not a cluster change.

### GPU follow-up jobs submitted (2026-11-15, 22:33 IST)

All five jobs were created on RCE using `~/amcas-pro/gpu/69_schedule_followup.sh` after both one-minute node01 synthetic preflights passed (the second checks every synthetic address with DRAMsim3's actual Config decoder). New cluster scripts: `gpu_followup.py`, `gpu_moe_followup.py`, `gpu_decode_check.cc`, `68_followup_job.sh`, `69_schedule_followup.sh`, `70_publish_followup.sh`. The initial local runner draft was removed after cluster-specific revisions; authoritative follow-up sources are frozen under `results/gpu/followup_20261115/code/` and published with results, rather than copied back over laptop sources.

| Job | Stage | Resources | Dependencies |
|---|---|---|---|
| 125717 | replay correctness audit | node05, 4 CPU, 16 GB, 1 h | none |
| 125718 | shared mapping/timing/feed comparison | node05, 8 CPU, 64 GB, 24 h | afterok 125717 |
| 125719 | beginning/middle/end windows | node05, 8 CPU, 64 GB, 24 h | afterok 125718 |
| 125720 | 30B two-V100 capture smoke | node04, 2 GPUs, 4 CPU, 48 GB, 2 h | afterok 125717 |
| 125721 | full 30B decode capture + mapping replay | node04, 2 GPUs, 8 CPU, 64 GB, 24 h | afterok 125718 and 125720 |

`--kill-on-invalid-dep=yes` prevents expensive successors from running after failed validation. The comparison uses 8M requests per cell, four 32k GPU/FA cells and both interleaves; matrices are trained once and held identical across the three timing/feed controls and both simulators. Window sensitivity freezes the beginning-window matrices and the full-stream first-touch page assignment. Matched timing aligns common resolved Ramulator fields, enables A16 bank groups in DS3 to match Ramulator, and uses unified DS3 queues to avoid the known write-drain deadlock. These are controlled comparisons; residual controller/PC differences are recorded, not claimed eliminated.

MoE smoke captures four-launch windows separately for each CUDA context; full capture streams through two independent per-device cache filters without retaining raw NVBit addresses. Gates check native versus traced generated output, launch-ID/kernel sequence, stable context partition, per-launch DRAM counts and paired stream lengths. Replay is independent per device and does not model inter-GPU transfer or token time.

Each job writes source hashes, allocation metadata, exit status and run `context.md`; on exit (including failure) it invokes the existing publish.sh, then appends a dated entry to the repository root context.md and pushes from the cluster under the publisher lock. Initial results publication confirmed at commit 6b51354; the scheduler also pushed the root-context submission record. At first verification 125717 was RUNNING on node05 and the other four were waiting on dependencies. Both V100s are requested explicitly, so MoE jobs may wait for node04 resources after dependencies clear.

Audit job 125717 finished its computation successfully at 22:33:31 IST (exit 0): synthetic streams and 100,000-request captured short-context windows passed on both standards, including the actual DRAMsim3 decoder and request accounting. At 22:33:49 it was still in its end-of-job publication step, so afterok dependencies had not yet been released; this is expected because Slurm waits for the full wrapper to exit.

Follow-up verification: 125717 is COMPLETED (0:0, 53 seconds including publication); 125718 is RUNNING on node05; 125720 has cleared its audit dependency and is PENDING on Priority for two V100s; 125719 and 125721 retain their intended dependencies. Audit results pushed as d69db36 and root-context completion entry as 92de00b. Submission records were pushed as 6b51354 / 5cb0c15. Detailed scheduling notes are also kept in the run result directory for the next automatic publish.

### HPC cache/trace audit and GPU follow-up findings (2026-11-16, about 01:55 IST)

User asked for an estimate of a full L2-inclusive rerun across all CPU experiment configurations, whether traces were deleted, exact HPC cache geometry, and live GPU job findings. No full CPU rerun has been submitted in response to this conditional request.

Live sysfs inspection on node01 (Xeon Gold 6226R, 2 sockets, 16 physical cores/socket, SMT2): per physical core L1D=32 KiB/8-way, L1I=32 KiB/8-way, unified L2=1 MiB/16-way; unified L3=22 MiB/11-way PER SOCKET; all line sizes 64 B. cpu0's L1/L2 are shared by SMT siblings 0,32; its LLC is shared by even-numbered logical CPUs (socket0). Hardware geometry can be represented in drcachesim's custom config; actual Intel replacement/prefetch, slice hashing, coherence and physical placement remain approximations. A hardware-oriented recapture should pin distinct physical cores and memory to one NUMA socket; old unpinned captures cannot be retroactively made single-socket hardware measurements.

Trace inventory: node01 retains full drmemtrace trees for early 0.5B and 7B short drburst, early 0.5B drphases, 0.5B ub64 smoke prefill and mmap/nommap feasibility runs. node03 has no retained drmemtrace/raw/trace directories in the inspected /scratch/$USER and /tmp/$USER trees. node01 drattr/drbatch/bmiss directories are empty. There are 21 preserved CPU miss.gz files in results/fullscale, including 7B and 30B long-context/batched/prefill. These are post-LLC streams from the old hierarchy, not full accesses: adding L2 requires fresh captures for most scientific workloads. Earlier 30B B4 P32768 took 10h21 overall; 7B ub64 prefill took 7h05. A complete recapture/cache/replay campaign is therefore a multi-day task; any 48-120h planning range excludes scheduler delay and is provisional, not a measured total.

Validated installed drcachesim custom hierarchy syntax using results/fullscale/l2pilot/node01_4core_l2_pilot.cfg: four private 32KiB 8-way L1I/L1D pairs, four private 1MiB 16-way unified L2s, one shared 22MiB 11-way LLC, 64B lines, LRU approximation, no prefetch, 10M-reference cap. In bounded srun 125784, the retained 0.5B trace finished the capped simulation (67.18 seconds) and printed the intended hierarchy. The following 7B pilot was terminated by the allocation's one-minute limit; it supplies no timing estimate. This partial/cold pilot cannot be linearly extrapolated to the full campaign.

GPU statuses: 125717 audit COMPLETED 0:0 (53s), 125718 compare COMPLETED 0:0 (48m48), 125719 windows COMPLETED 0:0 (29m24). Results and root-context updates published through 5f1b098. 125720 two-V100 smoke remains PENDING Priority (node04 currently has one GPU allocated to another job); 125721 full capture/replay PENDING Dependency. No experiment jobs under this account were RUNNING at the live check.

Findings, 8M-request windows, Open page, 256B interleave, same matrix per simulator: matching timing and independent channel feeds narrows the gap substantially. A16 FA-off channel cycle ratio Ram/DS3 is 0.919/0.928 (legacy global feed 0.354/0.884); V100 FA-on channel ratio 0.871/0.896 (legacy 0.727/0.896); A16 FA-on bank XOR ratio 0.908/0.932. These are saturated replay-cycle reductions, not measured token gains. Residual coupled-HBM-PC and controller differences remain.

Window sensitivity with frozen beginning-window matrix and full-stream first-touch placement: V100 FA-on channel ratio begins at 0.871/0.896, middle 0.881/0.902, ends 0.998/1.000. A16 FA-off channel begins 0.919/0.928, middle 0.685/0.669, ends 1.000/1.000. A16 FA-on bank XOR begins 0.908/0.932, middle 0.937/0.961, ends 1.001/0.997. Therefore beginning-window gains are not representative whole-token speedups; whole-step or phase-weighted replay is needed before a headline GPU gain.

Pulled cluster publications with rebase/autostash. Both cluster and local append-only context additions conflicted at the EOF; preserved both blocks and removed only Git conflict markers, then marked context resolved. Existing user code edits remain preserved; the autostash remains available.

### Presentation learning: CPU experiment tables (2026-11-16, IST)

User requested prompts, responses, exact prefill/decode counts and performance for every CPU experiment/scenario. Built outputs/01a1e00e-fd93-7ee0-b738-e23370cdfcd4/cpu_experiment_tables.xlsx from recorded files, with 44 full-scale capture records, 10 gem5 captures, 9,020 main-grid replay candidates (internal OracleRec recorders excluded), 608 region-study records, 48 gem5 Ramulator rows, 5,428 additional CPU replay/control records, 32 gem5 timed DRAMsim3 rows, native thread benchmarks, and 103 per-sequence saved response rows. Source URLs and missing-data explanations are included. Main grid and earlier region-study systems/windows are kept separate; lower saturated replay cycles are not claimed as native token-time speedups or as the final closed-loop proposal.

Recovered response text from original native token IDs with the existing llama.cpp vocabulary (vocab_only load; no inference rerun). New temporary extraction artifacts are under ~/amcas-pro/results/presentation_review on the cluster. Many drattr/drburst stdout files were deleted by scratch cleanup, so their response text is unavailable rather than regenerated. Long prompts are previews with full saved-prompt or corpus/construction source links; batched responses are reported separately for every sequence.

Important presentation corrections: single-sequence c1k/c8k/c32k labels are context settings, not exact prompt lengths (961; 7363 for 7B / 7356 for MoE; 31001 for Qwen2 / 30993 for MoE). The short *_pf32 captures actually prefill 25 tokens. Batched/precomputed P=512/8192/32768 prompts are truncated to exactly P tokens per sequence. Four sampled IDs in drbatch normally correspond to three generated-token evaluation calls, plus sampling after the last prompt-token call; only the selected decode call is traced. The 512-token prefill study traces just the final 32- or 64-token microbatch. Its full prompt is evaluated at step -1 and is not B/E stamped; step0 timings are already subsequent decode and MUST NOT be labeled full-prefill latency.

Gem5 short Qwen prompt has 24 tokens; the DynamoRIO short prompt has 12. Earlier conversational wording calling those prompts identical was too strong; the 0.7% agreement is a cross-case traffic-count check with differing prompt/ISA/thread settings. Gem5 stdout echoes prompt text with CLI escape processing, so escaped newline/tab sequences must be normalized when extracting the generated prefix, especially the 4096-token stories15M captures. Captures stopped by work-item limits may emit fewer tokens than requested by -n.

Git pull/rebase autostash again conflicted only at the append-only context EOF. Retained upstream and local blocks and removed conflict markers; preserved user code edits and the retained autostash.

### GPU presentation audit and MoE validation repair (2026-11-16, about 22:35 IST)

User requested an exactly four-slide PowerPoint covering the GPU-only problem statement, workflow/progress, hardware/experiments/observations, and proposed mapping/page policy, plus an audit against the problem statement. Interpreted "A30" as Qwen3-30B-A3B (model), since the two actual GPU devices used are V100 and A16, not NVIDIA A30.

Live accounting at 22:30 IST: 125717 audit,125718 comparison,125719 window checks COMPLETED; 125720 MoE smoke FAILED 1:0 after3m40 at17:37:51; dependent125721 CANCELLED. Failure assertion was stream/per-launch count mismatch. Direct inspection on node04 established147985 decompressed DRAM records, matching147985 per-launch reads,0writes. The raw-path name still referred to a FIFO with stat size0; stream_count incorrectly used exists() and treated it as a regular raw trace. This is a validation bookkeeping bug, not evidence of missing captured records. The smoke stopped after the first device window, so two-device validation is still incomplete.

Repaired in a versioned copy at ~/amcas-pro/results/gpu/followup_20261116/code (exists -> is_file in the trace loader/count paths), with separate scratch/output directories and a comparison gate explicitly copied from successful125718. Resubmitted the previously authorized remaining stages as126467 (two-V100 smoke,4CPU48G2h) and126468 (two-V100 full capture/replay,8CPU64G24h,afterok126467). Do not mark MoE tracing complete until these pass.

Hardware audit on CPU-only allocations: node04=dual Xeon Gold6226R2.90GHz,32 physical cores/64threads,OS MemTotal394931752kB (~376.64GiB),L3=22MiB/socket. node05=dual Gold6226 2.70GHz,24cores/48threads,MemTotal195980632kB (~186.90GiB),L3=19.25MiB/socket. Both L1I/D32KiB,L2=1MiB. Host DIMM speed remains unavailable; no assumed host-DRAM rate should be presented as measured. CPU-only Slurm allocations hide GPU devices, so queried A16 in an allocated GPU slot:15356MiB usable,driver610.43.02,maxSM1755MHz,maxmemory6251MHz,ECCenabled. node04 GPU slot was busy, so latest V100 driver was not re-read; earlier verified hardware metadata remains the source for V100.

Deck audit conclusion: the7B dense decode characterization/byte validation is complete within the model assumptions; MoE validated tracing remains incomplete;GPU prefill/serving coverage and full-token/closed-loop mapping benefit remain gaps. Open plus a channel hash is a provisional GPU direction, with A16 bank XOR an additional candidate. Do not transfer the CPU's universal bank-XOR conclusion to GPU. Matching simulator injection/timing reduces disagreement, while window sensitivity prevents first-window gains being claimed as whole-token/native speedups. The GPU branch tested baselineOpen/ClosedCAP and strictDS3close, not the fullCPUten-policy sweep. Full source citations and per-GPU/controller/cache assumptions are included in slide notes.

Presentation delivery check (2026-11-16, 22:45 IST): created the four-slide GPU presentation at `outputs/01a1e00e-fd93-7ee0-b738-e23370cdfcd4/amcas_gpu_presentation.pptx`, with editable workflow, hardware table, chart, proposal diagram and source/qualification notes. Rendered all four slides and inspected layout; corrected hardware-table overflow. Repair jobs 126467 and 126468 remain pending (Priority / Dependency). Cluster publication pushed the repair submission records in commit a6b32f2. The presentation distinguishes validated 7B characterization from pending MoE and closed-loop/whole-token validation.

### GPU presentation revision: read/write traffic (2026-11-16)
Revised slide3 in the four-slide presentation to replace the read-only chart with measured Nsight Compute read/write GB per decode token at32k: V100 FAon7.41/0.59, FAoff13.13/0.99; A16 FAon7.21/0.18, FAoff20.50/0.55. Source: dram-measure-owner table above. Explicitly label byte volumes rather than exact individual request counts. Hardware configuration and native timing observations retained. PPTX schema validation passed.

### GPU deck expansion: comparison graphs and exact capture flow (2026-11-16)
User requested comparison graphs and real prompt/token/access counts. Expanded the same PPTX to six slides: retained hardware/read-write table, added editable hardware read/write-volume and native-latency comparison charts at32k, added prompt-to-trace flow and four-config exact counts, then retained provisional proposal. Source measurements are results/measure summary JSON; native timings remain independent graph-enabled runs. V10032k x13 per-launch TSV sums independently agree with job124562 totals. A16 counts come from job124966 log. Exact total DRAM records: V100 FAon229103229, FAoff394172275; A16 FAon193126767, FAoff541639704. These are modeled32B records, not measured DRAM commands. Captured warp memory instructions and coalesced LOAD sectors have different units; active-lane total and coalesced-store total were not logged.

Prompt reconstruction: retrieved first116000bytes of current cluster results/sessions/corpus.txt; historical logs give31001tokens. Exported as gpu_32k_prompt.txt beside deck, SHA256 a611e6fb18f85bbca79a1a33aef9b1c081da9874c8ec08c73c6f64a3a54a9385. No historical prompt hash was retained, so current corpus prefix cannot be hash-verified against original capture. Prompt excerpt starts Android/Build GUI binding using Android Studio. Short prompt is The capital of France is (5tokens);8k corpus-prefix prompt is6980tokens. Generation limit was3tokens; logs show2decode evaluations; final response text/token IDs were filtered out by capture wrapper, so final response-token count is explicitly unavailable, not fabricated. One selected decode step is traced; these totals exclude prefill/model loading. Exported exact32k count data as gpu_trace_counts.csv. PPTX schema checks passed; rendered comparison/flow slides and corrected wrapping.
Follow-up verification: read A16 node05-local per-launch TSVs via a one-minute CPU-only allocation. Unique launch IDs in42987–43524 sum to208301617L2reads/192881465DRAMreads/245302writes;56679–57244 sum to791411897/535292279/6347425. Both agree exactly with historical job124966 logs and the delivered table. Final graph/flow slides rendered and visually checked after label corrections;6-slide deck validated.

### GPU slide correction: actual short prompt (2026-11-16)
User clarified that the prompt must be short enough to fit completely on the slide. Replaced slide5 long-corpus example with actual short-run prompt "The capital of France is" (5 prefill tokens), and replaced all four configurations with their MATCHING short-capture counts from jobs124965/124966 and124561. V100FAon160033156warp instructions,1075696720coalesced load sectors,137213135DRAMread+219930write=137433065records. V100FAoff161664660/1081845632/137213403+224115=137437518. A16FAon159819348/1073136064/137245495+219286=137464781. A16FAoff161664660/1081845632/137189896+246450=137436346. Exported gpu_short_trace_counts.csv. Previous32k graph slide remains explicitly labeled32k; short flow is labeled short prompt. Response evidence limitation retained. Six-slide PPTX validated.

### Single-experiment slide and recovered output (2026-11-16)
User requested response string/token count and removal of the comparison table from slide5. Archived previous deck as outputs/01a1e00e-fd93-7ee0-b738-e23370cdfcd4/reference/amcas_gpu_with_comparison_table.pptx and copied short count CSV there. Revised slide5 entirely around V100/7B/FAon short-prompt capture, keeping exact matching counts and showing5prefill tokens,3generated tokens,1selected decode-step trace (538launches).
Correction to earlier missing-output statement: NVBit capture wrapper discarded response text, but the matching Nsight short-run raw output DID retain it. cluster-data/results/gpu/q7_v100_short_faauto_ncu.raw line19 shows "The capital of France is Paris. The"; suffix " Paris. The". The same log records n_predict3,5prompt tokens,8tokens in sampler accounting and2decode evaluations, supporting3generated tokens. Slide visibly labels that output is recovered from the matching Nsight run and memory counts from NVBit, rather than claiming same invocation. Response token count3 and traced decode step count1 are separate quantities. Earlier statements that response was unavailable anywhere were too broad. PPTX validated; prior comparison retained for future use.

### Slides3/4 switched to short-prompt measurements (2026-11-16)
User requested short-prompt data on slides3/4. Replaced32k read/write table and all3comparison charts with exact results/measure/q7_{v100,a16}_short_fa{auto,off}.summary.json counter values; native timing V1007.81/8.47ms,A1638.11/40.11ms FAon/off. ReadGB rounds4.41/4.41V100,5.25/5.25A16; writeGB1.28/1.35V100,0.16/0.16A16. Updated conclusions: short-prompt read traffic nearly unchanged;FA native latency reductions7.8%V100/5.0%A16. Both slides now explicitly show the5token prompt. Hardware specifications retained; source notes switched to short results. Slide6 remains explicitly32k mapping evidence. Six-slide deck validated and rendered.

### Presentation reframed around one short V100 pilot (2026-11-16)
User requested archival of last-slide proposal, a future-work closing slide, V100-only hardware and space for algorithm explanation, and moving previousslide5 between2and3. Saved fullpreviousdeck as reference/amcas_gpu_before_next_steps.pptx and original proposal slide source/data/notes as reference/archived_mapping_policy_slide_source.txt beside output. Final order:1problem,2workflow,3singleexperimentflow(previous5),4V100hardware+coalescing/cachealgorithm(previous3),5V100-onlyFAcomparison(previous4),6pilotfindings+nextsteps. Removed A16 from hardware and visible graphs to focus the narrative. Explained active-lane/access-width32Bsector deduplication,per-SMsector-validLRU L1load-hit suppression,sharedhashedLRUL2,loadsector miss→DRAMread,dirtyeviction→write,L1write-through/L2dirty stores,explicitapproximationlimits. Closing slide separates short-prompt flow/findings from planned systematic8k/32k,prefill/multi-step,validatedMoEexpert-routing/multiGPU,readANDwritecountercalibration,andfulltokenDRAMsim/Ramulatorpolicy/mappingvalidation. Existing longcontextpilot results are preserved; future phrasing means systematic validation campaign, not denial of earlier runs. Deck validated; new layouts rendered/inspected. Native V100 timechart rescaled0–10ms; graph caption now refers back toslide3.

### Cache filter audit and controlled repair pilot (2026-11-17)
User authorized full pipeline audit, quickest defensible repair or Accel-Sim migration, shortV100experiment andGitHubpublication. Found missing kernel-boundary L1invalidation, stream discards cacheoperators/atomicsemantics/actualSM IDs, modulo-only dynamicSMplaceholder, whole-sector write-validation for partialstores, cold-window/ordering/physicalplacement limitations. Dirty residentendstate is a measurement-boundary choice; blindly flushing is not hardware validation. Created isolated cluster results/gpu/cache_audit_20261117 with originalsource, opt-in boundaryinvalidation variant, audit.md and run.sh. Submitted126566 onnode04CPU2/mem8G using retained42GB rawshortFAontrace, comparing original/boundary/L1bypass. Synthetic cross-kernel read test must pass before fullrun; originalcountreproduction is enforced. Exittrap publishesresults through existingpublish.sh. OfficialAccelSimsource cloning to tracing/accel-sim-audit-20261117 for pinnedcompatibilityaudit. Memory-onlytrace cannot supply instructiondependencies needed byAccelSim; no speed claim orfullvalidationclaim yet.

### Controlled cache audit results (2026-11-17)
126566 completed successfully; originalcounts reproducedexactly. Threevariant onepass42GB replay elapsed8m55.78,115.61suser17.86ssystem,93.7MBpeakRSS,24%CPUindicatingI/O/wait overhead. OriginalL2reads146095769/DRAMreads137213135/writes219930. Boundaryinvalidation147631773/137220351/219690;L1bypass1075696720/137220686/213689. BoundaryDRAMreads increase7216(~0.00526%);thus missingkernelL1invalidation has tinyreadimpact inTHISshortcase,notproofothererrorsharmless. Job auto-published9a569d8. AccelSim126567node05failedmissingCUDA;correctednode04job126574setup succeeded butbuildfailedmissingmakedepend(Error127). Pinnedframework/GPGPUSimsource auditpublished e684469,builddiagnostics00804c3andlaterautopublish. AccelSim compatibleinprinciplewithV100,butnoapplicationrunorspeedcomparisoncompleted;needsfreshSASStraceandDRAMexportintegration. Partialsectorstores/cacheoperatorloss/atomics/actualSMremainunresolved;isolatedboundaryfixnotfullyvalidatedreplacement.

### Accel-Sim proper build retry (2026-11-17)
User requested another properAccelSimattempt and clarificationofvariants. Boundaryvariant clearsprivateL1onlaunchchange whilekeepingL2warm; bypassvariantdisablesL1. Samecapturedtrace,notnewinference. RetriedofficialCMakepath126599 onnode04;gotpastmakedependbutCMakeunconditionallyrequiresOpenGLdevelopmentfiles. Createdisolateddependencybuild126606onnode04 withmicromamba prefixtracing/accel-build-deps,packagesxorg-makedepend/libgl-devel/libglx-devel;thenofficialsetup+CMakebuild. Snapshotresults/gpu/cache_audit_20261117/accel_deps_build.sh,logsameprefix;autopublishesonexit. As ofthisentrysolverstarted,noapplicationtraceorsimulationyet. Next:inspect126606dependency/buildresult,resolveconcreteerrors,buildNVBit1.8tracerthenboundedvalidationbeforellamashortcapture. Preservealloldresults.
AccelSimretry126606reached99%compilationafterisolateddepsinstallation;defaultalltargetfailedlinkingunrelatedlibcudartshim(-lGL). Retryingstandalonetracesimulatortargetaccel-sim.out instead;tracerbuildsubmittedindependently(theoldafterok126607cannotrunafterfailedalltarget). Scriptsaccel_target_build.sh/accel_tracer_retry.sh inrunfolder,autopublishlogs. Thisisbuildprogress,notcompletedinferenceorsimulation.

### Accel-Sim build, microbenchmark and fresh capture (2026-11-17)
Delegated continuation: fixed actual linker search paths with isolated libGL development prefix and -L/rpath. Official simulator target built successfully in126620 (published24612c1); officialNVBit1.8 sm70tracer/postprocessor built126609 (d0fa773). Tinyknown-output CUDAkernel traced/postprocessed/replayed under SM7_GV100 in126621, COMPLETED0:0 in2s. Added optional AMCAS_DRAM_EXPORT hook at dram_t::push (controlleradmission), fieldssequence/globalcorecycle/partition/address/write/bytes/accesstype, no cachedecisionchanges; built126624,published00714d66. Nonzero-readmicrobenchmark126631 validates nativeoutput and4CSVreadrecords×32B against sumcontroller n_req4. This establishes exporter plumbing, not workloadaccuracy.
Fresh officialSASS capture126622 runs Qwen2.5-7B Q4_K_M prompt The capital of France is,-n3,greedy,seed1,FAauto,graphsdisabled ononeV100. Officialtracer1-based kernelrange1296–1833 corresponds to old0-based1295–1832;538kernelwindowcheck enforced in dependentreplay126630. Officialtracesincludeallinstructions/registerdependencies and optionalcoreIDs; oldmemory-onlytracesareincompatible. Fullstdout/runtime saved, raws to /scratch/$USER/accel-q7short-20261117. At~3min capture had~263MB compressed,443GBscratchfree; can take tensofminutes. Sources/scripts/export schema/results in cluster results/gpu/cache_audit_20261117; source snapshots published automatically. Parent authorized PPT update only once actualnewresultsvalidated; none yet. Currentdeck remains theoriginalpilotstory.

### Real Qwen kernel replay and format repair (2026-11-17)
Representative Qwenkernel1298 (matvec) exposed stockTOOL_TRACE_CORE=1 postprocessor incompatibility:2SM/warpidcolumns were left ininstructionpayload, shiftingPC/mask/registerfields andassertingMAX_DST. FixedpostprocessorwithexplicitAMCAS_TRACE_CORE/per-rundotmarker opt-in; rawhardwarecoreIDs remainpreserved. Afterfix,126717COMPLETED,96.23swall/~949648simulatedthreadinst/sec. Controllerrequestexport226366reads=7243712bytes,0writes reconcilesexactly withsumcontroller n_req. MatchingoldNsightkernelID2readbytes7247552 (~0.05%difference),writes1479104; writeagreementNOTestablished. Dirtyresidentcacheboundaries/profilerreplay/omittedoldlocaltraffic needinvestigation. Anapplication-replay NsightcheckispreparedtorunonlyaftercaptureGPUworkfinishesduringCPUpostprocess,avoidingoverlap.
Fullfreshcapture126622healthyat~65min,kernel1597(~302/538),~5GBcompressedandhundredsofGBfree. Fullreplay126630 andfinalcountreconciliation126640 arequeueddependencies. Fullstepremainsunvalidated;PPTdataunchanged. Isolatedsourcepatches,testconfigs,andreal-kernelvalidationpublished9eaa280andnextautopublication. Thisrouteismuchslowerthancustomfilter;nospeedadvantageclaimed.

### Controlled V100 hardware counter correction and full replay restart (2026-11-17)
Fresh official SASS capture completed successfully. All 538 selected kernels match the original capture's names and launch geometries. Native output and timing logs are retained in cluster results/gpu/cache_audit_20261117. Initial full replay126630 stopped in its tiny preflight: the original store-only test had zero DRAM admissions, so the opt-in exporter never opened dram.csv. This was a fixture/validator error, not a model-trace failure. Replaced it with the already validated nonzero-read fixture and restarted simulation126821, reconciliation126822, and trace-release upload126823. Current full model data is still pending.

Four independent two-counter Nsight runs completed successfully with the expected 538 FA-on / 566 FA-off kernels. Application replay FA-on: 4,394,874,880 read bytes and 9,745,664 write bytes; FA-off: 4,395,132,384 / 10,352,096. Kernel replay with the same two counters: FA-on 4,395,018,432 / 9,783,648; FA-off 4,395,084,384 / 10,513,216. Reads agree within 0.0033%; writes within 1.56%. The earlier larger-metric-set measurements of 1.282 / 1.352 GB writes were not reproduced. Their exact discrepancy cause is not established; do not attribute it solely to replay mode. Supersede those write volumes with the controlled application-replay reference, preserving old results. Hardware controls and JSON published in e9e2a37.

Updated the PowerPoint's V100 comparison graphs with these cross-checked hardware values: reads round to 4.395 GB for both settings; writes are 9.75 / 10.35 MB. Native latency remains the separate earlier measurement of 7.81 / 8.47 ms. Archived the prior deck under reference/amcas_gpu_before_two_metric_validation.pptx. Added detailed source/method notes, retained original custom-filter counts while full Accel-Sim replay is pending, and passed PPTX schema validation plus rendered-slide inspection. No unvalidated full-model values were inserted.

### Exact pre-coalescing reference counts recovered (2026-11-17)
CPU-only job126829 scanned the retained 44,468,789,584-byte short V100 raw trace without rerunning inference. It counted 5,075,923,776 active-lane load references and 2,573,840 store references: 5,078,497,616 total. Coalescing yields 1,075,696,720 load sectors plus 1,185,698 store sectors: 1,076,882,418 total. The 158,939,256 warp-load and 1,093,900 warp-store instructions sum to the prior 160,033,156 baseline; existing load-sector counts also match exactly. There are538 launch markers. The parser checks complete input consumption and encoded widths; a ten-address fixture checks deduplication, a cross-sector store, and marker handling. Source/test/results published4354be3.

Updated slide3's original-prototype flow to use active-thread references → total coalesced sectors → modeled DRAM records, rather than mixing warp-instruction count with load-only sector count. Numbers are5,078,497,616 →1,076,882,418 →137,433,065. Coverage is global/generic accesses; local/shared/constant traffic was excluded by the old tracer. These are event counts, not byte volumes, and vector-width/transfer-granularity effects remain distinct. Full Accel-Sim output is still pending; it was not inserted. Deck schema and rendered-slide checks passed.
Fresh instrumented output validation now also passes directly on the official SASS capture, rather than relying only on a matching profiling run: stripping tracer diagnostic lines reconstructs "The capital of France is Paris. The". Perf accounting is5prompt tokens and8sampler tokens, supporting3generated tokens;2decode evaluations were executed and1selected step traced. Saved capture_output_validation.json. Final model acceptance now requires538launched/completed kernelUIDs, exactcontroller/export reconciliation, and this output/token check. Full replay remains active.
Trace-release uploader invocation was corrected to pass the GitHub token only through stdin, as required by repository logistics. Replaced only our own pending upload126823 with corrected dependency126858 (verify currentID in cluster results/gpu/cache_audit_20261117/retry_jobs.txt; this line's numeric value must follow actual submission output). Simulation126821 and reconciliation126822 remain unchanged. No secrets were printed or placed in arguments.
Correction: the newly submitted upload job is **126832**, confirmed by sbatch output and retry_jobs.txt. The preceding entry's126858 was a transcription error and is not part of this pipeline. Current chain:126821simulation →126822reconciliation →126832upload.

### Completed Accel-Sim result verified live (2026-11-17)
Live check:126821/126822/126832finished; simulationexit0; all538kernels completed, exportedcontrollerrequests137288021reconcileexactly. Read137233077requests/4391458464bytes;write54944requests/7028672bytes(variable32/64/96/128B). ControlledNsightFAonreference4394874880readbytes/9745664writebytes => modelread~0.078%lower,write~27.9%lower. Therefore internallyvalidatedexportandstrongaggregate-readagreement,butwriteaccuracyincomplete,nottiming/hardwareaddressvalidation. ColdselectedstepV100preset. Finalsummarypublished96ed5a9;releaseasset+metadatapublished8880de5;assetaccelsim-v100-q7short-dram-admissions-20261117.csv.gz1.383GB,SHA2560e62a6e750c412a5d3bcdd8a69f6327f75a0f25cf46f879100057ded8e046a78. Fullmodeldataavailable;PPTmustretainqualificationsifupdated.

### Authorized GPU campaign and latency-classification validation (2026-11-17)
User explicitly authorized latency-distribution study alongside fullCPU-equivalentGPUcampaign conditional onAccelSimreputation. Primaryresearchconfirmation: AccelSimISCA2020,https://people.ece.ubc.ca/aamodt/publications/papers/accelsim.isca2020.pdf andIEEE9138922;counter-by-counterGPUvalidationinclVolta,sectorcache/coalescing/subsectorwritepolicymodels. Reputablearchitecture-researchframework;customcontrolleradmissionexportstilloursandmustbevalidatedseparately;notexacthardwaretraceguarantee. Resumedagentaccelsim_v100forGPUcampaign+nativeV100latencydistributioncalibration/heldoutclassification. CPUmatrixdiscovered12mappings(identity,xor_global,region,chan,xor_global_chan,mop,mopxor,bitrev,chra,naive,expert_hash,expert_lin),10realpolicies(Open,ClosedCAP,Timeout16/64/256,Region,HAPPY,HAPPYcnt,Adaptive,Adaptive7_4),Oracleboundseparate. MusttranslateCPU64B/DDRassumptionsforGPUvariable32–128Brequests/HBM2;markunsupportedtensor/expertmappingsexplicitly. Initialtasksize-preservingexistingexportreplay+policy/mappingvalidation,writegapinvestigation,andlatencycalibrationbeforehugefreshSASScampaign. Fullmodel/context/batchcoverageexpandedonlywithcompatibility/scratch/runtimegates;noassumptionallCPUmodelsfitV100. AgentcurrentlyrestoringiiitVPNafterSSH DNSfailure;continuingautonomously.

### Accel-Sim export conversion and GPU policy campaign (2026-11-17)
Conversion job127120 passed:137288021controllerpackets preserve4391458464readbytes/7028672writebytes and yield137452723canonical32Bbursts. Source trace-pipeline/gpu/accelsim/export_sectors.cc validates sequence,monotonicglobalcorecycles,32Balignment,payload32–128B,andwithin128Bline. CSV lacksdirty-sectormasks: contiguousoffsetsforvariablewritesareexplicitcanonicalconvention,notreconstructedhardwarebytepositions. Row-onlybank/channelhashesandbitreversalabovebankfieldpreservelowcolumnbits; additionalmappersrequireHBMdecoderreview. TimestampsretainedinoriginalCSVbutnotusedinsaturatedreplay.
Scheduled127121 onnode04CPU6/mem48G:54combination1000burstpreflight then162combinationbegin/middle/end100kand1mwindowgrids. Sixtranslatedmappingsidentity,xor_global,naive,bitrev,chan,xor_global_chan;9policiesOpen,ClosedCAP,Timeout16/64/256,HAPPY,HAPPYcnt,Adaptive,Adaptive7_4. Usesramulator2-grid'sHBM12/HBM2_2Gb/HBM2_1600Mbps,tCK1.25ns,32controllers,256Bchannelinterleave,32Bbursts. Firsttouch2MiBplacementfrozenacrossfullstreambeforewindowselection;hashestrainedonfirst1mburstsandfrozenforheldoutmiddle/end. Region/expertvariantsneedfreshsemanticattribution;MOP/ChRamappersneedHBM-specificauditbeforeadmission. Thesearecoldwindowopen-loopDRAMmodelcomparisons,nottoken-timeperformance. Sources/resultsinresults/gpu/accelsim_campaign_20261117,autopublishonexit. Independentlatencycalibrationagentactive;fullwritegapremainsunresolved.

### Independent native V100 latency calibration submitted (2026-11-17)
A one-minute node04 preflight compiled and executed the dependent-pointer CUDA microbenchmark successfully on Tesla V100-PCIE-16GB (80 SMs, 6 MiB L2). Host validation of final dependent pointers passed for L1-hot, L2-hot, DRAM-cold and timer-floor controls. Job 127124 performs eight held-out-by-seed/allocation runs, 8192 samples per regime: 8 KiB ld.global.ca with in-kernel warmup; 1 MiB ld.global.cg with preparation; 64 MiB randomized ld.global.cg with reuse distance exceeding L2. Ending timestamp depends on the loaded pointer, and actual SM and effective clock are recorded. Separate Nsight application-replay counter groups check aggregate intended regimes. Sources and raw results live at results/gpu/latency_validation_20261117 on the cluster; publication is scheduled inside the job. This calibrates controlled latency distributions, not exact cache-level truth or an ordered hardware LLM DRAM trace. Agent will verify results before claiming classifier accuracy.

### Native V100 latency calibration completed and published (2026-11-17)
Job 127124 completed successfully in34seconds. Each regime has65536 unprofiled samples across8seeds; raw median cycles are timer floor10, L1-hot38, L2-hot230, DRAM-cold428; DRAM p99645. Four training seeds and four held-out seeds use distinct live allocations. Held-out intended-label hard-classifier accuracy is100%; conservative unique p1–p99 bands accept99.1597% with100% accepted accuracy (826of98304held-out samples abstain). These are controlled microbenchmark regime labels, not per-access measured cache-level truth. All dependent pointer outputs validated. Separate Nsight groups show8192measured L1hits after64warmupmisses; intendedL2hot8192hits/0misses; intendedDRAMcold0L2hits. However DRAMcold L2lookupmiss counter16384 exceeds8192loadsectors and DRAMreads530048B exceed naive262144B, so exact per-load/controller-count reconciliation remains unresolved. Counter semantics, translation and hardwarefetchgranularity must be investigated before claiming exactDRAMlabels. Raw samples, source, SASS, hardwaremetadata, confusion matrices and reports published in cluster-data/results/gpu/latency_validation_20261117; initialcommit59b2711. Counter caveats are saved separately in counter_validation.md as well as report.md. No LLM classification or presentation claims were changed.

### Replay preflight rejection and isolated drain repair (2026-11-17)
Cancelled onlyour127121 aftertiny1000readpreflightrevealedLoadStoreTraceendsatlastSEND,notcontrollerdrain;32cycleoutputandpartialrowaccountingwererejected,notacceptedfindings. Installedlegacygpu_replayusesthisfrontend:historicaltail-boundarytimingsneedqualification. Isolatedcopytracing/ramulator2-gpu-drainbuilt127241;frontendstopssendingatEOF,tracksreadANDwritecallbacks(includingforwarded/coalescedrequests),requiresallcompleteplus4096frontendtickguard(128memorycyclesatissue32). Pilotassertsacceptedinputcountandserved+forwarded/coalescedcounts. Guardincludedinreportedcyclesexplicitly. Existinginstallationsunchanged. Correctedreplay127242scheduledafterbuild.
Alsofoundsource-confirmedchannelplacementmismatch:GPUofflineLayoutused6+ib,butCacheLineInterleaveusestx_offset+ib,andHBM2Pythonpayloadis32B(tx_offset5). Thuslegacyib2actually128BinRamulatorwhileofflinehelperassumed256B. NewpilotusesRamulatorinterleave_bits3withofflinebitshift8fortrue256B;metadatarecordsboth. Needauditotherhistoricalfollowupcopiesbeforeclaimingextent. Currentcanonicalwriteoffsetconventionremainslimitedtodecoderspreservingwithin128Bbank/row/channelinvariance. Thisdoesnotfixorvalidateaggregatehardwarewritegap.

### Full GPU campaign explicitly resumed (2026-11-20)
User asks whether experiments complete and ordersremainingexecution. FullcampaignNOTcomplete;knownpublishedresultsremainshortV100pilot. Resumedaccelsim_v100agentwithconcreteinstructiontoenumerate19CPUworkloadsandcompleteGPU-equivalentcapturereplaycampaign,notstopatpilotstatus. Agentownsclusterjobmutations toavoidduplicates. SSHcurrentlyconnectsbutshellcommandsstallbeforeoutput;cycledexistingiiitVPNsuccessfully,testingServerAliveboundedconnection. No new submissions confirmed atthisentry. Agentcontinuesconnectiondiagnosis,coverage manifest,stagedcensus/capture/replay/publication. Preservevalidations,writegapqualificationandGPUmemory/runtimeconstraints.

### GPU workload campaign resumption and SSH blocker (2026-11-20)
User explicitly requested completing remaining experiments. ResumedGPUagent; enumerated20CPUworkloads fromgrid_report.CLASS, includingq30_pf32. Createdlocaltrace-pipeline/gpu/campaign_20261120/coverage.json separatingCPUworkloads,GPUadmission/nativefitgates,mapping/policygaps andactualcompletion(notclaimed). PreparedGPUportofCPUdrbatchdriver preservingprompt/token/graphselectionlogic,fullGPUoffloadrequest,tensordump,andCUDAprofilerselectedgraphcontrol;notcompiledorvalidatedyet.
PinnedofficialNVBittracerdocumentationadvertisesACTIVE_FROM_START=0ProfilerStart/Stop,butowningsourceenter_kernel_launchunconditionallyresetsactive_region=falseinthatmode. Preparedisolatedpatchtoleaveprofiler-ownedstateunchangedwhenactive_from_startfalse,andthree-kernelfixture(outside/insideROI/outside)requiringonlyinsideinstructionsrecorded. Requiresbuildandexplicitfixturebeforefreshcaptures;existingrange-basedshortpilotunaffected.
ClusterSSHcurrentlyauthenticates/bannerthenkeepalivetimesout;multiplehostname/directIPattemptsfailed,includingafteriiitVPNreconnect. DirectIPtestusedknownHostKeyAlias,withoutbypassinghostverification. NoNov20jobsweresubmittedandnolivestatusclaimcanbemadeuntilremotecommandaccessreturns. Continuedlocalpreparationandrecoveredexacthistoricalcapture/replayfilesfromGitHubconnector. Do notconflatelocalpreparationwithclusterexecution.

### HPC slow login recovered (2026-11-20)
User requested waiting up to10minutes rather than abandoningSSHearly. Command timeout600 ssh -F/dev/null BatchMode=yes ConnectTimeout600 ServerAliveInterval60 ServerAliveCountMax10 succeeded afterroughly2minutes,exit0;hostrce.iiit.ac.in,squeueempty. Earlier30secondtimeout diagnostics didnotestablishpersistentserverfailure;loginisveryslow. Resumedaccelsim_v100agentforprepared20workloadcampaignsync,build/ROIpreflightandactualsubmissions. UsegenerousSSHtimeoutandallowauthenticationdelaygoingforward.

### Slow SSH recovered and full workload campaign running (2026-11-20)
Correction: cluster authentication eventually succeeds when allowedseveralminutes. Previous30skeepalive diagnosis waspremature; usercorrectlyidentifiedslowlogin. Using600sconnectiontimeout,60skeepalive/10retries,andpersistentSSHcontrolsocket. Nohostverificationbypass. Syncedcampaignsourcesdirectlytoresults/gpu/campaign_20261120. GPUdriverbuild131037failedCUDA-driverlink;131039fixedstubdriverlink;final131044alsoloadsCUDAbackends explicitlyandvalidatesfullGPUoffload. ProfilerROIgate131038actuallyrecordedcorrectonlyinsidekernel,butvalidatorincorrectlyreadbinarytracezasUTF8;fixedheaderreadand131040PASSED(actualthree-kernelfixtureonlykernel2insidecaptured,nativeoutputverified).
Native40GBallocation131045wouldwaituntilNov22becauseotherusersreserve160GB/191GBnode04memory. Replacedonlypendingourjobswith131047dense(1V100,16GB),131048MoE(2V100,24GB,afterdense),and131049fresh0.5Bcapture(afterdense). Dense131047finished3m32s:all11casesPASSEDnativeincluding7B1k/8k/32k/secondtoken,B16P512,B4P8192,B4P32768,30tokenprefill,512tokenub32prefill,and0.5Bshort/32k. Confirmed25/25or29/29layersGPU-offloaded;recordsactualprompt/generatedtokenIDs,steps,tensors,prompthashes. DensecasevisibilityrestrictedtooneGPUevenwhenparentallocationhastwo. MoE131048nowrunning;freshcapture131049pendingresources. Nativewhole-invocationwallincludesmodelsetupand5spollquantization;notreportedastokenlatency.
CPUprovenancecorrection:q7_32k_t2meansSECONDtoken(source_s2n1t4),not2CPUthreads. SavedCPUinputs:c1k4000chars(actual961tokens),c8k32000chars,c32k116000chars,pf32is120chars/30tokens. q7_ub32isP512/u32lastgraph. Actual0.5Bmodelisqwen25-05b-q4_0.gguf.20coverageentriesincludeq30_B4P32768whichhasNOcompletedCPUgridfolder;recordthisdifferenceexplicitly.
Optionaldirty-sectorCSVexportbuilt131050withoutchangingdefaultlegacy7columnschema:AMCAS_DRAM_EXPORT_MASKSaddsactualmem_fetchsector_mask. Maskfixture131051PASSED32controllerpackets(16read/16write),exactcounterreconciliation,maskpopcount/bytechecksandnoncontiguousmask5→offset0/64test. Test-onlyminiaturewrite-throughcache,notV100workloaddata. Publishedb0eeb8b;densefeasibilitypublication0fd40a0. Freshmaskedexportswillavoidoldcanonicalwriteoffsetlimitationoncefullworkloadmaskchecksalsopass. Hardwarewritegap27.9%stillunresolved;notclaimgpuvalidationcomplete.

### NVBit sampled load-latency observer calibration submitted (2026-11-20)
Independent observer job 131095 uses an isolated tool at results/gpu/latency_llm_20261120, one node04 GPU, 1 GiB requested memory and a 12-minute cap. It pairs BEFORE/AFTER instrumentation on LDG instructions and passes the actual loaded destination register into the AFTER callback to enforce load-result readiness. Stores, atomics and LDGSTS are excluded; samples are deliberately limited to CTA0/thread0. A minute-bounded injection preflight precedes matched native/instrumented eight-seed L1/L2/DRAM intended-regime calibration. Observer time includes NVBit argument, trampoline, register-save/restore and scratch-buffer overhead; thresholds from the previous native-only calibration will not be reused. No LLM per-access cache classification is claimed until this observer demonstrates useful held-out discrimination. The parent campaign jobs remain untouched.

### main session: GPU campaign recovery, node03 replays (2026-11-22)

Overnight 21-22 Nov failures: all 11 Accel-Sim replay jobs (131083..132446) exited 1 within minutes because the in-script `sbatch` for the Nsight hardware job hit a Slurm "Socket timed out" under `set -e`. The Nsight jobs were created anyway (132742-132752), so their IDs were written to each `runs/<case>/hardware_job.txt` to avoid duplicates. Capture tasks q7_B4P32768 (131056_7), q7_pf32 (131056_8) and q7_ub32 (131056_9) stopped at run_case.py's 6 h AMCAS_CAPTURE_SECONDS limit (passed=false). MoE (Qwen3-30B, 2 V100) captures q30_short, q30_1k, q30_n4 segfaulted (exit -11) under the NVBit tracer after native gates passed; q30_32k hit the 6 h limit. gem5-workload captures 131089_6/7 ran out of memory at 5G.

Actions (user-approved): deleted the failed partial traces for q7_B4P32768 and q7_pf32 on node04 /scratch (they would mix into a new trace and count against the 80 GiB cap); held MoE array 131057 tasks 5-8 (task 4 already running); resubmitted dense captures 7,8 as 132993 and 9 (ub32) as 133005 on node04 V100s with AMCAS_CAPTURE_SECONDS=72000 and -t 22:00:00, all else unchanged. ub32 partial trace (8.4 GB) not deleted before resubmission.

Replays moved to node03: node04 memory is queued for another user's 64 GB jobs, so CPU-only replays could not start there. New `simulate_case_node03.sh` differs from `simulate_case.sh` only by node, an rsync of the capture trace from node04:/scratch to the same path on node03 (file count and byte total compared, logged to runs/<case>/trace_copy_node03.txt) and the upload job pinned to node03. Accel-Sim binary, V100 configs (cache_audit_20261117/preflight), case JSON and Python steps unchanged; Accel-Sim output does not depend on host CPU. Jobs 133007-133019 (q7_1k, q7_8k, q7_32k, q7_32k_t2, q05_32k, q7_B16P512, q7_B4P8192, 4 stories15M, 2 gem5 q05) all RUNNING on node03. node05's 8 GPUs are A16 (Ampere), not usable for captures that replay on Accel-Sim's V100 model.

### main session: MoE two-GPU tracer segfault, root cause and fix (2026-11-22)

All four MoE Accel-Sim captures (q30_short, q30_1k, q30_n4, q30_B4P512; Qwen3-30B split over two V100s) died with SIGSEGV right after `AMCAS_GRAPH_BEGIN step=1`, the first kernel inside the profiler window; native two-GPU runs pass and single-GPU dense captures trace fine. Cause, in tracer_tool-amcas20/tracer_tool.cu: one receiver thread is created in nvbit_tool_init with the first CUDA context and writes every record to `ctx_resultsFile[<that context>]`. A traced kernel on the second GPU opens its own per-context trace file, but its records go through the first context's FILE* (closed/never opened in the window), so fprintf crashes. Secondary: nvbit_tool_init can run per context (re-initialising the shared channel, a second receiver), and the first nvbit_at_ctx_term stops the receiver for all contexts. The channel buffer itself is pinned host memory and channel_dev is __managed__, so device-side pushes from both GPUs are fine.

Fix (`patch_tracer_mgpu.py`, isolated build `tracer_tool-amcas20mgpu`, ARCH=sm_70): receiver writes to the context of the kernel being traced (std::atomic set in enter_kernel_launch before recv_thread_receiving; traced kernels are serialised by leave_kernel_launch), tool init runs once, receiver stops when the last context ends. Single-context behaviour unchanged; dense captures keep using the unpatched tracer_tool-amcas20.so. run_case.py selects the mgpu tracer only when case gpu_devices==2 (backup run_case.py.bak_20261122). Validation job 133038 (`mgpu_check.sh`): rebuild, profiler-window fixture with the patched tracer, then the real q30_short capture with a check that both contexts have traced kernels and the output gate passes. Crashed partial capture directories are renamed, not deleted. capture_array.sh now queues replays through simulate_case_node03.sh (affects new submissions only).

### main session: capture resubmission error and fix (2026-11-22, 19:05 IST)

Error (main session): resubmissions 132993 (q7_B4P32768, q7_pf32) and 133005 (q7_ub32) were submitted without `--gres=gpu:1`; capture_array.sh has no gres in its header (131056 was submitted with it). llama.cpp reported "no CUDA-capable device", fell back to CPU, and the jobs died in 10-60 s (OOM / abort at first traced graph). These are not tracer or model failures. The 4 KB capture dirs were renamed `capture.failed_nogpu_202611221904`; resubmitted as 133237 (--array=7,8,9 --gres=gpu:1 -c 4 --mem=10G -t 22:00:00, AMCAS_CAPTURE_SECONDS=72000), matching 131056's resource request. Replay 133012 (q7_B16P512, 45 GB trace) was OOM-killed at 4 GB on node03; rerun as 133238 with --mem=16G, configuration otherwise unchanged. Completed node03 replays with policy grids: q05_32k and the five gem5 workloads (s15 short prefill/decode, s15 1024/4096 decode, q05 short/1024 decode). Multi-GPU tracer check 133038: rebuilt with ARCH=sm_70 and profiler-window fixture PASS; real q30_short capture running past the point where the unpatched tracer crashed.

### main session: MoE tracer fix validated; multi-context kernel lists (2026-11-22, 21:55 IST)

133038 (mgpu-check): profiler-window fixture with the patched tracer PASS; real two-V100 capture q30_short PASSED (exit 0, native/capture tokens match, 4571 s). Its final check reported FAIL only because it counted `kernelslist.g`, which holds the first context: post-traces-processing writes one list per context (kernelslist.g 535 kernels, kernelslist_ctx_0x55988075c100.g 575 kernels). Corrected result recorded in runs/q30_short/mgpu_check.json without recapturing. Slurm then cancelled the afterok MoE arrays 133241/133249 (DependencyNeverSatisfied); replaced by 133320 (--array=1-8%1, 2 GPUs, 22 GB, 22 h, AMCAS_CAPTURE_SECONDS=72000).

Same single-list assumption was in simulate_case.py (would have simulated only the first GPU) and hardware_case.py (Nsight kernel-count gate). `patch_multictx_lists.py` makes both read every kernelslist*.g except kernelslist_sim_*.g (replay inputs written into the trace dir); single-context results unchanged; backups *.bak_<ts>. simulate_case_node03.sh's copy check now also skips kernelslist_sim_*.g: rerun 133239 of q7_B16P512 had failed because node03 held one 37 KB kernelslist_sim file from the first attempt (1919 vs 1918 files). Resubmitted: sim-q30_short 133321 (node03, 4 GB), sim-q7_B16P512 133322 (node03, 16 GB). Dense captures 133237 tasks 7, 8 (with GPU) running on node04; task 9 pending.

### main session: replay time limits (2026-11-23, 00:20 IST)

At 14 h, kernels simulated of kernels traced: q7_1k 456/538, q7_8k 391/538, q7_32k 245/538, q7_32k_t2 247/538, q7_B4P8192 280/817; q7_B16P512 66/957 at 2.1 h. Decode runs the layers in order, so progress is about linear in kernels: q7_32k, q7_32k_t2, q7_B4P8192 and q7_B16P512 projected 30-41 h against a 24 h limit. Users cannot raise a running job's TimeLimit (partition MaxTime=UNLIMITED); Accel-Sim has no checkpoint. With user approval, cancelled 133009/133010/133013/133322 and resubmitted unchanged except -t 3-00:00:00: 133426 (q7_32k), 133427 (q7_32k_t2), 133428 (q7_B4P8192), 133429 (q7_B16P512, 16 GB). simulate_case_node03.sh default raised to 3 days for replays queued by future captures. q30_short replay 133321: first context 343/535 kernels at ~2.3 h, both contexts projected ~8 h.

### main session: overnight results and failures (2026-11-23, 08:40 IST)

From GitHub publications (VPN to RCE down since ~08:30: OpenVPN "TLS key negotiation failed to occur within 60 seconds", server 218.185.248.70:1194; no cluster commands possible). Passed: captures q7_pf32 (896 kernels, 7.3 h) and q7_B4P32768 (817 kernels, 9.6 h) with the 20 h limit; replays q7_1k (538 kernels, 18.2 h), q7_8k (538, 21.0 h) and MoE q30_short (two contexts 535 + 575 kernels, 5.0 + 4.2 h, passed). Failed: restarted replays 133426 (q7_32k) and 133427 (q7_32k_t2) OOM-killed at 4 GB about 4 h in (they had run 14 h at 4 GB before the restart); q7_pf32 replay 134005 died when its in-script Nsight sbatch hit a Slurm socket timeout (empty hardware_job.txt).

Regression found: the 00:20 copy of simulate_case_node03.sh from the laptop overwrote the cluster-only kernelslist_sim_* exclusion in its copy check. Local fix (not yet on the cluster): restore the exclusion, retry the Nsight sbatch four times and continue without it if it never submits, default --mem 16G. To do when VPN returns: copy the script, resubmit q7_32k, q7_32k_t2 (16 GB, 3 days) and q7_pf32.

q7_1k grid (analysis/gpu/latency.md): Ramulator best xor_global + Timeout64 0.94-0.95 cycles, xor_global + Open 0.96-0.99, mopxor + Open 1.15-1.17 (worse); DRAMsim3 mopxor + Open 0.87-0.95, rank correlation with Ramulator -0.14/-0.57. The two simulators disagree on mapping for 7B decode.

## 2026-11-23 — GPU: why DRAMsim3 and Ramulator disagree on mapping (diagnosis, no new jobs)

Data: q7_8k and q30_short policy grids (packed, 256 B and 2 KiB interleave, Open). Script: scratchpad `cabus.py` + inline Little's-law check; all inputs are the published `dramsim3.json.gz` / `results.json`.

- The disagreement is confined to the MOP4CLXOR family. For identity, xor_global, chan, xor_global_chan, naive, region and expert_* both simulators agree in direction and roughly in size (e.g. q7_8k 256 B xor_global: Ramulator 0.954, DRAMsim3 0.898). Only mop/mopxor flips: Ramulator 1.17–1.18 (q7) / 0.99–1.06 (q30), DRAMsim3 0.86–0.96 / 0.89–0.94. The negative Spearman rho comes from mop/mopxor ranks.
- Row hit rates agree across simulators for every mapping (identity 0.816 vs 0.827; mop 0.41 vs 0.50 on q7_8k). The simulators agree on row-buffer locality and disagree on what a miss costs.
- Ruled out: timings (DS3 ini matches Ramulator HBM2_1600Mbps: CL/RCD/RP 12, CCD_S 2, CCD_L 4, RRD 4, FAW 12); decode (MopView = Ramulator MOP4CLXOR incl. row^col XOR; DS3 decode validated per request); writes (q7_8k window is 3,990,805 reads / 9,195 writes, 0.23 %).
- Cause: controller queue depth under saturated replay. Throughput = requests in flight / latency (Little). DRAMsim3 (32 transactions + 16 banks x 8 commands per PC, 64 independent PCs) holds about 53–66 reads in flight per channel under identity and about 96–103 under mop (latency 82-88 -> 122-137 cycles), so MOP's extra misses are hidden by deeper queues and its bank/PC spreading wins. Ramulator HBM12 has read_buffer_size 32 per 2-PC controller (one row + one column command slot per cycle, shared by both PCs), so it cannot add in-flight requests and MOP's longer miss latency shows up as cycles (residency 71–75 -> 87–89 cycles on q7_8k).
- Secondary: under JEDEC HBM2 a 2-cycle ACT on a row bus shared by both PCs, DS3's mop schedule on q7_8k would need 1.13–1.15 row-bus cycles per cycle per PC pair (infeasible; lower bound 0.98x of identity at 256 B, 1.08x at 2 KiB). Ramulator counts ACT as one row slot, so this is not what slows Ramulator, but DS3's 64 independent command buses also overstate MOP.
- Calibration point: Accel-Sim's own SM7_QV100 config (the replay model) uses 32 channels with `-gpgpu_frfcfs_dram_sched_queue_size 64` per channel. Ramulator (32) is below it; DRAMsim3 (~100 in flight) is above it.
- Consequence: the mapping verdict depends on assumed controller queue depth; neither grid as run is calibrated to V100. Confirming run proposed (not submitted): Ramulator read_buffer_size 32/64/128 on q7_8k and q30_short, packed 256 B, identity/xor_global/mopxor, Open.

## 2026-11-23 — GPU queue-depth test (134623, node07): hypothesis refuted

`queue_depth.py`/`queue_depth.sh` rebuild the policy_grid traces (packed, 256 B, identity/xor_global/mopxor, Open) and replay Ramulator HBM12 with read_buffer_size 32/64/128. Depth 32 reproduces the published grid exactly (q30_short identity 256,464 cycles; q7_8k 292,187). Results in `cluster-data/results/gpu/campaign_20261120/queue_depth/`.

Cycle ratio vs identity at the same depth:

| case | depth | identity cycles | xor_global | mopxor (hit) |
|---|---|---|---|---|
| q7_8k | 32 | 292,187 | 0.954 | 1.169 (0.41) |
| q7_8k | 64 | 215,407 | 0.930 | 1.192 (0.44) |
| q7_8k | 128 | 176,709 | 0.916 | 1.177 (0.50) |
| q30_short | 32 | 256,464 | 0.990 | 1.049 (0.59) |
| q30_short | 64 | 191,281 | 0.991 | 1.059 (0.61) |
| q30_short | 128 | 159,761 | 1.002 | 1.047 (0.64) |

- Queue depth explains the absolute gap between the simulators (Ramulator identity at depth 128 is 176.7k cycles vs DRAMsim3 194.3k on q7_8k), not the mop/mopxor flip. mopxor stays 5–19 % worse than identity in Ramulator at every depth, including 64 (Accel-Sim V100 queue size). The previous section's "Cause: controller queue depth" is wrong for the ranking; keep it only as the explanation of absolute throughput.
- xor_global's gain over identity holds or grows with depth on q7_8k (0.954 -> 0.916); on q30_short it is about 1 % and vanishes at 128.
- Remaining candidate for the flip: DRAMsim3 models 64 independent pseudo-channels, each with its own row and column command bus; Ramulator HBM12 (and JEDEC HBM2) share one row and one column bus per channel across both PCs. DS3's mop schedule uses 0.75 row commands per cycle per PC pair (ACT counted once; 1.13 with the JEDEC two-cycle ACT). Not yet tested; the direct test is Ramulator with 64 single-PC controllers.
- DRAMsim3 command-bus check script kept as `trace-pipeline/gpu/campaign_20261120/ds3_cmdbus.py`.

Also today: node04 /scratch was 99 % full (24 GB free), so all 8 MoE captures 133320 aborted at the 90 GiB check. After re-verifying file count and bytes against node03, deleted node04 capture dirs for q7_32k, q7_32k_t2, q7_pf32, q7_B4P8192, q7_B16P512, q7_B4P32768 (log: runs/../node04_scratch_cleanup_20261123.txt); 291 GB free. Kept q7_1k, q7_8k, q30_short on node04: their node03 copies were pruned after upload, so node04 holds the only full copy. MoE resubmitted as 134627 (1-8%1, gpu:2, 22G, 22 h). q7_ub32 replay 134546 failed when rsync from node04 dropped at 5.2 GB; resubmitted as 134626 (rsync --partial resumes).

## 2026-11-23 — GPU pseudo-channel command-bus test (134672, node07): cause of the mop flip found

`pc_split.py`/`pc_split.sh`: same window, packed, 256 B, identity/xor_global/mopxor, Open. Each request's effective (channel, bank, row, col) from the mapping's view is re-encoded and replayed with RoBaRaCoCh in two Ramulator organisations with identical timings and matched buffer per channel pair D: shared = 32 HBM12 controllers x 2 PCs (grid model; one row + one column command bus per channel, as JEDEC HBM2 pseudo-channel mode); split = 64 controllers x 1 PC (each PC its own command buses and D/2 buffer, as DRAMsim3). Encoding check: shared D32 reproduces the grid exactly (q7_8k 292,187 / 0.954 / 1.169; q30_short 256,464 / 0.990 / 1.049), which also confirms MopView = Ramulator MOP4CLXOR. Results: `cluster-data/results/gpu/campaign_20261120/pc_split/`.

Cycle ratio vs identity, same organisation and depth:

| case | org | D | identity cycles | xor_global | mopxor |
|---|---|---|---|---|---|
| q7_8k | shared | 32 | 292,187 | 0.954 | 1.169 |
| q7_8k | shared | 64 | 214,937 | 0.926 | 1.194 |
| q7_8k | split | 32 | 387,463 | 0.938 | 0.779 |
| q7_8k | split | 64 | 261,416 | 0.923 | 0.818 |
| q30_short | shared | 32 | 256,464 | 0.990 | 1.049 |
| q30_short | shared | 64 | 189,182 | 0.995 | 1.071 |
| q30_short | split | 32 | 332,083 | 0.985 | 0.776 |
| q30_short | split | 64 | 221,900 | 0.997 | 0.862 |

- Splitting the pseudo-channels alone flips mopxor from 5–19 % worse to 14–22 % better inside Ramulator, reproducing DRAMsim3's direction (0.86–0.94). The simulator disagreement is the pseudo-channel model, not the mapping or the simulator: identity streams stay in one PC and suffer when PCs are independent (split identity is 25–33 % slower than shared), and mop's PC/bank spreading pays only when each PC has its own command bus.
- Real HBM2 pseudo-channel mode shares the channel's row/column command bus between the two PCs, so the shared (Ramulator grid) model is the faithful one for V100. DRAMsim3's HBM2 runs with channels = 64 overstate MOP4CLXOR. mopxor is not proposed for the GPU.
- xor_global beats identity in all 8 cells (0.92–0.997; 5–8 % on 7B decode, about 0.5–1.5 % on MoE), in both organisations, both depths and both simulators. GPU proposal candidate: xor_global mapping + Open (Ramulator adds a short timeout, Timeout64/256, worth a few % more).

## 2026-11-26 — node04 scratch full again; MoE captures 6-8 resubmitted

- 134627_6/7/8 (q30_B4P8192, q30_B4P32768, q30_pf32) aborted at run_case.py's 90 GiB free-scratch check: node04 /scratch (1.8 TB, shared; other users hold ~1.3 TB) had 71 GB free. Cause: every capture writes 15–92 GB to node04 and the pipeline never removes the node04 copy; replays copy it to node03, and upload_case.py prunes only node03's regenerable .tracez.
- With user approval, deleted node04 capture dirs of q7_1k, q7_8k, q30_short, q7_ub32, q30_32k, q30_B4P512 after a per-file check: every node04 file is identical on node03 or listed in retention.json as a regenerable .tracez (original trace kept on node03), and trace_assets.json validated_simulation is true. Log appended to node04_scratch_cleanup_20261123.txt. node04 free: 244 GB.
- Resubmitted as 140984 (--array=6-8%1, gpu:2, 22G, 22 h, AMCAS_CAPTURE_SECONDS=72000).
- Since 2026-11-24: completed replay + grid + upload for q7_B4P8192, q7_32k, q7_32k_t2, q7_pf32, q7_ub32, q30_32k, q30_B4P512; q30_n4 replay done (grids 140851/140852 queued); all Nsight jobs completed. 134150 q7_B4P32768 TIMEOUT at 3 days (about 520/817 kernels; trace kept on node03). q30_B16P512 replay 139898 running.

## 2026-11-26 — GPU: long-context decode is channel-interleave bound (new grids)

Report regenerated (eac5198) with q7_32k, q7_32k_t2, q7_B4P8192, q7_pf32, q7_ub32, q30_32k, q30_B4P512. Ramulator cycles vs identity + Open, packed, 256 B interleave unless stated:
- q7_32k / q7_32k_t2 (7B, 32k context): chan + Open 0.743–0.745, xor_global_chan + Open 0.742–0.744, DS3 chan 0.687–0.689 (both simulators agree; chan is a RoBaRaCoCh channel hash, not affected by the pseudo-channel artifact). 2 KiB interleave with identity: 0.640; 2 KiB xor_global_chan + Timeout256: 0.609. xor_global (bank hash only) 0.96–0.97.
- q7_B4P8192 (4 x 8k): chan 0.85–0.86 (DS3 0.83–0.84); 2 KiB identity 0.686; 2 KiB xor_global_chan + Timeout256 0.664.
- q7_8k: 2 KiB identity 1.013 (slightly worse than 256 B); 256 B xor_global + Timeout64 0.932 stays best. So the gain is specific to long-context KV streaming, where 256 B channel interleave concentrates traffic on few channels.
- Policy alone (identity + Timeout256) 0.99 on long contexts; the long-context gain is channel mapping, not page policy. Closed page 1.06–1.34 in every case.
- MoE (q30_1k, q30_32k, q30_B4P512): xor_global_chan / chan + Timeout256 best, 0.935–0.982; expert-aware mapping no gain beyond XOR hashes.
- Batched/prefill (q7_B16P512, q7_pf32, q7_ub32): DRAMsim3 mopxor 1.09–1.35 (worse); simulators agree on ranking.
- Proposal direction for GPU: channel-bit XOR hash (or coarser 2 KiB channel interleave) + bank XOR hash, Open page with Timeout256. To confirm: whether 2 KiB interleave hurts short-context cases (q7_8k 1.013, q05_short) enough to prefer the 256 B + chan hash.

Also: 140851/140852 (q30_n4 grids) moved to node01 (node03 memory fully reserved by other users); node04 memory is also fully reserved by other users, so MoE captures 140984 wait though both V100s are idle.

## 2026-11-26 evening: q30_n4 grids done; queue state
- 140851/140852 (q30_n4 policy grids, node01) and 140853 (upload) COMPLETED. Ramulator, vs identity+Open: best 0.962-0.977 (xor_global / xor_global_chan / expert_lin + Timeout256), hit 0.88-0.91, ClosedCAP 1.24-1.35. DS3 closed 2.6-2.8x. Same picture as the other MoE cases.
- gpu_latency_report.py now labels two-context (two-GPU) cases as case/<last 4 hex of ctx id>; latency.md regenerated (104 cells).
- 139898 q30_B16P512 replay: ctx_..a070 at 626 kernels after 22.3 h (~28/h); second context not started. Projected total ~64-73 h vs 72 h limit. Fallback: grid on the first context only.
- 140984 [6-8] MoE captures still PENDING (node04 memory reserved by others).

## 2026-11-27 11:50: MoE captures restarted with 3-day limits
- 140984_6 (q30_B4P8192) was still in NVBit prefill after 13.9 h, with no decode tokens captured. Slurm refuses user time extensions on a running job, and run_case.py's own stop (AMCAS_CAPTURE_SECONDS=72000) was fixed at submission. With the user's OK it was cancelled. Its partial scratch trace was renamed (not deleted) to node04:/scratch/.../q30_B4P8192/capture.partial_140984_6.
- Resubmitted with -t 3-00:00:00 and AMCAS_CAPTURE_SECONDS=255000: 141277_6 (q30_B4P8192, running on node04 since 11:50) and 141272_7/_8 (q30_B4P32768, q30_pf32). At the user's request 7/8 have no dependency or throttle and are scheduled independently. In practice they still run one at a time, because each needs both node04 V100s.

## 2026-11-27 17:40: GPU pipeline audit (why captures are slow, plus validity checks)
- **Capture slowness, measured:** the running q30_B4P8192 capture issues about 3110 kernel launches/min (~19 ms each). Its main thread and the NVBit receiver thread are each at 100% CPU, and both GPUs show 0% utilisation. That makes 841k launches in 5.5 h, about 44k per prefill graph (MoE plus quantize/convert kernels).
  - Cause: tracer_tool.cu enter/leave_kernel_launch does the full per-launch work even when active_region is false: cudaDeviceSynchronize before and after, the flush_channel<<<1,1>>> launch and sync, a pthread_yield spin on recv_thread_receiving, and two fopen/fclose of the stats file.
  - Across all finished cases, capture takes 430-2700x native time (e.g. q30_32k 65141 s vs 55 s native).
  - ptrace is blocked on node04 (yama ptrace_scope=3), so no stack samples were possible.
  - Proposed fix, not applied: early-return on inactive kernels (keep only the kernel-id increment). Validate by recapturing a finished case and checking that the traced kernel files and kernel lists are byte-identical.
- **Window coverage:** prepare_policy takes one 4M-sector (128 MB) window at 30% of each context's stream. That is 2-5% of the stream for 7B and 30B (q7_32k 2.0%, q30_n4 3.0%), i.e. roughly half of one 7B layer at 32k. The XOR/channel masks are fitted and evaluated on that same window (in-sample, as policy_grid.py's docstring says). Held-out and multi-window checks have not been run on GPU.
- **Hardware agreement (validation.json):** Accel-Sim reads match Nsight within 0-3% for most cases, except q7_32k/q7_32k_t2 at -15.5% and q7_B4P8192 at -8.9%. Writes are undercounted 8-56%, but writes are <2% of the decode windows (up to 10.8% for q7_ub32).
- The core scripts on the laptop match the cluster copies (md5).

## 2026-11-27 evening: read-gap cause, held-out job, pipeline walkthrough
- **q7_32k -15.5% read gap explained** (`kernel_read_gap.py`, per-kernel Nsight vs Accel-Sim): it is all in flash_attn_ext_vec, 3125 MB on hardware vs 1963 MB simulated. Every other kernel is within 0.2% (mul_mat_vec_q 3118 vs 3116 MB). The KV cache at 32k is 1.88 GB, so Accel-Sim reads it about once while the V100 reads it about 1.66x (GQA heads re-fetch after L2 eviction). This is the same 1.6-1.9x KV amplification the CPU study found with a 22 MB LLC. GPU attention share is 31% of DRAM reads in simulation vs 42% on hardware.
- **Held-out window job 141862** (holdout.py/holdout.sh, node03): q7_32k, q30_32k, q30_B4P512, q7_8k. Masks fitted on the 30% window are evaluated at 10% and 70% (transfer vs oracle refit); identity/xor_global/chan/xor_global_chan, Open/Timeout256, 256 B and 2 KiB interleave. Output: results/gpu/campaign_20261120/holdout/<case>_<ctx4>.json. Smoke-tested at 100k-sector windows on q7_8k.
- **PIPELINE_WALKTHROUGH.md:** step-by-step CPU and GPU pipeline, what is compared, problem-statement mapping, gaps. Main open gap: the proposal mask is fitted per trace; a cross-workload fixed-mask test is still needed.

## 2026-11-27 evening: cross-workload fixed-mask test submitted
- `xworkload.py`/`xworkload.sh` (node01, 6 cores, resumable): XOR masks (xor_global, chan, xor_global_chan) are fitted once on a trainer and applied unchanged to all 26 GPU policy windows.
  - Trainers: q7_32k, q30_32k/e520, and pooled (q7_32k + q30_32k/e520 + q05_32k).
  - Compared against identity and the per-window oracle refit.
  - Grid: packed/thp, 256 B / 2 KiB interleave, Open/Timeout256. 104 replays per window, 2704 in total.
  - Output: results/gpu/campaign_20261120/xworkload/results.json.
  - Smoke test (XW_SMOKE=100000, 2 windows, 208 replays) passed.

## 2026-11-27 ~21:15 — GPU held-out window check done (job 141862)

- 141862 COMPLETED (1h44m, node03). 6 contexts × 72 replays: q7_32k, q7_8k, q30_32k (2 ctx), q30_B4P512 (2 ctx). Output: `results/gpu/campaign_20261120/holdout/*.json`.
- Setup: masks fitted on the grid's 30% window, applied unchanged to 4M-sector windows at 10% and 70% of the same stream ("transfer"); compared with masks refitted on those windows ("oracle") and identity. Packed placement, s=8/11, Open/Timeout256. Metric: Ramulator HBM2 saturated cycles / identity cycles on the same window (lower is better).
- Best-of-3 maps (xor_global, chan, xor_global_chan), 8 held-out cells per context, mean transfer vs oracle:
  - q7_32k: 0.894 vs 0.885 (best 0.738 vs 0.740); 2 of 8 cells slightly worse than identity (max 1.010).
  - q7_8k: 0.947 vs 0.945 (best 0.900 vs 0.901).
  - q30_32k e520: 0.995 vs 0.994; 6900: 0.978 vs 0.988.
  - q30_B4P512 7450: 0.979 vs 0.979; f070: 0.982 vs 0.979.
- Conclusion: masks fitted on one window keep within about 1 percentage point of the refitted mask at other points in the same decode stream. The in-sample window choice does not inflate the GPU gains. The cross-workload test (142132) is still running: 41 of 104 window×placement×interleave groups done at 21:02.
- 139898 (q30_B16P512) ctx 2: kernel 847 of 856 at 21:10. Mask export, reconcile, prepare and upload follow automatically.
