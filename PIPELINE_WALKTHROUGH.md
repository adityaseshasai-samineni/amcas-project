# AMCAS experiment pipeline: step-by-step walkthrough and audit

Written 2026-11-27. This explains what each experiment actually executes, what is compared, and where it does and does not meet the problem statement. Job IDs and raw numbers are in `context.md`.

## 0. The question each experiment answers

The problem statement asks us to:
1. Characterise DRAM row-buffer hit rates for representative attention/KV and MoE traces, using DRAMsim3 and Ramulator.
2. Propose an address-interleaving or row-buffer policy for that access class.
3. Report the hit-rate and latency improvement over baseline configurations.

Both pipelines (CPU and GPU) follow the same three stages:

```
real inference ──► instruction/memory trace ──► cache model ──► DRAM request stream
                                                                   │
                       same stream, replayed many times ◄──────────┘
                       (mapping × placement × page policy) in Ramulator 2.1 and DRAMsim3
                                                                   │
                       compare cycles, row hit/miss/conflict, read latency vs identity + Open
```

**What is replayed into the DRAM simulators is never the raw NVBit or DynamoRIO address trace.** The raw trace contains every load and store, and most of those hit in the caches. It first goes through a cache model: drcachesim on the CPU side, the full Accel-Sim V100 timing model on the GPU side. Only what leaves the last-level cache (misses, write-backs, prefetches) becomes the DRAM request stream.

---

## 1. GPU pipeline (V100, HBM2), step by step

### 1a. Capture on node04 (the 2× V100 node): `capture_array.sh` → `run_case.py`

1. **Case definition** (`cases/<case>.json`). Each case sets:
   - the model (Qwen2.5-0.5B/7B, or Qwen3-30B-A3B Q4_K_M MoE, which needs both GPUs);
   - `AMCAS_B` sequences, each with an `AMCAS_P`-token prompt taken from the shared corpus (the CPU runs use the same text);
   - `AMCAS_NGEN` decode steps, of which `AMCAS_SKIP_STEPS=1` are skipped and `AMCAS_TRACE_STEPS=1` is traced;
   - full GPU offload (`AMCAS_GPU_LAYERS=99`) and flash-attention auto.
2. **Native run** (`run_case.py --mode native`, no tools). The custom driver `llama-gpu-batch-v3` (`gpu_batch_main.cpp`) runs as follows:
   - It loads the model, tokenises the prompts and prefills all B sequences.
   - It then decodes token by token.
   - Around the selected decode step it calls `cudaProfilerStart()` on every GPU, and `cudaProfilerStop()` after that step. This is the region of interest (ROI).
   - It writes the generated token IDs, the prompt tokens, the step boundaries and a tensor address map (`native.tensors.tsv`).
   - Gate: the run must finish, offload every layer to the GPU and produce valid output.
3. **Capture run** (`--mode capture`). The same binary runs with `LD_PRELOAD` = the NVBit Accel-Sim tracer, patched for multiple GPUs and the profiler window (`ACTIVE_FROM_START=0`).
   - The tracer instruments every kernel's SASS, but enables recording only while the profiler window is open.
   - For each kernel inside the ROI it writes a **full SASS instruction trace**: every warp instruction with its PC, active mask, and register and memory operands. Load and store instructions carry the address of every active thread.
   - It writes one compressed `kernel-N-ctx_X.trace.xz` per kernel and one `kernelslist` per CUDA context, i.e. one per GPU.
   - Gate: the capture's generated tokens and prompt tokens must be byte-identical to the native run's. This proves the tracer didn't change the computation.
   - Then `post-traces-processing` converts the traces into Accel-Sim input.
   - **Why it is slow (measured 2026-11-27):** even outside the ROI, the tracer's launch hook does the following on every kernel launch:
     - `cudaDeviceSynchronize()` twice;
     - an extra `flush_channel` kernel, followed by another sync;
     - a busy-wait for its receiver thread;
     - opens and closes a stats file twice.

     That costs about 19 ms per launch with both GPUs idle. A 30B MoE prefill launches about 44k kernels per 512-token batch. As a result, capture takes 430–2,700× the native time, and the 4 × 8192 MoE prefill alone takes about 17 h. A fix (early return for kernels outside the ROI) has been proposed but not applied.
4. **Hardware reference** (`hardware_case.py`, Nsight Compute):
   - The same binary and environment are run without NVBit, under `ncu --profile-from-start off`, so only the same ROI kernels are profiled.
   - It collects `dram__bytes_read.sum` and `dram__bytes_write.sum` per kernel.
   - Gates: the kernel count must equal the traced kernel count, and the output tokens must equal the native run's.
   - This yields per-kernel DRAM byte counts measured on the real V100. It is not a DRAM trace (no addresses or ordering).

### 1b. Replay on node03: `simulate_case_node03.sh` → `simulate_case.py`

5. **Copy** the traces from node04 to node03 with rsync (retried), and check that the file count and byte total match exactly.
6. **Accel-Sim timing simulation, once per GPU context.** `accel-sim.out` runs with the validated V100 configuration:
   - SM7_QV100: 80 SMs, L1/shared memory, a 6 MB L2 in 32 slices and 32 memory channels with FR-FCFS queues of 64.
   - It **re-executes every traced instruction of every warp** through this model, including the cores, the L1 and L2 caches and the interconnect, kernel by kernel in launch order. The L2 stays warm across the ROI kernels, but the window starts cold.
   - A patch (`AMCAS_DRAM_EXPORT`) writes every L2 → DRAM request to `dram.csv`: sequence, core cycle, partition, address, read/write, bytes and sector mask.
   - Checks:
     - every kernel launched and completed;
     - the launch UID equals the kernel count;
     - the per-channel `n_req` totals equal the number of exported packets.
   - Scope: each GPU is modelled independently, and GPU-to-GPU transfers aren't timed.
7. **Sector export** (`export_masked_sectors`):
   - Each DRAM packet (32–128 B, with a 4-bit sector mask) is split into 32 B sectors, one 64-bit word each: `(sector address >> 5) << 1 | write`. This produces `masked.dram`, the request stream.
   - **Physical placement:** each 2 MiB virtual page gets a physical frame in first-touch order over the whole stream (`masked.dram.packed`). This is the "packed" placement.
8. **Reconcile** (`reconcile_case.py`): compares total model read and write bytes against Nsight's hardware totals and records the relative errors in `validation.json`. Section 3 has the results.

### 1c. Policy grid on node03/node01: `prepare_policy.py` → `policy_case.sh` → `policy_grid.py`

9. **Window:** take 4,000,000 sectors (128 MB) starting 30% into each context's stream. For the 7B and 30B models this is 2–5% of one decode step.
10. **Placements:**
    - `packed` (first-touch, as above);
    - `thp`: the same 2 MiB pages permuted randomly over 16 GB (seed 1), imitating an OS with transparent huge pages.
11. **Channel interleave:** 256 B or 2 KiB. The HBM2 layout has 32 channels, 32 banks (pseudo-channel, bank group and bank), 14 row bits and 5 column bits per 32 B sector.
12. **Mappings.** Each one is an invertible bit transform of the physical address (injectivity is asserted).

    | mapping | what it is |
    |---|---|
    | `identity` | RoBaRaCoCh (baseline) |
    | `xor_global` | bank bits XOR a GF(2) combination of row bits, searched with a row-conflict cost model |
    | `chan` | channel bits XOR hashed bits |
    | `xor_global_chan` | both of the above, searched jointly |
    | `mop`, `mopxor` | Ramulator MOP4CLXOR, optionally with the searched bank XOR |
    | `chra` | ChRaBaRoCo |
    | `bitrev` | bit-reversal (Shao 2005) |
    | `naive` | a simple row-XOR |
    | `region` | per-tensor-class XOR, only when tensor attribution covers ≥ 50% |
    | `expert_hash`, `expert_lin` | expert-aware XOR, for MoE only |

    **All searched masks are fitted on the same window they are evaluated on (in-sample).**
13. **Page policies:**
    - Open, ClosedCAP;
    - Timeout 16, 64 and 256;
    - Region;
    - HAPPY (two variants; Ghasempour 2016);
    - Adaptive (two counter settings).
14. **Ramulator 2.1 replay** (one per mapping × policy):
    - 32 HBM12 controllers, HBM2 2Gb at 1600 Mbps, FR-FCFS, all-bank refresh, read and write buffers of 32.
    - The trace has no timestamps, so requests are fed **saturated**: as fast as the controllers accept them.
    - Recorded: cycles to drain the window, and row hit, miss and conflict counts.
    - Asserts: read and write request counts equal the trace's, and every read is either served or forwarded.
15. **DRAMsim3 replay** (`ds3_cross.py`) of the same transformed stream, with timings matched to Ramulator. DRAMsim3 models 64 independent pseudo-channels, each with its own command bus. Real HBM2 pseudo-channels share their channel's command bus. This explains the earlier mopxor disagreement, so DRAMsim3's mapping ranking is not trusted where pseudo-channel balance matters.
16. **Metrics** (`gpu_latency_report.py`), each a ratio against identity + Open on the same case, placement and interleave:
    - cycle ratio;
    - row-hit rate;
    - DRAMsim3 *measured* mean read latency;
    - Ramulator *modelled* latency: hit × CL + miss × (tRCD + CL) + conflict × (tRP + tRCD + CL).
17. **Upload:** the DRAM CSV goes to the GitHub release, and results are pushed to the repo.

---

## 2. CPU pipeline (Xeon, DDR4/DDR5), step by step

1. **Native run:** llama.cpp `build-fast` (AVX-512, OpenMP, 4 threads) on node01/node03, with the same models and corpus. `decode_stamp.so` (LD_PRELOAD) logs the wall-clock start and end of each `llama_process` call. A batched driver, `drbatch`, is used for B > 1.
2. **DynamoRIO memtrace** of the whole run: every load and store with its PC and virtual address, converted with `drraw2trace -compress gzip`.
3. **Windows:** `record_filter` trims the trace to the timestamp window of one decode token, using the decode_stamp times. Model loading is discarded.
4. **Cache simulation:**
   - `drcachesim` runs on each window from a cold start: per-core L1/L2 and a 22 MB 11-way LLC (`-cores 4`), with the stride prefetcher on.
   - DRAM reads = LL misses + prefetch misses. A second run without the prefetcher writes the demand-miss address stream.
   - Validated: Qwen 0.5B per-token DRAM reads match gem5 within 0.7%.
5. **DRAM request stream:**
   - Each miss is attributed to a tensor through the tensor address map: weights, KV, activations or experts.
   - It is placed physically with 4 KB pages or with 2 MB THP frames in a random permutation.
6. **Grid:** Ramulator 2.1 (DDR4 × 1 channel, DDR5 × 8 or × 16, plus HBM3, GDDR6 and LPDDR5/6 variants in `dramtech/`) and DRAMsim3, over the same mapping and policy families. Most replays are saturated open-loop; `22_chanloop.sh` adds a closed-loop check.
7. **Held-out test (already done on the CPU side):**
   - The XOR searched on decode token 1 is applied to token 2 (7B at 32k, THP).
   - Held-out gain: 14–16% on DDR5 Open, 29% on DDR5 ClosedCAP. This is within 1–2 points of the in-sample gain.

---

## 3. What is compared, exactly

There are two different comparisons. They answer different questions.

**(a) Is the simulated DRAM stream right? (validation)**
- **Output identity:** native, capture and hardware runs generate byte-identical tokens.
- **Kernel identity:** the traced kernel count equals Nsight's kernel count, in every finished case.
- **DRAM byte totals:** Accel-Sim's L2 → DRAM read bytes and write bytes vs Nsight `dram__bytes_read/write` over the same ROI kernels. This is a **count comparison (bytes), not a trace comparison.** Hardware gives no addresses or order, so the row-buffer behaviour itself cannot be validated against hardware; only the volume can.

| result | cases |
|---|---|
| reads within 0–3% of hardware | most cases |
| reads −8.9% | q7_B4P8192 |
| reads −15.5% | q7_32k, q7_32k_t2 |
| writes −8% to −56% (writes are < 2% of decode traffic, up to 11% for ubatch/prefill cases) | all cases |

**The −15.5% read gap at 32k, explained (2026-11-27, per-kernel comparison):**
- The gap is entirely in `flash_attn_ext_vec`:

  | | flash-attention reads |
  |---|---|
  | hardware | 3,125 MB |
  | Accel-Sim | 1,963 MB |

- Every other kernel agrees within 0.2%; the matrix-vector kernels (`mul_mat_vec_q`), for example, read 3,118 MB vs 3,116 MB.
- The 7B KV cache at 32k is 1.88 GB. So the simulator reads the KV cache about once, while the real V100 reads it about 1.66 times: on real hardware, query heads that share a KV head re-fetch it after the 6 MB L2 has evicted it.
- The CPU study found the same effect (1.6–1.9× KV amplification at 32k with a 22 MB LLC).
- So the GPU simulation **under-represents KV re-reads**: attention is 31% of traced DRAM reads in the simulation but 42% on real hardware.

**(b) Which mapping or policy is better? (the study)**
- One fixed request stream is replayed under each configuration. Request counts are identical by construction (asserted).
- What differs is the time to serve the stream (cycles) and the row-buffer outcome (hit, miss or conflict), plus latency.
- Everything is a ratio against identity + Open on the same stream.

---

## 4. Does the experiment meet the problem statement?

| problem statement asks | what we have | status |
|---|---|---|
| Representative attention and MoE traces | Dense 0.5B/7B (short to 32k context, batches 4/16, prefill) and MoE 30B-A3B (short, 1k, 32k, B4, n4, B16 running), on CPU and GPU, from real llama.cpp inference with output-identity gates | Met. Caveat: llama.cpp KV is contiguous per sequence, so the "irregular gather over KV" of paged attention (vLLM-style) is not represented. |
| Use DRAMsim3 / Ramulator | Both, with matched timings, plus a diagnosis of where they disagree (DRAMsim3's pseudo-channel command bus) | Met |
| Characterise row-buffer hit rates | Hit, miss and conflict rates per case, placement, interleave, mapping and policy | Met |
| Propose an interleaving or row-buffer policy | Bank XOR hash (row bits into bank bits), plus a channel XOR hash on GPU, with Open + Timeout256. Closed page is rejected in every case. | Partly met (see gap 1) |
| Hit-rate and latency improvement numbers | Cycle ratio and hit rate in every cell; DRAMsim3 measured latency; Ramulator modelled latency | Met, but as DRAM-side numbers only (see gap 3) |
| Compare with baselines and existing techniques | identity, ChRaBaRoCo, MOP, bit-reversal (Shao), XOR (Rotaru-style), HAPPY, Adaptive, Timeout | Met |

**Gaps that weaken the claims:**
1. **The proposed mapping is fitted per trace.**
   - A real controller has one fixed mapping. The CPU side shows the mask transfers to the next token of the same workload, but neither side has shown it transfers *across workloads*: does one mask fitted on 7B at 32k work for the 30B MoE?
   - The GPU side hasn't yet shown transfer along the stream either. Job 141862 is running that check (fit at 30%, evaluate at 10% and 70%).
   - **Needed:** a cross-workload test, with one fixed mask evaluated on every case.
2. **Window representativeness on GPU.** One 128 MB window per context, which is only half a 7B layer at 32k. The held-out windows (141862) also test whether the 30% window is typical.
3. **DRAM-side metric only.**
   - Saturated open-loop replay measures how fast the DRAM drains a fixed request stream. It does not model how GPU stalls feed back into request timing.
   - So "26–39% fewer cycles" means DRAM service time for that window, not token latency. This must be stated in the report, and an end-to-end figure would need Accel-Sim with the mapping inside its own DRAM model.
4. **GPU attention traffic is understated by about 1.66×** (section 3). The long-context channel finding is, if anything, conservative: real hardware would put more KV traffic through DRAM.
5. **Write volume is undercounted on GPU.** The effect is small for decode, since writes are < 2% of traffic.
6. **DRAMsim3's pseudo-channel model** is unlike HBM2, so its mapping rankings shouldn't be quoted for mop and mopxor.

**Bottom line:**
- The pipeline is internally consistent and checked at each stage: output identity, kernel counts, request counts, injective mappings, and both simulators.
- It answers the problem statement for row-buffer characterisation and for comparing policies.
- The weakest link is the *proposal*: a mask searched per trace is an oracle, not a deployable mapping, until the held-out and cross-workload tests confirm it.
