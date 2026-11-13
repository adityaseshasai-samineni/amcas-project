# Full-scale analysis: DRAM address mapping and row-buffer policy for LLM inference

Final report, 2026-11-14. It replaces the 2026-11-08/11 draft. `ANALYSIS.md` covers the earlier gem5-era runs and still stands.

**Sources.** Numbers come from three places:
- `analysis/fullscale/headline.md` (tables H1 to H5) and `analysis/fullscale/tables.md`. `trace-pipeline/tools/fullscale_report.py` generates both from the raw replay files in `cluster-data/results/fullscale/`.
- The dated sections of `context.md`, cited in brackets. Read later sections over earlier ones: they correct them (see section 9).
- The main session's checks against raw files ("main session: verification of the 2026-11-11/12 agent results", "check of serving-owner results").

One run is still going, and its results are marked **Pending**: l2fix-owner's fix to the GPU L2 model at 32k. The MOP4CLXOR comparison is folded in (section 3).

## 0. Summary

**The problem statement asks for three things.** Characterize row-buffer hit rates of attention/MoE DRAM traffic with DRAMSim/Ramulator. Propose an address-interleaving or row-buffer policy for it. Report hit-rate and latency improvements.

1. **Characterization.**
   - LLM decode on a CPU is mostly a sweep over the weights. The default row-interleaved mapping already gets 0.96–0.98 row hit for that.
   - At long context the KV cache becomes most of the traffic: 61% of 7B reads at 32k.
   - With 2 MiB (huge) pages, row hit drops to 0.80–0.86. The cause is that the KV buffers of different heads sit at power-of-two offsets inside each huge page and collide in the same banks.
2. **Proposal.** One global XOR bank hash: each bank-index bit is XORed with a few row bits, using a matrix searched offline once. Hardware cost is a handful of XOR gates on the bank bits.
   - It gives the best result among those tested.
   - It transfers to an unseen token and across models.
   - It never makes things worse in the placement sweep.
   - Per-region, expert-aware, bit-reversal and channel-hash variants add nothing in closed loop.
   - Ramulator's MOP4CLXOR wins only in saturated DDR4 replay (by 5–25%), and loses in closed loop (+4% to +38% cycles).
   - No row-buffer policy beats open page by more than 3.2%. Open page stays.
3. **Improvement numbers, 7B at 32k, 2 MiB pages, CPU stream:**
   - Row hit rises from 0.85 to 0.95, and read latency falls 20–25%.
   - DRAM cycles fall 16.7% (open page) and 29.1% (closed page) on DDR5 with 16 sub-channels.
   - Other memory systems: −14.7% / −27.7% on 8-channel DDR5-4800; 24–55% on LPDDR5/6, HBM3/4 and GDDR6; −9.9% on the node's real DDR4-2933 (6 channels).
   - Qwen3-30B-A3B (MoE): −22.4% / −34.4%.
   - Token time depends on whether decode is memory-bound. On the 4-thread test node it is 0% (compute-bound: 8.4–10 GB/s). At the native plateau of about 43 GB/s it is 1–2%. With cores fast enough to make memory the limit (~19 GB/s demand at the model's latency) it is up to 16%.
   - The gain needs huge pages and long context: 17% at 32k, 6% at 8k, 2–4% at 1k, under 2% for prefill.
4. **GPU.** The V100 (HBM2) and A16 (GDDR6) behave differently from the CPU:
   - row hit is low because about 70 streaming multiprocessors read different rows at once;
   - the bank XOR does nothing on the V100;
   - channel interleaving or a channel hash is the lever there.

   The GPU 32k numbers are overstated by up to 32%; pending the L2 fix.

**Contribution, relative to published work** (`related_work_2023_2026.md`):
- The measurement and the cause are the new parts.
- Applying an XOR bank hash is not new. On DDR5 with open page, the textbook permutation (bank bit j ^= row bit j) gets within about 1 point of our searched matrix.

## 1. Method and validation

**Models.**
- Qwen2.5-0.5B (Q4_0).
- Qwen2.5-7B-Instruct (Q4_K_M).
- Qwen3-30B-A3B (Q4_K_M; MoE with 128 experts, 8 active).
- On gem5 only: stories15M and tinymoe4.

The engine is llama.cpp throughout.

**Three capture routes.**

| Route | Where | What it records | Cache filter | Validation |
|---|---|---|---|---|
| gem5 | simulated CPU, SE mode, 1 thread, SSE2 | requests at the DRAM controller (CommMonitor), plus a patched virtual-address field | gem5's own caches | text and instructions/token match native; counters match the trace |
| DynamoRIO | native AVX-512 llama.cpp, 4 threads; one decode step attached by our driver `llama-drattr` / `llama-drbatch` | every load and store | drcachesim: 4 × 32 KiB L1, shared 22 MiB 11-way LLC (the node's real L3) | DRAM reads/token within **0.7%** of gem5 on Qwen 0.5B ("Phase split works; DynamoRIO matches gem5") |
| NVBit | V100 and A16 GPUs, llama.cpp CUDA build | global loads/stores per warp (patched NVBit 1.7.7.3 mem_trace, streamed) | own L1/L2 model in 32 B sectors | 0.5B: 349 vs 354 MB DRAM reads (Nsight Compute), stores within 0.03%; V100 7B short context 4.39 vs 4.39 GB; 8k +1.7%; **32k +32% (FlashAttention on), +13% (off): pending l2fix-owner**; A16 L2 misses within 0.4% but DRAM reads +16% (likely inline ECC) |

**Tensor attribution.**
- Our llama.cpp hook dumps every tensor's address range.
- Each DRAM read is then classed by name as expert weights (`_exps`), KV (`cache_k`/`cache_v`), dense weights or activations ("what did we change for segregating" is in `context.md` under the drattr notes).
- Check: KV appends measure exactly 57,344 B/token for 7B (28 × 2 × 4 × 128 × 2 B).

**Physical placement** is modelled, since no root means no physical addresses:
- `packed`: virtual order with untouched 4 KiB pages dropped;
- `4k`: shuffled 4 KiB pages;
- `thp`: 2 MiB pages on random frames.

**DRAM replay.**
- **Ramulator 2.1:** DDR4 (1 ch; DDR4-2933 6 and 12 ch), DDR5 (16 sub-channels; DDR5-4800 8 ch), LPDDR5, LPDDR6, HBM3, HBM4, GDDR6, HBM2 (GPU).
- **DRAMsim3:** a cross-check.
- **Saturated replay:** issue as fast as the controller accepts. This gives a bandwidth-bound upper bound.
- **Closed loop:** 4 SimpleO3 cores calibrated to native speed:
  - short-context 7B: 26.2 GB/s modelled vs 25.7 native;
  - 32k at 4 threads: 8.6–10.8 GB/s;
  - the 24–32-thread plateau: ~40–46 GB/s.

  Calibration is in policy-owner parts 2 and 5.

**Validation gate for captures.** Every capture checks that the traced run produces the same text as the untraced run. For batched runs the check is positional: no sequence may diverge before the traced step. DynamoRIO can flip a near-tie token at the traced step itself.

## 2. Characterization of row-buffer behaviour

**Traffic per token (CPU, 4 threads, 22 MiB LLC).**
- **Short context:** one pass over the weights. That's 4.40 GB for 7B (94% of the file), 0.35 GB for 0.5B and 1.94 GB for 30B-A3B (8 of 128 experts per layer).
- **7B at 32k:** 9.8–11.5 GB, 1.6–1.9× the weights plus one KV pass (6.18 GB). The KV re-reads miss the LLC. At ≥ 64 MiB of LLC the traffic falls to 6.2 GB.
- **30B at 32k:** 14.3 GB, about 2.9×.
- **Batching:**
  - dense 7B traffic per sequence falls 14× at B=16;
  - for the MoE only 3.2×, because the experts touched per layer grow 8 → 23 → 57 at B = 1/4/16 (49 for a 30-token prefill, 15.8 over 4 consecutive tokens).

**Row hit with the default mapping** (H1; `tables.md`):

| Stream | Row hit (open page) | Why |
|---|---|---|
| Weights, any context | 0.96–0.98 | near-sequential sweep |
| 7B 32k, 2 MiB pages | 0.85 (DDR5), 0.86 (DDR4 1ch) | KV head buffers collide in banks |
| 30B 32k, 2 MiB pages | 0.80 | same, larger KV share |
| 7B batch 4 × 32k, 2 MiB pages | 0.71 (DDR4) | same |
| 4 KiB pages | 0.53–0.75 | scattered pages; no bank hash can help |
| 512-token prefill window | 0.90 | weights reused across 32 tokens |
| Closed page (ClosedCAP), any | 0.15–0.25 lower | rows closed before reuse |

**Per data class** (offline, `drmiss_export.py`, 2 MiB pages):
- weights 0.98;
- KV 0.72–0.81, rising to 0.92–0.96 with the XOR;
- activations 0.74, rising to 0.82.

## 3. Mappings tested and the proposal

**Tested:**
- Ramulator RoBaRaCoCh (the default), MOP4CLXOR, ChRaBaRoCo;
- DRAMsim3 rochrababgco / rocorababgch;
- bit reversal;
- **global XOR bank hash (`xor_global`)**;
- per-region XOR (`region`, per 2 MiB granule tagged as weights / KV / other);
- channel-bit hash (`chan`, `xor_global_chan`);
- expert-aware (`expert_hash`, `expert_lin`, `region_expert`);
- the textbook permutation (`naive`);
- fixed matrices from other models or a pooled mix (`T7`, `T30`, `pool`).

**Evidence for the global XOR**, all with 2 MiB pages:

| Test | Result | Source |
|---|---|---|
| 7B 32k, DDR5 16 sub-ch, saturated | −16.7% Open, −29.1% ClosedCAP; hit 0.847 → 0.951; latency 249 → 199 | H1 |
| Same, whole token (179M reads), DDR5-4800 8 ch | −14.7% / −27.7% | H2 |
| Held-out token (fitted on token 1, tested on token 2) | 0.844 / 0.712, against 0.840 / 0.698 fitted on token 2 itself | mapping-owner |
| Qwen3-30B-A3B 32k | −22.4% / −34.4% | H1 |
| Qwen 0.5B 32k | −11.7% / −18.5% (DDR5) | H1 |
| One fixed matrix across models and contexts | within 1.1 pt (Open) / 3.2 pt (ClosedCAP) of a self-searched one | transfer-owner |
| Placement sweep (random to fully contiguous 2 MiB) | never hurts: all ratios ≤ 0.874 | transfer-owner |
| Closed loop, ~19 GB/s demand | −16.4% cycles, −21% latency | H3 |
| Real node DDR4-2933, 6 ch | −9.9% saturated, −4.6% at fast core, −1.2% at the plateau | H4 |
| DRAMsim3, same streams | −2.7% (6 ch), −6.6% (12 ch): same direction and mechanism, a third to a half of Ramulator's size | validation-owner |
| Writes included | unchanged (−19.9% with and without) | policy-owner part 3 |

**Why the gain exists.**
- Within a 2 MiB page, physical offset equals virtual offset.
- llama.cpp's KV cache stores each head's K and V at power-of-two strides, so they share the bank bits and differ only in row bits.
- The XOR folds row bits into the bank bits and spreads them.
- With 4 KiB pages the stride pattern is already broken up, so there's nothing to fix.
- This is a property of the layout, not of fragmentation: random and contiguous 2 MiB placements give the same default conflicts (transfer-owner, fragmentation sweep).

**Negative results:**
- **`region`:** equal to the global XOR within 0.5–2%. At batch 4 it's slightly ahead on DDR4 (0.872 vs 0.885 at 32k) and behind on DDR5 (1.000 vs 0.938). Not worth a tag table and allocator support.
- **Expert-aware:** within 1% of the default at B = 1, 4, 16 and over 4 tokens.
- **Bit reversal:** up to 2.4× slower; it breaks the weight stream's bank spread.
- **Channel hash:** −35% to −47% saturated, but 0–2% beyond the bank XOR in closed loop (H3, "xor+chan" column).
- **Textbook permutation:** within ~1 pt on DDR5 open page, but 4–8 pt worse with closed page and 6–10 pt worse on DDR4.
- **A matrix fitted only on random frames** loses up to 5 pt (Open) and 15 pt (ClosedCAP) on 7B when pages are contiguous. Fit the hardware matrix on both cases.

**MOP4CLXOR, Ramulator's built-in XOR mapper** ("mop-owner: MOP4CLXOR vs xor_global (2026-11-14)", jobs 124413, 124414, 124444, 124445). It puts the low column bits below the bank fields and XORs each bank level with the next column bits, so the 4 lines of a 256 B chunk go to 4 different bank groups. That trades row hits for bank parallelism. The row is XORed with col>>5. It was tested on:
- 7B at 32k and 8k, 30B at 32k and short context, the held-out token, and 7B at B=16;
- all three placements;
- DDR5 16 sub-channels and 8 channels, DDR4 6 channels and 1 channel;
- Open and ClosedCAP;
- saturated and closed loop.

Also tested: MOP4CLXOR combined with our searched XOR (`mopxor`, checked one-to-one) and ChRaBaRoCo (+40% to 3× cycles everywhere).

| Regime | Winner |
|---|---|
| **Closed loop** (fast core, plateau, 4-thread native) | **`xor_global` wins or ties everywhere.** MOP4CLXOR takes 4–38% more cycles at ~22 GB/s demand and is worse than the default mapping |
| Saturated DDR5, 2 MiB pages | three-way tie within 1–3% (`xor_global` 0.82, MOP4CLXOR 0.83, `mopxor` 0.82) |
| **Saturated DDR4** (6 ch and 1 ch) | **MOP4CLXOR / `mopxor` win by 5–25%,** even with contiguous pages (6 ch: 0.90 vs 1.01) |

So `xor_global` is the best mapping whenever the memory isn't flooded, which covers every realistic operating point we measured: native 4-thread decode runs at ~10 GB/s, far from DDR4's saturation. MOP4CLXOR is better only when DDR4 is bandwidth-saturated. A controller could select MOP4CLXOR-style column interleaving only when the queues are saturated, but we didn't evaluate that.

## 4. Row-buffer policies

- **Tested:** Open, ClosedCAP, close-page (DRAMsim3), and our Ramulator plugins: Timeout (close after 64 idle cycles), Region (close-page for chosen data classes) and HAPPY-style (address-hashed reuse predictor).
- **Result:** across 150 non-Open cells the best is 0.968 of Open (HAPPY, one closed-loop cell), and nothing else beats 0.976.
  - ClosedCAP costs 12–38% saturated.
  - Timeout is worst in closed loop (row hit 0.21).
  - Source: `analysis/policy_results.md`, policy-owner part 1.
- **Verdict:** for this traffic the mapping, not the page policy, moves the row-hit rate. Keep open page.

## 5. Headline numbers, and what they mean for token time

The full tables are in `analysis/fullscale/headline.md` (H1–H5). For the proposal on 7B at 32k with 2 MiB pages:

| Condition | DRAM cycles | Row hit | Read latency |
|---|---|---|---|
| DDR5 16 sub-ch, saturated, Open / ClosedCAP | −16.7% / −29.1% | 0.85 → 0.95 | 249 → 199 cycles |
| DDR5-4800 8 ch, whole token | −14.7% / −27.7% | 0.84 → 0.95 | – |
| DDR4-2933 6 ch (real node), saturated | −9.9% | 0.91 → 0.95 | – |
| Closed loop, 19 GB/s demand (memory-bound) | **−16.4%** | 0.62 → 0.79 | −21% |
| Closed loop, plateau ~40–46 GB/s | −1.2% (DDR5) / −1.2% (DDR4 6 ch) | 0.78 → 0.82 | −25% |
| Closed loop, 4-thread native rate (8.6–10.8 GB/s) | **~0%** | 0.74 → 0.77 | −10% to −16% |

**Native decode time, 7B at 32k** (`native/ctx32k/dec_t*.err`):

| Threads | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| ms per token | 1,149 | 576 | 358 | 264 |
| GB/s (11.5 GB/token) | ~10 | ~20 | ~32 | ~43 |

Bandwidth plateaus at about 43 GB/s, roughly 15% of the modelled DRAM peak, set by the cores and the two-socket layout. **On this server the proposal saves 1–2% of token time at best.** The DRAM-side gains (row hit, latency, DRAM busy time) are large. They turn into token time only where decode is memory-bound: more or faster cores per channel, fewer channels, or the LPDDR and HBM systems in section 6.

## 6. Other DRAM technologies (CPU streams)

Sources: H5 and "dramtech-owner: LPDDR5/6, HBM3/4, GDDR6". Each standard's bit layout was derived from Ramulator's source and checked with a probe. The XOR was searched again for each layout.

| 7B 32k, 2 MiB pages | Saturated Open | Saturated ClosedCAP | Closed loop, fast core | Latency, fast core | 4-thread native |
|---|---|---|---|---|---|
| DDR4-2933 6 ch | −9.9% | −8.7% | −4.6% | – | 0% |
| DDR5 16 sub-ch | −16.7% | −29.1% | −16.4% | −21% | ~0% |
| LPDDR5 8 ch / 32 ch (Grace-like, 410 GB/s) | −24.3% / −35.6% | −26.4% / −39.7% | −9.5% / −6.8% | −15% / −8% | ≤1% |
| LPDDR6 8 / 32 ch | −25.5% / −42.4% | −39.5% / −41.4% | −18.7% / −11.7% | −25% / −13% | ≤1% |
| HBM3 (16 ch) | −26.0% | −35.8% | −20.8% | −26% | 0% |
| HBM4 (32 ch) | −31.0% | −39.2% | −19.5% | −24% | 0% |
| GDDR6 (16 ch; a what-if for a CPU stream) | −54.6% | −41.1% | −6.8% | −12% | 0% |

- **Results:** 30B at 32k shows the same order, saturated 26–56%. 7B at 8k gives 6–9% (31% on GDDR6). 30B short context gives 0%.
- **Why the gain grows:** smaller rows (1–2 KiB vs ~8 KiB on DDR4/5), many narrow channels and fewer banks per channel make the KV bank collisions more costly.
- **Where it doesn't help:** contiguous-page (packed) placements gain nothing from the bank XOR. With 4 KiB pages only the channel hash helps, and only saturated.

## 7. GPU (V100 HBM2, A16 GDDR6)

Source: "gpu-owner: GPU trace route". Results are for 7B decode, 2 MiB pages, 256 B channel interleave, saturated replay of the first 80M requests.

| | V100 32k, FlashAttention on | V100 32k, FlashAttention off | A16 32k, FlashAttention on / off |
|---|---|---|---|
| DRAM per token (model) | 9.75 GB (Nsight Compute: 7.40) | +13% vs Nsight Compute | × ~1.16 for ECC |
| KV share of reads | 55% | 69% | 29% / 73% |
| Bank XOR | 0.99–1.01× | ~1× | 1.14–1.28× (FlashAttention on) |
| Channel hash | 1.46× | 1.7–2.1× | 1.7–2.1× (off) |
| 2 KiB channel interleave | 2.3× (128 → 297 of 819 GB/s); channel hash adds 1.07× on top | – | – |

- **Why GPU row hit is low:** about 70 streaming multiprocessors read different rows at the same moment. A single multiprocessor's weight stream hits 0.87; merged, 0.76. A static bank hash can't fix that.
- **The lever:** channel parallelism (interleave granularity, channel hash).
- **Closed page:** worse than open with FlashAttention on.
- **Pending: l2fix-owner.** The 32k model reads 32% more than Nsight Compute measures (FlashAttention on), so the 32k KV shares and bandwidth figures are overstated. The relative comparisons above are indicative until the L2 model is fixed. It is also checking whether ECC explains the A16's +16%.

## 8. Limits

- **Physical placement and the memory controller's address hash** are modelled; reading or changing them needs root, and Intel and NVIDIA don't document their hashes.
- **Saturated replays** are bandwidth-bound upper bounds. The closed-loop model is 4 cores with constant compute gaps, calibrated to native bandwidth.
- **Windows:** 8–12M (CPU) or 80M (GPU) requests, starting at 30% of the step (CPU) or at the start (GPU). The whole-token check (H2) matches the window to within a point.
- **One decode step per configuration,** except the held-out token and the 4-token MoE run.
- **The 32k absolute traffic varies 1.6–1.9×** between traces, because drcachesim's thread interleaving matters when the LLC thrashes. Quote it as a range.
- **Writes** are modelled only in the writes study (0.1–0.4% of requests).
- **drcachesim has no private L2.** Earlier 32k captures used an unoptimised llama-drattr build, which changes timing but not the addresses read.
- **Prefill** is covered by a 30-token prompt and a 32-token micro-batch of a 512-token prompt.

## 9. Corrections to earlier drafts

- (a) **The cause is the KV head layout inside 2 MiB pages,** not an aged allocator scattering pages. "packed" is an artificial 4 KiB compaction, not a real allocator.
- (b) **The channel hash's −35% to −47% is saturated-only.**
- (c) **The batch-4 32k replay window starts at 30% of the step,** not at its start.
- (d) **The batch-4 contiguous-page DDR5 −35%** is an artifact of the placement model and is not claimed.
- (e) **The 4-thread native 32k rate is 8.4–10 GB/s;** the earlier 4.8 GB/s came from an unoptimised binary.
- (f) **GPU 32k traffic is overstated** (pending the L2 fix). A16 absolute bytes need × ~1.16.

## 10. Position against related work

From `related_work_2023_2026.md`:

| Claim | Status |
|---|---|
| (1) Decode is a weight sweep with high row hit | Assumed in the literature; we give a full-scale measurement |
| (2) KV in 2 MiB pages loses row locality, and a bank XOR recovers it | Looks new as a measurement and explanation. XOR mapping itself is old (permutation-based interleaving, Rotaru 2026, Flatfish), and the textbook permutation captures most of the gain on DDR5 open page |
| (3) Per-region mapping equals the global XOR | New, as a negative result |
| (4) MoE routing gives no mapping benefit | Partly published (expert reuse); the DRAM-level negative looks new |

Still to do before claiming novelty:
- check the 2025–26 ISCA, MICRO, HPCA and ASPLOS programs by hand;
- verify the Flatfish, AttAcc and RH+ citations;
- drop UMDAM (withdrawn).

## 11. Open items

- **Pending:** l2fix-owner (GPU L2 at 32k). Its per-kernel check found the whole 32k overcount is in `flash_attn_ext_vec`, and an XOR-hashed L2 set index brings it into Nsight Compute's range.
- **Not evaluated:** a controller that switches to MOP4CLXOR-style column interleaving only when its queues saturate.
- **Optional:**
  - DRAMsim3 in closed loop;
  - 30B MoE at batch 4 with 8k / 32k prompts;
  - prefill with 64-token micro-batches;
  - a gate-level cost estimate of the XOR.
