# AMCAS presentation prep

Prepared 2026-11-14 by a Haiku agent (present-coach) from the project's own files. The main session then checked it against the files and corrected it.

**Sources.** File paths are relative to the repository root.
- "ctx" means a dated section heading of `context.md`.
- "H1–H5" are the tables in `analysis/fullscale/headline.md`.
- "§" means a section of `ANALYSIS_FULLSCALE.md`.

The PNG figures were not opened, so check each one before it goes on a slide.

**Corrections applied:**
- The report's token-time statement is fixed in `ANALYSIS_FULLSCALE.md` (commit 2389e01). The gain is about 0% at 4 threads, about 7% at 16 threads (~32 GB/s) and 1–2% at the 24–32-thread plateau.
- The report now covers DRAMsim3 closed loop, the XOR hardware cost and the novelty verdicts.
- In this file, Q13, Q44 and Q76 were corrected.

---

## 1. Talk outline (14 slides; drop 13 for a 12–13-slide talk)

1. **Title and the question.** Can the DRAM address mapping and row-buffer policy be tuned for LLM inference traffic? Show: `problem_stmt_and_literature_survey.tex` (Problem Statement).
2. **Decode is a weight sweep.** Short-context decode streams the weights, and the default mapping gives 0.96–0.98 row hit. Show: `analysis/fullscale/fig_reads_per_token.png`; H1 rows "7B 1k" and "30B short".
3. **Three capture routes:**
   - gem5: exact controller traffic, but slow;
   - DynamoRIO: native AVX-512 at full size;
   - NVBit: GPUs.

   Show: the §1 table; CLAUDE.md "Pipeline architecture".
4. **Validation.** DynamoRIO matches gem5 within 0.7% per token on Qwen 0.5B, and every capture passes the same-text gate. Show: ctx "Phase split works; DynamoRIO matches gem5 (job 117350)"; §1.
5. **Where the reads come from.** At 32k, KV is 61% of 7B reads. LLC size controls the re-reads: 1.6–1.9× at 22 MB, 1.0× at 64 MB and above. Show: `analysis/fullscale/fig_llc_sweep.png`; §2.
6. **The cause.** With 2 MiB pages, KV head buffers at power-of-two offsets share bank bits, so row hit falls from 0.96 to 0.85 (7B 32k). Show: H1 row "7B 32k DDR5" (0.847 → 0.951 with the XOR); §3 "Why the gain exists".
7. **The proposal.** One global XOR bank hash, searched offline: 5–6 XOR2 gates typical (at most 13), at most 2 logic levels. Show: `docs/xor_hardware_cost.md`.
8. **Evidence it isn't overfitting:**
   - held-out token 0.844 / 0.712 vs in-sample 0.840 / 0.698;
   - one matrix transfers across models within 1.1 / 3.2 points.

   Show: the §3 evidence table.
9. **What we dropped, and why:** region-aware, expert-aware, bit reversal, the channel hash, MOP4CLXOR (wins only in saturated DDR4). Show: tracker "How we got here" steps 6, 9, 19; §3 negatives.
10. **Row-buffer policy: keep open page.** Across 150 non-open cells, no policy beats open by more than 3.2%; closed page costs 12–38% saturated. Show: `analysis/policy_results.md`; §4.
11. **Headline numbers, with conditions.** Show H1, H2 and H4 as tables, not charts:
    - DDR5 16 sub-ch saturated: −16.7% (open) / −29.1% (closed);
    - whole token on DDR5-4800 8 ch: −14.7% / −27.7%;
    - the real node's DDR4-2933 6 ch: −9.9%.
12. **DRAM cycles vs token time.** The token gain depends on whether decode is memory-bound. Show: H3; the §5 table.
    - 4 threads: ~0%;
    - 16 threads (~32 GB/s): ~7%;
    - 24–32-thread plateau (~43 GB/s): 1–2%;
    - memory-bound fast core: −16%.
13. **Other memory systems and GPUs.** LPDDR5/6, HBM3/4 and GDDR6 give −24% to −55% saturated. On GPUs the lever is channel interleaving, not the bank hash (V100 bank XOR ~1.0×). Show: H5; §6, §7.
14. **Corrections we made, and limits:**
    - the cause is the KV layout, not the allocator;
    - the GPU L2 model overstated 32k by 32% until it was fixed;
    - placement and the controller hash are modelled.

    Show: §8, §9.

---

## 2. Anticipated audience questions

### Problem and scope

**Q1. What is the problem?** The problem statement asks to characterize row-buffer hit rates of attention and MoE DRAM traffic with DRAMSim/Ramulator, propose an address-interleaving or row-buffer policy, and report hit-rate and latency gains. Source: `problem_stmt_and_literature_survey.tex`; §0.

**Q2. What did you deliver against it?**
- characterization of weights, KV, batching, MoE and prefill;
- one mapping proposal, the global XOR bank hash;
- a negative result for row-buffer policies.

The hit-rate and latency numbers are in H1 and H3. Source: §0.

**Q3. Why is the default mapping good for weights?** Decode is a near-sequential sweep over the weights, so consecutive lines fall in one open row: 0.96–0.98 row hit. Source: the §2 table; H1 "7B 1k" and "30B short".

**Q4. Where is the problem, then?** At long context. At 32k the KV cache is 61% of 7B reads, and with 2 MiB pages row hit falls to 0.80–0.86. Source: §0, §2.

**Q5. Does 4 KiB paging have this problem?** No. With 4 KiB pages the stride pattern is broken up. Row hit is 0.53–0.75 for other reasons, which no bank hash fixes. Source: the §2 table; §3 "Why the gain exists".

**Q6. What is the scope of the claim?**
- CPU llama.cpp decode and prefill;
- Qwen2.5-0.5B, Qwen2.5-7B and Qwen3-30B-A3B;
- 4-thread native runs and modelled memory systems;
- GPUs reported separately.

Source: §1, §8.

**Q7. What does "row hit" mean here?** The fraction of DRAM accesses that find their row already open in the bank. Source: §2; `tools/analyze.py` (`open_page_stats`).

### Why CPU, llama.cpp, these models

**Q8. Why llama.cpp?** It's a widely used CPU inference engine that runs quantized models without a GPU, and it runs the same way under gem5 and under native tracing. Source: §1.

**Q9. Why CPU and not GPU for the main study?** We can trace and replay the CPU route without root. The GPU route needed its own build, tracer and L2 model, and is reported in §7. Source: §7; tracker step 16.

**Q10. Why Qwen2.5-7B?** It's a realistic dense model whose weights (~4.4 GB read per token) and KV cache are both DRAM-sized. Source: §2.

**Q11. Why Qwen3-30B-A3B?** It's a real MoE with 128 experts and 8 active per token, so expert routing is real rather than a toy. The gem5-era MoE had 2 experts. Source: §1; tracker step 2.

**Q12. Why is Qwen 0.5B in the study?** It's the validation model: its per-token DRAM reads match gem5 within 0.7%. It's also the cheapest model for the GPU checks. Source: ctx "Phase split works; DynamoRIO matches gem5"; §1.

**Q13. Why 4 threads for the traces?** *(corrected)* 4 threads was the project's default from the first DynamoRIO runs, which kept traces and cache simulation affordable (one 7B token is billions of instructions). It wasn't chosen to match a server. The native thread scan shows its consequence: at 4 threads, 7B 32k decode runs at 8.4–10 GB/s and is compute-bound, and bandwidth plateaus only at 24–32 threads (~43 GB/s). So token-time results are reported separately at each operating point. Source: §5 table; ctx "policy-owner" parts 4–5.

### Trace capture

**Q14. How does gem5 capture work?** A `CommMonitor` sits between the memory bus and the DRAM controller and records exactly the requests the controller receives: LLC misses, writebacks and prefetches. The simulation fast-forwards atomically with warm caches, then switches to the O3 core at a work-item boundary. Source: CLAUDE.md "gem5 capture"; `configs/llm_trace.py`.

**Q15. What did you change in gem5?** A patch adds a `vaddr` field to the packet trace (`patches/apply_patches.py`, idempotent). Source: CLAUDE.md.

**Q16. What did you change in llama.cpp for gem5?**
- The gem5 build is static and SSE2-only: gem5's X86 lacks AVX and miscomputes SSSE3.
- Logging is inline.
- `-DAMCAS_M5` adds `m5_work_begin/end` around each graph compute and dumps tensor addresses to `amcas_tensors.tsv`.

Source: CLAUDE.md.

**Q17. How does DynamoRIO capture work?**
- DynamoRIO instruments the unmodified native AVX-512 llama.cpp and records every load and store.
- `drraw2trace -compress gzip` converts the raw trace; zip output fails past ~4 GB per thread.
- `drcachesim` models 4 × 32 KiB L1 plus a shared 22 MiB 11-way LLC.

Source: §1; CLAUDE.md "DynamoRIO route".

**Q18. How do you trace only one decode step?** Our drivers attach DynamoRIO at a chosen `llama_process` call and detach after it (`dr_app_start` / `dr_app_stop_and_cleanup`). `tools/decode_stamp.c` (LD_PRELOAD) logs the wall-clock start and end of each step, and DynamoRIO timestamps (µs since 1601) line up with it for window cutting with `record_filter`. Source: CLAUDE.md; ctx "Phase split works".

**Q19. Why not drcachesim's `-skip_to_timestamp`?** It doesn't work on these traces, because they have an empty CPU schedule. `record_filter -trim_before/after_timestamp` is used instead. Source: CLAUDE.md "Windows".

**Q20. What is `llama-drattr`?** Our driver. It links llama.cpp's libraries with DynamoRIO's static tracer and intercepts `ggml_backend_sched_graph_compute_async`, writing one line per tensor (name, address, size, type, op, shape) to `tensors.tsv`. No llama.cpp source is changed for the DynamoRIO route. Source: `trace-pipeline/tools/drburst/drattr_main.cpp`.

**Q21. What are `llama-drbatch` and `llama-drburst`?** `llama-drbatch` is the batched-decode driver used for the B=4 and B=16 captures (`scripts/18_drbatch.sh`); `llama-drburst` traces single steps. The batch check is positional: no sequence may diverge before the traced step. Source: ctx "batch-dev"; §1 "Validation gate".

**Q22. How is each DRAM read attributed to a tensor?** The tensor dump gives each tensor's address range. Each read is classed by name: expert weights (`_exps`), KV (`cache_k`/`cache_v`), dense weights, activations, or unattributed. Source: §1; `tools/drmiss_export.py`.

**Q23. How do you know the attribution is right?** 7B KV grows by exactly 57,344 B per token (28 layers × 2 × 4 heads × 128 × 2 B). Source: §1.

**Q24. Are the addresses physical?** No. DynamoRIO and NVBit see virtual addresses, and physical placement is modelled, because reading physical addresses needs root. Source: §1, §8.

**Q25. What are the windows?** 8–12M requests on the CPU, starting at 30% of the step; 80M requests on the GPU, from the start. The whole-token check (H2) agrees with the window to within a point. Source: §8.

### Cache model

**Q26. Why model the caches at all?** The trace records every load and store, and most of them never reach DRAM. A cache model filters them down to the DRAM requests. Source: the §1 table.

**Q27. Why a 22 MiB LLC?** It's the node's real L3 size. The LLC sweep shows the 32k result depends on it: KV re-reads 1.6–1.9× at 22 MB, 1.0× at 64 MB and above. Source: §1, §2; ctx "LL sweep at 32k".

**Q28. How stable is the 32k traffic?** At 22 MB it varies 1.6–1.9× between runs, because of drcachesim's thread interleaving while the LLC thrashes. Quote 32k absolute numbers as a range. Source: §8; ctx "The 9.8 / 10.3 / 11.5 GB discrepancy".

**Q29. What does drcachesim not model?** A private L2. Earlier 32k captures also used an unoptimised llama-drattr build, which changes timing but not the addresses read. Source: §8.

**Q30. How do you count DRAM reads from drcachesim?** LL demand misses plus LL prefetch misses, at 64 B. "Compulsory misses" includes first touches served by prefetch. Source: CLAUDE.md "Counting DRAM reads".

**Q31. How is the GPU cache modelled?**
- NVBit records are streamed into an L1 per multiprocessor (128 KB, 4-way) and an L2 (6 MiB on the V100, 2 MiB on the A16; 16-way; 128 B lines with 32 B sectors; write-back).
- Only the DRAM stream is written, gzipped.

Source: ctx "gpu-owner"; `trace-pipeline/gpu/gpu_filter.c`.

**Q32. What was wrong with the GPU L2 model?** It indexed sets with `line % 3072` and overcounted V100 32k DRAM reads by 32% (9.75 vs 7.40 GB). Kernel by kernel, the whole error was in `flash_attn_ext_vec` (4,917 vs 3,167 MB). Source: §7; ctx "l2fix-owner".

**Q33. What is the hashed-L2 fix?** An XOR-folded set index, `(line ^ line>>13) % sets`. V100 32k goes to 7.32 GB against 7.40 GB measured. Source: §7.

**Q34. Is the shift of 13 fitted?** Yes. It's calibrated, not unique: an ideal hash gives −10% to −14%, plain modulo +13% to +32%. Treat 32k GPU miss counts as ±10%. Source: §7.

**Q35. Did replacement policy or L1 size matter?** No. Replacement policy, L1 size, associativity and thread-block order didn't change DRAM bytes; only the set index did. Source: ctx "l2fix-owner".

### Validation

**Q36. How closely does DynamoRIO match gem5?** Per-token DRAM reads on Qwen 0.5B: 5,473,141 vs 5,434,710, 0.7% apart. Source: ctx "Phase split works" (job 117350).

**Q37. Are the two routes really comparable?** Yes, with caveats:
- gem5 is single-thread SSE2; DynamoRIO is 4-thread AVX-512, with ~11× fewer instructions per token;
- their prefetchers differ;
- they see physical vs virtual addresses.

The 0.7% is one model and one step. Source: same section.

**Q38. What did you check with Nsight Compute?** GPU DRAM reads per token and per-kernel L2 misses. The counters are readable without root. 0.5B: 349 vs 354 MB on the V100. Source: §1; ctx "gpu-owner".

**Q39. What is the same-text gate?** Each traced run must produce the same text as the untraced run. For batched runs, no sequence may diverge before the traced step, because DynamoRIO can flip a near-tie token at the traced step itself. Source: §1.

**Q40. Is the replay engine calibrated?** The closed-loop model gives 26.2 GB/s against 25.7 native at short context, and is matched to the native 4-thread 32k rate and the ~43 GB/s plateau. Source: §1; ctx "policy-owner" part 2.

**Q41. Did the checks find real errors?** Yes:
- the GPU L2 overcount (32%);
- an unoptimised drattr build that gave 4.8 instead of 8.4–10 GB/s at 4 threads (build type explains ~25% of the gap; the rest is probably node contention);
- the batch-4 window start.

Source: §9; ctx "policy-owner" part 5.

### Placement models

**Q42. Why three placements?** Physical placement is unknown without root, so we bracket it:
- `packed`: virtual order with untouched 4 KiB pages dropped;
- `4k`: shuffled 4 KiB pages;
- `thp`: 2 MiB pages on random frames.

Source: §1; tracker step 3.

**Q43. Which placement is realistic?** 2 MiB pages, for a system using huge pages, and that's the only placement where the bank hash helps. "packed" is artificial. Source: §9(a).

**Q44. Does fragmentation matter?** *(owner corrected)* No. Random and contiguous 2 MiB placements give the same default conflicts; the cause is the KV layout inside the page. Source: §3; ctx "transfer-owner" (fragmentation sweep).

**Q45. Does the gain need huge pages?** Yes. Packed and 4 KiB placements gain nothing from the bank XOR. Source: §6.

**Q46. Is the gain from the placement model or the KV layout?** From the layout. Inside a 2 MiB page the physical offset equals the virtual offset, so the power-of-two KV head stride reaches the bank bits. Source: §3; tracker step 13.

**Q47. What was the batch-4 artifact?** The "packed DDR5 −35%" at batch 4 comes from the VA-compacted placement model creating bank aliasing, so it isn't claimed. Source: §9(d).

### DRAM simulators

**Q48. Why Ramulator 2 and DRAMsim3?** The problem statement names both. Ramulator is the main simulator, with the most standards; DRAMsim3 is the cross-check. Source: §1.

**Q49. Saturated vs closed loop?** Saturated replay issues as fast as the controller accepts, a bandwidth-bound upper bound. Closed loop models 4 cores that wait on loads, calibrated to native speed. Source: §1, §8.

**Q50. Does DRAMsim3 agree?**
- Saturated: same direction, smaller size (−2.7% / −6.6% vs Ramulator −9.9% / −12.8%).
- Closed loop: fast core −4.3% / −9.2% vs −4.6% / −7.2%; plateau −0.4% / −1.0%; 4 threads ~0%.

Source: §3; ctx "extras-sim-owner".

**Q51. Why did DRAMsim3's saturated gain look smaller?** Open-loop replay amplifies the simulators' differences. With the loop closed they agree within ~2 points. Source: ctx "extras-sim-owner".

**Q52. Which memory systems?** DDR4 1 ch; the real node's DDR4-2933 6/12 ch; DDR5 16 sub-ch; DDR5-4800 8 ch; LPDDR5/6; HBM3/4; GDDR6. Source: §1, §6; H4.

**Q53. How were the other standards' layouts derived?** From Ramulator's source, checked with a bit-by-bit probe on each of 8 configurations. The XOR is searched again for each layout. Source: §6.

### Characterization

**Q54. What does 7B at 32k look like per token?** 9.8–11.5 GB, against 4.40 GB at short context: 1.6–1.9× the weights plus one KV pass (6.18 GB). Source: §2.

**Q55. What happens with batching?** Dense traffic per sequence falls 14× at B=16; MoE only 3.2×, because the experts touched per layer grow 8 → 23 → 57. Source: §2.

**Q56. And MoE routing?** At B=1 experts are read as whole contiguous slices (0.96 row hit), and expert-aware mappings are within 1% of the default. Source: §3.

**Q57. Does prefill change the picture?** No. The 512-token prefill window has 0.90 row hit, and every mapping is within 1%. Source: §2; H1.

**Q58. 30B attribution?** At short context, expert weights are most of the traffic. A 30-token prefill touches 49 of 128 experts per tensor (84% expert weights). Source: ctx "30B prefill attribution rerun".

**Q59. Are writes modelled?** In the writes study: 0.1–0.4% of requests, and the gain is unchanged (−19.9% with and without). Source: §3, §8.

### The proposal

**Q60. What exactly is it?** One global bank-index hash: each bank bit is XORed with a few row bits, with the matrix searched offline once. Row, column and 2 MiB granule are unchanged, so it's one-to-one. Source: `docs/xor_hardware_cost.md`; §3.

**Q61. What is the evidence?**
- 7B 32k on DDR5: −16.7% / −29.1%;
- whole token on DDR5-4800: −14.7% / −27.7%;
- 30B: −22.4% / −34.4%;
- held-out token;
- closed loop: −16.4% at 18.7 GB/s demand.

Source: H1–H3; §3.

**Q62. Does one matrix work everywhere?** Nearly: within 1.1 pt (open) and 3.2 pt (closed) of a self-searched one. A matrix fitted only on random frames loses up to 5 / 15 points on contiguous pages, so fit both. Source: §3.

**Q63. Is it safe?** In the placement sweep it never makes things worse: all ratios ≤ 0.874. Source: §3.

**Q64. Is the search method new?** No. XOR mapping searches are published (Rotaru 2026, Flatfish 2022). We claim the measurement, the cause and the negative results. Source: `related_work_2023_2026.md` §9.4.

**Q65. Does it help in closed loop?** In memory-demanding regimes: −16.4% cycles and −21% latency at 18.7 GB/s demand, and about −7% at 32 GB/s (16 threads). About 0% at 4 threads. Source: H3; §5.

### Why not the alternatives

**Q66. Per-region mapping?** It equals the global XOR within 0.5–2%. At batch 4 it's ahead on DDR4 but behind on DDR5 (1.000 vs 0.938). And it needs a tag table and allocator support. Source: §3.

**Q67. Expert-aware?** Within 1% of the default at B = 1, 4 and 16 and over 4 tokens. Source: §3.

**Q68. MOP4CLXOR?** It wins only in saturated DDR4 (by 5–25%). In closed loop it takes 4–38% more cycles, worse than the default. Source: §3; ctx "mop-owner".

**Q69. The channel hash?** −35% to −47% saturated, but only 0–2% beyond the bank XOR in closed loop. Source: §3; H3.

**Q70. Bit reversal?** Up to 2.4× slower: it breaks the weight stream's bank spread. Source: §3.

**Q71. The textbook permutation?** Within ~1 pt on DDR5 open page, so it captures most of the gain. But it's 4–8 pt worse with closed page and 6–10 pt worse on DDR4. Say this openly. Source: §3, §10.

**Q72. ChRaBaRoCo?** +40% to 3× cycles everywhere. Source: §3.

### Row-buffer policies

**Q73. Which policies?** Open, ClosedCAP, close page (DRAMsim3), Timeout (64 idle cycles), Region (close page for chosen classes), HAPPY-style reuse predictor. Source: §4.

**Q74. Did any beat open page?** The best non-open cell is 0.968 of open (HAPPY, one closed-loop cell); nothing else beats 0.976. Source: §4; `analysis/policy_results.md`.

**Q75. Why is closed page bad here?** Rows are closed before the next hit arrives. ClosedCAP costs 12–38% saturated and lowers row hit by 0.15–0.25. Source: §2, §4.

**Q76. How strong is the HAPPY negative?** *(source corrected)* HAPPY was built as a simplified Ramulator plugin: an address-hashed 2-bit reuse predictor, without the paper's global/per-thread history or tournament. It won one closed-loop cell by 3.2%. So it's a negative result for this traffic and this simplified predictor, not for HAPPY in general. Source: ctx "policy-dev: row-policy plugins" (`trace-pipeline/ramulator-ext/rowpolicy/`); `analysis/policy_results.md`.

### DRAM technologies

**Q77. Does the gain hold on LPDDR, HBM and GDDR?** Yes, and it's larger: −24% to −55% saturated (open page), vs −16.7% on DDR5; −7% to −23% in closed loop. Source: H5; §6.

**Q78. Why larger?** Smaller rows (1–2 KiB vs ~8 KiB), many narrow channels and fewer banks per channel make KV bank collisions cost more. Source: §6.

**Q79. What does a 4-thread CPU see there?** About 0% token change (≤1% on LPDDR); it's compute-bound. Source: §6; H5.

**Q80. Is GDDR6 meaningful for a CPU?** It's a what-if, not a real CPU system. Source: §6.

### GPU

**Q81. Does the bank XOR help on GPUs?** About 1.0× on the V100 (HBM2); +14–28% on the A16 (GDDR6) with FlashAttention on. Source: §7.

**Q82. What's the GPU lever?** Channel parallelism. At 32k: 2 KiB channel interleaving 1.82× (FlashAttention on); channel hash 1.35× (on) or 1.8× (off). Source: §7.

**Q83. Why is GPU row hit low?** About 70 multiprocessors read different rows at once. One multiprocessor's weight stream hits 0.87; merged, 0.76. A static bank hash can't fix concurrency. Source: §7.

**Q84. Did the L2 fix change the conclusions?** No, the sizes shrank 10–25%: interleave 2.3× → 1.82×, channel hash 1.46× → 1.35×. Source: §7.

**Q85. How do you know the A16 has ECC on?**
- ECC is enabled;
- 15,356 of 16,384 MiB is usable, the 1/16 inline-ECC reservation;
- DRAM reads run 14.3% above L2 misses.

It isn't directly isolated: there's no ECC counter, and turning ECC off needs admin. Source: §7.

### Token time vs DRAM time

**Q86. Why does token time barely improve at 4 threads?** At 4 threads decode is compute-bound (8.4–10 GB/s). DRAM savings show up in token time only when memory is the bottleneck. Source: §5.

**Q87. At what thread count does it pay?** At 16 threads native decode reads ~32 GB/s (358 ms/token), and the matched closed-loop point gives about −7%. At the 24–32-thread plateau (~43 GB/s) it's 1–2%. Source: §5; H3 "window 512" and "plateau".

**Q88. Then where does the 16% come from?** A memory-bound closed-loop point at 18.7 GB/s demand (fast cores), meaning a machine with faster cores or fewer channels per core, not this node at native settings. Source: H3; §5.

**Q89. Why is the gain smaller at the plateau than at 16 threads?** At the plateau the cores and two-socket layout limit throughput at ~15% of DRAM peak, so even large DRAM latency savings (−25%) barely move cycles. Source: H3; §5.

**Q90. What does it mean for serving?** At batch 4 the gain returns only with long per-sequence context: global XOR 0.884 (DDR4) / 0.937 (DDR5) open page at 32k, 1–4% at 8k. Source: ctx "serving-owner" and its check; `region/7B_B4_P32768`.

### Hardware cost

**Q91. What does the hash cost?** 5–6 XOR2 gates typical, at most 13 (11 with sharing), at most 2 levels (~3–5 FO4). A channel hash adds 3–4 gates. Source: `docs/xor_hardware_cost.md`.

**Q92. Area and latency?** Negligible area, and off the DRAM-timing critical path. Gate counts are exact; picoseconds and mm² are estimates good to about 2×. Source: same.

**Q93. Does it need to be programmable?** Matrices differ per DRAM organisation, so a product might expose a register: about 144 bits and ~800 gate equivalents, set at boot. Source: same.

**Q94. Do real CPUs already do this?** Yes. Intel server controllers use XOR functions of physical-address bits for channel and bank selection (DRAMA, Pessl et al., USENIX Security 2016; Sudoku, DRAMSec 2025). Source: `docs/xor_hardware_cost.md`; `related_work_2023_2026.md`.

### Limitations

**Q95. What is modelled, not measured?** Physical placement, the memory controller's address hash (undocumented by Intel and NVIDIA), and the GPU's thread-block-to-multiprocessor mapping. Source: §8.

**Q96. How many decode steps per configuration?** One, except the held-out token and the 4-token MoE run. Source: §8.

**Q97. Prefill coverage?** A 30-token prompt and a 32-token micro-batch of a 512-token prompt; 64-token micro-batches are running. Source: §8, §11.

**Q98. What isn't evaluated?** A controller that switches to MOP4CLXOR-style interleaving only when its queues saturate. Source: §11.

### Corrections (present them as a strength)

**Q99. What did you get wrong at first?**
- (a) The cause was first attributed to an aged allocator; it's the KV layout inside 2 MiB pages.
- (b) The channel hash's −35% to −47% is saturated-only.
- (c) The 4-thread rate was 4.8 GB/s from an unoptimised binary; it's 8.4–10.
- (d) The GPU L2 model overstated 32k by 13–32%.
- (e) Token time is ~7% at 16 threads, not "1–2% at best".

Source: §9.

**Q100. Why does admitting these help?** Each was caught by checking against an independent source: gem5, Nsight Compute, a held-out token, a second simulator, or raw-file re-derivation. Headline numbers were re-derived from raw replay files before publishing. Source: ctx "main session: verification…"; "check of serving-owner results".

### Novelty and related work

**Q101. Is XOR bank hashing new?** No: permutation-based interleaving (Zhang et al., MICRO 2000), Rotaru 2026 and Flatfish 2022 cover it. New: the measurement, the cause (KV layout in 2 MiB pages), and the negative results. Source: §0, §10.

**Q102. Closest prior work?**
- FACIL (HPCA 2025, per-huge-page selectable mapping);
- RoMe (HPCA 2026, LLM streams are bulk sequential);
- RH+ (arXiv 2606.05511);
- STARC (ASPLOS 2026, KV-to-bank placement on PIM).

Source: `related_work_2023_2026.md` §9.

**Q103. Has anyone measured this before?** Not that we found: no DRAM row-buffer study of CPU llama.cpp decode, and no open- vs closed-page comparison for LLM decode. The search isn't exhaustive. Source: same.

**Q104. Which conference programs were checked?** At title level: ISCA 2025/2026, MICRO 2025/2026, HPCA 2026. ASPLOS 2026 and DBLP were not fully checked. Source: same, §9.1.

**Q105. What not to cite?** UMDAM (withdrawn); HiKV and Mamo et al. for DRAM claims. Source: same.

**Q106. What to watch?** DUALign (MICRO 2026), a hardware-assisted memory mapping for PNM LLM serving. Only its title is verified, so read it before any "first" claim. Source: same.

### Next steps

**Q107. What would you do next?**
- a controller that switches to MOP4CLXOR-style interleaving only when saturated;
- a CPU channel-interleave granularity sweep (only 256 B was used);
- the remaining 30B batch-4 and 64-token prefill runs (running now).

Source: §11; tracker "Which row policy and which interleaving?".

**Q108. Would you build it?** The hardware is cheap. It pays off where decode is memory-bound with huge pages, so the next step is measuring on a memory-bound machine. Source: `docs/xor_hardware_cost.md`; §5.

---

## 3. Tough questions

**T1. Isn't the gain only in saturated replay?** No. Closed loop gives −16.4% when memory-bound and about −7% at the 16-thread native rate. It's ~0% at 4 threads. Saturated −16.7% is an upper bound. Source: H1, H3; §5.

**T2. Your placement is modelled, not real.** Correct; reading it needs root. We bracketed it with three models plus a contiguity sweep. The gain needs only 2 MiB pages, random or contiguous, and the XOR never makes things worse (≤ 0.874). A real kernel with 4 KiB pages would see no gain. Source: §1, §3, §8.

**T3. The textbook XOR already does this.** On DDR5 open page it's within ~1 point of ours, and we say so. Our claim is the measurement and the cause, not a new hash. With closed page and on DDR4 the searched matrix is 4–10 pt better. Source: §3, §10.

**T4. Why does token time barely improve?** At 4 threads decode is compute-bound (8.4–10 GB/s). At 16 threads (~32 GB/s) it's about 7%, and at the plateau 1–2%, because the cores cap demand at ~15% of DRAM peak. The DRAM-side gains are large, but token time follows only when memory is the bottleneck. Source: §5; H3.

**T5. Was the 32k number cherry-picked?** No. 32k traffic is quoted as a range (1.6–1.9×), and the gain is shown at every context: 17% at 32k, 6% at 8k, 2–4% at 1k, under 2% for prefill. Source: §0; H1.

**T6. One token and one matrix: overfitting?** Held-out token: 0.844 / 0.712 vs 0.840 / 0.698 in-sample. One matrix across models: within 1.1–3.2 pt. Source: §3.

**T7. MOP4CLXOR is simpler and beats you on DDR4.** Only saturated, by 5–25%. In every closed-loop case `xor_global` wins or ties, and MOP4CLXOR is worse than the default. Source: §3; ctx "mop-owner".

**T8. Your hardware cost is an estimate.** Gate counts and depths are exact for the searched matrices; area and delay are scaled estimates good to about 2×. Source: `docs/xor_hardware_cost.md`.

**T9. Your GPU numbers changed.** Yes. The L2 model overstated 32k by 32%, and we fixed it against Nsight Compute kernel by kernel. The corrected numbers are in §7, ±10% at 32k. Source: §7, §9.

**T10. The DynamoRIO–gem5 match could be luck.** It's one model and one step, across different instruction sets, thread counts and address types. Say so. Later checks (Nsight Compute on GPU, KV-append bytes, the LLC sweep) support the pipeline independently. Source: ctx "Phase split works"; §1.

**T11. The 16% on your slide is a what-if.** It's a memory-bound regime this node reaches only partially; at 16 threads it's ~7%. State the condition on the slide. Source: H3; §5.

**T12. Your row-policy negative could be a modelling artifact.** 150 cells across Ramulator built-ins and our plugins, open and closed loop; closed page is worse everywhere. HAPPY is a simplified version. It's a negative result for these streams, not a proof. Source: §4; `analysis/policy_results.md`.

**T13. Isn't DRAMsim3 weaker?** Saturated, yes: −2.7% vs −9.9% on 6 channels. In closed loop the two agree within 2 points. Source: §3; ctx "extras-sim-owner".

**T14. How much does the calibration drive the plateau result?** The plateau (~43 GB/s) is measured natively, and the model is calibrated to it. A different core model could move the result by a few points. Source: §1, §5.

**T15. Is the novelty claim safe?** Only as "to our knowledge". ASPLOS 2026 and DBLP aren't fully checked, and DUALign is unread. The defensible claim is the measurement, the cause and the negative results. Source: `related_work_2023_2026.md` §9.

---

## 4. Numbers cheat sheet

| # | Figure | Condition | Source |
|---|---|---|---|
| 1 | 0.7% | DRAM reads per token, DynamoRIO vs gem5, Qwen 0.5B (5,473,141 vs 5,434,710) | ctx "Phase split works" |
| 2 | 0.96–0.98 | Row hit, weight sweep, default mapping | §2 |
| 3 | 61% | KV share of 7B reads at 32k | §0 |
| 4 | 9.8–11.5 GB | 7B 32k DRAM reads per token (22 MiB LLC) | §2 |
| 5 | 0.847 → 0.951 | Row hit, 7B 32k, 2 MiB, DDR5 16 sub-ch, with XOR | H1 |
| 6 | −16.7% / −29.1% | Cycles, same, saturated, open / closed | H1 |
| 7 | 249 → 199 cycles | Read latency, same | H1 |
| 8 | −14.7% / −27.7% | Whole token, DDR5-4800 8 ch | H2 |
| 9 | −9.9% | Real node DDR4-2933 6 ch, saturated | H4 |
| 10 | −16.4% / −21% | Closed loop, 18.7 GB/s demand: cycles / latency | H3 |
| 11 | −7.1% / −7.9% | Closed loop ~32 GB/s (≈ native 16 threads), DDR5 16 / 8 ch | H3; §5 |
| 12 | −1.2% | Closed loop at the plateau (~40 GB/s) | H3 |
| 13 | ~0% | 4-thread native rate (8.4–10.8 GB/s) | §5; H3 |
| 14 | 1,149 / 576 / 358 / 264 ms | Native 7B 32k token at 4 / 8 / 16 / 32 threads (~10 / 20 / 32 / 43 GB/s) | §5 |
| 15 | −22.4% / −34.4% | Qwen3-30B-A3B 32k, DDR5 | H1 |
| 16 | −11.7% / −18.5% | Qwen 0.5B 32k, DDR5 | H1 |
| 17 | 0.844 / 0.712 vs 0.840 / 0.698 | Held-out vs in-sample, open / closed | §3 |
| 18 | ≤ 0.874 | Placement sweep, worst ratio (never hurts) | §3 |
| 19 | 0.968 | Best non-open policy (HAPPY, one cell); none beats open by > 3.2% | §4 |
| 20 | +12% to +38% | ClosedCAP cost, saturated | §4 |
| 21 | −35% to −47% / 0–2% | Channel hash: saturated / beyond bank XOR in closed loop | §3 |
| 22 | −5% to −25% / +4% to +38% | MOP4CLXOR: saturated DDR4 / closed loop | §3 |
| 23 | 0.5–2% | Region-aware vs global XOR | §3 |
| 24 | 5–6 XOR2 (max 13), ≤ 2 levels | Bank-hash hardware cost | `docs/xor_hardware_cost.md` |
| 25 | ~144 bits / ~800 GE | Programmable version | `docs/xor_hardware_cost.md` |
| 26 | 7.32 vs 7.40 GB | V100 32k, FlashAttention on, model vs Nsight Compute (fixed L2) | §7 |
| 27 | 1.82× / 1.35× | V100 32k: 2 KiB interleave / channel hash | §7 |
| 28 | −54.6% / −41.1% | GDDR6 (what-if), 7B 32k, saturated | H5 |
| 29 | −4.3% / −9.2% | DRAMsim3 closed loop, fast core, DDR4-2933 6 / 12 ch | §3 |
| 30 | −2.7% / −6.6% | DRAMsim3 saturated, DDR4 6 / 12 ch | §3 |

---

## 5. Sixty-second summary

LLM decode on a CPU is mostly a sweep over the weights, and the default DRAM mapping already keeps those reads in open rows (0.96–0.98 row hit). The trouble comes at long context. At 32k the KV cache is most of the traffic. With 2 MiB huge pages, the KV buffers of different heads sit at power-of-two offsets inside each page and collide in the same DRAM banks, which drops row hit to about 0.85.

We traced full-size Qwen models natively with DynamoRIO, validated within 0.7% of gem5, attributed every DRAM read to a tensor, and replayed the streams through Ramulator and DRAMsim3. A single global XOR bank hash, searched offline, recovers the lost row hits: about −17% DRAM cycles on DDR5 (−29% with closed page). It transfers to unseen tokens and other models, and costs about five XOR gates. Row-buffer policies, per-region and expert-aware mappings, and a channel hash add nothing in closed loop, so open page stays.

Token time follows only when decode is memory-bound: about 0% at 4 threads, about 7% at 16 threads, 1–2% at the 24–32-thread plateau, and up to 16% on a memory-bound machine. GPUs need a different lever, channel interleaving. What's new is the measurement and the cause; XOR bank hashing itself is old.
