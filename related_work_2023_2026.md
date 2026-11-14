# Related work 2023-2026: novelty check for AMCAS

Date of search: 2026-11-11. Method: web search, then each cited paper's arXiv abstract page was read (WebFetch) to confirm title, authors and date. Entries marked "(not verified at source)" were seen only through search snippets or aggregators. The search tool is not exhaustive: absence of a hit is weak evidence, not proof, that nothing exists. The survey in `problem_stmt_and_literature_survey.tex` already covers Shao 2005 (bit-reversal), Rotaru 2026 (GF(2) mappings), Yuan 2026 (HSRAI), Ghasempour 2016 (HAPPY) and Banasik 2025; those are not repeated below except where they matter.

Our findings, for reference:
1. Decode DRAM streams are weight sweeps with very high row-buffer hit under a default row-interleaved mapping.
2. Long-context KV-heavy streams under randomized (THP-shuffled) physical placement lose row locality; an XOR bank hash restricted to KV/activation pages recovers up to 17% (open) / 29% (closed page) cycles on DDR5.
3. A per-region (class-tagged 2 MiB granule) mapping equals but does not beat a global XOR hash.
4. MoE routing at batch 1 gives no mapping benefit; batching grows the expert union.

## 1. DRAM address mapping / interleaving for LLM inference

**UMDAM: A Unified Data Layout and DRAM Address Mapping for Heterogenous NPU-PIM.** Hai Huang. arXiv:2511.03293, Nov 2025 (cs.DC). https://arxiv.org/abs/2511.03293
Proposes a column-major tiled data layout plus a configurable DRAM address mapping so one layout serves both NPU and PIM in edge LLM decode; reports up to 3.0x lower time-to-first-token on OPT models. Confirmed 2026-11-14: arXiv shows "This paper has been withdrawn by Hai Huang" (v1 and v2 removed by arXiv administrators, licence issue). DROP it from the citation list.
Overlap/difference: it is the only paper found whose title ties DRAM address mapping to LLM inference, but it designs a layout for PIM/NPU co-execution on OPT models, not a bank-hash for a CPU/host DRAM stream, and it does not study physical-placement randomization.

**RACAM: Enhancing DRAM with Reuse-Aware Computation and Automated Mapping for ML Inference.** Siyuan Ma, Jiajun Hu, Jeeho Ryoo, Aman Arora, Lizy Kurian John. arXiv:2512.09304, Dec 2025. https://arxiv.org/abs/2512.09304
A bit-serial processing-in-DRAM design with locality buffers and an automated mapping of workloads onto DRAM parallelism; claims 9x-102x speedups over GPUs.
Overlap/difference: "mapping" here means compute-to-DRAM-hierarchy placement for PUD, not the host physical-address-to-bank function. Little relevance beyond showing the word is in use.

**RoMe: Row Granularity Access Memory System for Large Language Models.** Hwayong Nam, Seungmin Baek, Jumin Kim, Michael Jaemin Kim, Jung Ho Ahn. arXiv:2512.01541, Dec 2025 (cs.AR). https://arxiv.org/abs/2512.01541
Argues LLMs stream KB-to-MB blocks, so it replaces cache-line DRAM access with row-granularity commands and drops columns, bank groups and pseudo channels, using the saved pins for +12.5% bandwidth.
Overlap/difference: it rests on the same premise as our finding (1), that LLM streams are sequential bulk sweeps with row locality essentially guaranteed. It changes the interface; we show the existing interface plus a mapping already reaches ~0.96-0.98 hit on weights and identify where it does not (KV under random placement). Good citation to motivate finding (1).

**Flatfish: a Reinforcement Learning Approach for Application-Aware Address Mapping.** X. Li, Z. Yuan, Y. Guan, G. Sun, T. Zhang, R. Wei, D. Niu. IEEE TCAD 41(11):4758-4770, 2022. DOI 10.1109/TCAD.2022.3146204 (verified 2026-11-14 via OpenAlex record: authors, volume, issue, pages, year match).
Learns, from trace characteristics, a binary invertible matrix (a generalization of XOR mappings) as the application-specific address mapping.
Overlap/difference: outside the 2023-2026 window but the closest precedent for "per-application learned XOR/BIM mapping", alongside Rotaru 2026 already in the survey. It is not LLM-specific and does not consider OS page placement. Our greedy GF(2) search (`xor_gf2`) is conceptually the same family, so we must not claim the search method as new.

## 2. KV-cache access characterization (CPU, GPU/HBM, PIM)

**Dynamic Sparse Attention: Access Patterns and Architecture.** Noam Levy. arXiv:2603.13430, Mar 2026 (cs.AR). https://arxiv.org/abs/2603.13430
Logs which KV entries dynamic sparse attention touches per layer during decoding, shows many cache misses, and proposes an LLC reservation scheme with token-level LRU.
Overlap/difference: characterizes KV access at cache level for sparse attention; no DRAM row/bank analysis. Complements our dense-attention DRAM view.

**Sparse Attention Remapping with Clustering for Efficient LLM Decoding on PIM (STARC).** Zehao Fan, Garrett Gagnon, Zhenyu Liu, Liu Liu. arXiv:2505.05772; ASPLOS 2026 per the arXiv comments. https://arxiv.org/abs/2505.05772
Clusters KV pairs semantically and places each cluster in memory regions aligned with PIM bank structure; 19-31% lower attention latency on an HBM-PIM system.
Overlap/difference: a KV-to-bank data-placement scheme, the closest in spirit to our "KV pages get their own bank mapping". It is a software data layout for sparse attention on PIM, with accuracy implications, not a hardware address hash evaluated under random OS placement.

**PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System.** Hyucksung Kwon, Kyungmo Koo, Janghyeon Kim et al. arXiv:2412.20166 (v3 Dec 2025); accepted to HPCA 2026 per arXiv. https://arxiv.org/abs/2412.20166
Token-centric partitioning across channels, dynamic command scheduling and dynamic KV allocation for long contexts on PIM; up to 11.3x throughput on PIM-only systems.
Overlap/difference: addresses channel utilization and KV capacity management, not row-buffer behaviour in a conventional DRAM stream.

**Comparative Characterization of KV Cache Management Strategies for LLM Inference.** Oteo Mamo, Olga Kogiou, Hyunjin Yi, Weikuan Yu. arXiv:2604.05012, Apr 2026. https://arxiv.org/abs/2604.05012
Compares vLLM, InfiniGen and H2O on latency, throughput and memory usage. No DRAM-level measurement (confirmed from the abstract page). Not relevant to our claims, listed only so we do not cite it for DRAM behaviour.

**A Systematic Characterization of LLM Inference on GPUs.** Haonan Wang et al. arXiv:2512.01644, Dec 2025. https://arxiv.org/abs/2512.01644
Broad GPU phase and microarchitecture characterization. Abstract does not mention DRAM row behaviour; at most a general citation for decode being memory bound.

HiKV (Fang, Yin, Shi, Verhelst; arXiv:2607.22389, Jul 2026, to appear IEEE TCAS-I, https://arxiv.org/abs/2607.22389) is a KV compression accelerator; the abstract page does not discuss row behaviour (a search snippet claimed it notes row misses for sparse selection; not confirmed from the abstract). Not recommended as a citation.

## 3. Row-buffer / page policy for ML workloads

No paper found that compares open versus closed (or timeout/predictive) page policies for LLM decode streams. Closest:
- RoMe (above) sidesteps page policy by making every access row-granular.
- RH+ "Row-Hit-Optimized Scheduling for PIM-based LLM Inference" (now verified, arXiv:2606.05511, see Section 9; 8-12x speedup in simulation by changing the address stride so consecutive MACs share a row). It concerns PIM all-bank MAC command streams.
- HAPPY (Ghasempour 2016) is already in the survey.

Our open versus ClosedCAP numbers on THP-placed KV streams (29% vs 17% gain) appear to be a gap in the literature found.

## 4. MoE memory access, expert placement/offloading, DRAM

**Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference.** Haiyang Huang, Newsha Ardalani, Anna Sun, Liu Ke, Hsien-Hsin S. Lee, Anjali Sridhar, Shruti Bhosale, Carole-Jean Wu, Benjamin Lee. arXiv:2303.06182, 2023. https://arxiv.org/abs/2303.06182
Proposes dynamic gating, expert buffering (hot experts in GPU memory, rest in CPU memory) and load balancing. Capacity/placement level, not DRAM rows.

**Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models.** Jingcong Liang, Siyuan Wang, Miren Tian, Yitong Li, Duyu Tang, Zhongyu Wei. arXiv:2505.16056 (v4 Feb 2026). https://arxiv.org/abs/2505.16056
Metrics for expert reuse across neighboring tokens over 20 MoE models; cache of about 2x the active experts is a practical balance.
Overlap: supports our MoE-across-tokens measurement (4 consecutive tokens' union = 1.55x one token's experts, i.e. strong reuse). They measure routing; we measure DRAM traffic and footprint.

**In-depth Analysis on Caching and Pre-fetching in Mixture of Experts Offloading.** Shuning Lin, Yifan He, Yitong Chen. arXiv:2511.05814, Nov 2025. https://arxiv.org/abs/2511.05814
LRU/LFU expert-cache analysis and speculative prefetch. Same level as above.

**Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching.** Sungmin Yun, Kwanhee Kyung, Juhwan Cho, Jaewan Choi, Jongmin Kim, Byeongho Kim, Sukhan Lee, Kyomin Sohn, Jung Ho Ahn. MICRO 2024 (per arXiv listing); arXiv:2409.01141. https://arxiv.org/abs/2409.01141
xPU plus Logic-PIM; MoE experts with few tokens go to PIM. Argues MoE and GQA attention with continuous batching are low-Op/B and DRAM-access heavy.
Overlap: agrees that batching changes the MoE access picture; it is an architecture paper and does not study address mapping.

MoNDE (CXL near-data MoE, experts and activations on separate DRAM banks) was seen only in a search snippet (not verified at source) and is not cited.

No paper found that characterizes MoE expert reads at DRAM row/bank level or evaluates an expert-aware address mapping.

## 5. PIM attention (row behaviour only)

**NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing.** Guseul Heo, Sangyeop Lee, Jaehong Cho, Hyunmin Choi, Sanghyeon Lee, Hyungkyu Ham, Gwangsun Kim, Divya Mahajan, Jongse Park. ASPLOS 2024; arXiv:2403.00579. https://arxiv.org/abs/2403.00579
Dual row buffers per bank so normal accesses overlap PIM GEMV, plus sub-batch interleaving; 3x over GPU-only. Row-buffer-aware at the circuit level; no mapping study.

**AttAcc (Unleashing the Potential of PIM: Accelerating Large Batched Inference of Transformer-Based Generative Models).** Jaehyun Park et al. See Section 9 for the corrected citation (the 2023 CAL letter and the ASPLOS 2024 paper are two different papers, and the author list given here earlier was wrong). PIM for batched attention. Not about row behaviour.

**CHIME** (Liu et al., arXiv:2504.17584, https://arxiv.org/abs/2504.17584) is a DIMM-PIM attention/FC disaggregation system; mapping details are not in the abstract. Not used.

## 6. CPU LLM inference memory studies

**Memory Access Characterization of Large Language Models in CPU Environment and its Potential Impacts.** Spencer Banasik. arXiv:2506.01827, Jun 2025 (cs.LG). https://arxiv.org/abs/2506.01827
Already in our survey: llama.cpp plus Qwen 0.5B traces, cache/prefetcher evaluation, single-threaded. Remains the only CPU llama.cpp memory-trace paper found, and it stops at the cache level. Our work extends it to full-scale models, DRAM replay and DRAM mapping.

No other peer-reviewed or arXiv study of llama.cpp DRAM row behaviour found (a Hugging Face forum post and a blog post report DRAM bandwidth only, not cited).

## 7. Learned / application-specific XOR mappings

Beyond Rotaru 2026 and Flatfish (section 1): the search surfaced Kumar and Desai 2009 (RL mapping for MPSoC, lookup table) and AIMM (NMP page/computation remapping), neither relevant to bank hashes for LLM. Optimizing Memory Mapping Using Deep Reinforcement Learning (mallocMuZero, Wang et al., arXiv:2305.07440) is about tensor-to-memory-level placement in compilers, not DRAM address bits; do not cite as DRAM mapping. No 2024-2026 learned XOR bank hash for LLM workloads found.

## 8. Physical placement (THP / huge pages) versus DRAM mapping

Searches returned only classical bank-partitioning and page-coloring literature (e.g. Erez et al. HPCA 2012 bank partitioning; noted that with 2 MiB huge pages the bank bits above bit 21 cannot be controlled by the OS). Nothing on THP-induced row-locality loss for LLM KV streams was found. This supports our mechanism explanation but I did not verify the individual papers; cite them only after reading.

## Novelty assessment

**Claim 1 (decode is a high-row-hit weight sweep under default mapping).** Not novel as an idea: RoMe (2512.01541) explicitly builds on LLMs streaming large blocks, and the PIM papers assume the same. What looks new is the measurement: full-scale Qwen2.5-7B and Qwen3-30B-A3B captured at the LLC-miss level on a CPU, replayed through Ramulator with row-hit numbers (0.96-0.98). Cite RoMe and Banasik as context, present ours as quantitative confirmation.

**Claim 2 (KV under randomized physical placement loses row locality; XOR hash on KV/activation pages recovers 17%/29%).** No paper found makes this claim. Nearest are STARC (KV-to-bank placement on PIM, software layout, sparse attention) and PIMphony/UMDAM (PIM/NPU layouts). The interaction of OS huge-page placement with KV-stream bank conflicts, and the closed-page amplification, look new. Hedge: our XOR matrices are searched in-sample on a single token (as the project log notes), and the search method itself is not new (Rotaru 2026, Flatfish 2022). Claim the finding and the class-restricted application, not the hash-search method.

**Claim 3 (per-region mapping equals but does not beat global XOR).** Looks new, but it is a negative result; no directly comparable published work found. Region-aware mapping resembles STARC's bank-aligned placement in spirit only.

**Claim 4 (MoE at batch 1 gives no mapping benefit; batching grows expert union).** The union-growth part is consistent with Liang et al. (routing consistency), Huang et al. 2023 (expert buffering) and Duplex (batching and MoE). The DRAM-level negative result (whole contiguous expert slices already stream at ~0.96 row hit) and the batched-DRAM-traffic numbers (4.9x reads for B=16) look new; no DRAM-level MoE characterization found.

**What we should cite:** Rotaru 2026 and Flatfish (learned/XOR mapping precedent); RoMe (LLM streams are sequential bulk accesses); Banasik 2025 (CPU llama.cpp characterization, cache-level only); STARC and PIMphony/NeuPIMs/Duplex (PIM works that treat row/bank placement for attention and MoE); Levy 2026 (KV access at cache level); Liang 2025 and Huang 2023 (MoE routing locality). Avoid citing UMDAM (withdrawn), HiKV, and Mamo et al. for DRAM claims.

**Caveats.** The search tool is US-centric and snippet-based; conference proceedings (ISCA/MICRO/HPCA/ASPLOS 2025-2026) may contain DRAM-level LLM characterizations not indexed. Before making a "first" claim, do a targeted manual pass over those venues' 2025-2026 programs and DBLP for "address mapping" + "LLM".


## 9. Conference program check (2025-2026)

Date of check: 2026-11-14. Method: fetched the program pages with WebFetch and filtered titles by keyword (DRAM, memory, HBM, bank, row, address, mapping, interleav, page, KV, MoE, expert, CPU); web search and Consensus for arXiv. Caveats: titles only (programs carry no abstracts), so a title can hide a relevant paper; the keyword filter runs over a model-summarised page, not a raw grep; the MICRO 2026 page was read in two halves, the HPCA 2026 page only up to Tuesday 3 Feb afternoon (last 45k characters unread).

### 9.1 What was checked

| Venue | Source | Status |
|---|---|---|
| ISCA 2025 | https://iscaconf.org/isca2025/program/ | Program online, checked (18 keyword hits). No mapping/interleaving/row-buffer-for-LLM title. |
| ISCA 2026 (Raleigh, 27 Jun-1 Jul 2026) | https://www.iscaconf.org/isca2026/program/ | Program online, checked (37 keyword hits). No address-mapping, row-buffer or bank-conflict title. Nearest: "Combating the Memory Walls: Optimization Pathways for Long-Context Agentic LLM Inference" (Wu et al.), "Patterns Behind Chaos: Forecasting Data Movement for Efficient Large-Scale MoE LLM Inference" (Yu et al.), "ConServe: Contiguity-Preserving Memory Management for Multi-Turn LLM Serving" (Li). Abstracts not read, so relevance of these three is unconfirmed (they look like capacity/system-level work). |
| MICRO 2025 (18-22 Oct 2025) | https://microarch.org/micro58/program/ | Program online, checked (19 keyword hits). Hits: Stratum (3D-DRAM MoE), Kelle, LongSight, "Leveraging Chiplet-Locality for Efficient Memory Mapping in Multi-Chip Module GPUs" (Park et al.; a chiplet page/memory mapping paper, not DRAM bank mapping), ColumnDisturb, Citadel. No DRAM bank/row mapping for LLM. |
| MICRO 2026 (31 Oct-4 Nov 2026, Athens) | https://microarch.org/micro59/program/ | Program online, fully read. Relevant: DUALign, Dual-View Memory (PFM), RowScope, PARE, Helios, AMMA, C3, SYMPHONY (details in 9.2). |
| HPCA 2025 | no program fetched; FACIL found via web search (project page of first author) | Partial. |
| HPCA 2026 (Sydney, 31 Jan-4 Feb) | https://2026.hpca-conf.org/program/program-hpca-2026/ | Program online, read to Tuesday 3 Feb afternoon. Relevant: RoMe, BARD (DDR5 write latency via bank parallelism), AQPIM, LILo, PIM-malloc. PIMphony is HPCA 2026 per its arXiv listing only (not seen in the portion read). |
| ASPLOS 2026 | official ASPLOS page returned 404; DBLP blocked (Anubis); only a search snippet (167 papers, Pittsburgh 24-26 Mar) | NOT checked in full. STARC is ASPLOS '26 per the authors' revised arXiv note. The claim "no ASPLOS 2026 DRAM-mapping-for-LLM paper" is unverified. |
| DAC 2026 | web search only; REFLEX found on the SKKU lab page | Not systematically checked. |
| IISWC, SC, MEMSYS, IEEE CAL 2025-2026 | web search only | Programs not retrieved; searches surfaced nothing relevant at DRAM level (CAL: Fang et al., "Accelerating LLM Inference via Dynamic KV Cache Placement in Heterogeneous Memory System", arXiv 2508.13231, is placement across HBM and DRAM at the capacity/bandwidth level, not rows). DBLP is blocked from this environment, so these remain a gap. |
| arXiv 2025-2026 | WebSearch plus Consensus (6 queries) | Done, non-exhaustive. |

### 9.2 New relevant papers

Verified means the arXiv abstract page or the conference program page was opened and the title and authors read.

1. **FACIL: Flexible DRAM Address Mapping for SoC-PIM Cooperative On-device LLM Inference.** Seong Hoon Seo, Junghoon Kim, Donghyun Lee, Seonah Yoo, Seokwon Moon, et al. (the full author list was not read at source; check before citing). HPCA 2025. Source: first author's publication page (not opened successfully; the facts come from a search result summary, so treat as partly verified). Memory controller applies a different DRAM address mapping to the page-offset bits of each 2 MiB huge page, chosen by a user library, so SoC and PIM can both read LLM weights with the layout each wants (2.37x lower time-to-first-token). Overlap: closest published precedent for "per-huge-page selectable DRAM mapping" (our claim 3 region mapping, and the interplay of huge pages with mapping in claim 2). It targets SoC-vs-PIM layout for weights, not KV bank conflicts or hashing, and gives no row-hit analysis of a host CPU stream.
2. **DUALign: A Hardware-Assisted Memory Mapping Strategy for PNM-Accelerated LLM Serving.** Euijun Kim, Joshua Gabriel Dela Rosa, Bogil Kim, Youngin Kim, Seungyong Lee, Joonseop Sim, Youngpyo Joo, William Song. MICRO 2026 (program page, https://microarch.org/micro59/program/). Verified title and authors from the program only; no abstract or arXiv found. Likely a PNM (processing-near-memory) layout/mapping paper. Overlap: possibly claims 2/3; MUST read the paper when available before any "first" claim.
3. **Rethinking Unified Memory for NPU-PIM Systems: Dual-View Memory for Dynamic Inference of LLM (PFM).** Shixin Zhao, Lian Liu, Tianhua Han, Mengdi Wang, Yinhe Han, Ying Wang. arXiv:2608.06989, accepted MICRO'26. https://arxiv.org/abs/2608.06989 (verified). Decouples physical data layout from per-accessor logical views with accessor-aware address translation, 2.32x throughput; the abstract does not mention bank hashing or row-buffer hits (a search snippet said the full text uses 2 MB superpages; unverified). Overlap: claim 3, loosely claim 2.
4. **RoMe** (already in section 1) is confirmed HPCA 2026 (arXiv comments: "accepted at HPCA 2026"; verified; also on the HPCA 2026 program). Cite as HPCA 2026, not arXiv-only.
5. **RH+: Row-Hit-Optimized Scheduling for PIM-based LLM Inference.** Yongchan Jung, Shafayat Mowla Anik, Byeong Kil Lee, Jeeho Ryoo. arXiv:2606.05511, 3 Jun 2026 (cs.ET); no venue on the page. https://arxiv.org/abs/2606.05511 (verified). Argues host-centric address interleaving puts every all-bank MAC command in a different row; changes stride so 32 consecutive MACs share a row (8.25-11.88x in Ramulator 2.0 HBM3-PIM). Overlap: it is a PIM-specific address-stride change motivated by the same row-locality loss; it does not cover host DRAM streams, page placement or XOR hashing. Relevant to claim 3 only as "row-aware interleaving matters for LLM streams".
6. **REFLEX: Rewrite-Free Row-Aligned Sparse Attention for Efficient LLM Execution on PIM.** Juhong Park, Yintao He, Sangheum Yeon, Yiran Chen, Jong Hwan Ko. DAC 2026 (63rd), per the SKKU lab publication page (https://iris.skku.edu/publication/c94_dac_2026/, verified). Places the needed sparse-attention KV entries in one DRAM row to avoid scattered activations on PIM (1.64x). Overlap: KV-to-row placement, claim 2 in spirit; software layout on PIM.
7. **STARC** (section 2) is confirmed ASPLOS '26 (author's revised-version note, arXiv:2505.05772, verified). The title has changed: now "STARC: Selective Token Access with Remapping and Clustering for Efficient LLM Decoding on PIM Systems" (Consensus record; arXiv abstract page still shows the older name). Use the ASPLOS title in the bib.
8. **Sudoku: Decomposing DRAM Address Mapping into Component Functions.** Minbok Wi, Seungmin Baek, Seonyong Park, Mattan Erez, Jung Ho Ahn. DRAMSec 2025, arXiv:2506.15918 (verified). Reverse-engineers CPU DRAM address functions (XOR bank functions) on Intel/AMD. Overlap: supplies the premise that real CPU controllers already use XOR bank functions, which affects how we describe our "default" mapping and our simulated hash; no LLM content.
9. **SGDRC: Software-Defined Dynamic Resource Control for Concurrent DNN Inference on NVIDIA GPUs.** Yongkang Zhang et al. arXiv:2407.13996 (2024, last revised Mar 2025; verified abstract page). Reverse-engineers the VRAM channel-hash mapping of NVIDIA GPUs and uses cache coloring to avoid channel conflicts for co-located DNN services. Overlap: claim 4 (GPU channel hash exists and matters); it is a multi-tenant resource-control paper, not KV/attention bank or channel mapping.
10. **Nazaraliyev et al., USENIX Security 2025** (GPU GDDR physical-page-to-bank mapping reverse engineering). The PDF could not be parsed, title and authors NOT verified; do not cite until read. Overlap: claim 4 (GPU DRAM addressing is hashed and known).
11. **ProfInfer: An eBPF-based Fine-grained LLM Inference Profiler.** arXiv:2601.20755, Jan 2026 (seen via search result only, not opened). Per-operator HPC (cache misses) for llama.cpp. Overlap: claim 1 context only; no DRAM rows.
12. Hits that were opened or seen and judged NOT relevant to our claims: Kelle, LongSight, Stratum, Helios, SYMPHONY, AMMA, C3 (collaborative CPU-DRAM computation for LLM, MICRO 2026; title only), LILo (HPCA 2026, CPU on-chip accelerators for compressed LLM inference), BARD (HPCA 2026, DDR5 write latency via bank parallelism; general scheduling, not LLM), PARE (MICRO 2026, high-bandwidth DRAM controller), RowScope (MICRO 2026, ML reverse engineering of DRAM organisation), Folded Banks (ISCA 2025, HBM design for random access), CHIME (cited as ISCA '26 by a secondary source only; not in the ISCA 2026 keyword hits, so venue unconfirmed), KVNAND, vAttention (2 MB allocation granularity for KV, relevant only as evidence that GPU KV buffers are allocated in 2 MiB multiples). None reports DRAM bank/row behaviour of dense LLM KV streams on host DRAM.

Searches for: bank conflicts of KV caches under huge pages, XOR bank hashing for transformer traffic, open vs closed page for LLM, MoE expert reads at DRAM-row level, llama.cpp DRAM row behaviour, HBM/GDDR channel mapping for attention kernels returned nothing beyond the above. Absence of a hit remains weak evidence.

### 9.3 Corrected citations

- **Flatfish** (verified via OpenAlex): Xingchen Li, Zhihang Yuan, Yijin Guan, Guangyu Sun, Tao Zhang, Rongshan Wei, Dimin Niu. IEEE TCAD 41(11):4758-4770, 2022 (published online 25 Jan 2022). DOI 10.1109/TCAD.2022.3146204. The earlier BibTeX was right; only the "unverified" flag is removed.
- **AttAcc: two different papers, our earlier entry mixed them up.**
  (a) Choi, Park, Kyung, Kim, Ahn, "Unleashing the Potential of PIM: Accelerating Large Batched Inference of Transformer-Based Generative Models", IEEE CAL 22(2):113-116, 2023, DOI 10.1109/LCA.2023.3305386 (verified at Illinois experts page). The earlier entry attributed it to "Jaehyun Park et al." and the "3.19x" number; the author order is Jaewan Choi first.
  (b) Park, Choi, Kyung, Kim, Kwon, Kim, Ahn, "AttAcc! Unleashing the Power of PIM for Batched Transformer-based Generative Model Inference", ASPLOS '24, vol. 2, pp. 103-119, DOI 10.1145/3620665.3640422 (verified via Crossref). The name AttAcc belongs to this one. Cite (b).
- **RH+**: exact title "RH+: Row-Hit-Optimized Scheduling for PIM-based LLM Inference", Yongchan Jung, Shafayat Mowla Anik, Byeong Kil Lee, Jeeho Ryoo, arXiv:2606.05511 (3 Jun 2026), preprint with no venue. Not peer reviewed as far as the page shows.
- **UMDAM**: withdrawn by the author (arXiv status text confirmed). Dropped from BibTeX. FACIL, not UMDAM, is the reference for huge-page-aware mapping. (Beware: the UMDAM text is still quoted by search aggregators.)
- **RoMe**: HPCA 2026. **PIMphony**: arXiv comments say HPCA 2026 (not yet seen on the program page I read). **STARC**: ASPLOS '26, renamed.

### 9.4 Revised novelty assessment (supersedes the earlier "Novelty assessment" where they differ)

**Claim 1 (CPU decode DRAM traffic is a weight sweep, 0.96-0.98 row hit under default mapping).** Verdict: partly published. The premise (LLM decode streams large sequential blocks, row locality is easy) is stated in RoMe (HPCA 2026) and RH+ (arXiv 2026), and Banasik (2025) characterised llama.cpp memory traces at cache level. No paper found that measures row-buffer hit rate of full-scale CPU llama.cpp decode through a DRAM simulator. The measurement is new; the idea is not. Cite: RoMe, Banasik, RH+.

**Claim 2 (huge-page power-of-two KV layout collides in banks, row hit 0.80-0.86; a global XOR bank hash recovers -16..-29% DRAM cycles; 0-2% token time on a compute-bound Xeon).** Verdict: new as a finding and as an end-to-end negative result; the mechanism ingredients are published. Published: XOR bank hashing in general (Rotaru 2026, Flatfish 2022, Shao 2005 bit reversal, and the Sudoku reverse engineering showing CPUs already use XOR bank functions), per-huge-page selectable mapping (FACIL, HPCA 2025), KV-to-row/bank placement on PIM (STARC, REFLEX, PIMphony), row-aware interleaving for PIM (RH+). Not found: any study of KV streams on a conventional host controller under 2 MiB pages and power-of-two head strides, the transfer of one searched hash across tokens and models, or the finding that the DRAM gain does not become token time on a core-bound CPU. Caveat: DUALign (MICRO 2026) is unread and might overlap; re-check before submission. We must not claim the hash-search method; we must cite FACIL as the huge-page/mapping precedent.

**Claim 3 (per-region class-tagged and MoE-expert-aware mapping give nothing over global XOR; no page policy beats open page by more than 3.2%; MOP4CLXOR wins only in saturated DDR4).** Verdict: new as negative results (no directly comparable publication found), though the idea of class- or region-specific mapping is published (FACIL, PFM/Dual-View Memory, DUALign pending). No paper found comparing open/closed/timeout/HAPPY-style policies on LLM streams; RoMe argues policy becomes moot with row-granularity access, which is consistent with our result but not evidence for it. Cite: FACIL, PFM, RoMe, HAPPY (Ghasempour 2016).

**Claim 4 (GPU: V100 HBM2 and A16 GDDR6 are SM-concurrency/channel-parallelism bound; bank XOR ~1.0x, channel interleave/hash 1.35-1.82x; V100 L2 needs a hashed set index for KV-heavy attention).** Verdict: partly published. That NVIDIA VRAM uses a channel hash and that channel conflicts matter is published (SGDRC arXiv 2024; USENIX Security 2025 on GDDR bank mapping, citation unverified). GPU characterisations (Wang 2025, "Mind the Memory Gap" arXiv:2503.08311 on DRAM-bandwidth saturation in attention) say decode is DRAM-bandwidth bound. Not found: a quantified comparison of bank hash vs channel mapping for attention kernels, or the L2 set-hash correction for KV-heavy kernels. The simulator-modelling observation is a methodological result of ours; present it as such.

**Must-cite list (updated):** FACIL (HPCA 2025), RoMe (HPCA 2026), RH+ (arXiv 2026), STARC (ASPLOS 2026), PIMphony (HPCA 2026), REFLEX (DAC 2026), PFM/Dual-View Memory (MICRO 2026), Sudoku (DRAMSec 2025), SGDRC (arXiv 2024), AttAcc (ASPLOS 2024, version b), Flatfish (TCAD 2022), Rotaru 2026, Banasik 2025, HAPPY 2016. Drop UMDAM. Watch: DUALign (MICRO 2026). Not to be cited for DRAM claims: HiKV, Mamo et al.

**Open gaps in this check:** ASPLOS 2026, HPCA 2026 (last part), ISCA 2025 sessions beyond keyword filters, IISWC/SC/MEMSYS/CAL programs, and DBLP (blocked). A first-claim sentence in a paper should say "to our knowledge" and be re-checked against DBLP by hand.

## BibTeX

```bibtex
@misc{ma2025racam,
  title={RACAM: Enhancing DRAM with Reuse-Aware Computation and Automated Mapping for ML Inference},
  author={Ma, Siyuan and Hu, Jiajun and Ryoo, Jeeho and Arora, Aman and John, Lizy Kurian},
  year={2025}, eprint={2512.09304}, archivePrefix={arXiv}
}
@inproceedings{nam2026rome,
  title={RoMe: Row Granularity Access Memory System for Large Language Models},
  author={Nam, Hwayong and Baek, Seungmin and Kim, Jumin and Kim, Michael Jaemin and Ahn, Jung Ho},
  booktitle={HPCA}, year={2026}, eprint={2512.01541}, archivePrefix={arXiv}
}
@article{li2022flatfish,
  title={Flatfish: a Reinforcement Learning Approach for Application-Aware Address Mapping},
  author={Li, Xingchen and Yuan, Zhihang and Guan, Yijin and Sun, Guangyu and Zhang, Tao and Wei, Rongshan and Niu, Dimin},
  journal={IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems},
  volume={41}, number={11}, pages={4758--4770}, year={2022},
  doi={10.1109/TCAD.2022.3146204}
}
@misc{levy2026dsa,
  title={Dynamic Sparse Attention: Access Patterns and Architecture},
  author={Levy, Noam},
  year={2026}, eprint={2603.13430}, archivePrefix={arXiv}
}
@inproceedings{fan2026starc,
  title={{STARC}: Selective Token Access with Remapping and Clustering for Efficient {LLM} Decoding on {PIM} Systems},
  author={Fan, Zehao and Gagnon, Garrett and Liu, Zhenyu and Liu, Liu},
  booktitle={ASPLOS}, year={2026}, eprint={2505.05772}, archivePrefix={arXiv}
}
@inproceedings{kwon2026pimphony,
  title={PIMphony: Overcoming Bandwidth and Capacity Inefficiency in PIM-based Long-Context LLM Inference System},
  author={Kwon, Hyucksung and Koo, Kyungmo and Kim, Janghyeon and others},
  booktitle={HPCA}, year={2026}, eprint={2412.20166}, archivePrefix={arXiv}
}
@inproceedings{heo2024neupims,
  title={NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing},
  author={Heo, Guseul and Lee, Sangyeop and Cho, Jaehong and Choi, Hyunmin and Lee, Sanghyeon and Ham, Hyungkyu and Kim, Gwangsun and Mahajan, Divya and Park, Jongse},
  booktitle={ASPLOS}, year={2024}, eprint={2403.00579}, archivePrefix={arXiv}
}
@inproceedings{yun2024duplex,
  title={Duplex: A Device for Large Language Models with Mixture of Experts, Grouped Query Attention, and Continuous Batching},
  author={Yun, Sungmin and Kyung, Kwanhee and Cho, Juhwan and Choi, Jaewan and Kim, Jongmin and Kim, Byeongho and Lee, Sukhan and Sohn, Kyomin and Ahn, Jung Ho},
  booktitle={MICRO}, year={2024}, eprint={2409.01141}, archivePrefix={arXiv}
}
@misc{huang2023moe,
  title={Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference},
  author={Huang, Haiyang and Ardalani, Newsha and Sun, Anna and Ke, Liu and Lee, Hsien-Hsin S. and Sridhar, Anjali and Bhosale, Shruti and Wu, Carole-Jean and Lee, Benjamin},
  year={2023}, eprint={2303.06182}, archivePrefix={arXiv}
}
@misc{liang2025routing,
  title={Not All Models Suit Expert Offloading: On Local Routing Consistency of Mixture-of-Expert Models},
  author={Liang, Jingcong and Wang, Siyuan and Tian, Miren and Li, Yitong and Tang, Duyu and Wei, Zhongyu},
  year={2025}, eprint={2505.16056}, archivePrefix={arXiv}
}
@misc{lin2025moecache,
  title={In-depth Analysis on Caching and Pre-fetching in Mixture of Experts Offloading},
  author={Lin, Shuning and He, Yifan and Chen, Yitong},
  year={2025}, eprint={2511.05814}, archivePrefix={arXiv}
}
@misc{banasik2025memory,
  title={Memory Access Characterization of Large Language Models in CPU Environment and its Potential Impacts},
  author={Banasik, Spencer},
  year={2025}, eprint={2506.01827}, archivePrefix={arXiv}
}
@misc{wang2025gpuchar,
  title={A Systematic Characterization of LLM Inference on GPUs},
  author={Wang, Haonan and Xiao, Xuxin and Yan, Mingyu and others},
  year={2025}, eprint={2512.01644}, archivePrefix={arXiv}
}
@inproceedings{seo2025facil,
  title={FACIL: Flexible DRAM Address Mapping for SoC-PIM Cooperative On-device LLM Inference},
  author={Seo, Seong Hoon and Kim, Junghoon and Lee, Donghyun and Yoo, Seonah and Moon, Seokwon and others},
  booktitle={IEEE International Symposium on High-Performance Computer Architecture (HPCA)}, year={2025},
  note={Author list incomplete; verify before submission}
}
@misc{jung2026rhplus,
  title={{RH+}: Row-Hit-Optimized Scheduling for {PIM}-based {LLM} Inference},
  author={Jung, Yongchan and Anik, Shafayat Mowla and Lee, Byeong Kil and Ryoo, Jeeho},
  year={2026}, eprint={2606.05511}, archivePrefix={arXiv}
}
@inproceedings{park2024attacc,
  title={{AttAcc}! Unleashing the Power of {PIM} for Batched Transformer-based Generative Model Inference},
  author={Park, Jaehyun and Choi, Jaewan and Kyung, Kwanhee and Kim, Michael Jaemin and Kwon, Yongsuk and Kim, Nam Sung and Ahn, Jung Ho},
  booktitle={ASPLOS (Volume 2)}, pages={103--119}, year={2024}, doi={10.1145/3620665.3640422}
}
@article{choi2023pimcal,
  title={Unleashing the Potential of {PIM}: Accelerating Large Batched Inference of Transformer-Based Generative Models},
  author={Choi, Jaewan and Park, Jaehyun and Kyung, Kwanhee and Kim, Nam Sung and Ahn, Jung Ho},
  journal={IEEE Computer Architecture Letters}, volume={22}, number={2}, pages={113--116}, year={2023},
  doi={10.1109/LCA.2023.3305386}
}
@misc{zhao2026pfm,
  title={Rethinking Unified Memory for {NPU-PIM} Systems: Dual-View Memory for Dynamic Inference of {LLM}},
  author={Zhao, Shixin and Liu, Lian and Han, Tianhua and Wang, Mengdi and Han, Yinhe and Wang, Ying},
  year={2026}, eprint={2608.06989}, archivePrefix={arXiv}, note={MICRO 2026}
}
@inproceedings{park2026reflex,
  title={{REFLEX}: Rewrite-Free Row-Aligned Sparse Attention for Efficient {LLM} Execution on {PIM}},
  author={Park, Juhong and He, Yintao and Yeon, Sangheum and Chen, Yiran and Ko, Jong Hwan},
  booktitle={Design Automation Conference (DAC)}, year={2026}
}
@misc{wi2025sudoku,
  title={Sudoku: Decomposing {DRAM} Address Mapping into Component Functions},
  author={Wi, Minbok and Baek, Seungmin and Park, Seonyong and Erez, Mattan and Ahn, Jung Ho},
  year={2025}, eprint={2506.15918}, archivePrefix={arXiv}, note={DRAMSec 2025}
}
@misc{zhang2024sgdrc,
  title={{SGDRC}: Software-Defined Dynamic Resource Control for Concurrent {DNN} Inference on {NVIDIA} {GPUs}},
  author={Zhang, Yongkang and Yu, Haoxuan and Han, Chenxia and Wang, Cheng and Lu, Baotong and Li, Yunzhe and Jiang, Zhifeng and Li, Yang and Chu, Xiaowen and Li, Huaicheng},
  year={2024}, eprint={2407.13996}, archivePrefix={arXiv}
}
```
