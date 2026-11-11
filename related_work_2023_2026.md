# Related work 2023-2026: novelty check for AMCAS

Date of search: 2026-11-11. Method: web search, then each cited paper's arXiv abstract page was read (WebFetch) to confirm title, authors and date. Entries marked "(not verified at source)" were seen only through search snippets or aggregators. The search tool is not exhaustive: absence of a hit is weak evidence, not proof, that nothing exists. The survey in `problem_stmt_and_literature_survey.tex` already covers Shao 2005 (bit-reversal), Rotaru 2026 (GF(2) mappings), Yuan 2026 (HSRAI), Ghasempour 2016 (HAPPY) and Banasik 2025; those are not repeated below except where they matter.

Our findings, for reference:
1. Decode DRAM streams are weight sweeps with very high row-buffer hit under a default row-interleaved mapping.
2. Long-context KV-heavy streams under randomized (THP-shuffled) physical placement lose row locality; an XOR bank hash restricted to KV/activation pages recovers up to 17% (open) / 29% (closed page) cycles on DDR5.
3. A per-region (class-tagged 2 MiB granule) mapping equals but does not beat a global XOR hash.
4. MoE routing at batch 1 gives no mapping benefit; batching grows the expert union.

## 1. DRAM address mapping / interleaving for LLM inference

**UMDAM: A Unified Data Layout and DRAM Address Mapping for Heterogenous NPU-PIM.** Hai Huang. arXiv:2511.03293, Nov 2025 (cs.DC). https://arxiv.org/abs/2511.03293
Proposes a column-major tiled data layout plus a configurable DRAM address mapping so one layout serves both NPU and PIM in edge LLM decode; reports up to 3.0x lower time-to-first-token on OPT models. Note: arXiv marks this version as withdrawn (license issue), so cite with care or not at all.
Overlap/difference: it is the only paper found whose title ties DRAM address mapping to LLM inference, but it designs a layout for PIM/NPU co-execution on OPT models, not a bank-hash for a CPU/host DRAM stream, and it does not study physical-placement randomization.

**RACAM: Enhancing DRAM with Reuse-Aware Computation and Automated Mapping for ML Inference.** Siyuan Ma, Jiajun Hu, Jeeho Ryoo, Aman Arora, Lizy Kurian John. arXiv:2512.09304, Dec 2025. https://arxiv.org/abs/2512.09304
A bit-serial processing-in-DRAM design with locality buffers and an automated mapping of workloads onto DRAM parallelism; claims 9x-102x speedups over GPUs.
Overlap/difference: "mapping" here means compute-to-DRAM-hierarchy placement for PUD, not the host physical-address-to-bank function. Little relevance beyond showing the word is in use.

**RoMe: Row Granularity Access Memory System for Large Language Models.** Hwayong Nam, Seungmin Baek, Jumin Kim, Michael Jaemin Kim, Jung Ho Ahn. arXiv:2512.01541, Dec 2025 (cs.AR). https://arxiv.org/abs/2512.01541
Argues LLMs stream KB-to-MB blocks, so it replaces cache-line DRAM access with row-granularity commands and drops columns, bank groups and pseudo channels, using the saved pins for +12.5% bandwidth.
Overlap/difference: it rests on the same premise as our finding (1), that LLM streams are sequential bulk sweeps with row locality essentially guaranteed. It changes the interface; we show the existing interface plus a mapping already reaches ~0.96-0.98 hit on weights and identify where it does not (KV under random placement). Good citation to motivate finding (1).

**Flatfish: a Reinforcement Learning Approach for Application-Aware Address Mapping.** X. Li, Z. Yuan, Y. Guan, G. Sun, T. Zhang, R. Wei, D. Niu. IEEE TCAD 41(11):4758-4770, 2022. DOI 10.1109/TCAD.2022.3146204 (not verified at source; seen via dblp/OpenAlex search listings and an aggregator).
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
- RH+ "Row-Hit-Optimized Scheduling for PIM-based LLM Inference" (title and claims from an aggregator, newx.sg, not verified at source; reported 8-12x speedup in simulation by remapping so consecutive MACs share a row). Treat as unverified; it concerns PIM MAC command streams.
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

**AttAcc (Unleashing the Potential of PIM: Accelerating Large Batched Inference of Transformer-Based Generative Models).** Jaehyun Park et al. IEEE Computer Architecture Letters, 2023, DOI 10.1109/LCA.2023.3305386 (author list and DOI not verified at source; seen via search listings). PIM for batched attention, 3.19x over DGX A100 on GPT-3. Not about row behaviour.

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

## BibTeX

```bibtex
@misc{huang2025umdam,
  title={UMDAM: A Unified Data Layout and DRAM Address Mapping for Heterogenous NPU-PIM},
  author={Huang, Hai},
  year={2025}, eprint={2511.03293}, archivePrefix={arXiv},
  note={Withdrawn by arXiv administrators}
}
@misc{ma2025racam,
  title={RACAM: Enhancing DRAM with Reuse-Aware Computation and Automated Mapping for ML Inference},
  author={Ma, Siyuan and Hu, Jiajun and Ryoo, Jeeho and Arora, Aman and John, Lizy Kurian},
  year={2025}, eprint={2512.09304}, archivePrefix={arXiv}
}
@misc{nam2025rome,
  title={RoMe: Row Granularity Access Memory System for Large Language Models},
  author={Nam, Hwayong and Baek, Seungmin and Kim, Jumin and Kim, Michael Jaemin and Ahn, Jung Ho},
  year={2025}, eprint={2512.01541}, archivePrefix={arXiv}
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
  title={Sparse Attention Remapping with Clustering for Efficient LLM Decoding on PIM},
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
```
