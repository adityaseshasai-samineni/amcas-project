# Native V100 load-latency calibration

Controlled intended cache regimes; this is not an exact per-access hardware cache-level oracle or an LLM DRAM trace. Native unprofiled timing samples alone train/test the classifier. Separate Nsight runs check aggregate cache regimes.

One CTA, one active lane, randomized dependent 128-byte-stride pointer cycle. L1: 8 KiB, ld.global.ca, full-cycle warmup within measurement kernel. L2: 1 MiB, ld.global.cg, full-cycle preparation kernel. DRAM: 64 MiB, ld.global.cg, full-cycle preparation followed by first 8192 nodes; reuse distance exceeds 6 MiB L2. Allocation addresses remain live and distinct across all runs. SM placement is recorded, not pinned. Eight seeds, 8192 samples per regime per seed. Timer-only floor shares output-store instrumentation. Inline PTX predicates the ending clock read on loaded pointer, preserving completion dependency. Pointer outputs are host-validated.

| Regime | N | p1 | median | p95 | p99 |
|---|---:|---:|---:|---:|---:|
| floor | 65536 | 10 | 10 | 10 | 10 |
| l1 | 65536 | 38 | 38 | 38 | 38 |
| l2 | 65536 | 201 | 230 | 247 | 251 |
| dram | 65536 | 398 | 428 | 537 | 645 |

All values are raw inclusive SM-clock cycles. Timer-floor subtraction is not claimed to remove all observer overhead. Global output stores, translation, clock changes, cache replacement and profiler effects remain limitations.

Training seeds101–104; held-out seeds105–108, distinct allocations. Median-midpoint hard-classifier accuracy: 100.0000%. Conservative unique p1–p99 band classifier coverage 99.1597%, accepted accuracy 100.0000%, unconditional correct fraction 99.1597%. These numbers assess controlled intended regimes, not per-access verified truth.

Confusion matrices and per-run statistics: analysis.json. Raw samples: samples.csv. Separate counter exports: ncu_*.csv. Inspect counter errors before treating regime labels as counter-confirmed. Latency overlap and unknown results must be retained when adapting this method to real kernels. No direct end-to-end LLM performance conclusion follows.

## Separate profiler checks
All nine application-replay groups completed. L1 intended-hot: 8256 load sectors including64 warmup;8192 hits and64 misses, exactly the8192 measured loads hitting after warmup. L2 intended-hot:8192 read sectors,8192 L2 hits,0 misses; L1 bypass yields0 L1 hits. DRAM intended-cold:8192 load sectors and0 L1 hits;8192 L2 read sectors and0 L2 hits. However its reported L2 lookup-miss counter is16384 rather than8192, and total DRAM reads530048 bytes rather than a naive8192×32B estimate262144. Thus counter groups support the aggregate hot-versus-cold distinction, but **do not reconcile one measured load to one DRAM-sector event**. Translation/page-table traffic, hardware fetch granularity and counter semantics must be investigated before using these counters as exact request counts. Profiled output stores also cause writes:107648B L1 case,48512B L2 case,1792B DRAM case; writeback completion is not constrained to the kernel window. Do not interpret those writes as pointer-chase payload stores.

Every class has32768 held-out samples. The hard-classifier confusion matrix is diagonal32768/class, so precision and recall are1.0 for each controlled intended label. The conservative classifier abstains on826/98304 held-out samples; these are not silently discarded from unconditional accuracy. Per-access cache-level truth remains unavailable, and thresholds are not validated on arbitrary LLM concurrency, translated addresses, atomics, or writes.
