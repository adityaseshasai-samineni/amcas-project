# Controlled V100 short-prompt hardware counter measurements

The fresh official Accel-Sim capture completed successfully in 2h10m03s; all 538 selected kernel names and launch geometries match the previous short-prompt capture. The full GPGPU-Sim replay is still running and its final request counts are not yet accepted.

Independent Nsight Compute runs collected only dram__bytes_read.sum and dram__bytes_write.sum over one selected decode step. Qwen2.5-7B Q4_K_M, prompt "The capital of France is" (5 tokens), greedy, generation limit3, no warmup, contextcapacity4096, FlashAttention auto/off, graphsdisabled, cache-controlnone, clock-controlnone.

| Replay | FA | Kernels | Read bytes | Write bytes |
|---|---|---:|---:|---:|
| Application | on | 538 | 4,394,874,880 | 9,745,664 |
| Kernel | on | 538 | 4,395,018,432 | 9,783,648 |
| Application | off | 566 | 4,395,132,384 | 10,352,096 |
| Kernel | off | 566 | 4,395,084,384 | 10,513,216 |

The two replay modes agree within0.0033% for reads and1.56% for writes. The earlier larger-metric-set measurements of1.282/1.352GBwrites were not reproduced. Their discrepancy is under investigation; the data does not establish that replay mode alone caused it. Preserve those historical files but supersede their write volumes with these matched two-counter controls. Application replay is the presentation reference; write units are now decimalMB, reads decimalGB.

The presentation's native7.81/8.47ms timing is a separate earlier graph-enabled mean over63decode evaluations, not profiler runtime. Original custom-filter model counts are retained and explicitly marked as prototype results until full GPGPU-Sim replay passes reconciliation and comparison. Hardware counters do not provide an ordered hardware DRAM trace.

Raw controls and exact summary: ncu_*_two_metrics.raw and ncu_replay_controls.json. Revised PowerPoint has archived previous values and detailed source notes; no pending full-model counts were inserted.
