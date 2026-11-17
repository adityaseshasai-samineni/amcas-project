# GPU MoE capture repair, 2026-11-16

125720 failed because stream_count used a FIFO's zero stat size, despite 147985 valid compressed read records matching per-launch counters. New version uses is_file and separate output/scratch directories. Comparison gate derives from successful 125718. Two-device validation/replay remain pending.

### GPU follow-up moe-smoke, job 126467 (2026-11-17T18:06:58+05:30)

Exit code 0. Results: results/gpu/followup_20261116.
