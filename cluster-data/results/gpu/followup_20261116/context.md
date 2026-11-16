# GPU MoE capture repair, 2026-11-16

125720 failed because stream_count used a FIFO's zero stat size, despite 147985 valid compressed read records matching per-launch counters. New version uses is_file and separate output/scratch directories. Comparison gate derives from successful 125718. Two-device validation/replay remain pending.
