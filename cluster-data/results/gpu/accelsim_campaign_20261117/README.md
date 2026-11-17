# Accel-Sim controller-export replay

`export_sectors.cc` converts the optional AMCAS `dram_t::push` CSV into
little-endian uint64 events `(byte_address / 32 << 1) | write`.
The original CSV remains authoritative for controller packet size, timestamps,
partitions and order. Conversion preserves bytes, not controller packet count.
Variable-size write packets omit dirty-sector masks: contiguous write offsets
are a canonical convention, not recovered GPU dirty-byte positions.

`policy_pilot.py` admits row-only bank/channel XOR, identity, naive row XOR and
bank/row bit reversal. These leave within-128-byte column bits unchanged.
Region mappings/policies require tensor labels from the *fresh capture*;
expert mappings additionally require an MoE workload. MOP4CLXOR and ChRaBaRoCo
require separate HBM field-decoder checks before admission.

Nine policies are admitted: Open, ClosedCAP, Timeout16/64/256, HAPPY binary,
HAPPY count, Adaptive15/6, Adaptive7/4. Timeout units are HBM controller cycles.
The CPU grid's Oracle is a future-aware bound, not a realizable policy.

The cluster pilot uses an isolated `ramulator2-gpu-drain` installation with a
finite-input frontend that tracks completion callbacks, then waits 4096 frontend
ticks (128 memory cycles at issue32). Reported cycles include that guard.
Acceptance requires all input reads/writes accepted and all requests completed
or explicitly forwarded/coalesced. The original LoadStoreTrace terminates at
last injection, so its output cannot establish a fully drained comparison.

HBM2 has a Python payload override of32B. CacheLineInterleave uses
`log2(transaction_bytes) + interleave_bits`; true256B therefore requires
interleave_bits3, not2. The pilot explicitly synchronizes this with its offline
bit layout. Historical GPU helper versions need separate review.

Replay is cold-window, saturated and open-loop. First-touch2MiB physical
placement is synthetic, frozen over the complete trace before window selection.
Bank/channel hashes train on the first million bursts and remain fixed for
middle/end evaluations. None of these times are measured inference latency.
The Accel-Sim pilot read-byte agreement does not resolve its27.9% write gap.
