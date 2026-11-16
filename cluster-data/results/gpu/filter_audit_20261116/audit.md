# GPU filter audit — 2026-11-16
## Confirmed implementation
Custom NVBit memory-only tracer + custom C functional sector cache filter; no Accel-Sim/GPGPU-Sim integration found in project code or installed tracing directories.
## Findings
- L1 state retained across launches: no dependent-grid invalidation. Test conservative clear-at-every-launch (appropriate serial-window sensitivity; not a general concurrent-stream fix).
- smmap=dyn is a no-op alias of CTA modulo SM count; no hardware SM IDs captured.
- compact opcode encoding retains width/store bit but loses cache operators, atomic semantics and instruction dependencies; cannot convert losslessly to Accel-Sim SASS traces.
- L1 store updates validity; test bypassing store allocation/update in L1.
- Store partial-byte masks not retained; sector write-validation assumes full sector validity. Atomics treated as stores: insufficient for general workloads.
- sector coalescing correct for represented access width, but width inferred from opcode text; not independently checked against NVBit memory-reference size.
- L2 dirty data remaining at end is not written back: report it separately, do NOT force flush and call it measured traffic.
- CPU sequential trace order and CTA modulo placement approximate GPU scheduling. No timing/MSHR merging; selected decode window cold; omitted local/shared/constant traffic and cache hints.
- aggregate read-byte agreement is insufficient validation; previous short V100 modeled writes ~7 MB vs hardware ~1282 MB is a major discrepancy. Terminal dirty upper bound is only L2 capacity (~6 MiB), so flushing cannot explain it.
## Controlled repair experiment
Preserve baseline source; run baseline, boundary-invalidation+L1-store-bypass, and L1-bypass variants on SAME retained short V100 FA-on capture. No production filter replacement until read AND write validation passes. Retained prompt: The capital of France is,5tokens; one selected decode step; matched Nsight output Paris. The (3tokens) is a separate run.
## Accel-Sim evaluation
Official source: https://github.com/accel-sim/accel-sim-framework ; gpu-simulator/README.md and util/tracer_nvbit/README.md.
Cycle-level GPGPU-Sim backend models instruction dependencies/scheduling and memory system. Requires its own SASS tracer with PCs/register dependencies/all instructions; existing memory-only stream insufficient. NVBit tracer, postprocessor, simulator build and V100 configuration required. Selective kernel capture available. Not established faster; detailed cycle-level simulation adds work compared with our functional filter. Need benchmark, not a speed claim. V100 ISA/config support must be checked in pinned checkout. DRAM stream export to our replay formats requires integration. Even Accel-Sim requires hardware validation.
