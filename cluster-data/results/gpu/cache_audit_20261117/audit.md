# Cache pipeline audit — 2026-11-17
Decision: run a controlled repair/sensitivity experiment on the retained V100 short trace before a full simulator migration. This is NOT full hardware validation.

Confirmed defects/limitations from source:
- Stream encoding keeps only width/store flag. Cache operators, atomic read-modify-write semantics, and actual SM IDs are lost. Old trace cannot recover these.
- Both smmap branches use CTA modulo SM; documented dynamic assignment is not implemented.
- L1 state survives dependent kernel boundaries. Added opt-in invalidate=1 variant, restricted interpretation to serialized dependent kernel pilot. Not a universal concurrency fix.
- Partial stores mark an entire 32B sector valid/dirty without byte-valid tracking. Can suppress later reads incorrectly; requires byte masks or explicit write-policy model.
- L2 dirty survivors are not flushed at end. This is a window-boundary definition, not automatically an error: artificial flushing must not be compared to counters in the original window.
- No real timing, concurrent ordering, atomics, physical-page translation, or measured SM placement. Read-byte agreement alone is insufficient; old write bytes differ greatly from hardware.
- Cold selected window, approximate LRU/hash and full128KiB L1 capacity remain assumptions.

Experiment: original x13 versus per-kernel L1 invalidation versus bypass-L1. Same retained raw capture, one pass, exact per-launch counts and elapsed time. Preserve original source and outputs. No silent replacement of published results.

Accel-Sim: https://github.com/accel-sim/accel-sim-framework uses an NVBit SASS tracer plus GPGPU-Sim timing model. Our memory-only trace lacks non-memory instructions, registers/dependencies and required opcode metadata; cannot plug it in or recover a complete SASS trace by conversion. A fresh Accel-Sim capture and V100 configuration are needed; DRAM stream export needs integration. Detailed simulation is not demonstrated faster than this streaming filter. Do not use H100 validation claims as V100 validation. Clone official sources for pinned compatibility audit next.

Pinned feasibility evidence:
Accel-Sim 7ba54ded60c4ba0083410c37d22d719166acd6dc; GPGPU-Sim4195dd4f229cbf03f4f38c49139ae94d01d23f03. SM7_GV100 config has80SMs,32memorycontrollers,gpgpu_flush_l1_cache=1, sector caches and partitioned L2. The L1/L2 associativities/configuration differ from our assumed4way/16way. Tracer supports DYNAMIC_KERNEL_RANGE and optional TOOL_TRACE_CORE; requires separateNVBit1.8. Its SASS/frontend tracks dependencies absent from our trace. Write path includes explicit write-allocation policies. This supports migration feasibility, not a validated application run or speed advantage. Our pilot job126566 testsboundaryfixfirst.

## Executed Accel-Sim validation
Simulator and officialNVBit1.8sm70tracer build successfully.126621tinyknown-output CUDAtrace+postprocess+SM7_GV100replay COMPLETED0:0. OptionalAMCAS_DRAM_EXPORThook in dram_t::push preservescontrolleradmissionsequence,address,size,direction,accessenum andglobalcorecycle.126631nonzero-readtest:4readrecords×32B exactlyequalssummedcontroller n_req4. Syntheticwrite-pathtest (preflight_write) usesexplicitlymodifiedtest-onlyminiaturewrite-throughL2;16reads+16writes exactlyequalcontroller n_req32. Its modifiedcache is NOT theV100experimentconfiguration. InitialIPOLYsetcount1testasserted;linearindextestfixedthis. Write-backversionretaineddirtydata,sozeroDRAMwriteswasnotanexportfailure. No forcedendflushaddedtorealmodel.

FreshSASScapture126622 is runningoneQwen2.5-7Bdecodewindow; officiallaunchIDs1296..1833inclusive maptoold1295..1832. Firstkernelname/gridcheckedagainstoldcensus; full538windowcountrequired. CapturehasGPU100% andadvancingstats; approximately1GBcompressedthroughkernel1352at12min. It is muchslowerthancustommemory-onlytracing,notafasterdrop-inreplacement.126630fullreplayand126640streamreconciliationwillrunaftercapture/exportbuild. Finalworkloaddataunavailableuntilthosefinish. CurrentPPTshouldnotbeswitchedtounvalidatednumbers.

## Representative real-kernel validation
Stockpostprocessor cannotconsumeTOOL_TRACE_CORE=1 extraSM/warpidfields: fullQwenmatvecreplay initiallyassertedMAX_DSTbecauseofshiftedPC/mask/registercolumns. FixedwithanexplicitAMCAS_TRACE_CORE/per-rundotmarker opt-in toskipthose2fieldsduringpostprocessing,retainingrawhardwarecoreIDrecords. Original/sourcepatchsaved; core0microbenchunaffected. Representativekernel1298thenCOMPLETED(126717),96.23swall,949648simulatedthreadinstructions/sec; exactexportvscontrollern_reqreconciledinrepresentative_validation.json. This confirmsrealQweninstructioncompatibility,notfull-tokenaccuracyorspeedadvantage. Bothpostprocessandsimulatorrequireparent-bearingkernelslistpaths; useabsolutepaths(baresinglefilenamecausedbadparentpaths). Fullselecteddecodecapture/replayremainsinprogress.
