# HBM2 decoder compatibility audit, 2026-11-17

Owning source: installed `~/amcas-pro/tracing/ramulator2-grid`, inspected without modifying it. HBM2_2Gb has levels Channel,PseudoChannel,Sid,BankGroup,Bank,Row,Column, sizes32externalcontrollers,2,1,4,4,16384,128. Prefetch4 reduces column bits7→5; transaction size32B gives offset5. Source: `src/ramulator/dram/impl/HBM2.cpp:27,55,65`, `python/ramulator/dram/hbm2.py:238`, `src/ramulator/controller/addr_mapper/addr_mapper_base.cpp:10-17,20-39`.

Channel extraction is separate. `memory_system/channel_mapper/impl/cache_line_interleave.cpp:21-44` sets shift=tx_offset+interleave_bits, takes5channelbits, removes them, preserving lower bytes. To use256B channel interleave, interleave_bits must be3, not2. More generally offline s=6+ib requires simulator interleave_bits=ib+1 for32B HBM2 transactions. Legacy `gpu/gpu_replay.py:85` constructs offline s=6+ib, while `tools/dramtech_cfg.py:83-97` passes ib unadjusted: **legacy replay has a one-bit channel-position mismatch**, affecting XOR training/transforms relative to controller decode. Existing results remain legacy-model results, not validated physical V100 mappings. New `gpu/accelsim/policy_pilot.py` corrects this by passingib+1.

Below, `i` is byte address after channel removal and `a=i>>5`; offset bits0–4 are not decoded. Combined bank index includes pseudochannel+sid+bankgroup+bank, but integer bit order differs by mapper. All figures concern modeled addresses, not recovered V100 physical addresses.

## RoBaRaCoCh

`ro_ba_ra_co_ch.cpp:17-25`: column=a[4:0]; pseudochannel=a[5]; sid=0; bankgroup=a[7:6]; bank=a[9:8]; row=a>>10. In intra-byte positions: col5–9,PC10,BG11–12,bank13–14,row15+. A Layout(off=5,col=5,bank=5,row=14,s=8,W=5) exactly describes its aggregate bank low-order PC thenBGthenbank. All four offsets0,32,64,96 within128B affect only column, so channel/bank/row invariant at256B interleave.

## MOP4CLXOR

`mop4clxor.cpp:13-40` explicitly supports arbitrary preceding levels including pseudochannel and zero-width sid. Parse low2columnbits first, then PC1,SID0,BG2,BANK2, remaining3columnbits, row14; then XOR column slices into levelsPC,BG,BANK,Row in that order. Therefore raw bank region begins intra bytebit7; column combinesbits5–6 withbits12–14. Effective aggregate bank=raw5 XOR(column&31); row is unchanged because five column bits are consumed by preceding bank levels, leaving no column bits for row XOR. The existing Python MopView implements these aggregate fields correctly when given HBM2 Layoutoff5/col5/bank5.

**Counterexample, intra addresses0,32,64,96:** effective(PC,BG,BANK)=(0,0,0),(1,0,0),(0,1,0),(1,1,0), row0. Stock MOP4CLXOR therefore violates128B block invariance in every tested block. It is not admissible as an offset-invariant replacement for the current exporter, which lacks original per-sector masks/offsets for larger writes. MOPxor does not repair this. Either preserve exact original sector offsets/masks and explicitly model32B striping, or design and separately name a new128B-preserving mapper; do not call that new mapper stockMOP4CLXOR.

## ChRaBaRoCo

`ch_ra_ba_ro_co.cpp:17-23` reverses complete level extraction: column5bits, row14bits, bank2bits, BG2bits,SID0bits,PC1bit. In intra-byte positions: column5–9,row10–23,bank24–25,BG26–27,PC28. A ChRaView can return rawtop5bank=(i>>24)&31,row=(i>>10)&16383,col=(i>>5)&31; bank's integer bit order isBANKthenBGthenPC, notPCthenBGthenBANK. For cost/rate analysis permutation of bank IDs is harmless; converting fields into another decoder must reorder component bits. To express common aggregate indexPC|(BG<<1)|(BANK<<3), use PC=(raw>>4)&1,BG=(raw>>2)&3,BANK=raw&3. Within128B only column varies, so it is admissible under the current synthetic-sector convention provided channel shift>=7 (our256B baseline shift8).

## Fixture and recommendation

`hbm_decoder_fixture.py` is an independent scalar transcription of the owning C++ with all seven output fields; deterministic examples and10000random128Bblocks/mapper are in JSON. RoBa andChRa have0non-column changes;MOP has10000. This proves/transparently tests decoder algebra, not simulator timing or physical V100mapping. AdmitRoBa+row-onlyXOR/channel hashes whose sources exclude sectoroffsetbits; admitChRa after the aggregate-bit-order conversion is accounted for; exclude stockMOP/MOPxor from the offset-invariant campaign. Keep Oracle separately as a bound. No existing source was modified.
