### Codex continuation: GPU follow-up scheduling (2026-11-15, IST)

User requested all five GPU follow-ups be prepared and scheduled, then explicitly directed that job files be created on the cluster and results pushed from the cluster using the existing logistics. Read the previous GPU scripts, context handoffs, and live queue before proceeding. Pulled origin/main with rebase/autostash (11f22ba -> 1ee2278), preserving existing local edits.

Five stages: (1) synthetic + captured-stream replay correctness audit; (2) shared-map/timing/feed comparison for V100/A16 32k FA on/off, 256 B/2 KiB; (3) beginning/middle/end windows using a mapping frozen from the beginning; (4) separate-device 30B two-V100 capture smoke; (5) gated full-step 30B capture and independent-device mapping replay. Outputs: ~/amcas-pro/results/gpu/followup_20261115; compressed address streams remain on node-local scratch/tmp. New code is being created in ~/amcas-pro/gpu, with source snapshots under the result directory for reproducibility and GitHub publication. No existing traces or scripts are being overwritten.

Audit of old gpu_ds3_study.py found global ordered Ramulator injection versus independent per-PC/channel DRAMsim3 injection, and simulator-specific XOR searches. New comparisons hold matrices identical between simulators and add independent-channel Ramulator injection. Resolved timing/geometry differences are saved explicitly; HBM coupled PCs versus separate DRAMsim3 PC instances remain a model difference. A one-minute node01 srun preflight passed for both GPU standards on synthetic row-hit, row-conflict, channel-spread and mixed read/write streams (2 CPUs/4 GB); submissions and further decoder checks follow below.

Connectivity: VPN profile iiit appeared active but had lost its tunnel IP/routes. Reconnecting with nmcli down/up restored SSH. SSH uses -F /dev/null because this host rejects /etc/ssh/ssh_config.d/20-systemd-ssh-proxy.conf ownership/permissions. This is a client workaround, not a cluster change.

### GPU follow-up jobs submitted (2026-11-15, 22:33 IST)

All five jobs were created on RCE using `~/amcas-pro/gpu/69_schedule_followup.sh` after both one-minute node01 synthetic preflights passed (the second checks every synthetic address with DRAMsim3's actual Config decoder). New cluster scripts: `gpu_followup.py`, `gpu_moe_followup.py`, `gpu_decode_check.cc`, `68_followup_job.sh`, `69_schedule_followup.sh`, `70_publish_followup.sh`. The initial local runner draft was removed after cluster-specific revisions; authoritative follow-up sources are frozen under `results/gpu/followup_20261115/code/` and published with results, rather than copied back over laptop sources.

| Job | Stage | Resources | Dependencies |
|---|---|---|---|
| 125717 | replay correctness audit | node05, 4 CPU, 16 GB, 1 h | none |
| 125718 | shared mapping/timing/feed comparison | node05, 8 CPU, 64 GB, 24 h | afterok 125717 |
| 125719 | beginning/middle/end windows | node05, 8 CPU, 64 GB, 24 h | afterok 125718 |
| 125720 | 30B two-V100 capture smoke | node04, 2 GPUs, 4 CPU, 48 GB, 2 h | afterok 125717 |
| 125721 | full 30B decode capture + mapping replay | node04, 2 GPUs, 8 CPU, 64 GB, 24 h | afterok 125718 and 125720 |

`--kill-on-invalid-dep=yes` prevents expensive successors from running after failed validation. The comparison uses 8M requests per cell, four 32k GPU/FA cells and both interleaves; matrices are trained once and held identical across the three timing/feed controls and both simulators. Window sensitivity freezes the beginning-window matrices and the full-stream first-touch page assignment. Matched timing aligns common resolved Ramulator fields, enables A16 bank groups in DS3 to match Ramulator, and uses unified DS3 queues to avoid the known write-drain deadlock. These are controlled comparisons; residual controller/PC differences are recorded, not claimed eliminated.

MoE smoke captures four-launch windows separately for each CUDA context; full capture streams through two independent per-device cache filters without retaining raw NVBit addresses. Gates check native versus traced generated output, launch-ID/kernel sequence, stable context partition, per-launch DRAM counts and paired stream lengths. Replay is independent per device and does not model inter-GPU transfer or token time.

Each job writes source hashes, allocation metadata, exit status and run `context.md`; on exit (including failure) it invokes the existing publish.sh, then appends a dated entry to the repository root context.md and pushes from the cluster under the publisher lock. Initial results publication confirmed at commit 6b51354; the scheduler also pushed the root-context submission record. At first verification 125717 was RUNNING on node05 and the other four were waiting on dependencies. Both V100s are requested explicitly, so MoE jobs may wait for node04 resources after dependencies clear.

Audit job 125717 finished its computation successfully at 22:33:31 IST (exit 0): synthetic streams and 100,000-request captured short-context windows passed on both standards, including the actual DRAMsim3 decoder and request accounting. At 22:33:49 it was still in its end-of-job publication step, so afterok dependencies had not yet been released; this is expected because Slurm waits for the full wrapper to exit.
