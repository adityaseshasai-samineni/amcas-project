#!/bin/bash
# Publish results from the cluster to GitHub. Called at the end of every job.
# Usage: publish.sh "what finished" [RELEASE_RUN ...]
#
# Copies small result files (under 50 MB) and logs into a clone of the repo at
# ~/amcas-repo, regenerates the analysis, commits and pushes. Each RELEASE_RUN
# names a results/traces run whose dram.trc.gz goes to the trace release.
#
# Needs a fine-grained token for this repository only (Contents: read and write)
# in ~/.config/amcas/gh_token, mode 600. Without it, nothing is published and the
# results stay on the cluster. The token is passed to git and curl on stdin or
# through a credential helper, never on a command line.
set -uo pipefail
MSG=${1:-results}; shift || true
TOK=$HOME/.config/amcas/gh_token
REPO=adityaseshasai-samineni/amcas-project
R=$HOME/amcas-repo
A=$HOME/amcas-pro
E=$A/tracing/mamba/envs/analysis
export OPENBLAS_NUM_THREADS=1
[ -s "$TOK" ] || { echo "publish: no token at $TOK; results stay on the cluster"; exit 0; }
[ "$(stat -c %a "$TOK")" = 600 ] || { echo "publish: $TOK must be mode 600"; exit 1; }

# One publisher at a time: several jobs can finish together on a shared home.
L=$HOME/.config/amcas/publish.lock
for i in $(seq 1 120); do mkdir "$L" 2>/dev/null && break; sleep 30; done
[ -d "$L" ] || { echo "publish: could not get the lock"; exit 1; }
trap 'rmdir "$L"' EXIT

# Credential helper: prints the token for git to read on stdin. `cat` gets the
# file path as an argument, not the token.
git_() { git -C "$R" -c credential.helper= -c credential.helper="!f() { echo username=x-access-token; echo password=\$(cat $TOK); }; f" "$@"; }
if [ ! -d "$R/.git" ]; then
  git -c credential.helper="!f() { echo username=x-access-token; echo password=\$(cat $TOK); }; f" \
    clone -q "https://github.com/$REPO.git" "$R" || { echo "publish: clone failed"; exit 1; }
  git -C "$R" config user.name "adityaseshasai-samineni"
  git -C "$R" config user.email "sassai2003@gmail.com"
fi
git_ pull -q --rebase || { git_ rebase --abort 2>/dev/null; echo "publish: pull failed; nothing published"; exit 1; }

mkdir -p "$R/cluster-data"
rsync -a --max-size=50M --exclude='*.trc.gz' --exclude='*.trc' --exclude='*.bin' --exclude='*.session' \
  --exclude='*.gguf' "$A/results" "$A/logs" "$R/cluster-data/"
# Scripts and tools are versioned from the laptop, not copied back from here: an
# older cluster copy must never overwrite newer edits.

# Regenerate the analysis from everything now in the repo.
(cd "$R" && "$E/bin/python" trace-pipeline/tools/report.py cluster-data/results analysis) || echo "publish: report.py failed; publishing results without new analysis"
[ -f "$A/tools/fullscale_report.py" ] && (cd "$R" && "$E/bin/python" trace-pipeline/tools/fullscale_report.py cluster-data/results analysis/fullscale) \
  || true

git_ add -A
if git_ diff --cached --quiet; then
  echo "publish: nothing new to commit"
else
  git_ commit -q -F - <<M
Results: $MSG

Published automatically from the RCE cluster by tools/publish.sh.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
M
  for i in 1 2 3 4 5; do git_ push -q && { echo "publish: pushed $(git_ log -1 --format=%h)"; break; }; git_ pull -q --rebase || { git_ rebase --abort; break; }; sleep 20; done
fi

# Large traces go to the release, if any were named.
if [ $# -gt 0 ]; then
  ID=$(curl -s -K <(printf 'header = "Authorization: Bearer %s"\n' "$(cat $TOK)") \
    "https://api.github.com/repos/$REPO/releases/tags/traces-2026-11-03" | grep -o '"id": *[0-9]*' | head -1 | grep -o '[0-9]*$')
  [ -n "$ID" ] && bash "$A/scripts/09_upload_traces.sh" "$ID" "$@" < "$TOK"
fi
