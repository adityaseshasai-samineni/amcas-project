#!/bin/bash
# Upload each run's dram.trc.gz to a GitHub release as <run>.dram.trc.gz, plus a
# SHA256SUMS file. Assets already present are skipped, so it can be re-run.
# Usage (token on stdin, never in argv or on disk):
#   ssh rce 'bash ~/amcas-pro/scripts/09_upload_traces.sh RELEASE_ID [RUN ...]' < tokenfile
set -uo pipefail
REPO=adityaseshasai-samineni/amcas-project
ID=$1; shift
read -r T
[ -n "$T" ] || { echo "no token on stdin"; exit 1; }
auth() { printf 'header = "Authorization: Bearer %s"\nheader = "Accept: application/vnd.github+json"\n' "$T"; }
cd $HOME/amcas-pro/results/traces
ALL=$([ $# -eq 0 ] && echo 1)
RUNS=${*:-$(ls -d */ | tr -d / | grep -v -x moe_l3-22m_stride)}
have=$(curl -s -K <(auth) "https://api.github.com/repos/$REPO/releases/$ID/assets?per_page=100" | sed -n 's/.*"name": "\([^"]*\)".*/\1/p')
sums=$(mktemp); trap 'rm -f $sums' EXIT
for r in $RUNS; do
  f=$r/dram.trc.gz; name=$r.dram.trc.gz
  [ -f $f ] || { echo "$r: no trace"; continue; }
  (cd $r && sha256sum dram.trc.gz | sed "s/dram.trc.gz/$name/") >> $sums
  if grep -qx "$name" <<< "$have"; then echo "$name: already uploaded"; continue; fi
  out=$(curl -s --fail -K <(auth) -H "Content-Type: application/gzip" --retry 3 \
    --data-binary @$f "https://uploads.github.com/repos/$REPO/releases/$ID/assets?name=$name" \
    | grep -o '"size": *[0-9]*' | head -1 | grep -o '[0-9]*$')
  [ "$out" = "$(stat -c %s $f)" ] && echo "$name: uploaded $((out / 1000000)) MB" || echo "$name: FAILED (reported size '$out')"
done
# Only a full run knows every trace, so only it writes SHA256SUMS.
if [ -n "$ALL" ] && ! grep -qx SHA256SUMS <<< "$have"; then
  curl -s --fail -K <(auth) -H "Content-Type: text/plain" --data-binary @$sums \
    "https://uploads.github.com/repos/$REPO/releases/$ID/assets?name=SHA256SUMS" >/dev/null && echo "SHA256SUMS: uploaded"
fi
