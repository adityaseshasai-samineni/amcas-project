#!/bin/bash
set -Eeuo pipefail
MSG=$1
A=$HOME/amcas-pro
R=$HOME/amcas-repo
bash "$A/tools/publish.sh" "GPU follow-up: $MSG"
TOK=$HOME/.config/amcas/gh_token
[ -s "$TOK" ] && [ "$(stat -c %a "$TOK")" = 600 ] || exit 1
LOCK=$HOME/.config/amcas/publish.lock
owned=0
for i in $(seq 1 120); do
 if mkdir "$LOCK" 2>/dev/null; then owned=1; break; fi
 sleep 5
done
[ "$owned" = 1 ] || exit 1
trap 'rmdir "$LOCK"' EXIT
git_(){ git -C "$R" -c credential.helper= -c credential.helper="!f() { echo username=x-access-token; echo password=\$(cat $TOK); }; f" "$@"; }
git_ pull -q --rebase
printf '\n### GPU follow-up cluster update (%s)\n\n%s. See cluster-data/results/gpu/followup_20261115/ for source snapshots, manifests, validation gates and results. Saturated memory replay results are not measured token-time speedups.\n' "$(date -Is)" "$MSG" >> "$R/context.md"
git_ add context.md
git_ commit -q -m "context: GPU follow-up $MSG"
git_ push -q
