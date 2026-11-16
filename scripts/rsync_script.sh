#!/usr/bin/env bash
# rsync_contrastive_run.sh
#
# Run FROM YOUR LOCAL MACHINE to pull all contrastive_run folders from the
# cluster, oldest first (so you can delete already-synced ones to free space).
#
# Key behaviours:
#   - Folders are processed oldest-first (by cluster mtime)
#   - Already-completed folders are skipped (tracked in rsync_done.log)
#   - Safe to re-run / interrupt and resume at any time
#   - NO --delete: deleting a folder from the cluster does NOT remove your
#     local copy. Future rsync runs simply skip it (it's already in the log).
#
# Usage (from your local machine):
#   chmod +x rsync_contrastive_run.sh
#   ./rsync_contrastive_run.sh
#   ./rsync_contrastive_run.sh --dry-run    # show what would be synced
#
# To sync only specific folder patterns, set FILTER, e.g.:
#   FILTER="le_b1b2" ./rsync_contrastive_run.sh

CLUSTER_HOST="liyading@login.ds.uchicago.edu"
CLUSTER_SRC="/net/spaces/clinic/CLS/lding/data/fa_data_analysis/ae_results/contrastive_run"
LOCAL_DEST="/home/lding/lding/dsicluster_CLS_rsync_folder/data/fa_data_analysis/ae_results/contrastive_run"   # override with env var

LOG_FILE="$LOCAL_DEST/rsync_done.log"
DRY_RUN=0
[[ "$1" == "--dry-run" ]] && DRY_RUN=1

mkdir -p "$LOCAL_DEST"
touch "$LOG_FILE"

# SSH multiplexing: open one persistent connection, reuse it for all rsync calls.
# This avoids repeated SSH handshakes that trigger server-side rate limiting.
SSH_CTRL_DIR="$HOME/.ssh/ctl"
SSH_CTRL_SOCK="$SSH_CTRL_DIR/rsync-%r@%h:%p"
mkdir -p "$SSH_CTRL_DIR"
SSH_OPTS="-o ControlMaster=auto -o ControlPath=$SSH_CTRL_SOCK -o ControlPersist=60"

# Open the master connection (stays open for 60s of inactivity, then auto-closes)
ssh $SSH_OPTS -fN "$CLUSTER_HOST"
trap "ssh -o ControlPath=$SSH_CTRL_SOCK -O exit $CLUSTER_HOST 2>/dev/null" EXIT

echo "============================================================"
echo " rsync_contrastive_run.sh"
echo "  Source : ${CLUSTER_HOST}:${CLUSTER_SRC}"
echo "  Dest   : ${LOCAL_DEST}"
echo "  Log    : ${LOG_FILE}"
[[ $DRY_RUN -eq 1 ]] && echo "  Mode   : DRY RUN (no files transferred)"
echo "============================================================"
echo ""

# Get folder list from cluster sorted by mtime ascending (oldest first).
# -rt = reverse of time sort = oldest first
echo "Fetching folder list from cluster..."
FOLDERS=$(ssh $SSH_OPTS "$CLUSTER_HOST" \
    "ls -rt '$CLUSTER_SRC' 2>/dev/null | grep -v '^latent_distance'")

if [ -z "$FOLDERS" ]; then
    echo "ERROR: Could not list folders from cluster. Check SSH connection."
    exit 1
fi

# Optional pattern filter
if [ -n "$FILTER" ]; then
    FOLDERS=$(echo "$FOLDERS" | grep "$FILTER")
    echo "Filtering to pattern: $FILTER"
fi

N=$(echo "$FOLDERS" | wc -l | tr -d ' ')
DONE_COUNT=$(wc -l < "$LOG_FILE" | tr -d ' ')
echo "Found $N folders on cluster  ($DONE_COUNT already synced)"
echo ""

i=0
n_synced=0
n_skipped=0
n_failed=0

while IFS= read -r folder; do
    [[ -z "$folder" ]] && continue
    i=$((i + 1))

    # Skip if already logged as done
    if grep -qxF "$folder" "$LOG_FILE"; then
        n_skipped=$((n_skipped + 1))
        continue
    fi

    echo "[$i/$N] $folder"

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  (dry run — would rsync)"
        continue
    fi

    mkdir -p "$LOCAL_DEST/$folder"

    rsync -avh \
        --partial \
        --append-verify \
        --info=progress2,stats \
        --timeout=600 \
        -e "ssh $SSH_OPTS" \
        "${CLUSTER_HOST}:${CLUSTER_SRC}/${folder}/" \
        "${LOCAL_DEST}/${folder}/"

    RC=$?
    if [ $RC -eq 0 ]; then
        echo "$folder" >> "$LOG_FILE"
        n_synced=$((n_synced + 1))
        echo "  OK"
    else
        n_failed=$((n_failed + 1))
        echo "  FAILED (exit $RC) — will retry on next run"
    fi
    echo ""
done <<< "$FOLDERS"

echo "============================================================"
echo " Summary: $n_synced synced, $n_skipped already done, $n_failed failed"
echo " Log: $LOG_FILE"
echo "============================================================"
