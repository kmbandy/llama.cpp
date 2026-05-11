#!/usr/bin/env bash
# MAD-131: walk ${ssd_path}/paged and remove any instance-* subdir whose
# lockfile is no longer held (i.e. no live owner). Safe to run while
# other instances are alive — flock prevents touching live ones.

set -euo pipefail

SSD_PATH="${1:-${SSD_PATH:-/var/lib/army/ssd}}"
PAGED_DIR="$SSD_PATH/paged"

if [[ ! -d "$PAGED_DIR" ]]; then
    echo "cleanup-cold: $PAGED_DIR doesn't exist; nothing to do."
    exit 0
fi

removed=0
kept=0
for dir in "$PAGED_DIR"/instance-*; do
    [[ -d "$dir" ]] || continue
    lock="$dir/.lock"
    if [[ -f "$lock" ]]; then
        # Try a non-blocking exclusive lock. If someone holds it (live
        # llama-server with this instance ID), skip. flock returns 0
        # on success; we then immediately release.
        if flock -nx "$lock" -c true 2>/dev/null; then
            # We got the lock → no live holder. Safe to remove.
            echo "cleanup-cold: removing orphaned $dir"
            rm -rf "$dir"
            ((removed++))
        else
            echo "cleanup-cold: keeping $dir (live holder)"
            ((kept++))
        fi
    else
        # No lockfile → safe to remove
        echo "cleanup-cold: removing $dir (no lockfile)"
        rm -rf "$dir"
        ((removed++))
    fi
done

echo "cleanup-cold: removed=$removed kept=$kept"
