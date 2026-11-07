#!/usr/bin/env bash
# Restarts matricxon (Linux and macOS): stop.sh if running, then start.sh. On Debian/Ubuntu prefer
# `systemctl restart` with scripts/matricxon.service.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# As root, start.sh refuses with its message: checked before stop.sh, so nothing is left stopped.
if [ "$(id -u)" -eq 0 ] && [ "${MATRICXON_ALLOW_ROOT:-}" != 1 ]; then
    exec bash "$SCRIPT_DIR/start.sh"
fi

"$SCRIPT_DIR/stop.sh"
"$SCRIPT_DIR/start.sh"
