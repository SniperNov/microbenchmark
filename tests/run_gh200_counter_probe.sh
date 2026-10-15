#!/usr/bin/env bash
# Diagnostic only: make the zero counter initial value opaque to the compiler.
set -euo pipefail
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
export GH200_PROBE_KIND=counter
exec bash "$script_dir/run_gh200_entry_probe.sh" "$@"
