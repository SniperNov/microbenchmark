#!/usr/bin/env bash
# Read existing binaries; rebuild the historical OpenACC source in isolation.
# This collector never executes a benchmark binary or changes production code.
set -euo pipefail
publish=0
if [ "${1:-}" = --push ]; then publish=1; shift; fi
if [ "$#" -ne 0 ]; then printf 'Usage: bash %s [--push]\n' "$0" >&2; exit 1; fi
repo_dir="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$repo_dir"
historical_ref=fb158194914ab00f68cb86195c2aa89cfb9dbac1
for tool in nvc cuobjdump python3 git sha256sum; do command -v "$tool" >/dev/null; done
git cat-file -e "$historical_ref^{commit}"
mkdir -p result/GH200/openacc_instructions
relative_out=$(mktemp -d "result/GH200/openacc_instructions/$(date -u +%Y%m%dT%H%M%SZ).XXXXXX")
out="$repo_dir/$relative_out"
mkdir -p "$out/bin" "$out/tools" "$out/existing_candidates" "$out/historical_rebuild/source"
printf 'Output directory: %s\n' "$out"
cp tests/collect_gh200_openacc_instructions.sh tests/summarize_openacc_instructions.py "$out/tools/"
sha256sum src/core/driver.c src/backends/openmp/backend.c src/backends/openacc/backend.c src/backends/cuda/backend.cu > "$out/production_sources_before.sha256"
trap 'sha256sum --check "$out/production_sources_before.sha256" > "$out/production_sources_check.log" || { cat "$out/production_sources_check.log"; exit 1; }' EXIT
{
    hostname
    nvc -V
    cuobjdump --version
    printf 'Historical source commit: %s\n' "$historical_ref"
    printf 'No benchmark executable is run. Existing binary/time association is unverified.\n'
    if command -v nvidia-smi >/dev/null; then nvidia-smi --query-gpu=name,driver_version --format=csv; fi
} > "$out/environment.log" 2>&1

# Save available native binaries before compiling anything. bin/microbenchmark
# is commonly a shell wrapper; checking ELF magic prevents treating it as code.
for original in build/microbenchmark_openacc bin/microbenchmark_openacc bin/microbenchmark; do
    if [ ! -f "$original" ]; then continue; fi
    if ! python3 - "$original" <<'PY'
import sys
with open(sys.argv[1],'rb') as f:
    sys.exit(0 if f.read(4)==b'\x7fELF' else 1)
PY
    then continue; fi
    label="${original//\//_}"
    cp "$original" "$out/bin/$label"
    {
        printf 'Original path: %s\n' "$original"
        printf 'Status: existing candidate; association with historical timing is unverified.\n'
        sha256sum "$original"
        stat "$original"
    } > "$out/existing_candidates/$label.provenance.txt"
    if ! cuobjdump --dump-sass "$out/bin/$label" > "$out/existing_candidates/$label.sass.txt" 2> "$out/existing_candidates/$label.sass.stderr.log"; then
        printf 'Existing candidate could not be disassembled: %s; diagnostics retained.\n' "$original"
    fi
done

# Reproduce the source revision recorded by the historical OpenACC run.
# Keep these files outside the working source tree, including old method9.
for path in src/core/driver.c src/core/backend.h src/backends/openacc/backend.c Makefile Makefile.backend Makefile.defs.openacc.nvc; do
    mkdir -p "$out/historical_rebuild/source/$(dirname -- "$path")"
    git show "$historical_ref:$path" > "$out/historical_rebuild/source/$path"
done
old_metadata=result/GH200/OpenACC/NVC_26_3/git_state_gh_openacc_manual_20260825_103355.txt
if [ -f "$old_metadata" ]; then cp "$old_metadata" "$out/historical_rebuild/original_run_git_state.txt"; fi
cat > "$out/historical_rebuild/provenance.txt" <<EOF
Status: HISTORICAL_SOURCE_REBUILD, not the original timed binary.
Source commit: $historical_ref
Compiler options: nvc -acc -O0 -DPRINT_DISTRIBUTION -I src/core ... -lm
These are the historical openacc-distribution build options.
Current compiler/runtime versions are recorded in environment.log.
This does not establish binary identity with the historical measurement.
Historical method9 remains scalar in this isolated source snapshot.
Production OpenACC method9 and all working sources are untouched.
No new timing data are collected and no benchmark binary is executed.
EOF
compile=(nvc -acc -O0 -DPRINT_DISTRIBUTION -I src/core
         -o "$out/bin/openacc_historical_rebuild"
         src/core/driver.c src/backends/openacc/backend.c -lm)
printf '%q ' "${compile[@]}" > "$out/historical_rebuild/compile.log"
printf '\n' >> "$out/historical_rebuild/compile.log"
(
    cd "$out/historical_rebuild/source"
    "${compile[@]}"
) >> "$out/historical_rebuild/compile.log" 2>&1 || { cat "$out/historical_rebuild/compile.log"; exit 1; }
cuobjdump --dump-sass "$out/bin/openacc_historical_rebuild" > "$out/historical_rebuild/openacc.sass.txt" 2> "$out/historical_rebuild/openacc.sass.stderr.log"

# Also capture the current working implementation, including restored method9.
# It is a separate build and is not paired with the historical timing data.
mkdir -p "$out/current_rebuild/source/src/core" "$out/current_rebuild/source/src/backends/openacc"
for path in src/core/driver.c src/core/backend.h src/backends/openacc/backend.c; do
    cp "$path" "$out/current_rebuild/source/$path"
done
cat > "$out/current_rebuild/provenance.txt" <<EOF
Status: CURRENT_WORKING_SOURCE_REBUILD, not paired with historical timings.
The current working source files are copied here without editing them.
Historical and current OpenACC implementations remain in separate directories.
No benchmark is executed and no timing result is produced.
EOF
current_compile=(nvc -acc -O0 -DPRINT_DISTRIBUTION -I src/core
                 -o "$out/bin/openacc_current_rebuild"
                 src/core/driver.c src/backends/openacc/backend.c -lm)
printf '%q ' "${current_compile[@]}" > "$out/current_rebuild/compile.log"
printf '\n' >> "$out/current_rebuild/compile.log"
if (
    cd "$out/current_rebuild/source"
    "${current_compile[@]}"
) >> "$out/current_rebuild/compile.log" 2>&1; then
    cuobjdump --dump-sass "$out/bin/openacc_current_rebuild" > "$out/current_rebuild/openacc.sass.txt" 2> "$out/current_rebuild/openacc.sass.stderr.log"
    printf 'COMPILED_AND_DISASSEMBLED\n' > "$out/current_rebuild/status.txt"
else
    printf 'COMPILE_FAILED; historical SASS remains available; inspect compile.log\n' > "$out/current_rebuild/status.txt"
    cat "$out/current_rebuild/status.txt"
    cat "$out/current_rebuild/compile.log"
fi
sha256sum "$out/bin/"* > "$out/binary_sha256.txt"
python3 "$out/tools/summarize_openacc_instructions.py" "$out" | tee "$out/summary.log"
sha256sum --check "$out/production_sources_before.sha256" > "$out/production_sources_check.log"
printf '\nCompleted: %s\n' "$out"

if [ "$publish" = 1 ]; then
    git add -f -- tests/collect_gh200_openacc_instructions.sh tests/summarize_openacc_instructions.py
    find "$relative_out" -type f ! -path '*/bin/*' -print0 | xargs -0 git add -f --
    git commit --only -m 'Collect GH200 OpenACC GPU instructions' -- \
        tests/collect_gh200_openacc_instructions.sh tests/summarize_openacc_instructions.py "$relative_out"
    git push
fi
