#!/bin/sh

sanitize_compiler_tag()
{
    printf "%s" "$1" | tr '. -/' '___' | tr -cd 'A-Za-z0-9_'
}

format_compiler_tag()
{
    family="$1"
    version="$2"
    if [ -n "$version" ]; then
        printf "%s_%s" "$family" "$(sanitize_compiler_tag "$version")"
    else
        printf "%s_unknown" "$family"
    fi
}

compiler_tag_nvcc()
{
    version=$(nvcc --version 2>/dev/null | sed -n 's/.*V\([0-9][0-9.]*\).*/\1/p' | tail -n 1)
    if [ -n "$version" ]; then
        format_compiler_tag "NVCC" "$version"
    else
        printf "NVCC_unknown"
    fi
}

compiler_tag_nvc()
{
    version=$(nvc --version 2>/dev/null | sed -n 's/^nvc[[:space:]]*\([0-9][0-9.]*\).*/\1/p' | head -n 1)
    if [ -n "$version" ]; then
        format_compiler_tag "NVC" "$version"
    else
        printf "NVC_unknown"
    fi
}

compiler_tag_gcc()
{
    version=$(gcc -dumpfullversion -dumpversion 2>/dev/null | head -n 1)
    if [ -n "$version" ]; then
        format_compiler_tag "GCC" "$version"
    else
        printf "GCC_unknown"
    fi
}

compiler_tag_cray()
{
    version=$(cc --version 2>/dev/null | sed -n 's/.*Version \([0-9][0-9.]*\).*/\1/p' | head -n 1)
    if [ -n "$version" ]; then
        format_compiler_tag "CCE" "$version"
    else
        printf "CCE_unknown"
    fi
}

compiler_tag_amdclang()
{
    version=$(amdclang --version 2>/dev/null | sed -n 's/.*version \([0-9][0-9.]*\).*/\1/p' | head -n 1)
    if [ -n "$version" ]; then
        format_compiler_tag "AMDCLANG" "$version"
    else
        printf "AMDCLANG_unknown"
    fi
}

compiler_tag_sycl()
{
    if command -v icpx >/dev/null 2>&1; then
        version=$(icpx --version 2>/dev/null | sed -n 's/.* \([0-9][0-9.]*\).*/\1/p' | head -n 1)
        if [ -n "$version" ]; then
            format_compiler_tag "ICPX" "$version"
        else
            printf "ICPX_unknown"
        fi
    elif command -v dpcpp >/dev/null 2>&1; then
        version=$(dpcpp --version 2>/dev/null | sed -n 's/.* \([0-9][0-9.]*\).*/\1/p' | head -n 1)
        if [ -n "$version" ]; then
            format_compiler_tag "DPCPP" "$version"
        else
            printf "DPCPP_unknown"
        fi
    elif command -v acpp >/dev/null 2>&1; then
        version=$(acpp --version 2>/dev/null | sed -n 's/.*version \([0-9][0-9.]*\).*/\1/p' | head -n 1)
        if [ -n "$version" ]; then
            format_compiler_tag "ACPP" "$version"
        else
            printf "ACPP_unknown"
        fi
    else
        printf "SYCL_unknown"
    fi
}

# Parameters shared by every platform and programming model. Keeping these in
# one place prevents otherwise-identical experiments from silently drifting.
set_common_benchmark_parameters()
{
    N_DATA="1,4,16,64,256,1024,4096,8192,16382,32768,65536"
    N_FIXED="16382"
    N_ATORED="16,64,256,1024,4096,8192,16382"
    N_PARREPS="1,2,4,8,16,32,64,128"
    DELAY_SHORT="1,8096"
}

# Platform topology. Every API on a platform must consume the same pair.
set_platform_launch_parameters()
{
    case "$1" in
        A100)   PLATFORM_GROUPS=108; PLATFORM_WIDTH=32 ;;
        H100)   PLATFORM_GROUPS=132; PLATFORM_WIDTH=32 ;;
        H200)   PLATFORM_GROUPS=132; PLATFORM_WIDTH=32 ;;
        GH200)  PLATFORM_GROUPS=132; PLATFORM_WIDTH=32 ;;
        MI210)  PLATFORM_GROUPS=104; PLATFORM_WIDTH=64 ;;
        MI300X) PLATFORM_GROUPS=304; PLATFORM_WIDTH=64 ;;
        *)
            echo "Unknown platform topology: $1" >&2
            return 1
            ;;
    esac
}

archive_run_provenance()
{
    script_path="$1"
    outdir="$2"
    job_name="$3"
    timestamp="$4"
    run_id="${SLURM_JOB_ID:-manual}"
    script_base=$(basename "$script_path")

    mkdir -p "$outdir"
    cp "$script_path" "$outdir/submitted_${job_name}_${run_id}_${timestamp}_${script_base}"

    {
        echo "timestamp=$timestamp"
        echo "hostname=$(hostname)"
        echo "slurm_job_id=${SLURM_JOB_ID:-manual}"
        echo "slurm_job_name=${SLURM_JOB_NAME:-manual}"
        echo "script=$script_path"
        echo "working_directory=$(pwd)"
        echo
        if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
            git status --short --branch || true
            git log -1 --format=fuller || true
            echo
            git diff --no-ext-diff HEAD || true
        else
            echo "Git metadata unavailable: working directory is not a Git worktree."
        fi
    } > "$outdir/git_state_${job_name}_${run_id}_${timestamp}.txt"

    if [ -n "${SLURM_JOB_ID:-}" ]; then
        PROVENANCE_SLURM_LOG="${SLURM_SUBMIT_DIR:-$(pwd)}/slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
        PROVENANCE_SLURM_DEST="$outdir/slurm_${job_name}_${SLURM_JOB_ID}_${timestamp}.out"
        trap 'if [ -f "$PROVENANCE_SLURM_LOG" ]; then mv "$PROVENANCE_SLURM_LOG" "$PROVENANCE_SLURM_DEST"; fi' EXIT
    fi
}
