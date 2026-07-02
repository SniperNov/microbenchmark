#!/bin/sh

sanitize_compiler_tag()
{
    printf "%s" "$1" | tr '. -/' '___' | tr -cd 'A-Za-z0-9_'
}

compiler_tag_nvcc()
{
    version=$(nvcc --version 2>/dev/null | sed -n 's/.*V\([0-9][0-9.]*\).*/\1/p' | tail -n 1)
    if [ -n "$version" ]; then
        printf "NVCC%s" "$(sanitize_compiler_tag "$version")"
    else
        printf "NVCCunknown"
    fi
}

compiler_tag_nvc()
{
    version=$(nvc --version 2>/dev/null | sed -n 's/^nvc[[:space:]]*\([0-9][0-9.]*\).*/\1/p' | head -n 1)
    if [ -n "$version" ]; then
        printf "NVC%s" "$(sanitize_compiler_tag "$version")"
    else
        printf "NVCunknown"
    fi
}

compiler_tag_gcc()
{
    version=$(gcc -dumpfullversion -dumpversion 2>/dev/null | head -n 1)
    if [ -n "$version" ]; then
        printf "GCC%s" "$(sanitize_compiler_tag "$version")"
    else
        printf "GCCunknown"
    fi
}

compiler_tag_cray()
{
    version=$(cc --version 2>/dev/null | sed -n 's/.*Version \([0-9][0-9.]*\).*/\1/p' | head -n 1)
    if [ -n "$version" ]; then
        printf "CCE%s" "$(sanitize_compiler_tag "$version")"
    else
        printf "CCEunknown"
    fi
}

compiler_tag_amdclang()
{
    version=$(amdclang --version 2>/dev/null | sed -n 's/.*version \([0-9][0-9.]*\).*/\1/p' | head -n 1)
    if [ -n "$version" ]; then
        printf "AMDCLANG%s" "$(sanitize_compiler_tag "$version")"
    else
        printf "AMDCLANGunknown"
    fi
}

compiler_tag_sycl()
{
    if command -v icpx >/dev/null 2>&1; then
        version=$(icpx --version 2>/dev/null | sed -n 's/.* \([0-9][0-9.]*\).*/\1/p' | head -n 1)
        if [ -n "$version" ]; then
            printf "ICPX%s" "$(sanitize_compiler_tag "$version")"
        else
            printf "ICPXunknown"
        fi
    elif command -v dpcpp >/dev/null 2>&1; then
        version=$(dpcpp --version 2>/dev/null | sed -n 's/.* \([0-9][0-9.]*\).*/\1/p' | head -n 1)
        if [ -n "$version" ]; then
            printf "DPCPP%s" "$(sanitize_compiler_tag "$version")"
        else
            printf "DPCPPunknown"
        fi
    elif command -v acpp >/dev/null 2>&1; then
        version=$(acpp --version 2>/dev/null | sed -n 's/.*version \([0-9][0-9.]*\).*/\1/p' | head -n 1)
        if [ -n "$version" ]; then
            printf "ACPP%s" "$(sanitize_compiler_tag "$version")"
        else
            printf "ACPPunknown"
        fi
    else
        printf "SYCLunknown"
    fi
}
