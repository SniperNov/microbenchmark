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

