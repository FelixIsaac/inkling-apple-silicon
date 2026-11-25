export INKLING_SITU=1 INKLING_CLIP=1 INKLING_GATE_MERGE=1 INKLING_SHORTCONV=1
export INKLING_REL_MASK=1 INKLING_DEPTH_CACHE=1
export INKLING_DEPTH_MIX=1 INKLING_REL_BIAS=1
export INKLING_CTRL_FUSE=1 INKLING_CONV_STATE=1 INKLING_ORDER=1
if [ "$(sysctl -n machdep.cpu.brand_string 2>/dev/null | grep -c -E 'M5')" = 1 ]; then
    export INKLING_MOE64=32 INKLING_MOE64_WS=1
    export INKLING_W4A8=1 INKLING_CONV_FUSE=1 INKLING_METAL_ENCODERS=2
    case "${INKLING_MODEL:-}" in
        *tie64*) export INKLING_W4A8_G64=1 INKLING_W4A8_T64=1 INKLING_W4A8_ADAPT=1 ;;
    esac
fi
# inkling-m5-extras.patch: split loop-controller matmul (no concat); SiTU computed inside the int8 down-proj quantiser
export INKLING_SPLIT_CTRL=1
case "${INKLING_MODEL:-}" in *tie64*) export INKLING_SITU_A8=1 ;; esac
# Relative bias inside flash attention instead of a materialised per-head mask: up to 7x smaller compute buffer at long
# context and +10-30% prefill at 16K depth on the M5 Air; same quality (98.9% top-token vs default on the fold file)
export INKLING_REL_FA=1
