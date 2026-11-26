#!/bin/sh
# Offline coding server (M5 Air 24 GB), measured 26 Nov 2026: ./serve-coding.sh [CTX] [PORT]
#   best-quality file (97.3% top-token vs the original NVFP4), K8V4 cache with the SWA cache in f16 (99.0% docs /
#   98.5% code vs f16 KV), REL_FA, n-gram drafting (+42% decode on code edits, no loss on fresh code).
#   -lm none: weights read into memory, not mmap: 128K decode 10.6 -> 15.2 tok/s (ABBA), depth 0 unchanged.
#   CTX: 65536 default; ~200K fits with other apps closed, ~50K with ~4 GB of other apps open.
#   Add `-md weights/dspark-Inkling-2-Mini-Q8_0_Old.gguf --spec-type draft-dspark,ngram-mod` for edit-heavy sessions
#   (+83% on edits, -10% on fresh code).
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
m="$root/weights/Inkling-2-Mini-imx-Q4-ctrl16-fold.gguf"
INKLING_MODEL=$m . "$root/env.sh"
exec "$root/bin/llama-server" -m "$m" -a inkling -ngl all -c "${1:-65536}" -np 1 -fa on -b 2048 -ub 1024 \
    -ctk q8_0 -ctv q4_0 -lm none --spec-type ngram-mod --jinja --host 127.0.0.1 --port "${2:-18766}"
