# Roadmap: where this work stands and what to do next (M5 Air 24 GB, 26 Nov 2026)

Status on a MacBook Air M5 (10-core GPU, 24 GB), `Inkling-2-Mini-imx-Q4-ctrl16-fold.gguf`. Full numbers and method:
`results/2026-11-25-fold-split-situ.md`, `results/2026-11-25-why-fast-and-mixed-kv.md`, `results/2026-11-26-layer-map.md`.

## Done

| Area | Result |
|---|---|
| Prefill (pp2048) | 717 tok/s with the tie64 fold file (author's stack 680, stock master 197); 583 with the ctrl16 file |
| Decode | 35–36 tok/s at depth 0; +42% on code edits with `--spec-type ngram-mod` |
| Quality vs the original NVFP4 weights | 97.3% same top token (imatrix requant, ctrl16), KLD 0.0066 |
| KV cache quality vs f16 KV | 99.0% (K8V4 + `INKLING_SWA_KV_F16`), 99.2% at 64K depth |
| Context | 196K (K8V4, ub 1024) or 262K (q4/q4, ub 512) on the default GPU cap; needle test at 244K passed 3/3 |
| Long-context decode | `-lm none` (weights in memory, not mmap): 128K decode 10.6 → 15.2 tok/s |
| Upstream bugs fixed locally | Metal FA with K type ≠ V type (falls back to CPU, drops the compact bias); F16 dequant pass uses the K kernel for V |

Run it: `./serve-coding.sh [CTX] [PORT]`.

## Speed model (q4/q4 + SWA f16, `-lm none`)

- Decode: `ms/token ≈ 28 + 0.24 × K`. 28 ms = 2.92 GB of weights per token at ~104 GB/s. 0.24 ms per 1K tokens of
  context = reading their KV cache; the bus alone would allow ~0.09 (q4/q4 is 9.2 KB per token over 8 global passes).
- Prefill at depth K: `ms/token ≈ 1.65 + 0.044 × K`. 1.65 ms = ~9 GFLOP of matmuls per token (mostly int8 experts).
  0.044 ms per 1K = 98 MFLOP of attention (8 global passes × 24 heads × 128 × 4) at ~2.2 TFLOPS.
- Fill time for N tokens: `1.65·N/1000 + 0.022·(N/1000)²` seconds (244K measured 1,514 s, model 1,712 s).

## Open work, ranked

| # | Work | Expected (*estimate*) | Effort | How to start |
|---|---|---|---|---|
| 1 | q4_0 dequant in the FA vec kernel: the decode slope is ~2.6× the bandwidth floor at 256K | 256K decode 11 → 15–18 tok/s | 1–3 days | `INKLING_METAL_TIMING=1` + `INKLING_OP_PROFILE=2` on one decode graph at 128K; compare q8_0 vs q4_0 V |
| 2 | FA prefill at depth runs at ~2.2 TFLOPS; experts reach ~9.4 TOPS int8 | faster long fills (now 28 min for 256K) | days | profile the tensor FA path at 64K/128K; tile sizes; GQA packing in the tensor path |
| 3 | NVFP4 experts on the GPU, exact: an E2M1 value × E4M3 scale is exact in fp16, so an fp16 tensor path keeps the original weights bit-for-bit | 97.3% → ~99% top-token; decode same; prefill slower | ~1 week | port the NVFP4 type and Metal dequant from master into this pin, then a W4A16 path in `mul_mm_id` |
| 4 | Decode dispatch fusion: 5,608 nodes per token, 27.5 ms GPU vs ~24 ms at 120 GB/s | 35 → up to ~40 tok/s | days | `INKLING_OP_PROFILE_NAMES=1` decode profile; fuse norm/controller/router chains |
| 5 | Sparse top-k attention over the 8 global passes (Quest-style page selection) | the only route to 512K–1M on 24 GB; not exact | weeks, research | prior-art scan first; measure KLD at 64K before any build |
| 6 | Upstream: issue + PR for mixed K/V types in Metal FA | fixes every model on Metal | 1 day + review | needs a `test-backend-ops` case; issue text must be human-written (llama.cpp AI policy) |
| 7 | Rebase this patch onto current llama.cpp master | NVFP4 kernels, newer fixes | days | master `5220a2d40` (25 Nov) still lacks Metal for the compact bias (#32740) |

Ruled out (measured or computed): GQA heads as rows (`INKLING_FA_GQA_ROWS`, bit-identical, no gain); streaming experts
from SSD (decode collapses); 3-bit experts (quality); DSpark alone for code (−17% on new code).

## Method notes

- A fanless Air drifts 15–20% with heat: alternate arms (ABBA) with 45–90 s cool-downs.
- **Check swap before trusting a long-context number.** Results taken with ~6 GB of swap were 30–40% low and led to a
  wrong diagnosis here once (see the correction in `results/2026-11-26-layer-map.md`).
- Quality: `llama-perplexity --kl-divergence` against the same build's f16-KV (or reference-file) baseline, on both prose
  and code; code is the more sensitive corpus.
- zsh does not word-split `$VAR`; put multi-flag commands in `sh` scripts.
