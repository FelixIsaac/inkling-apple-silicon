# Why the patch is fast, current-upstream baselines, and a GPU mixed KV cache (M5 Air 24 GB, 25 Nov 2026)

MacBook Air M5, 24 GB, 10-core GPU, on AC, other apps closed. Fold file (`pure-Q4-tie64-fold`) unless noted.

## Baselines: is current llama.cpp already this fast? No.

`llama-bench -p 2048 -n 128 -ub 1024 -b 2048 -fa 1 -r 1`, two rounds in mirrored order with 60 s cool-downs (mean shown):

| Arm | pp2048 | tg128 |
|---|---:|---:|
| llama.cpp master `5220a2d40` (25 Nov), stock, unsloth `UD-Q4_K_M` (NVFP4 experts now on GPU) | 191 | 23.1 |
| llama.cpp master `5220a2d40`, stock, fold file | **197** | **28.1** |
| pinned `0a9105de` + this patch, all flags off | 201 | 26.5 |
| this patch, full M5 stack | 680 | 34.4 |
| + `inkling-m5-extras` | **717** | **34.8** (after the decode fix below) |

Homebrew `llama.cpp 0.9.0` (`cd959bae9`, 4 Nov) predates Inkling support and cannot load these files.

## Why the patch is 3.4× faster: same binary and file, flags off vs on

One prefill graph of 1,024 tokens (`INKLING_OP_PROFILE=2`, with `INKLING_OP_PROFILE_NAMES=1` from the extras patch):

| | Stock | Patched |
|---|---:|---:|
| Graph nodes | 11,744 | 6,208 |
| GPU time | 4,556 ms | 1,385 ms |
| Relative-position bias from generic ops (permute, CONT, CPY, GET_ROWS, SUB, ADD, f32 matmul) | ~1,560 ms (34%) | 37 ms (one mask kernel) |
| Expert matmuls | 1,116 ms | 668 ms (W4A8 int8, 64-token tiles) |
| Depth mixing and transposes | ~294 ms | 51 ms (fused `depth_mix`) |
| SiTU from SCALE, TANH, … | ~170 ms | ~0 (fused, then folded into the quantiser) |

For one decode token, stock uses 11,624 nodes and 35.0 ms of GPU time with a single encoder thread; patched uses 5,512 nodes and 28.1 ms with encoding split over two threads.

## Changes in this round

1. **Decode fix for `INKLING_SPLIT_CTRL`.** It is now gated to `n_tokens > 1`. In decode, two half-width matvecs plus an add cost more launches than one matvec saves: 33.4 → 34.8 tok/s.
2. **`INKLING_REL_FA` on by default in `env.sh`.** At 16K depth, alternated: pp1024 406/487 → 537/535, tg64 23.4/28.6 → 28.9/28.6. The 128K compute buffer drops from 7,625 to 1,057 MiB. Quality on the fold file: 98.9% same top token, KLD 0.0008.
3. **Mixed KV cache (K `q8_0`, V `q4_0`) on the Metal GPU.** Upstream (also in master `5220a2d40`) has two defects:
   - `ggml_metal_device_supports_op` rejects FLASH_ATTN_EXT whenever K type ≠ V type. Every mixed-type attention therefore falls back to the CPU, which is slow, and the CPU FA ignores `src[6..8]`. With `REL_FA` the relative bias is dropped on the global layers (89.6% same top).
   - The F16 dequant pass uses the K-type kernel to dequantize V (`get_pipeline_flash_attn_ext_kv_f16` is named from `src[1]`).

   Fixes: V is dequantized with its own kernel; `use_kv_f16` requires a kernel for both types and sends mixed ops that aren't vec-sized through the dequant path; there are 9 vec instantiations `kernel_flash_attn_ext_vec_q8_0q4_0_dk128_dv128*`; the pipeline is named by the type pair; `supports_op` and an assert now admit (q8_0, q4_0) at dk = dv = 128.

## KV cache quality (`llama-perplexity --kl-divergence`, 2 × 8,192 tokens, against the same build's F16 KV)

| K / V | Path | Same top | Mean KLD |
|---|---|---:|---:|
| q8_0 / q8_0 | vec (ub 16) | 99.04% | 0.0012 |
| **q8_0 / q4_0** | **vec, new mixed kernel (ub 16)** | **98.29%** | **0.0030** |
| **q8_0 / q4_0** | **`REL_FA`, tensor + dequant (ub 1024)** | **98.30%** | **0.0029** |
| q4_0 / q4_0 | `REL_FA` | 97.4–98.1% | 0.004–0.0075 |

For comparison, Gemma 4 26B-A4B on Homebrew 0.9.0 (same corpus, 3 chunks) is much more sensitive to KV quantization: q8/q8 gives 93.2%, q8/q4 88.4% and q4/q4 84.7%.

## Memory: the longest context that fits (MTL0 buffers from `llama-server -v`, default Metal working set ~18,186 MiB, weights 13,790 MiB)

| ctx | K / V | KV | SWA KV | Compute (`REL_FA`) | Total | Fits? |
|---:|---|---:|---:|---:|---:|---|
| 128K | q8 / q8 | 2,176 | 170 | ~1,067 | 3,413 | yes |
| 128K | q8 / q4 | 1,664 | 130 | ~1,067 | 2,861 | yes |
| 192K | q8 / q8 | 3,264 | 170 | ~1,452 | 4,886 | no |
| **192K** | **q8 / q4** | **2,496** | **130** | **~1,452** | **4,078** | **yes** |

The longest context at ≥98% top-token agreement goes from about 150K (q8/q8) to about 200K (q8/q4) on the default cap.
