# Folded expert scales, split controllers, SiTU in the int8 quantiser, 25 Nov 2026

MacBook Air M5, 24 GB, 10-core GPU, fanless, on AC. Same build as the int8 release plus `patches/inkling-m5-extras.patch`.
Weights: `pure-Q4-ctrl16` → `tie64` → `fold` (made from unsloth `UD-Q4_K_M`, see below).

## Why the unsloth file is slow

`Inkling-2-Mini-UD-Q4_K_M.gguf` stores 40 expert tensors (20 layers × gate_up and down, 10.9 GB) as NVFP4. This revision
has no Metal kernel for them, so llama.cpp leaves them in a CPU buffer: `CPU_Mapped 11404 MiB, MTL0_Mapped 4425 MiB`,
~23 prefill and ~10 decode. Converting the file with the `make-pure-q4.sh` quantize line (`--allow-requantize` works from
this file directly), then `ctrl_f16.py`, puts 13.8 GB on MTL0 and 0.25 GB on the CPU.

## What changed

Where prefill time went on the full M5 stack (`INKLING_OP_PROFILE=2`, keyed by name): two MULs,
`ffn_moe_gate_up_scaled` and `ffn_moe_down_scaled`, took 6.6%. They multiply every expert output by a per-expert scalar
(`blk.N.ffn_{gate_up,down}_exps.scale`, 128 values, the NVFP4 global scale).

1. **`scripts/fold_scales.py`** multiplies those scalars into the fp16 `d` (and `dmin` for Q4_K) of every block of that
   expert and drops the tensors. They load as `TENSOR_NOT_REQUIRED`, so the graph skips the MULs. tie64 pairs stay tied.
   Of 318M block scales, 8 became subnormal; the worst fp16 rounding is 4.9e-4 relative.
2. **`INKLING_SPLIT_CTRL`**: the loop controller's `concat(norm(cur), norm(msg)) @ W[4608×64]` becomes two matmuls on
   half-width views of `W` plus an add. CONCAT (2.2%) disappears; the two f16 matmuls take 29 ms against 34 ms for the
   wide one, and the add takes 0.7 ms.
3. **`INKLING_SITU_A8`**: when a SiTU node's only consumer is a W4A8 (G64) expert matmul, the node is not run. Instead
   the matmul's activation quantiser (`kernel_inkling_quant_a8g64_situ`) computes SiTU from the gate and up views, with
   arithmetic identical to `kernel_inkling_situ`. It runs 16 threads per 64-value group with simd-shuffle max and sum.
   The int8 kernels never read the f32 input, so it is never written. The gate and up views are registered as `src[3]`
   and `src[4]` of the MUL_MAT_ID, which keeps them allocated and gives the concurrency tracker the dependency. Without
   that, ggml-alloc reused their memory and quality collapsed (KLD 0.56).

## Speed, `llama-bench -p 2048 -n 128 -ub 1024 -b 2048 -fa 1 -r 1`, 4 rounds, arms alternating, 75 s cool-downs

| Arm | pp2048 runs | Mean | tg128 mean |
|---|---|---:|---:|
| full M5 stack, `tie64` | 585.0 / 589.8 / 569.0 / 586.2 | 582.5 | 31.7 |
| + fold + `SPLIT_CTRL` + `SITU_A8` | 694.6 / 657.5 / 668.8 / 680.1 | **675.2** | 32.0 |

Earlier pairwise rounds on the same machine: fold +5.7% (578 → 611), split +2.1% on top (639 → 653), SiTU-in-quantiser
about 22 ms per 1,024-token graph (~1.4%).

## Quality, `llama-perplexity --kl-divergence`, GSM8K sample, 4 × 2,048 tokens, against the full M5 stack on `tie64`

| Arm | Same top token | Mean KLD | PPL ratio |
|---|---:|---:|---:|
| fold | 98.83% | 0.00107 | 1.0042 |
| fold + `SPLIT_CTRL` | 98.90% | 0.00104 | 1.0042 |
| fold + `SPLIT_CTRL` + `SITU_A8` | 98.90% | 0.00104 | 1.0042 |

The last two rows are identical. The fold's cost is about the same as `INKLING_W4A8` alone (99.0%, 0.0011).

## Context fill on 24 GB (`tie64`, f16 KV, `-d`)

| Depth | pp1024 | tg64 |
|---|---:|---:|
| 0 | 624 | 33.4 |
| 8K | 478 | 26.0 |
| 16K | 423 | 23.8 |
| 32K | 167 | 11.3 (swapping) |

With `-ctk q8_0 -ctv q8_0`, 32K improves to 179 / 15.7, but 16K drops to 244 / 15.8.

## Tried, no gain on this machine

`ROUTE_MAP`, `REL_FA`, `GROUP_SHORT_EXPERTS`, `PAIRED_GATES`, `EXPERT_SIMDGROUPS=8`, `NAX_TILE=1`, `-ub 512/2048`.
`ADAPT` made no measurable difference; `T64` gave a small gain.

## Long context on 24 GB: use `INKLING_REL_FA` and a q8_0 KV cache

Without `REL_FA`, the relative-bias mask is materialised per head over the whole context, so the compute buffer grows
with context × ubatch. `llama-server -v`, MTL0 buffers, fold file:

| ctx | ubatch | KV | Global KV | Compute | Compute with `REL_FA` |
|---:|---:|---|---:|---:|---:|
| 32K | 1024 | f16 | 1,024 MiB | 2,053 MiB | — |
| 128K | 1024 | f16 | 4,096 MiB | 7,625 MiB | 1,057 MiB |
| 128K | 1024 | q8_0 | 2,176 MiB | — | 1,067 MiB |
| 192K | 1024 | q8_0 | 3,264 MiB | — | 1,452 MiB |
| 256K | 512 | q8_0 | 4,352 MiB | — | 1,435 MiB |

The SWA KV is fixed at 128–320 MiB. The weights use 13,790 MiB of a ~18,190 MiB Metal working set, which leaves room for
about 128K tokens with `REL_FA` and q8_0 KV. Quality against the default on the fold file: `REL_FA` gives 98.93% same top
and KLD 0.00082; adding q8_0 KV gives 98.95% and 0.00110.
