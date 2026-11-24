"""Fold the per-expert output scales (blk.N.ffn_{gate_up,down}_exps.scale, left over from the NVFP4
checkpoint) into the fp16 block scales of the Q4_K / IQ4_NL expert weights, and drop the scale tensors.

The graph then skips the two MUL passes over every expert output (~6.6% of prefill on an M5 Air).
Exact apart from fp16 rounding of the new block scales; tie64 pairing is preserved (both blocks of a
pair get the same factor).

usage: GGUF_PY=build/source/gguf-py python3 scripts/fold_scales.py SRC.gguf DST.gguf
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("GGUF_PY", "build/source/gguf-py"))
import gguf  # noqa: E402

BS = {"Q4_K": 144, "IQ4_NL": 18}
F16_MIN_NORMAL = 6.103515625e-05


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    src, dst = sys.argv[1], sys.argv[2]
    r = gguf.GGUFReader(src)
    by_name = {t.name: t for t in r.tensors}
    scales = {n[: -len(".scale")] + ".weight": np.array(t.data, np.float32).ravel()
              for n, t in by_name.items() if n.endswith("_exps.scale")}
    if not scales:
        sys.exit("no *_exps.scale tensors found")

    arch = r.fields[gguf.Keys.General.ARCHITECTURE].contents()
    w = gguf.GGUFWriter(dst, arch)
    for f in r.fields.values():
        if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF."):
            continue
        vt = f.types[0]
        st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), vt, sub_type=st)

    keep = [t for t in r.tensors if not t.name.endswith("_exps.scale")]
    for t in keep:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()

    n_sub = n_blk = 0
    worst_rel = 0.0
    for t in keep:
        data = t.data
        if t.name in scales:
            tn = t.tensor_type.name
            if tn not in BS:
                sys.exit(f"{t.name}: unsupported type {tn}")
            s = scales[t.name]
            bs = BS[tn]
            b = np.array(data, np.uint8).reshape(len(s), -1, bs)  # expert, block, bytes
            cols = [(0, 2), (2, 4)] if tn == "Q4_K" else [(0, 2)]
            for lo, hi in cols:
                d = b[:, :, lo:hi].copy().view(np.float16)[..., 0].astype(np.float32)
                exact = d * s[:, None]
                new = exact.astype(np.float16)
                nz = exact != 0
                rel = np.abs(new.astype(np.float32) - exact)[nz] / np.abs(exact[nz])
                worst_rel = max(worst_rel, float(rel.max()) if rel.size else 0.0)
                n_sub += int((np.abs(exact[nz]) < F16_MIN_NORMAL).sum())
                n_blk += int(nz.sum())
                b[:, :, lo:hi] = new[..., None].view(np.uint8).reshape(b.shape[0], b.shape[1], 2)
            data = b.reshape(data.shape)
        w.write_tensor_data(data, tensor_endianess=r.endianess)
    w.close()
    print(f"folded {len(scales)} scale tensors; block scales {n_blk}, subnormal {n_sub} "
          f"({100 * n_sub / max(n_blk, 1):.3f}%), worst fp16 rel rounding {worst_rel:.2e}")
    print("FOLD_DONE")


if __name__ == "__main__":
    main()
