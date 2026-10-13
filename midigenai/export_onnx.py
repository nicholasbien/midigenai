"""Export a checkpoint to ONNX for the browser runner (web/midigen-web.mjs).

    python -m midigenai.export_onnx --version v5 --out dist/v5/onnx
    python -m midigenai.export_onnx --version v5 --out dist/v5/onnx --upload    # then to the HF repo, needs HF_TOKEN

Two graphs of one decoder step with a KV cache, one per browser engine, because what is fast differs:

- `decoder-webgpu-q4.onnx` (WebGPU): inputs `ids` (1, S) and per layer `past_k{i}` / `past_v{i}` (1, H, P, D);
  outputs `logits` (1, V) for the last position and `present_k{i}` / `present_v{i}` (1, H, P+S, D). The page keeps
  the presents on the GPU and feeds them back, so the cache never crosses to the CPU. Weights 4-bit (MatMulNBits,
  block 32), which WebGPU runs well and wasm does not.
- `decoder-wasm-int8.onnx` (CPU, WebAssembly): the cache is time-major, `past_k{i}` (1, P, H, D) in and only the new
  rows `new_k{i}` (1, S, H, D) out, so a step copies the cache into wasm once and nothing back; the page appends the
  new rows into buffers it keeps. Copying the whole cache in and out each step made a 500-token prompt 2.5x slower.
  Weights int8 (dynamic quantization).

Both compute RoPE from the past length, so any context works (as torch's extrapolated table does), and both are
checked against the torch model before quantizing. `manifest.json` next to them lists the files, sizes, sha256s and
the shapes the runner needs. Needs torch and the `onnx` extra (`pip install "midigenai[onnx]"`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

GPU_FILE = "decoder-webgpu-q4.onnx"
CPU_FILE = "decoder-wasm-int8.onnx"
OPSET = 17


def _wrappers():
    import torch
    import torch.nn.functional as F

    class _Step(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
            d = m.cfg.head_dim
            self.register_buffer("inv_freq", 1.0 / (m.cfg.rope_base ** (torch.arange(0, d, 2).float() / d)))

        def rope(self, x, cos, sin):
            x1, x2 = x[..., 0::2], x[..., 1::2]          # interleaved pairs, as model.apply_rope
            return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)

        def run(self, ids, past, time_major: bool):
            m = self.m
            H, D = m.cfg.n_heads, m.cfg.head_dim
            S = ids.shape[1]
            P = past[0].shape[1 if time_major else 2]
            pos = torch.arange(S).float() + P
            freqs = torch.outer(pos, self.inv_freq)
            cos, sin = freqs.cos()[None, None], freqs.sin()[None, None]
            mask = (torch.arange(P + S)[None, :] > pos[:, None].long()).float() * -1e9     # (S, P+S) causal
            x = m.embed(ids)
            outs = []
            for i, b in enumerate(m.blocks):
                a = b.attn
                h = b.attn_norm(x)
                q = self.rope(a.q_proj(h).view(1, S, H, D).transpose(1, 2), cos, sin)
                k = self.rope(a.k_proj(h).view(1, S, H, D).transpose(1, 2), cos, sin)
                v = a.v_proj(h).view(1, S, H, D).transpose(1, 2)
                pk, pv = past[2 * i], past[2 * i + 1]
                if time_major:
                    pk, pv = pk.transpose(1, 2), pv.transpose(1, 2)
                kk, vv = torch.cat([pk, k], 2), torch.cat([pv, v], 2)
                outs += [k.transpose(1, 2), v.transpose(1, 2)] if time_major else [kk, vv]
                att = F.softmax(q @ kk.transpose(-1, -2) * D ** -0.5 + mask, -1)
                x = x + a.o_proj((att @ vv).transpose(1, 2).reshape(1, S, H * D))
                x = x + b.ffn(b.ffn_norm(x))
            logits = m.lm_head(m.norm(x[:, -1:, :]))[:, 0, :]
            return (logits, *outs)

    class PresentStep(_Step):
        def forward(self, ids, *past):
            return self.run(ids, past, time_major=False)

    class NewRowsStep(_Step):
        def forward(self, ids, *past):
            return self.run(ids, past, time_major=True)

    return PresentStep, NewRowsStep


def load_model(checkpoint: str | Path):
    import torch
    from .model import ModelConfig, MusicTransformer
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = MusicTransformer(ModelConfig(**ck["model_config"])).eval()
    model.load_state_dict(ck["model"])
    return model


def names(n_layers: int, time_major: bool) -> tuple[list[str], list[str]]:
    past = [f"past_{c}{i}" for i in range(n_layers) for c in "kv"]
    out = [f"{'new' if time_major else 'present'}_{c}{i}" for i in range(n_layers) for c in "kv"]
    return past, out


def check_step(model, step, time_major: bool, tol: float = 1e-3) -> float:
    """Max |logit difference| between the wrapper and the model's own forward, on a prefill and one cached step."""
    import torch
    cfg = model.cfg
    ids = torch.randint(0, cfg.vocab_size, (1, 40))
    shape = (1, 0, cfg.n_heads, cfg.head_dim) if time_major else (1, cfg.n_heads, 0, cfg.head_dim)
    empty = [torch.zeros(shape) for _ in range(2 * cfg.n_layers)]
    with torch.no_grad():
        ref, caches = model(ids)
        got = step(ids, *empty)
        d1 = (ref[:, -1] - got[0]).abs().max().item()
        past = [t.transpose(1, 2).contiguous() if time_major else t for kv in caches for t in kv]
        nxt = torch.tensor([[7]])
        ref2, _ = model(nxt, caches)
        d2 = (ref2[:, -1] - step(nxt, *past)[0]).abs().max().item()
    worst = max(d1, d2)
    if worst > tol:
        raise RuntimeError(f"ONNX wrapper differs from the model by {worst:g} (> {tol:g})")
    return worst


def export(model, step, path: Path, time_major: bool) -> None:
    import torch
    cfg = model.cfg
    shape = (1, 0, cfg.n_heads, cfg.head_dim) if time_major else (1, cfg.n_heads, 0, cfg.head_dim)
    empty = [torch.zeros(shape) for _ in range(2 * cfg.n_layers)]
    ids = torch.randint(0, cfg.vocab_size, (1, 8))
    past, outs = names(cfg.n_layers, time_major)
    p_axis, o_axis = (1, 1) if time_major else (2, 2)
    torch.onnx.export(
        step, (ids, *empty), str(path),
        input_names=["ids", *past], output_names=["logits", *outs],
        dynamic_axes={"ids": {1: "S"}, **{n: {p_axis: "P"} for n in past},
                      **{n: {o_axis: "S" if time_major else "T"} for n in outs}},
        opset_version=OPSET, dynamo=False)


def quantize_int8(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gather"])


def quantize_q4(src: Path, dst: Path) -> None:
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    q = MatMulNBitsQuantizer(onnx.load(str(src)), block_size=32, is_symmetric=True)
    q.process()
    onnx.save(q.model.model, str(dst))


def agreement(model, path: Path, ids: list[int], time_major: bool) -> dict:
    """Teacher-forced next-token agreement of an exported graph with the torch model (fp32) over `ids`,
    decoded one token at a time so the cached path is what is measured."""
    import numpy as np
    import onnxruntime as ort
    import torch
    cfg = model.cfg
    with torch.no_grad():
        ref = torch.log_softmax(model(torch.tensor([ids]))[0][0], -1).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    past_names, out_names = names(cfg.n_layers, time_major)
    shape = (1, 0, cfg.n_heads, cfg.head_dim) if time_major else (1, cfg.n_heads, 0, cfg.head_dim)
    past = {n: np.zeros(shape, np.float32) for n in past_names}
    rows = []
    for t in ids:
        out = sess.run(None, {"ids": np.array([[t]], np.int64), **past})
        rows.append(out[0][0])
        for n, o, new in zip(past_names, out_names, out[1:]):
            past[n] = np.concatenate([past[n], new], axis=1) if time_major else new
    lg = np.stack(rows)
    lp = lg - lg.max(-1, keepdims=True)
    lp = lp - np.log(np.exp(lp).sum(-1, keepdims=True))
    kl = float((np.exp(ref) * (ref - lp)).sum(-1).mean())
    return {"top1": float((ref.argmax(-1) == lp.argmax(-1)).mean()), "kl": kl, "tokens": len(ids)}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default=None, help="checkpoint version on the hub (default: the package default)")
    ap.add_argument("--checkpoint", help="a local ckpt .pt instead of the hub's (with --tokenizer)")
    ap.add_argument("--tokenizer", help="the checkpoint's tokenizer.json (with --checkpoint)")
    ap.add_argument("--out", required=True, help="output folder (gets the two .onnx files and manifest.json)")
    ap.add_argument("--check-midi", help="a MIDI file to measure agreement with the torch model on (default: none)")
    ap.add_argument("--upload", action="store_true", help="upload the folder to the HF repo as VERSION/onnx (HF_TOKEN)")
    ap.add_argument("--repo", default=None, help="HF repo for --upload (default: the package's)")
    a = ap.parse_args(argv)

    from .hub import DEFAULT_REPO, download_files, folder
    version = folder(a.version)
    if a.checkpoint:
        ckpt, tok = Path(a.checkpoint), Path(a.tokenizer) if a.tokenizer else None
    else:
        ckpt, tok = download_files(version=version)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_model(ckpt)
    cfg = model.cfg
    PresentStep, NewRowsStep = _wrappers()
    plan = [("gpu", PresentStep, False, GPU_FILE, quantize_q4, "q4"),
            ("cpu", NewRowsStep, True, CPU_FILE, quantize_int8, "int8")]
    check_ids = None
    if a.check_midi:
        from .generate import Generator
        g = Generator(checkpoint_path=None, tokenizer_path=tok, backend="remote", decode=lambda spec: [])
        check_ids = [g.bos_id, *g.encode_midi_file(a.check_midi)][:300]
    files = {}
    for engine, cls, time_major, name, quantize, quant in plan:
        step = cls(model).eval()
        diff = check_step(model, step, time_major)
        fp32 = out / (name + ".fp32.tmp")
        export(model, step, fp32, time_major)
        quantize(fp32, out / name)
        fp32.unlink()
        for extra in out.glob(name + "*.data"):     # external-data side files, if the exporter wrote any
            extra.unlink()
        entry = {"file": name, "bytes": (out / name).stat().st_size, "sha256": sha256(out / name), "quant": quant,
                 "cache": "new-rows, time-major (1,P,H,D)" if time_major else "present, (1,H,P,D)",
                 "fp32_max_logit_diff": diff}
        if check_ids:
            entry["agreement"] = agreement(model, out / name, check_ids, time_major)
        files[engine] = entry
        print(f"{engine}: {name} {entry['bytes'] / 1e6:.1f} MB" +
              (f", top-1 {entry['agreement']['top1']:.3f}" if check_ids else ""), flush=True)
    import onnxruntime
    manifest = {"version": version, "vocab_size": cfg.vocab_size, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                "head_dim": cfg.head_dim, "opset": OPSET, "exported_with_onnxruntime": onnxruntime.__version__,
                "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out}/manifest.json")
    if a.upload:
        from huggingface_hub import HfApi
        HfApi().upload_folder(repo_id=a.repo or DEFAULT_REPO, folder_path=str(out), path_in_repo=f"{version}/onnx",
                              commit_message=f"{version}: ONNX decoders for the browser runner")
        print(f"uploaded to {a.repo or DEFAULT_REPO}/{version}/onnx")


if __name__ == "__main__":
    main()
