%%writefile helm_analysis_core.py
"""
Shared analysis utilities for HELM routed-head experiments.

IMPORTANT NUMPY CONVERSION RULE
-------------------------------
Every Gram matrix is converted in exactly this order:

    tensor.detach().to(torch.float32).cpu().numpy().astype(np.float64)

Do not reorder that conversion.

The three companion scripts intentionally use the same:
- deterministic validation examples
- MLM masking
- CE implementation
- context/residual head geometry
- raw/directional effective-rank calculations

so their outputs are directly comparable.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import pyarrow.parquet as pq
except Exception as exc:
    raise RuntimeError("pyarrow is required") from exc

try:
    from huggingface_hub import hf_hub_download
except Exception as exc:
    raise RuntimeError("huggingface_hub is required") from exc


DATA_REPO = "JamesResearch1216/HELM-Easiness-Data-10B-Labeled-v6"
VALIDATION_FILE = "data/seq_1024/validation-00000.parquet"
CHECKPOINT_FILE = "checkpoint-006500.pt"
TRAINING_STATE_FILE = "training_state.json"


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def get_hf_token():
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        return None


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rankdata_np(x):
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and sx[j] == sx[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def spearman_np(x, y):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata_np(x), rankdata_np(y))[0, 1])


def safe_mean(x):
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def safe_std(x):
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=1)) if len(a) > 1 else float("nan")


def safe_sem(x):
    a = np.asarray(list(x), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else float("nan")


def gini_np(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, 0.0, None)
    if x.sum() <= 0 or len(x) == 0:
        return 0.0
    sx = np.sort(x)
    n = len(sx)
    return float(
        (2.0 * np.sum((np.arange(1, n + 1)) * sx) / (n * sx.sum()))
        - (n + 1) / n
    )


def effective_rank_from_gram(gram):
    gram = np.asarray(gram, dtype=np.float64)
    g = 0.5 * (gram + gram.T)
    vals = np.linalg.eigvalsh(g)
    vals = np.clip(vals, 0.0, None)
    total = vals.sum()
    if total <= 1e-18:
        return 0.0, 0.0, vals
    p = vals / total
    pp = p[p > 1e-15]
    erank = float(np.exp(-(pp * np.log(pp)).sum()))
    prank = float((total * total) / (np.square(vals).sum() + 1e-18))
    return erank, prank, vals


def directional_gram(raw_gram):
    raw_gram = np.asarray(raw_gram, dtype=np.float64)
    diag = np.clip(np.diag(raw_gram), 1e-18, None)
    denom = np.sqrt(np.outer(diag, diag))
    corr = raw_gram / denom
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def mean_abs_offdiag(mat):
    mat = np.asarray(mat, dtype=np.float64)
    if mat.shape[0] <= 1:
        return 0.0
    mask = ~np.eye(mat.shape[0], dtype=bool)
    return float(np.abs(mat[mask]).mean())


def parse_layers(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def write_csv(path: Path, rows: List[dict]):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def strip_state_prefixes(state):
    out = {}
    for key, value in state.items():
        k = key
        changed = True
        while changed:
            changed = False
            for prefix in ("module.", "_orig_mod."):
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    changed = True
        out[k] = value
    return out


def import_model_file(path: Path, module_name="analysis_arch"):
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

@dataclass
class DeviceContext:
    device: torch.device
    kind: str
    xm: object = None

    def mark_step(self):
        if self.kind == "xla" and self.xm is not None:
            self.xm.mark_step()

    def autocast(self):
        if self.kind == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        if self.kind == "xla":
            return torch.autocast("xla", dtype=torch.bfloat16)
        return contextlib.nullcontext()


def resolve_device(requested):
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            requested = "cuda"
        else:
            try:
                import torch_xla.core.xla_model as xm
                return DeviceContext(xm.xla_device(), "xla", xm)
            except Exception:
                requested = "cpu"

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return DeviceContext(torch.device("cuda"), "cuda")

    if requested == "xla":
        import torch_xla.core.xla_model as xm
        return DeviceContext(xm.xla_device(), "xla", xm)

    if requested == "cpu":
        return DeviceContext(torch.device("cpu"), "cpu")

    raise ValueError(requested)


# ---------------------------------------------------------------------------
# Assets/model/data
# ---------------------------------------------------------------------------

def download_file(repo, filename, repo_type, cache_dir, token, label):
    print(f"Downloading {label}: {repo}/{filename}")
    return Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            repo_type=repo_type,
            token=token,
            local_dir=str(cache_dir / label),
        )
    )


def resolve_checkpoint(local_path, repo, filename, cache_dir, token):
    if local_path:
        p = Path(local_path)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    return download_file(repo, filename, "model", cache_dir, token, "checkpoint")


def resolve_validation(local_path, repo, filename, cache_dir, token):
    if local_path:
        p = Path(local_path)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    return download_file(repo, filename, "dataset", cache_dir, token, "validation")


def maybe_training_state(repo, cache_dir, token):
    try:
        return download_file(
            repo, TRAINING_STATE_FILE, "model", cache_dir, token, "training_state"
        )
    except Exception as exc:
        print(f"WARNING: training_state.json unavailable: {exc}")
        return None


def read_breakpoints(path):
    if path is None:
        return None
    try:
        state = json.loads(Path(path).read_text())
        ed = state.get("easiness_dict")
        if isinstance(ed, dict) and ed.get("breakpoints"):
            return [float(x) for x in ed["breakpoints"]]
    except Exception:
        pass
    return None


def instantiate_config(module, breakpoints=None):
    kwargs = {}
    if breakpoints is not None:
        kwargs["easiness_cdf_breakpoints"] = breakpoints
    try:
        return module.HELMConfig(**kwargs)
    except TypeError:
        return module.HELMConfig()


def load_model(module, checkpoint, dev, breakpoints=None):
    cfg = instantiate_config(module, breakpoints)
    model = module.HELMForMaskedLM(cfg)
    payload = torch.load(str(checkpoint), map_location="cpu")
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    state = strip_state_prefixes(state)

    incompat = model.load_state_dict(state, strict=False)
    missing = list(incompat.missing_keys)
    unexpected = list(incompat.unexpected_keys)
    if missing or unexpected:
        print(f"State load: {len(missing)} missing, {len(unexpected)} unexpected")
        if missing:
            print(" missing:", missing[:8])
        if unexpected:
            print(" unexpected:", unexpected[:8])
        if len(missing) > 5 or len(unexpected) > 5:
            raise RuntimeError("Large checkpoint/architecture mismatch")

    model.to(dev.device)
    model.eval()
    if hasattr(model, "enable_efficient_inference"):
        try:
            model.enable_efficient_inference("dense", compile=False)
        except Exception:
            pass
    return model, cfg


def deterministic_span_mask(
    ids,
    config,
    seed,
    probability=0.30,
    span_length=3,
):
    ids = ids.clone().long()
    labels = torch.full_like(ids, -100)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))

    special = {
        int(config.bos_token_id),
        int(config.eos_token_id),
        int(config.pad_token_id),
        int(config.mask_token_id),
        int(config.unk_token_id),
    }
    candidates = [i for i, t in enumerate(ids.tolist()) if int(t) not in special]
    if not candidates:
        return ids, labels

    target = max(1, int(round(probability * len(candidates))))
    cset = set(candidates)
    perm = torch.randperm(len(candidates), generator=g).tolist()
    chosen = set()

    for pi in perm:
        if len(chosen) >= target:
            break
        st = candidates[pi]
        for p in range(st, min(st + span_length, ids.numel())):
            if p in cset:
                chosen.add(p)
                if len(chosen) >= target:
                    break

    pos = torch.tensor(sorted(chosen), dtype=torch.long)
    labels[pos] = ids[pos]
    r = torch.rand(len(pos), generator=g)

    mask_sel = r < 0.80
    rand_sel = (r >= 0.80) & (r < 0.90)
    ids[pos[mask_sel]] = int(config.mask_token_id)

    if rand_sel.any():
        ids[pos[rand_sel]] = torch.randint(
            0,
            int(config.vocab_size),
            (int(rand_sel.sum()),),
            generator=g,
        )
    return ids, labels


def prepare_examples(validation_path, config, num_examples, seq_len, seed):
    table = pq.read_table(
        str(validation_path), columns=["input_ids", "easiness_score"]
    )
    n = min(int(num_examples), table.num_rows)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(table.num_rows)[:n]

    input_col = table.column("input_ids")
    easy_col = table.column("easiness_score")
    examples = []

    for i, row_idx in enumerate(indices.tolist()):
        ids = torch.tensor(input_col[row_idx].as_py(), dtype=torch.long)[:seq_len]
        if ids.numel() < seq_len:
            ids = torch.cat(
                [
                    ids,
                    torch.full(
                        (seq_len - ids.numel(),),
                        int(config.pad_token_id),
                        dtype=torch.long,
                    ),
                ]
            )

        masked, labels = deterministic_span_mask(
            ids, config, seed=seed + 100003 * i
        )
        examples.append(
            {
                "input_ids": masked,
                "labels": labels,
                "attention_mask": (masked != int(config.pad_token_id)).long(),
                "easiness_score": torch.tensor(
                    float(easy_col[row_idx].as_py()), dtype=torch.float32
                ),
                "example_id": torch.tensor(i, dtype=torch.long),
            }
        )
    return examples


def make_batches(examples, batch_size):
    usable = (len(examples) // batch_size) * batch_size
    examples = examples[:usable]
    if not examples:
        raise RuntimeError("Not enough complete examples for one batch")

    batches = []
    for s in range(0, usable, batch_size):
        chunk = examples[s : s + batch_size]
        batches.append(
            {
                k: torch.stack([x[k] for x in chunk], dim=0)
                for k in chunk[0]
            }
        )
    return examples, batches


def make_subbatches(examples, indices, batch_size):
    idx = list(indices)
    usable = (len(idx) // batch_size) * batch_size
    idx = idx[:usable]
    out = []
    for s in range(0, usable, batch_size):
        chunk = [examples[i] for i in idx[s : s + batch_size]]
        out.append(
            {
                k: torch.stack([x[k] for x in chunk], dim=0)
                for k in chunk[0]
            }
        )
    return out


def move_batch(batch, dev):
    return {k: v.to(dev.device) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Forward / CE
# ---------------------------------------------------------------------------

def call_model(model, batch, pass_easiness=True, current_step=6500):
    kwargs = {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
    }
    if pass_easiness:
        kwargs["easiness_score"] = batch.get("easiness_score")
    else:
        kwargs["easiness_score"] = None
    kwargs["current_step"] = current_step

    try:
        out = model(**kwargs)
    except TypeError:
        kwargs.pop("current_step", None)
        try:
            out = model(**kwargs)
        except TypeError:
            kwargs.pop("easiness_score", None)
            out = model(**kwargs)

    return out[0] if isinstance(out, (tuple, list)) else out


def per_example_ce(logits, labels, chunk_tokens=128):
    B, S, V = logits.shape
    sums = torch.zeros(B, device=logits.device, dtype=torch.float32)
    counts = torch.zeros(B, device=logits.device, dtype=torch.float32)

    for s in range(0, S, chunk_tokens):
        e = min(s + chunk_tokens, S)
        lgt = logits[:, s:e, :].to(torch.float32)
        lab = labels[:, s:e]
        losses = F.cross_entropy(
            lgt.reshape(-1, V),
            lab.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(B, e - s)
        valid = (lab != -100).float()
        sums += (losses * valid).sum(dim=1)
        counts += valid.sum(dim=1)

    return sums / counts.clamp_min(1.0)


def evaluate_ce(
    model,
    batches,
    dev,
    pass_easiness=True,
    return_per_example=False,
):
    all_vals = {}
    total_sum = 0.0
    total_n = 0

    with torch.no_grad():
        for cpu_batch in batches:
            batch = move_batch(cpu_batch, dev)
            with dev.autocast():
                logits = call_model(
                    model, batch, pass_easiness=pass_easiness
                )
                vals = per_example_ce(logits, batch["labels"])
            dev.mark_step()

            vals_cpu = vals.detach().to(torch.float32).cpu().tolist()
            ids = cpu_batch["example_id"].tolist()
            for eid, val in zip(ids, vals_cpu):
                all_vals[int(eid)] = float(val)
                total_sum += float(val)
                total_n += 1

    mean = total_sum / max(1, total_n)
    return (mean, all_vals) if return_per_example else mean


# ---------------------------------------------------------------------------
# Mask adapters
# ---------------------------------------------------------------------------

def full_mask_from_model(model, variant):
    """Return [B,L,H] hard-forward masks from the most recent forward."""
    layers = []

    if variant in {"learned", "topk"}:
        P = int(model.config.num_permanent_heads)
        for block in model.model.blocks:
            elastic = (
                block.mlt_vw_rtr.save_hard_mask.detach()
                .to(torch.float32)
                .cpu()
            )
            perm = torch.ones(
                elastic.size(0), P, dtype=torch.float32
            )
            layers.append(torch.cat([perm, elastic], dim=-1))

    elif variant == "random":
        for block in model.model.blocks:
            m = (
                block.attn.save_random_mask.detach()
                .to(torch.float32)
                .cpu()
            )
            layers.append(m)

    else:
        raise ValueError(variant)

    return torch.stack(layers, dim=1)


def collect_masks_and_ce(
    model,
    batches,
    dev,
    variant,
    pass_easiness,
):
    masks = {}
    ce_by_id = {}

    with torch.no_grad():
        for cpu_batch in batches:
            batch = move_batch(cpu_batch, dev)
            with dev.autocast():
                logits = call_model(
                    model, batch, pass_easiness=pass_easiness
                )
                vals = per_example_ce(logits, batch["labels"])
            dev.mark_step()

            fm = full_mask_from_model(model, variant)
            ids = cpu_batch["example_id"].tolist()
            vals = vals.detach().to(torch.float32).cpu().tolist()

            for bi, eid in enumerate(ids):
                masks[int(eid)] = fm[bi].numpy().astype(np.float32)
                ce_by_id[int(eid)] = float(vals[bi])

    return masks, ce_by_id


class LearnedRouterOverride:
    """
    Forward-hook replacement for HELM_7c/7e/constant-K router outputs.

    mode:
      dense
      permanent
      random_same_count
      dense_minus (requires target layer/head)
      forced (requires dict layer->[B,H] full masks)
    """

    def __init__(
        self,
        model,
        mode,
        target_layer=None,
        target_head=None,
        forced_masks=None,
    ):
        self.model = model
        self.mode = mode
        self.target_layer = target_layer
        self.target_head = target_head
        self.forced_masks = forced_masks or {}
        self.handles = []

    def _hook(self, li):
        def hook(module, inputs, output):
            B, H, _, _ = output.shape
            P = int(self.model.config.num_permanent_heads)

            if self.mode == "dense":
                return torch.ones_like(output)

            if self.mode == "permanent":
                result = torch.zeros_like(output)
                result[:, :P, :, :] = 1
                return result

            if self.mode == "dense_minus":
                result = torch.ones_like(output)
                if li == self.target_layer:
                    result[:, int(self.target_head), :, :] = 0
                return result

            if self.mode == "random_same_count":
                # output forward values are hard 0/1 even though an STE exists.
                k_elastic = (
                    output[:, P:, 0, 0].detach().float().sum(dim=-1).long()
                )
                E = H - P
                scores = torch.rand(
                    B, E, device=output.device, dtype=torch.float32
                )
                # Random unique ranks 0..E-1 per example.
                order = torch.argsort(scores, dim=-1, descending=True)
                ranks = torch.argsort(order, dim=-1)
                elastic = (
                    ranks < k_elastic.view(B, 1)
                ).to(output.dtype)
                perm = torch.ones(
                    B, P, device=output.device, dtype=output.dtype
                )
                full = torch.cat([perm, elastic], dim=-1)
                return full.view(B, H, 1, 1)

            if self.mode == "forced":
                m = self.forced_masks[li].to(
                    device=output.device, dtype=output.dtype
                )
                return m.view(B, H, 1, 1)

            raise ValueError(self.mode)

        return hook

    def __enter__(self):
        for li, block in enumerate(self.model.model.blocks):
            self.handles.append(
                block.mlt_vw_rtr.register_forward_hook(self._hook(li))
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()


class RandomAttentionMaskOverride:
    """
    Override HELM_32_random._random_full_mask for controlled CE tests.

    mode:
      dense
      permanent
      dense_minus
      forced
    """

    def __init__(
        self,
        model,
        mode,
        target_layer=None,
        target_head=None,
        forced_masks=None,
    ):
        self.model = model
        self.mode = mode
        self.target_layer = target_layer
        self.target_head = target_head
        self.forced_masks = forced_masks or {}
        self.originals = []
        self.old_modes = []

    def __enter__(self):
        P = int(self.model.config.num_permanent_heads)
        H = int(self.model.config.num_attention_heads)

        for li, block in enumerate(self.model.model.blocks):
            attn = block.attn
            self.old_modes.append(attn.routing_mode)
            attn.set_routing_mode("random")
            original = attn._random_full_mask
            self.originals.append(original)

            def make_override(layer_idx, attn_obj):
                def fn(_self, batch_size, device, dtype):
                    if self.mode == "dense":
                        return torch.ones(
                            batch_size, H, device=device, dtype=dtype
                        )

                    if self.mode == "permanent":
                        result = torch.zeros(
                            batch_size, H, device=device, dtype=dtype
                        )
                        result[:, :P] = 1
                        return result

                    if self.mode == "dense_minus":
                        result = torch.ones(
                            batch_size, H, device=device, dtype=dtype
                        )
                        if layer_idx == self.target_layer:
                            result[:, int(self.target_head)] = 0
                        return result

                    if self.mode == "forced":
                        return self.forced_masks[layer_idx].to(
                            device=device, dtype=dtype
                        )

                    raise ValueError(self.mode)
                return types.MethodType(fn, attn_obj)

            attn._random_full_mask = make_override(li, attn)

        return self

    def __exit__(self, exc_type, exc, tb):
        for block, original, old_mode in zip(
            self.model.model.blocks, self.originals, self.old_modes
        ):
            block.attn._random_full_mask = original
            block.attn.set_routing_mode(old_mode)


def forced_masks_for_ids(masks_by_id, ids, dev, target_layer, target_head, value):
    L, H = next(iter(masks_by_id.values())).shape
    out = {}
    for li in range(L):
        arr = np.stack([masks_by_id[int(i)][li] for i in ids], axis=0)
        if li == target_layer:
            arr[:, target_head] = float(value)
        out[li] = torch.tensor(arr, dtype=torch.float32, device=dev.device)
    return out


# ---------------------------------------------------------------------------
# Attention geometry
# ---------------------------------------------------------------------------

class AttentionInputCapture:
    def __init__(self, model, layers):
        self.model = model
        self.layers = list(layers)
        self.data = {}
        self.handles = []

    def _hook(self, li):
        def hook(module, inputs):
            # All current models have hidden_states and attention_mask first.
            self.data[li] = (inputs[0].detach(), inputs[1].detach())
        return hook

    def __enter__(self):
        for li in self.layers:
            self.handles.append(
                self.model.model.blocks[li].attn.register_forward_pre_hook(
                    self._hook(li)
                )
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles.clear()


def unmasked_attention_context(module, attn, hidden_states, attention_mask):
    qkv_proj = module.cast_linear(hidden_states, attn.qkv)
    B, S, _ = hidden_states.shape

    q, k, v = qkv_proj.split(attn.total_head_dim, dim=-1)
    q = q.view(B, S, attn.num_attention_heads, attn.d_head).permute(0, 2, 1, 3)
    k = k.view(B, S, attn.num_attention_heads, attn.d_head).permute(0, 2, 1, 3)
    v = v.view(B, S, attn.num_attention_heads, attn.d_head).permute(0, 2, 1, 3)

    q = module.justnorm(q)
    k = module.justnorm(k)
    q = attn.RoPE(q)
    k = attn.RoPE(k)

    sqk = attn.sqk * (
        attn.ngpt_sqk_init_value / attn.ngpt_sqk_init_scale
    )
    sqk = sqk.view(
        1, attn.num_attention_heads, 1, attn.d_head
    ).to(q.dtype)

    q = sqk * q
    k = sqk * k

    context = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask.to(q.dtype),
        scale=math.sqrt(attn.d_head),
    )

    if attn.config.use_exclusive_attention:
        vn = F.normalize(v, dim=-1)
        context = context - (
            context * vn
        ).sum(dim=-1, keepdim=True) * vn

    return context


def add_gram(acc, key, value):
    if key not in acc:
        acc[key] = np.zeros_like(value, dtype=np.float64)
    acc[key] += value


def geometry_analysis(
    model,
    module,
    batches,
    dev,
    variant,
    pass_easiness,
    layers,
    max_batches,
    sample_tokens,
):
    """
    Produce potential and executed context/residual Gram matrices.

    IMPORTANT:
      Conversion is exactly:
      detach -> float32 -> cpu -> numpy -> float64
    """
    grams = {}
    pr_ratio_examples = {li: [] for li in layers}

    with torch.no_grad():
        for cpu_batch in batches[:max_batches]:
            batch = move_batch(cpu_batch, dev)

            with AttentionInputCapture(model, layers) as cap:
                with dev.autocast():
                    _ = call_model(
                        model, batch, pass_easiness=pass_easiness
                    )
                dev.mark_step()

            full_masks = full_mask_from_model(model, variant).to(dev.device)

            for li in layers:
                hidden, attn_mask = cap.data[li]
                attn = model.model.blocks[li].attn

                with dev.autocast():
                    context = unmasked_attention_context(
                        module, attn, hidden, attn_mask
                    )  # [B,H,S,d]

                    S = context.size(2)
                    T = min(int(sample_tokens), S)
                    positions = torch.linspace(
                        0, S - 1, steps=T, device=context.device
                    ).long()
                    c = context.index_select(2, positions)

                    mask = full_masks[:, li, :].to(
                        device=c.device, dtype=c.dtype
                    ).view(c.size(0), c.size(1), 1, 1)

                    c_exec = c * mask

                    W = attn.output.weight.to(c.dtype).view(
                        attn.hidden_size,
                        attn.num_attention_heads,
                        attn.d_head,
                    )

                    y = torch.einsum("bhtd,ohd->bhto", c, W)
                    y_exec = y * mask

                    # Aggregated Gram matrices.
                    for state, c_use, y_use in (
                        ("potential", c, y),
                        ("executed", c_exec, y_exec),
                    ):
                        cflat = (
                            c_use.permute(1, 0, 2, 3)
                            .contiguous()
                            .view(c_use.size(1), -1)
                        )
                        yflat = (
                            y_use.permute(1, 0, 2, 3)
                            .contiguous()
                            .view(y_use.size(1), -1)
                        )

                        cgram = cflat @ cflat.T
                        ygram = yflat @ yflat.T

                        # REQUIRED CONVERSION ORDER.
                        cgram_np = (
                            cgram.detach()
                            .to(torch.float32)
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )
                        ygram_np = (
                            ygram.detach()
                            .to(torch.float32)
                            .cpu()
                            .numpy()
                            .astype(np.float64)
                        )

                        add_gram(grams, (li, state, "context"), cgram_np)
                        add_gram(grams, (li, state, "residual"), ygram_np)

                    # Exact per-example context participation-rank ratio r_PR / K.
                    flat = c_exec.to(torch.float32).reshape(
                        c_exec.size(0), c_exec.size(1), -1
                    )
                    flat = flat / math.sqrt(float(max(1, flat.size(-1))))
                    g = torch.bmm(flat, flat.transpose(1, 2))
                    tr = torch.diagonal(g, dim1=-2, dim2=-1).sum(-1)
                    tr2 = g.square().sum(dim=(-2, -1))
                    rpr = tr.square() / (tr2 + 1e-12)
                    K = full_masks[:, li, :].sum(-1).clamp_min(1).to(rpr.device)
                    ratio = rpr / K
                    pr_ratio_examples[li].extend(
                        ratio.detach().to(torch.float32).cpu().tolist()
                    )

    rows = []

    for key, gram in sorted(grams.items()):
        li, state, space = key
        raw_er, raw_pr, _ = effective_rank_from_gram(gram)
        dgram = directional_gram(gram)
        dir_er, dir_pr, _ = effective_rank_from_gram(dgram)

        energy = np.clip(np.diag(gram), 0.0, None)
        ef = energy / max(1e-18, energy.sum())
        top_sorted = np.sort(ef)[::-1]

        rows.append(
            {
                "layer": li,
                "state": state,
                "space": space,
                "heads": gram.shape[0],
                "raw_entropy_rank": raw_er,
                "raw_entropy_ratio": raw_er / gram.shape[0],
                "raw_participation_rank": raw_pr,
                "raw_participation_ratio": raw_pr / gram.shape[0],
                "directional_entropy_rank": dir_er,
                "directional_entropy_ratio": dir_er / gram.shape[0],
                "directional_participation_rank": dir_pr,
                "directional_participation_ratio": dir_pr / gram.shape[0],
                "mean_abs_cos": mean_abs_offdiag(dgram),
                "energy_gini": gini_np(energy),
                "top1_energy_fraction": float(top_sorted[:1].sum()),
                "top4_energy_fraction": float(top_sorted[:4].sum()),
            }
        )

    ratio_rows = []
    for li, vals in pr_ratio_examples.items():
        ratio_rows.append(
            {
                "layer": li,
                "mean_context_pr_ratio_per_example": safe_mean(vals),
                "std_context_pr_ratio_per_example": safe_std(vals),
                "min_context_pr_ratio_per_example": float(np.min(vals)) if vals else float("nan"),
                "max_context_pr_ratio_per_example": float(np.max(vals)) if vals else float("nan"),
            }
        )

    return rows, ratio_rows


# ---------------------------------------------------------------------------
# Router/mask statistics
# ---------------------------------------------------------------------------

def mask_statistics(masks_by_id, model):
    ids = sorted(masks_by_id)
    arr = np.stack([masks_by_id[i] for i in ids], axis=0)  # [N,L,H]
    N, L, H = arr.shape
    P = int(getattr(model.config, "num_permanent_heads", 0))

    layer_rows = []
    head_rows = []

    for li in range(L):
        counts = arr[:, li, :].sum(axis=-1)
        unique = np.unique(arr[:, li, :], axis=0).shape[0]

        layer_rows.append(
            {
                "layer": li,
                "mean_total_heads": float(counts.mean()),
                "std_total_heads": float(counts.std()),
                "min_total_heads": float(counts.min()),
                "max_total_heads": float(counts.max()),
                "unique_masks": int(unique),
                "unique_mask_fraction": float(unique / N),
            }
        )

        for h in range(H):
            head_rows.append(
                {
                    "layer": li,
                    "head": h,
                    "is_permanent_slot": int(h < P),
                    "activation_frequency": float(arr[:, li, h].mean()),
                }
            )

    return layer_rows, head_rows



def learned_router_calibration(
    model,
    batches,
    dev,
    pass_easiness=True,
):
    """Collect per-example, per-layer learned-router counts/targets/errors."""
    rows = []

    with torch.no_grad():
        for cpu_batch in batches:
            batch = move_batch(cpu_batch, dev)
            with dev.autocast():
                _ = call_model(
                    model, batch, pass_easiness=pass_easiness
                )
            dev.mark_step()

            ids = cpu_batch["example_id"].tolist()
            easiness = cpu_batch["easiness_score"].tolist()

            for li, block in enumerate(model.model.blocks):
                router = block.mlt_vw_rtr
                actual = (
                    router.save_total_head_count.detach()
                    .to(torch.float32).cpu().tolist()
                )
                target = (
                    router.save_target_total_head_count.detach()
                    .to(torch.float32).cpu().tolist()
                )
                logits = (
                    router.save_router_logits.detach()
                    .to(torch.float32).cpu()
                )
                sig = torch.sigmoid(logits)

                for bi, eid in enumerate(ids):
                    rows.append(
                        {
                            "example_id": int(eid),
                            "layer": li,
                            "easiness_score": float(easiness[bi]),
                            "actual_total_heads": float(actual[bi]),
                            "target_total_heads": float(target[bi]),
                            "count_error": float(actual[bi] - target[bi]),
                            "mean_sigmoid": float(sig[bi].mean()),
                            "sigmoid_saturation_fraction": float(
                                ((sig[bi] < 0.05) | (sig[bi] > 0.95))
                                .float().mean()
                            ),
                            "mean_ste_derivative": float(
                                (sig[bi] * (1.0 - sig[bi])).mean()
                            ),
                        }
                    )

    return rows

# ---------------------------------------------------------------------------
# A_h / I_h common-coalition specialization
# ---------------------------------------------------------------------------

def common_coalition_specialization(
    model,
    examples,
    batches,
    dev,
    masks_by_id,
    variant,
    pass_easiness,
    layers,
    group_size,
    ablation_batch_size,
    seed,
):
    """
    For each elastic/non-permanent head h:
      A_h = examples where baseline policy selected h
      I_h = examples where baseline policy did not select h

    Evaluate BOTH groups under the SAME all-head coalition and remove h.

      utility = CE(all heads except h) - CE(all heads)

    Positive A-I gap means the policy tends to select h on examples where h has
    greater exact single-head utility under the common all-head context.

    For Random-32 this is a negative control: because A/I assignment is random
    wrt input, the expected gap is zero.
    """
    P = int(getattr(model.config, "num_permanent_heads", 0))
    H = int(model.config.num_attention_heads)
    rng = np.random.default_rng(seed)

    Override = (
        RandomAttentionMaskOverride
        if variant == "random"
        else LearnedRouterOverride
    )

    with Override(model, "dense"):
        dense_mean, dense_ce = evaluate_ce(
            model,
            batches,
            dev,
            pass_easiness=pass_easiness,
            return_per_example=True,
        )

    all_ids = sorted(masks_by_id)
    rows = []

    for li in layers:
        for h in range(P, H):
            A = [i for i in all_ids if masks_by_id[i][li, h] > 0.5]
            I = [i for i in all_ids if masks_by_id[i][li, h] <= 0.5]

            rng.shuffle(A)
            rng.shuffle(I)

            nA = min(group_size, len(A))
            nI = min(group_size, len(I))
            nA = (nA // ablation_batch_size) * ablation_batch_size
            nI = (nI // ablation_batch_size) * ablation_batch_size
            A = A[:nA]
            I = I[:nI]

            def eval_group(ids):
                if not ids:
                    return []
                bs = make_subbatches(examples, ids, ablation_batch_size)
                with Override(
                    model,
                    "dense_minus",
                    target_layer=li,
                    target_head=h,
                ):
                    _, vals = evaluate_ce(
                        model,
                        bs,
                        dev,
                        pass_easiness=pass_easiness,
                        return_per_example=True,
                    )
                return [vals[i] - dense_ce[i] for i in ids if i in vals]

            du_A = eval_group(A)
            du_I = eval_group(I)

            rows.append(
                {
                    "layer": li,
                    "head": h,
                    "activation_frequency": float(
                        np.mean([masks_by_id[i][li, h] for i in all_ids])
                    ),
                    "active_examples_used": len(du_A),
                    "inactive_examples_used": len(du_I),
                    "utility_active_mean": safe_mean(du_A),
                    "utility_active_sem": safe_sem(du_A),
                    "utility_inactive_mean": safe_mean(du_I),
                    "utility_inactive_sem": safe_sem(du_I),
                    "specialization_gap_A_minus_I": (
                        safe_mean(du_A) - safe_mean(du_I)
                        if du_A and du_I
                        else float("nan")
                    ),
                }
            )

            print(
                f"L{li:02d} h{h:02d} f="
                f"{rows[-1]['activation_frequency']:.3f} "
                f"A-I={rows[-1]['specialization_gap_A_minus_I']:+.5f}"
            )

    return dense_mean, rows


# ---------------------------------------------------------------------------
# CE mode helpers
# ---------------------------------------------------------------------------

def learned_ce_modes(
    model,
    batches,
    dev,
    pass_easiness,
    random_trials=3,
):
    rows = []

    routed = evaluate_ce(
        model, batches, dev, pass_easiness=pass_easiness
    )
    rows.append({"mode": "learned_routed", "ce": routed, "std": 0.0})

    with LearnedRouterOverride(model, "dense"):
        dense = evaluate_ce(
            model, batches, dev, pass_easiness=pass_easiness
        )
    rows.append({"mode": "forced_dense_all32", "ce": dense, "std": 0.0})

    with LearnedRouterOverride(model, "permanent"):
        perm = evaluate_ce(
            model, batches, dev, pass_easiness=pass_easiness
        )
    rows.append({"mode": "permanent_only", "ce": perm, "std": 0.0})

    vals = []
    for _ in range(int(random_trials)):
        with LearnedRouterOverride(model, "random_same_count"):
            vals.append(
                evaluate_ce(
                    model,
                    batches,
                    dev,
                    pass_easiness=pass_easiness,
                )
            )
    rows.append(
        {
            "mode": "random_same_count",
            "ce": safe_mean(vals),
            "std": safe_std(vals),
        }
    )

    return rows


def random32_ce_modes(
    model,
    batches,
    dev,
    random_trials=5,
):
    rows = []

    # Different random mask draw each trial.
    vals = []
    model.set_routing_mode("random")
    for _ in range(int(random_trials)):
        vals.append(
            evaluate_ce(model, batches, dev, pass_easiness=False)
        )
    rows.append(
        {
            "mode": "random_sparse",
            "ce": safe_mean(vals),
            "std": safe_std(vals),
        }
    )

    with RandomAttentionMaskOverride(model, "dense"):
        dense = evaluate_ce(model, batches, dev, pass_easiness=False)
    rows.append({"mode": "forced_dense_same_bank", "ce": dense, "std": 0.0})

    with RandomAttentionMaskOverride(model, "permanent"):
        perm = evaluate_ce(model, batches, dev, pass_easiness=False)
    rows.append({"mode": "permanent_only", "ce": perm, "std": 0.0})

    return rows


def zip_results(output_dir: Path, name: str):
    # ZIP MUST be outside output_dir to prevent recursive self-zipping.
    zip_base = output_dir.parent / name
    old = zip_base.with_suffix(".zip")
    if old.exists():
        old.unlink()
    return shutil.make_archive(
        str(zip_base), "zip", root_dir=output_dir
    )