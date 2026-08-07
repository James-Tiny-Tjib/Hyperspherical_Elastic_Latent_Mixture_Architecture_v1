##################################################
# HELM inference analysis harness
#
# Answers, automatically (no eyeballing number tables):
#   A. Quality        -- CE/ppl/acc/top5/F1, overall and split by difficulty
#   B. Head census    -- which heads are always-on / always-off / genuinely dynamic
#   C. Easiness test  -- THE hypothesis test: does head count track difficulty at
#                        inference, when the model never sees the easiness label?
#   D. Layer profile  -- depth gradient in head usage
#   E. Paths          -- recurring routing motifs, co-activation, route diversity
#   F. Polarization   -- sum(sigmoid) vs count(>0.5); catches the "ratio looks high
#                        but the budget is satisfied" artifact
#   G. Timing         -- dense vs flex vs gather vs forced-dense, per sequence length
#
# Outputs: results JSON + PNG figures + a console report with explicit verdicts.
# Runs on a routed model OR the no-router baseline (router sections auto-skip).
##################################################

import os
import re
import json
import time
import random
import argparse
import warnings
from typing import List, Optional, Dict, Any, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, f1_score

from bm_model_0 import HELMConfig, HELMForMaskedLM
from SpanMLMCollatorWithEasiness import SpanMLMCollatorWithEasiness


DEFAULT_MODEL_REPO_ID = "JamesResearch1216/v1-Architecture-phase05v1"
DEFAULT_DATA_REPO_ID = "JamesResearch1216/HELM-Easiness-Data-10B-Labeled-v6"
DEFAULT_TOKENIZER_PATH = "answerdotai/ModernBERT-base"
CURRICULUM_SUBSETS = {1024: "seq_1024", 2048: "seq_2048", 4096: "seq_4096"}
VALIDATION_SHARD_INDEX = {1024: 0, 2048: 1, 4096: 2}


# ============================================================
# 0. UTIL
# ============================================================

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _j(x):
    """Make numpy/torch scalars JSON-serializable."""
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, dict):
        return {str(k): _j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_j(v) for v in x]
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return x


# ============================================================
# 1. MODEL LOADING  (mirrors test_HELM_v1.load_helm_model)
# ============================================================

def _step_of(path):
    m = re.search(r"checkpoint-(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def load_model(checkpoint_path=None, step=None, repo_id=DEFAULT_MODEL_REPO_ID,
               hf_token=None, tokenizer_path=DEFAULT_TOKENIZER_PATH, device=None,
               config_overrides=None):
    import glob
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=hf_token)

    resolved = checkpoint_path
    if resolved is None:
        pattern = f"checkpoint-{step:06d}.pt" if step is not None else "checkpoint-*.pt"
        cands = glob.glob(os.path.join(os.getcwd(), pattern))
        if cands:
            cands.sort(key=_step_of)
            resolved = cands[-1]

    if resolved is None:
        if step is None:
            raise FileNotFoundError(
                "No local checkpoint found. Pass --checkpoint or --step so the "
                "exact weights being analyzed are unambiguous."
            )
        fn = f"checkpoint-{step:06d}.pt"
        print(f"Downloading {fn} from {repo_id} ...")
        resolved = hf_hub_download(repo_id=repo_id, filename=fn,
                                   repo_type="model", token=hf_token)

    config = HELMConfig(vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id)
    for k, v in (config_overrides or {}).items():
        setattr(config, k, v)

    model = HELMForMaskedLM(config)
    ckpt = torch.load(resolved, map_location="cpu", weights_only=False)
    state = ckpt["model_state"]
    state = {(k[len("module."):] if k.startswith("module.") else k): v
             for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    print(f"Loaded {resolved} onto {device}")
    return model, tokenizer, config, device, resolved


def has_router(model) -> bool:
    return hasattr(model.model.blocks[0], "mlt_vw_rtr")


# ============================================================
# 2. PER-SAMPLE ROUTING CAPTURE
# ============================================================
# get_telemetry() averages over the batch, which destroys the per-example
# information every path/motif question depends on. save_flat_mask is
# [b, 1, E] -- per example -- so read that directly instead.

def capture_routing(model) -> Optional[Dict[str, np.ndarray]]:
    """Returns {'mask': [B, L, E] binary, 'scores': [B, L, E] sigmoid} for the
    forward pass that just ran, or None for a no-router model."""
    if not has_router(model):
        return None
    masks, scores = [], []
    for blk in model.model.blocks:
        r = blk.mlt_vw_rtr
        masks.append(r.save_flat_mask.detach().float().cpu().squeeze(1))    # [B, E]
        scores.append(r.save_sigmoid_scores.detach().float().cpu().squeeze(1))
    return {
        "mask": torch.stack(masks, dim=1).numpy(),      # [B, L, E]
        "scores": torch.stack(scores, dim=1).numpy(),   # [B, L, E]
    }


class ForceDenseRouting:
    """Context manager: make every router emit an all-ones mask, so timing can be
    compared against a genuinely unrouted forward on identical weights."""

    def __init__(self, model):
        self.model = model
        self._orig = []

    def __enter__(self):
        if not has_router(self.model):
            return self
        for blk in self.model.model.blocks:
            r = blk.mlt_vw_rtr
            self._orig.append((r, r.forward))

            def patched(hidden_states, step_tensor, easiness_score=None, _r=r):
                b = hidden_states.size(0)
                H = _r.config.num_attention_heads
                m = torch.ones(b, H, 1, 1, device=hidden_states.device,
                               dtype=hidden_states.dtype)
                _r.aux_loss = torch.zeros((), device=hidden_states.device)
                _r.sparsity_loss = torch.zeros((), device=hidden_states.device)
                return m

            r.forward = patched
        return self

    def __exit__(self, *a):
        for r, fn in self._orig:
            r.forward = fn
        self._orig = []
        return False


# ============================================================
# 3. DATA
# ============================================================

def _is_valid_parquet(path):
    import pyarrow.parquet as pq
    try:
        pq.read_metadata(path)
        return True
    except Exception:
        return False


def load_validation_rows(seq_len, n, repo_id=DEFAULT_DATA_REPO_ID, hf_token=None,
                         seed=67, local_dir="./local_parquet_shards",
                         balance_easiness=True):
    import pyarrow.parquet as pq
    subset = CURRICULUM_SUBSETS[seq_len]
    idx = VALIDATION_SHARD_INDEX.get(seq_len, 0)
    repo_filename = f"data/{subset}/validation-{idx:05d}.parquet"
    local_path = os.path.join(local_dir, repo_filename)
    if os.path.exists(local_path) and not _is_valid_parquet(local_path):
        os.remove(local_path)
    if not os.path.exists(local_path):
        os.makedirs(local_dir, exist_ok=True)
        local_path = hf_hub_download(repo_id=repo_id, filename=repo_filename,
                                     repo_type="dataset", token=hf_token,
                                     local_dir=local_dir)
    rows = pq.read_table(local_path).to_pylist()

    if balance_easiness:
        # Deliberately span the difficulty range -- a random sample of a
        # right-skewed easiness distribution barely covers the easy tail, which
        # is exactly the region the correlation test needs.
        rows.sort(key=lambda r: r["easiness_score"])
        idx = np.linspace(0, len(rows) - 1, min(n, len(rows))).astype(int)
        return [rows[i] for i in idx]
    return random.Random(seed).sample(rows, min(n, len(rows)))


# ============================================================
# 4. EVALUATION LOOP
# ============================================================

def chunked_ce(logits, labels, vocab_size, n_chunks=8):
    lf = nn.CrossEntropyLoss(reduction="sum")
    fl = logits.reshape(-1, vocab_size)
    fb = labels.reshape(-1)
    total = fl.size(0)
    chunk = (total + n_chunks - 1) // n_chunks
    valid = (fb != -100).sum().clamp_min(1)
    s = fl.new_zeros(())
    for i in range(0, total, chunk):
        s = s + lf(fl[i:i + chunk].float(), fb[i:i + chunk])
    return s / valid


def mlm_metrics(logits, labels, loss):
    mask = labels != -100
    n = int(mask.sum().item())
    if n == 0:
        return {"accuracy": float("nan"), "top5_accuracy": float("nan"),
                "macro_f1": float("nan"), "perplexity": float("nan"),
                "num_masked_tokens": 0}
    ml = logits[mask]
    mt = labels[mask]
    preds = ml.argmax(-1).cpu().numpy()
    gold = mt.cpu().numpy()
    k = min(5, ml.size(-1))
    top5 = float((torch.topk(ml, k=k, dim=-1).indices == mt.unsqueeze(-1))
                 .any(-1).float().mean().item())
    return {
        "accuracy": float(accuracy_score(gold, preds)),
        "top5_accuracy": top5,
        "macro_f1": float(f1_score(gold, preds, average="macro", zero_division=0)),
        "perplexity": float(torch.exp(loss.detach()).item()),
        "num_masked_tokens": n,
    }


@torch.no_grad()
def evaluate(model, collator, rows, device, vocab_size, batch_size=4, backend="dense"):
    """Run the eval set, collecting per-sample metrics AND per-sample routing."""
    if has_router(model):
        model.enable_efficient_inference(backend=backend, compile=False)

    all_mask, all_scores, all_easiness = [], [], []
    per_sample = []

    for start in range(0, len(rows), batch_size):
        chunk = rows[start:start + batch_size]
        batch = collator(chunk)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attn = torch.ones_like(input_ids, dtype=torch.long, device=device)

        logits, _, _ = model(input_ids=input_ids, attention_mask=attn)

        routing = capture_routing(model)
        if routing is not None:
            all_mask.append(routing["mask"])
            all_scores.append(routing["scores"])

        # per-sample metrics so quality can be stratified by difficulty
        for bi in range(input_ids.size(0)):
            lg = logits[bi:bi + 1]
            lb = labels[bi:bi + 1]
            if (lb != -100).sum() == 0:
                continue
            ce = chunked_ce(lg, lb, vocab_size)
            m = mlm_metrics(lg, lb, ce)
            m["ce_loss"] = float(ce.item())
            m["easiness"] = float(chunk[bi]["easiness_score"])
            if routing is not None:
                m["active_elastic"] = float(routing["mask"][bi].sum())
                m["elastic_ratio"] = float(routing["mask"][bi].mean())
            per_sample.append(m)
            all_easiness.append(m["easiness"])

    out = {"per_sample": per_sample, "easiness": np.array(all_easiness)}
    if all_mask:
        out["mask"] = np.concatenate(all_mask, axis=0)      # [N, L, E]
        out["scores"] = np.concatenate(all_scores, axis=0)  # [N, L, E]
    if has_router(model):
        model.enable_efficient_inference(backend="dense")
    return out


# ============================================================
# 5. ANALYSES
# ============================================================

def analyze_quality(per_sample) -> Dict[str, Any]:
    ez = np.array([p["easiness"] for p in per_sample])
    keys = ["ce_loss", "accuracy", "top5_accuracy", "macro_f1", "perplexity"]
    res = {"overall": {k: float(np.nanmean([p[k] for p in per_sample])) for k in keys},
           "n_samples": len(per_sample)}
    # difficulty tertiles: "hard" = low easiness
    q1, q2 = np.quantile(ez, [1 / 3, 2 / 3])
    bins = {"hard": ez <= q1, "medium": (ez > q1) & (ez <= q2), "easy": ez > q2}
    res["by_difficulty"] = {}
    for name, sel in bins.items():
        sub = [p for p, s in zip(per_sample, sel) if s]
        if not sub:
            continue
        res["by_difficulty"][name] = {
            "n": len(sub),
            "easiness_range": [float(min(p["easiness"] for p in sub)),
                               float(max(p["easiness"] for p in sub))],
            **{k: float(np.nanmean([p[k] for p in sub])) for k in keys},
        }
    res["tertile_bounds"] = [float(q1), float(q2)]
    return res


def analyze_head_census(mask, always_on=0.95, always_off=0.05) -> Dict[str, Any]:
    """Which heads are structurally on/off vs genuinely input-dependent."""
    N, L, E = mask.shape
    freq = mask.mean(axis=0)                 # [L, E] activation frequency
    cls = np.full((L, E), "dynamic", dtype=object)
    cls[freq >= always_on] = "always_on"
    cls[freq <= always_off] = "always_off"

    counts = {c: int((cls == c).sum()) for c in ["always_on", "always_off", "dynamic"]}
    # A head is only doing routing work if its activation actually varies.
    dynamic_frac = counts["dynamic"] / (L * E)
    return {
        "freq_matrix": freq,                                  # kept as array for plots
        "counts": counts,
        "dynamic_fraction": float(dynamic_frac),
        "per_layer_dynamic": [int((cls[l] == "dynamic").sum()) for l in range(L)],
        "per_layer_always_off": [int((cls[l] == "always_off").sum()) for l in range(L)],
        "per_layer_always_on": [int((cls[l] == "always_on").sum()) for l in range(L)],
        "mean_freq": float(freq.mean()),
        "std_freq": float(freq.std()),
    }


def analyze_easiness_response(mask, easiness) -> Dict[str, Any]:
    """THE hypothesis test.

    The model is never given easiness_score at inference (_easiness_to_target only
    runs under self.training). So any correlation here is the router inferring
    difficulty from content alone -- which is the entire claim of the architecture.
    """
    from scipy.stats import spearmanr, kruskal
    N, L, E = mask.shape
    total = mask.reshape(N, -1).sum(axis=1)     # active elastic heads per sample

    rho, p = spearmanr(easiness, total)
    per_layer = []
    for l in range(L):
        r_l, p_l = spearmanr(easiness, mask[:, l, :].sum(axis=1))
        per_layer.append({"layer": l, "rho": float(r_l), "p": float(p_l)})

    q1, q2 = np.quantile(easiness, [1 / 3, 2 / 3])
    g_hard = total[easiness <= q1]
    g_med = total[(easiness > q1) & (easiness <= q2)]
    g_easy = total[easiness > q2]
    try:
        kw_h, kw_p = kruskal(g_hard, g_med, g_easy)
    except Exception:
        kw_h, kw_p = float("nan"), float("nan")

    # Expected sign: harder text (LOW easiness) should use MORE heads -> negative rho.
    direction = "correct" if rho < 0 else ("inverted" if rho > 0 else "none")
    significant = bool(p < 0.05)
    return {
        "spearman_rho": float(rho), "spearman_p": float(p),
        "direction": direction, "significant": significant,
        "mean_heads_hard": float(g_hard.mean()) if len(g_hard) else None,
        "mean_heads_medium": float(g_med.mean()) if len(g_med) else None,
        "mean_heads_easy": float(g_easy.mean()) if len(g_easy) else None,
        "spread_hard_minus_easy": (float(g_hard.mean() - g_easy.mean())
                                   if len(g_hard) and len(g_easy) else None),
        "kruskal_H": float(kw_h), "kruskal_p": float(kw_p),
        "per_layer": per_layer,
        "total_heads_mean": float(total.mean()),
        "total_heads_std": float(total.std()),
        "total_heads_min": float(total.min()),
        "total_heads_max": float(total.max()),
    }


def analyze_layer_profile(mask) -> Dict[str, Any]:
    from scipy.stats import spearmanr
    N, L, E = mask.shape
    per_layer_ratio = mask.mean(axis=(0, 2))          # [L]
    per_layer_std = mask.sum(axis=2).std(axis=0)      # variability across samples
    rho, p = spearmanr(np.arange(L), per_layer_ratio)
    return {
        "per_layer_ratio": per_layer_ratio.tolist(),
        "per_layer_sample_std": per_layer_std.tolist(),
        "depth_gradient_rho": float(rho), "depth_gradient_p": float(p),
        "shallowest_ratio": float(per_layer_ratio[0]),
        "deepest_ratio": float(per_layer_ratio[-1]),
    }


def _kmeans_numpy(X, k, n_init=10, max_iter=100, seed=0):
    """Minimal k-means (numpy only). Avoids sklearn.cluster, which pulls in
    sklearn.neighbors -- a compiled submodule that's blocked/broken on some
    Windows Application-Control setups even when sklearn.metrics works fine."""
    rng = np.random.default_rng(seed)
    best_labels, best_inertia = None, np.inf
    n = X.shape[0]
    for _ in range(n_init):
        centers = X[rng.choice(n, k, replace=False)].copy()
        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # [n, k]
            new_labels = d.argmin(axis=1)
            if np.array_equal(new_labels, labels) and _ > 0:
                labels = new_labels
                break
            labels = new_labels
            for c in range(k):
                pts = X[labels == c]
                if len(pts):
                    centers[c] = pts.mean(axis=0)
        inertia = ((X - centers[labels]) ** 2).sum()
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels
    return best_labels


def analyze_paths(mask, easiness, n_clusters=6, seed=0) -> Dict[str, Any]:
    """Do recurring routing motifs exist, and do they track difficulty?

    Route diversity also decides whether batched inference could ever be
    efficient: heterogeneous per-sequence subsets can't be grouped into
    same-shaped kernel work, so low diversity is a prerequisite, not a curiosity.
    """
    from scipy.stats import kruskal
    N, L, E = mask.shape
    flat = mask.reshape(N, -1)

    # exact-signature diversity
    sigs = {}
    for row in flat:
        key = row.astype(np.uint8).tobytes()
        sigs[key] = sigs.get(key, 0) + 1
    unique_full = len(sigs)
    top_sig_share = max(sigs.values()) / N

    # per-layer diversity -- the number that matters for grouped kernels
    per_layer_unique = []
    for l in range(L):
        s = {row.astype(np.uint8).tobytes() for row in mask[:, l, :]}
        per_layer_unique.append(len(s))

    out = {
        "unique_full_routes": unique_full,
        "unique_full_routes_frac": float(unique_full / N),
        "most_common_route_share": float(top_sig_share),
        "per_layer_unique_head_sets": per_layer_unique,
        "mean_per_layer_unique": float(np.mean(per_layer_unique)),
        "n_samples": int(N),
    }

    # pairwise Jaccard between routes (sampled, to stay cheap)
    rng = np.random.default_rng(seed)
    pairs = min(2000, N * (N - 1) // 2)
    js = []
    for _ in range(pairs):
        i, k = rng.integers(0, N, 2)
        if i == k:
            continue
        a, b = flat[i] > 0, flat[k] > 0
        union = (a | b).sum()
        js.append(((a & b).sum() / union) if union else 1.0)
    out["mean_pairwise_jaccard"] = float(np.mean(js)) if js else None

    # motif clustering
    k = min(n_clusters, max(2, N // 5))
    if N >= 2 * k and flat.std() > 0:
        lab = _kmeans_numpy(flat, k, n_init=10, seed=seed)
        groups = [easiness[lab == c] for c in range(k) if (lab == c).sum() > 0]
        try:
            H, p = kruskal(*groups) if len(groups) > 1 else (float("nan"), float("nan"))
        except Exception:
            H, p = float("nan"), float("nan")
        out["motifs"] = {
            "n_clusters": int(k),
            "cluster_sizes": [int((lab == c).sum()) for c in range(k)],
            "cluster_mean_easiness": [float(easiness[lab == c].mean())
                                      if (lab == c).sum() else None for c in range(k)],
            "cluster_mean_heads": [float(flat[lab == c].sum(axis=1).mean())
                                   if (lab == c).sum() else None for c in range(k)],
            "easiness_separation_H": float(H),
            "easiness_separation_p": float(p),
            "clusters_track_difficulty": bool(p < 0.05) if p == p else False,
        }

    # co-activation: strongest head pairs per layer
    coact = []
    for l in range(L):
        m = mask[:, l, :]
        C = (m.T @ m) / max(1, N)          # [E, E] P(i and j both on)
        np.fill_diagonal(C, 0.0)
        if C.size and C.max() > 0:
            i, jx = np.unravel_index(np.argmax(C), C.shape)
            coact.append({"layer": l, "top_pair": [int(i), int(jx)],
                          "p_joint": float(C.max())})
    out["top_coactivations"] = coact
    return out


def analyze_polarization(scores, mask) -> Dict[str, Any]:
    """sum(sigmoid) vs count(>0.5).

    The sparsity penalty constrains the SUM of sigmoid values; elastic_head_ratio
    reports the COUNT above 0.5. Those agree only when scores are polarized. If
    scores bunch near 0.5, the reported ratio can be near 1.0 while the budget is
    perfectly satisfied -- i.e. the model is soft-attenuating everything rather
    than routing. This section detects that directly.
    """
    N, L, E = scores.shape
    undecided = float(((scores > 0.4) & (scores < 0.6)).mean())
    soft_sum = scores.sum(axis=2)      # [N, L]
    hard_cnt = mask.sum(axis=2)        # [N, L]
    gap = (hard_cnt - soft_sum)
    return {
        "mean_sigmoid": float(scores.mean()),
        "std_sigmoid": float(scores.std()),
        "frac_in_undecided_band_0.4_0.6": undecided,
        "frac_below_0.1": float((scores < 0.1).mean()),
        "frac_above_0.9": float((scores > 0.9).mean()),
        "mean_soft_sum_per_layer": soft_sum.mean(axis=0).tolist(),
        "mean_hard_count_per_layer": hard_cnt.mean(axis=0).tolist(),
        "mean_count_minus_sum": float(gap.mean()),
        "per_layer_count_minus_sum": gap.mean(axis=0).tolist(),
        # near 1.0 = crisply polarized (count == sum); large gap = soft gating
        "polarization_index": float(1.0 - min(1.0, abs(gap.mean()) / max(1e-6, E * 0.5))),
    }


# ============================================================
# 6. TIMING
# ============================================================

@torch.no_grad()
def benchmark(model, collator, rows, device, seq_len, batch_sizes=(1, 4),
              backends=("dense", "flex"), reps=20, warmup=5) -> Dict[str, Any]:
    results = {}
    routed = has_router(model)

    def timed(bs, tag, ctx=None):
        chunk = rows[:bs]
        if len(chunk) < bs:
            chunk = (rows * ((bs // len(rows)) + 1))[:bs]
        batch = collator(chunk)
        ids = batch["input_ids"].to(device)
        attn = torch.ones_like(ids, dtype=torch.long, device=device)
        mgr = ctx if ctx is not None else _NullCtx()
        with mgr:
            for _ in range(warmup):
                model(input_ids=ids, attention_mask=attn)
            _sync(device)
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                model(input_ids=ids, attention_mask=attn)
                _sync(device)
                ts.append((time.perf_counter() - t0) * 1000.0)
        ts = np.array(ts)
        results[tag] = {"mean_ms": float(ts.mean()), "std_ms": float(ts.std()),
                        "median_ms": float(np.median(ts)), "batch_size": bs,
                        "seq_len": seq_len}
        print(f"    {tag:<34} {ts.mean():8.2f} ms  (+/- {ts.std():.2f})")

    for bs in batch_sizes:
        for be in backends:
            if routed:
                try:
                    model.enable_efficient_inference(backend=be, compile=False)
                except Exception as e:
                    print(f"    backend {be} unavailable: {e}")
                    continue
            elif be != "dense":
                continue
            timed(bs, f"bs{bs}/{be}")
        # forced-dense on identical weights = the true "no routing" reference
        if routed:
            model.enable_efficient_inference(backend="dense")
            timed(bs, f"bs{bs}/forced_all_heads", ctx=ForceDenseRouting(model))

    if routed:
        model.enable_efficient_inference(backend="dense")
    return results


class _NullCtx:
    def __enter__(self): return self
    def __exit__(self, *a): return False


# ============================================================
# 7. PLOTS
# ============================================================

def make_plots(out_dir, seq_len, head_census, layer_profile, easiness_resp,
               per_sample, polarization, mask):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = f"seq{seq_len}"

    # 1. activation heatmap (layer x head)
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(head_census["freq_matrix"], aspect="auto", cmap="viridis",
                   vmin=0, vmax=1)
    fig.colorbar(im, label="Activation frequency")
    ax.set_xlabel("Elastic head index"); ax.set_ylabel("Layer")
    ax.set_title(f"Head activation frequency ({tag}, N={mask.shape[0]})")
    fig.tight_layout(); fig.savefig(f"{out_dir}/heatmap_{tag}.png", dpi=130); plt.close(fig)

    # 2. easiness vs heads used -- the hypothesis test, visually
    ez = np.array([p["easiness"] for p in per_sample])
    hu = np.array([p.get("active_elastic", np.nan) for p in per_sample])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(ez, hu, alpha=0.6, s=22)
    if len(ez) > 2 and np.isfinite(hu).all():
        z = np.polyfit(ez, hu, 1)
        xs = np.linspace(ez.min(), ez.max(), 50)
        ax.plot(xs, np.poly1d(z)(xs), "r--",
                label=f"rho={easiness_resp['spearman_rho']:.3f}, "
                      f"p={easiness_resp['spearman_p']:.2g}")
        ax.legend()
    ax.set_xlabel("Easiness score (model never sees this)")
    ax.set_ylabel("Active elastic heads")
    ax.set_title(f"Difficulty response ({tag})")
    fig.tight_layout(); fig.savefig(f"{out_dir}/easiness_{tag}.png", dpi=130); plt.close(fig)

    # 3. layer profile
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layer_profile["per_layer_ratio"], "o-")
    ax.set_xlabel("Layer"); ax.set_ylabel("Mean elastic ratio")
    ax.set_title(f"Depth profile ({tag}, rho={layer_profile['depth_gradient_rho']:.2f})")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out_dir}/layers_{tag}.png", dpi=130); plt.close(fig)

    # 4. polarization histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(polarization["_scores_flat"], bins=60)
    ax.axvline(0.5, color="r", ls="--", label="threshold")
    ax.set_xlabel("Sigmoid score"); ax.set_ylabel("Count")
    ax.set_title(f"Score polarization ({tag}) -- "
                 f"{polarization['frac_in_undecided_band_0.4_0.6']*100:.1f}% undecided")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{out_dir}/polarization_{tag}.png", dpi=130); plt.close(fig)


# ============================================================
# 8. REPORT
# ============================================================

def print_report(seq_len, quality, census, ez_resp, layers, paths, polar, timing):
    W = 78
    print("\n" + "=" * W)
    print(f"  ANALYSIS REPORT -- sequence length {seq_len}")
    print("=" * W)

    print("\n[A] QUALITY")
    o = quality["overall"]
    print(f"  N={quality['n_samples']}  CE={o['ce_loss']:.4f}  ppl={o['perplexity']:.2f}  "
          f"acc={o['accuracy']:.4f}  top5={o['top5_accuracy']:.4f}  F1={o['macro_f1']:.4f}")
    for name in ["hard", "medium", "easy"]:
        d = quality["by_difficulty"].get(name)
        if d:
            print(f"    {name:<7} (n={d['n']:>3}, ez {d['easiness_range'][0]:.2f}-"
                  f"{d['easiness_range'][1]:.2f}): CE={d['ce_loss']:.4f}  acc={d['accuracy']:.4f}")

    if ez_resp is None:
        print("\n  [no router in this checkpoint -- routing sections skipped]")
    else:
        print("\n[B] HEAD CENSUS")
        c = census["counts"]
        print(f"  always-on={c['always_on']}  always-off={c['always_off']}  "
              f"dynamic={c['dynamic']}  ({census['dynamic_fraction']*100:.1f}% dynamic)")
        print(f"  per-layer always-off: {census['per_layer_always_off']}")
        if census["dynamic_fraction"] < 0.10:
            print("  >> VERDICT: routing is near-static. Heads are structurally on/off,")
            print("     not input-dependent -- this is a learned pruning, not routing.")
        elif census["dynamic_fraction"] > 0.60:
            print("  >> VERDICT: most heads are genuinely input-dependent.")

        print("\n[C] DIFFICULTY RESPONSE  (model never sees easiness at inference)")
        print(f"  Spearman rho={ez_resp['spearman_rho']:+.4f}  p={ez_resp['spearman_p']:.2e}")
        print(f"  heads used -- hard={ez_resp['mean_heads_hard']:.2f}  "
              f"medium={ez_resp['mean_heads_medium']:.2f}  easy={ez_resp['mean_heads_easy']:.2f}")
        print(f"  hard-minus-easy spread = {ez_resp['spread_hard_minus_easy']:+.2f} heads")
        print(f"  Kruskal-Wallis H={ez_resp['kruskal_H']:.2f}  p={ez_resp['kruskal_p']:.2e}")
        if ez_resp["significant"] and ez_resp["direction"] == "correct":
            print("  >> VERDICT: PASS. The router infers difficulty from content alone")
            print("     and allocates more heads to harder text. This is the core claim.")
        elif ez_resp["significant"] and ez_resp["direction"] == "inverted":
            print("  >> VERDICT: INVERTED. Significant, but MORE heads on EASIER text.")
        else:
            print("  >> VERDICT: FAIL. No significant difficulty response at inference.")

        print("\n[D] DEPTH PROFILE")
        print(f"  ratio by layer: {[f'{r:.2f}' for r in layers['per_layer_ratio']]}")
        print(f"  depth gradient rho={layers['depth_gradient_rho']:+.3f} "
              f"(p={layers['depth_gradient_p']:.2g})  "
              f"L0={layers['shallowest_ratio']:.2f} -> L{len(layers['per_layer_ratio'])-1}="
              f"{layers['deepest_ratio']:.2f}")

        print("\n[E] ROUTE STRUCTURE")
        print(f"  unique full routes: {paths['unique_full_routes']}/{paths['n_samples']} "
              f"({paths['unique_full_routes_frac']*100:.0f}%)  "
              f"most common route = {paths['most_common_route_share']*100:.0f}% of samples")
        print(f"  unique head-sets per layer: {paths['per_layer_unique_head_sets']}")
        print(f"  mean pairwise Jaccard = {paths['mean_pairwise_jaccard']:.3f}")
        if "motifs" in paths:
            m = paths["motifs"]
            print(f"  motifs: sizes={m['cluster_sizes']}  "
                  f"mean easiness={[f'{x:.2f}' for x in m['cluster_mean_easiness']]}")
            print(f"  motif<->difficulty association p={m['easiness_separation_p']:.2e} "
                  f"-> {'TRACKS difficulty' if m['clusters_track_difficulty'] else 'no association'}")
        if paths["unique_full_routes_frac"] > 0.9:
            print("  >> NOTE: routes are almost all distinct. Batched inference cannot")
            print("     group them into same-shaped kernel work (no clustering to exploit).")

        print("\n[F] POLARIZATION  (sum-of-sigmoids vs count-above-0.5)")
        print(f"  mean sigmoid={polar['mean_sigmoid']:.3f}  "
              f"undecided(0.4-0.6)={polar['frac_in_undecided_band_0.4_0.6']*100:.1f}%  "
              f"<0.1={polar['frac_below_0.1']*100:.1f}%  >0.9={polar['frac_above_0.9']*100:.1f}%")
        print(f"  mean (hard count - soft sum) = {polar['mean_count_minus_sum']:+.2f} heads/layer")
        if polar["frac_in_undecided_band_0.4_0.6"] > 0.35:
            print("  >> VERDICT: scores are NOT polarized. The reported head ratio")
            print("     overstates real routing -- this is soft attenuation of most heads,")
            print("     and the sparsity budget can be satisfied while the ratio looks high.")
        else:
            print("  >> VERDICT: scores are polarized; ratio and budget agree.")

    if timing:
        print("\n[G] TIMING")
        for k, v in timing.items():
            print(f"  {k:<34} {v['mean_ms']:8.2f} ms  (+/- {v['std_ms']:.2f})")
        base = {k: v for k, v in timing.items() if "forced_all_heads" in k}
        for bk, bv in base.items():
            bs = bv["batch_size"]
            for k, v in timing.items():
                if f"bs{bs}/" in k and "forced" not in k:
                    sp = bv["mean_ms"] / v["mean_ms"]
                    verdict = "FASTER" if sp > 1.03 else ("slower" if sp < 0.97 else "no change")
                    print(f"  routing vs all-heads @ {k:<20} {sp:.3f}x  ({verdict})")
    print("=" * W)


# ============================================================
# 9. MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="HELM inference analysis")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--repo-id", type=str, default=DEFAULT_MODEL_REPO_ID)
    ap.add_argument("--data-repo-id", type=str, default=DEFAULT_DATA_REPO_ID)
    ap.add_argument("--hf-token", type=str, default=os.getenv("HF_TOKEN"))
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[1024, 2048, 4096])
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--backends", type=str, nargs="+", default=["dense", "flex"])
    ap.add_argument("--timing-batch-sizes", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--out-dir", type=str, default="./helm_analysis")
    ap.add_argument("--label", type=str, default="model",
                    help="Tag for this run, e.g. 'routed' or 'noroute'.")
    ap.add_argument("--seed", type=int, default=67)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    model, tokenizer, config, device, ckpt_path = load_model(
        checkpoint_path=args.checkpoint, step=args.step, repo_id=args.repo_id,
        hf_token=args.hf_token)
    routed = has_router(model)
    print(f"Router present: {routed}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"capability {torch.cuda.get_device_capability(0)}")

    collator = SpanMLMCollatorWithEasiness(tokenizer=tokenizer)

    report = {
        "label": args.label,
        "checkpoint": os.path.basename(ckpt_path),
        "routed": routed,
        "config": {
            "num_attention_heads": config.num_attention_heads,
            "num_permanent_heads": getattr(config, "num_permanent_heads", None),
            "num_hidden_layers": config.num_hidden_layers,
            "hidden_size": config.hidden_size,
            "num_router_latents": getattr(config, "num_router_latents", None),
            "head_target_min": getattr(config, "head_target_min", None),
            "head_target_center": getattr(config, "head_target_center", None),
            "head_target_max": getattr(config, "head_target_max", None),
            "use_sigmoid_scaling": getattr(config, "use_sigmoid_scaling", None),
            "use_exclusive_attention": getattr(config, "use_exclusive_attention", None),
        },
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "num_samples": args.num_samples,
        "by_seq_len": {},
    }

    for seq_len in args.seq_lens:
        print(f"\n{'#'*70}\n# Sequence length {seq_len}\n{'#'*70}")
        try:
            rows = load_validation_rows(seq_len, args.num_samples,
                                        repo_id=args.data_repo_id,
                                        hf_token=args.hf_token, seed=args.seed)
        except Exception as e:
            print(f"  could not load validation data for {seq_len}: {e}")
            continue
        print(f"  loaded {len(rows)} validation rows "
              f"(easiness {min(r['easiness_score'] for r in rows):.3f} - "
              f"{max(r['easiness_score'] for r in rows):.3f})")

        print("  running evaluation ...")
        ev = evaluate(model, collator, rows, device, config.vocab_size,
                      batch_size=args.batch_size, backend="dense")

        quality = analyze_quality(ev["per_sample"])
        census = ez_resp = layers = paths = polar = None

        if "mask" in ev:
            mask, scores, ez = ev["mask"], ev["scores"], ev["easiness"]
            census = analyze_head_census(mask)
            ez_resp = analyze_easiness_response(mask, ez)
            layers = analyze_layer_profile(mask)
            paths = analyze_paths(mask, ez, seed=args.seed)
            polar = analyze_polarization(scores, mask)
            polar["_scores_flat"] = scores.reshape(-1)
            try:
                make_plots(args.out_dir, seq_len, census, layers, ez_resp,
                           ev["per_sample"], polar, mask)
                print(f"  figures -> {args.out_dir}/*_seq{seq_len}.png")
            except Exception as e:
                print(f"  plotting failed: {e}")
            polar.pop("_scores_flat", None)
            census.pop("freq_matrix", None)   # too big for JSON; it's in the PNG

        timing = None
        if not args.skip_timing:
            print("  timing ...")
            timing = benchmark(model, collator, rows, device, seq_len,
                               batch_sizes=args.timing_batch_sizes,
                               backends=args.backends, reps=args.reps)

        print_report(seq_len, quality, census, ez_resp, layers, paths, polar, timing)

        report["by_seq_len"][str(seq_len)] = _j({
            "quality": quality, "head_census": census,
            "easiness_response": ez_resp, "layer_profile": layers,
            "paths": paths, "polarization": polar, "timing": timing,
        })

    path = os.path.join(args.out_dir, f"results_{args.label}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {path}")
    print("Send me that JSON (and the PNGs) and I'll interpret it.")


if __name__ == "__main__":
    main()