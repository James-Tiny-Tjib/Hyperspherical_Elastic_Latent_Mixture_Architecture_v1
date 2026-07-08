##################################################
# Inference / benchmarking harness for HELM v1
#
# 1) load_helm_model()      -- rebuild the architecture from model.py and load a
#                               training checkpoint's weights into it (local dir
#                               first, then the HF Hub).
# 2) run_inference()        -- MLM inference on raw text (user-provided or scripted),
#                               auto-batches, times the forward pass, and reports
#                               router head usage.
# 3) run_validation_test()  -- pulls real validation sequences (1024/2048/4096) from
#                               the training data repo, masks them with the same
#                               SpanMLMCollatorWithEasiness used in training, and
#                               reports loss + sklearn MLM metrics.
##################################################

import os
import re
import glob
import json
import time
import random
import argparse
from typing import List, Optional, Union, Dict, Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, f1_score

from model import HELMConfig, HELMForMaskedLM
from SpanMLMCollatorWithEasiness import SpanMLMCollatorWithEasiness


# ============================================================
# Defaults (mirrors phase_05_model_tuning.py / prepare_data.py)
# ============================================================

DEFAULT_MODEL_REPO_ID = "JamesResearch1216/v1-Architecture-phase05v1"
DEFAULT_DATA_REPO_ID = "JamesResearch1216/HELM-Easiness-Data-10B-Labeled-v6"
DEFAULT_TOKENIZER_PATH = "answerdotai/ModernBERT-base"
CURRICULUM_SUBSETS = {1024: "seq_1024", 2048: "seq_2048", 4096: "seq_4096"}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _sync(device: torch.device):
    # Only CUDA needs an explicit sync for wall-clock timing to be meaningful.
    if device.type == "cuda":
        torch.cuda.synchronize()


# ============================================================
# 1. MODEL LOADING
# ============================================================

def _step_of(path: str) -> int:
    m = re.search(r"checkpoint-(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _find_local_checkpoint(step: Optional[int], search_dirs: Sequence[str]) -> Optional[str]:
    candidates = []
    for d in search_dirs:
        pattern = f"checkpoint-{step:06d}.pt" if step is not None else "checkpoint-*.pt"
        candidates.extend(glob.glob(os.path.join(d, pattern)))
    if not candidates:
        return None
    candidates.sort(key=_step_of)
    return candidates[-1]


def _valid_steps_from_hub(repo_id: str, hf_token: Optional[str]) -> List[int]:
    """
    Peek at training_state.json on the hub for non-deleted checkpoint steps, newest first.

    training_state.json's recorded `status` can go stale relative to what actually still
    exists in the repo (e.g. a partially-failed upload, manual cleanup); unlike
    CheckpointDriver.resume_training's fuller reconciliation, this doesn't re-check the
    repo's file listing up front -- instead, load_helm_model() falls back to the next
    step down if the "latest" one 404s on download.
    """
    try:
        path = hf_hub_download(
            repo_id=repo_id, filename="training_state.json", repo_type="model", token=hf_token
        )
        with open(path, "r") as f:
            state = json.load(f)
        valid_steps = [
            int(s) for s, v in state["checkpoints"].items() if v.get("status") != "deleted"
        ]
        return sorted(valid_steps, reverse=True)
    except Exception:
        return []


def load_helm_model(
    checkpoint_path: Optional[str] = None,
    step: Optional[int] = None,
    repo_id: str = DEFAULT_MODEL_REPO_ID,
    hf_token: Optional[str] = None,
    config: Optional[HELMConfig] = None,
    config_overrides: Optional[Dict[str, Any]] = None,
    tokenizer_path: str = DEFAULT_TOKENIZER_PATH,
    device: Optional[torch.device] = None,
    search_dirs: Optional[Sequence[str]] = None,
):
    """
    Rebuild HELMForMaskedLM from model.py and load a checkpoint's weights into it.

    Resolution order for the weights file:
      1. `checkpoint_path` if given explicitly.
      2. A `checkpoint-{step:06d}.pt` (or, if `step` is None, the highest-step one
         found) inside `search_dirs` (defaults to CWD + this file's directory).
      3. Downloaded from `repo_id` on the HF Hub, using `step` if given, else the
         latest step recorded in that repo's training_state.json.

    Config values that only matter for training-time loss terms (router warmup /
    annealing schedules, easiness breakpoints) aren't saved in the checkpoint, so
    they're left at HELMConfig()'s defaults -- they don't affect eval-mode forward
    passes. vocab_size/pad_token_id are re-derived from the tokenizer to guarantee
    the embedding/classifier shapes match the saved weights.
    """
    device = device or get_device()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=hf_token)

    resolved_path = checkpoint_path
    if resolved_path is None:
        dirs = list(search_dirs) if search_dirs is not None else [
            os.getcwd(), os.path.dirname(os.path.abspath(__file__))
        ]
        resolved_path = _find_local_checkpoint(step, dirs) if dirs else None

    if resolved_path is None:
        if step is not None:
            # An explicit step is a direct request -- fail loudly rather than silently
            # substituting a different checkpoint than the one asked for.
            filename = f"checkpoint-{step:06d}.pt"
            print(f"No local checkpoint found -- downloading {filename} from {repo_id}...")
            resolved_path = hf_hub_download(
                repo_id=repo_id, filename=filename, repo_type="model", token=hf_token
            )
        else:
            candidate_steps = _valid_steps_from_hub(repo_id, hf_token)
            if not candidate_steps:
                raise FileNotFoundError(
                    f"No local checkpoint found and couldn't resolve a step from "
                    f"{repo_id}/training_state.json. Pass `step=` or `checkpoint_path=` explicitly."
                )
            # training_state.json's recorded status can be stale (see _valid_steps_from_hub),
            # so fall back to the next-newest step if the "latest" one 404s on download.
            resolved_path = None
            last_error: Optional[Exception] = None
            for candidate_step in candidate_steps:
                filename = f"checkpoint-{candidate_step:06d}.pt"
                print(f"No local checkpoint found -- downloading {filename} from {repo_id}...")
                try:
                    resolved_path = hf_hub_download(
                        repo_id=repo_id, filename=filename, repo_type="model", token=hf_token
                    )
                    break
                except Exception as e:
                    print(f"  {filename} unavailable ({e}) -- trying the next-newest checkpoint.")
                    last_error = e
            if resolved_path is None:
                raise FileNotFoundError(
                    f"None of the checkpoints recorded in {repo_id}/training_state.json "
                    f"({candidate_steps}) could be downloaded."
                ) from last_error
    else:
        print(f"Using local checkpoint: {resolved_path}")

    if config is None:
        config = HELMConfig(vocab_size=len(tokenizer), pad_token_id=tokenizer.pad_token_id)
    if config_overrides:
        for k, v in config_overrides.items():
            setattr(config, k, v)

    model = HELMForMaskedLM(config)

    ckpt = torch.load(resolved_path, map_location="cpu", weights_only=False)
    model_state = ckpt["model_state"]

    # Checkpoints saved from a DDP-wrapped model carry a "module." prefix; strip it.
    new_state = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in model_state.items()
    }
    model.load_state_dict(new_state, strict=True)
    model.to(device)
    model.eval()

    print(f"Loaded HELM checkpoint from {resolved_path} onto {device}.")
    return model, tokenizer, config, device


# ============================================================
# 2. RAW INFERENCE: timing / auto-batching / head usage
# ============================================================

def _tokenize_batch(tokenizer, texts: List[str], device: torch.device, max_length: int):
    # Cap at the model's own RoPE table size (max_position_embeddings), not the tokenizer's
    # possibly-larger default -- ModernBERT-base's tokenizer allows up to 8192 tokens, which
    # would overrun HELM's default 4096-position RoPE buffer and crash on a shape mismatch.
    enc = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    return input_ids, attention_mask


def _head_usage_from_telemetry(unwrapped_model) -> Dict[str, Any]:
    telemetry = unwrapped_model.get_telemetry()
    cfg = unwrapped_model.config
    num_elastic = cfg.num_attention_heads - cfg.num_permanent_heads

    per_layer_active = []
    for i in range(cfg.num_hidden_layers):
        elastic_ratio = telemetry[f"layer_{i}_elastic_head_ratio"]
        per_layer_active.append(elastic_ratio * num_elastic + cfg.num_permanent_heads)

    total_active = float(sum(per_layer_active))
    total_possible = cfg.num_hidden_layers * cfg.num_attention_heads
    return {
        "per_layer_active_heads": per_layer_active,
        "total_active_heads": total_active,
        "total_possible_heads": total_possible,
        "usage_fraction": total_active / total_possible,
    }


def _print_head_usage(head_usage: Dict[str, Any]):
    print(
        f"  Head usage: {head_usage['total_active_heads']:.1f}/"
        f"{head_usage['total_possible_heads']} active "
        f"({head_usage['usage_fraction'] * 100:.1f}%)"
    )


@torch.no_grad()
def run_inference(
    model,
    tokenizer,
    inputs: Union[str, List[str]],
    device: Optional[torch.device] = None,
    top_k: int = 5,
    compile_backend: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run MLM inference on raw text, auto-batching if given a list of >1 items.

    Backend selection (see model.py's HELMSelfAttention eval paths):
      - single input  -> benchmarks "dense" (compute-all-heads baseline) against
                         "gather" (per-example active-head slicing) and reports
                         the speedup, since "gather" only special-cases batch==1.
      - batched input -> uses "flex" (batched FlexAttention over active heads).

    Also reports router head usage (active heads / total heads) and, for any
    [MASK] tokens present in the input, the top-k predicted fill-ins.
    """
    device = device or next(model.parameters()).device
    unwrapped = model.module if hasattr(model, "module") else model

    texts = [inputs] if isinstance(inputs, str) else list(inputs)
    if not texts:
        raise ValueError("run_inference() received no input text (empty string/list).")
    is_single = len(texts) == 1

    input_ids, attention_mask = _tokenize_batch(
        tokenizer, texts, device, max_length=unwrapped.config.max_position_embeddings
    )

    def _timed_forward(backend: str):
        unwrapped.enable_efficient_inference(backend=backend, compile=compile_backend)
        _sync(device)
        t0 = time.perf_counter()
        logits, _, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        _sync(device)
        return logits, time.perf_counter() - t0

    results: Dict[str, Any] = {}
    if is_single:
        _, dense_time = _timed_forward("dense")
        logits, gather_time = _timed_forward("gather")
        speedup = (dense_time - gather_time) / dense_time * 100 if dense_time > 0 else 0.0
        results.update(
            backend_used="gather",
            dense_time_sec=dense_time,
            gather_time_sec=gather_time,
            speedup_pct=speedup,
        )
        if verbose:
            print(f"[single] dense forward:  {dense_time * 1000:.2f} ms")
            print(f"[single] gather forward: {gather_time * 1000:.2f} ms  ({speedup:+.1f}% vs dense)")
    else:
        logits, batch_time = _timed_forward("flex")
        results.update(backend_used="flex", batch_time_sec=batch_time)
        if verbose:
            print(f"[batch={len(texts)}] flex-attention forward: {batch_time * 1000:.2f} ms")

    head_usage = _head_usage_from_telemetry(unwrapped)
    results["head_usage"] = head_usage
    if verbose:
        _print_head_usage(head_usage)

    mask_id = tokenizer.mask_token_id
    predictions = []
    for bi in range(input_ids.size(0)):
        positions = (input_ids[bi] == mask_id).nonzero(as_tuple=True)[0].tolist()
        row_preds = []
        for pos in positions:
            topk = torch.topk(logits[bi, pos].float(), k=top_k)
            row_preds.append({
                "position": pos,
                "top_k_tokens": tokenizer.convert_ids_to_tokens(topk.indices.tolist()),
                "top_k_scores": topk.values.tolist(),
            })
        predictions.append(row_preds)
        if verbose and row_preds:
            print(f"  Input {bi}: {texts[bi]!r}")
            for p in row_preds:
                print(f"    pos {p['position']}: {p['top_k_tokens']}")

    results["predictions"] = predictions
    unwrapped.enable_efficient_inference(backend="dense")  # leave model in the safe default
    return results


# ============================================================
# 3. VALIDATION-SET MLM EVALUATION
# ============================================================

def chunked_ce(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int, n_chunks: int = 8) -> torch.Tensor:
    """Same chunked cross-entropy used in the training/validation loop (avoids one giant softmax)."""
    loss_fct_sum = nn.CrossEntropyLoss(reduction="sum")
    flat_logits = logits.reshape(-1, vocab_size)
    flat_labels = labels.reshape(-1)
    total = flat_logits.size(0)
    chunk = (total + n_chunks - 1) // n_chunks
    valid = (flat_labels != -100).sum().clamp_min(1)
    loss_sum = flat_logits.new_zeros(())
    for i in range(0, total, chunk):
        loss_sum = loss_sum + loss_fct_sum(flat_logits[i:i + chunk].float(), flat_labels[i:i + chunk])
    return loss_sum / valid


def _mlm_metrics(logits: torch.Tensor, labels: torch.Tensor, loss: torch.Tensor) -> Dict[str, float]:
    """
    `loss` is the mean cross-entropy over these same masked tokens (already computed by
    the caller via chunked_ce) -- perplexity is derived from it instead of re-running a
    second cross-entropy pass, so the two never have a chance to silently disagree.
    """
    mask = labels != -100
    num_masked = int(mask.sum().item())
    if num_masked == 0:
        return {"accuracy": float("nan"), "top5_accuracy": float("nan"),
                "macro_f1": float("nan"), "perplexity": float("nan"), "num_masked_tokens": 0}

    masked_logits = logits[mask]
    masked_labels_t = labels[mask]
    masked_labels = masked_labels_t.cpu().numpy()
    preds = masked_logits.argmax(dim=-1).cpu().numpy()

    accuracy = accuracy_score(masked_labels, preds)
    macro_f1 = f1_score(masked_labels, preds, average="macro", zero_division=0)

    # Softmax is monotonic, so top-k membership can be read straight off raw logits --
    # no need to materialize a full-vocab softmax just to rank them.
    k = min(5, masked_logits.size(-1))
    top5_indices = torch.topk(masked_logits, k=k, dim=-1).indices
    top5 = float((top5_indices == masked_labels_t.unsqueeze(-1)).any(dim=-1).float().mean().item())

    perplexity = float(torch.exp(loss.detach()).item())

    return {
        "accuracy": accuracy,
        "top5_accuracy": top5,
        "macro_f1": macro_f1,
        "perplexity": perplexity,
        "num_masked_tokens": num_masked,
    }


def _is_valid_parquet(path: str) -> bool:
    import pyarrow.parquet as pq
    try:
        pq.read_metadata(path)
        return True
    except Exception:
        return False


def _download_validation_parquet(
    seq_len: int,
    index: int = 0,
    repo_id: str = DEFAULT_DATA_REPO_ID,
    hf_token: Optional[str] = None,
    local_dir: str = "./local_parquet_shards",
) -> str:
    subset = CURRICULUM_SUBSETS[seq_len]
    repo_filename = f"data/{subset}/validation-{index:05d}.parquet"
    local_path = os.path.join(local_dir, repo_filename)
    if os.path.exists(local_path):
        if _is_valid_parquet(local_path):
            return local_path
        print(f"Cached file at {local_path} is not a valid parquet file (partial download / "
              f"corrupted?) -- deleting and re-downloading.")
        os.remove(local_path)
    os.makedirs(local_dir, exist_ok=True)
    return hf_hub_download(
        repo_id=repo_id, filename=repo_filename, repo_type="dataset",
        token=hf_token, local_dir=local_dir,
    )


def _load_validation_pool(
    seq_len: int,
    pool_size: int,
    repo_id: str = DEFAULT_DATA_REPO_ID,
    hf_token: Optional[str] = None,
    seed: int = 67,
) -> List[Dict[str, Any]]:
    import pyarrow.parquet as pq

    path = _download_validation_parquet(seq_len, repo_id=repo_id, hf_token=hf_token)
    table = pq.read_table(path)
    n = table.num_rows
    pool_size = min(pool_size, n)
    idx = random.Random(seed).sample(range(n), pool_size)
    return table.take(idx).to_pylist()


def _select_examples(
    rows: List[Dict[str, Any]], num_examples: int, balance_easiness: bool, seed: int = 67
) -> List[Dict[str, Any]]:
    num_examples = min(num_examples, len(rows))
    if not balance_easiness:
        return random.Random(seed).sample(rows, num_examples)

    # Sort by easiness and pick evenly-spaced percentile points to guarantee a spread.
    ordered = sorted(rows, key=lambda r: r["easiness_score"])
    if num_examples == 1:
        return [ordered[len(ordered) // 2]]

    # num_examples <= len(ordered) is guaranteed above, so a free slot always exists --
    # search outward from each rounded position (instead of only forward) for the nearest
    # unused index, so a cluster of collisions near either end can never fall back to
    # silently picking the same row twice.
    positions = np.linspace(0, len(ordered) - 1, num_examples)
    picked, seen = [], set()
    for p in positions:
        center = int(round(p))
        idx = center
        offset = 0
        while idx in seen:
            offset += 1
            if center + offset < len(ordered):
                idx = center + offset
            elif center - offset >= 0:
                idx = center - offset
            else:
                raise AssertionError("ran out of unique indices despite num_examples <= len(ordered)")
        seen.add(idx)
        picked.append(ordered[idx])
    return picked


def run_validation_test(
    model,
    tokenizer,
    config: HELMConfig,
    seq_lengths: Sequence[int] = (1024, 2048, 4096),
    num_examples: int = 4,
    batch: bool = True,
    balance_easiness: bool = False,
    pool_size: int = 50,
    device: Optional[torch.device] = None,
    data_repo_id: str = DEFAULT_DATA_REPO_ID,
    hf_token: Optional[str] = None,
    mlm_probability: Optional[float] = None,
    mlm_use_span_masking: Optional[bool] = None,
    mlm_span_length: Optional[int] = None,
    seed: int = 67,
) -> Dict[int, Dict[str, Any]]:
    """
    For each of `seq_lengths`, pull `pool_size` random validation rows, select
    `num_examples` of them (spread across easiness scores if `balance_easiness`),
    mask them with SpanMLMCollatorWithEasiness, and run/timing/score the model
    (batched in one forward pass if `batch`, else one-at-a-time).
    """
    device = device or next(model.parameters()).device
    unwrapped = model.module if hasattr(model, "module") else model
    model.eval()

    collator = SpanMLMCollatorWithEasiness(
        tokenizer=tokenizer,
        mlm_probability=mlm_probability if mlm_probability is not None else config.mlm_probability,
        mlm_use_span_masking=mlm_use_span_masking if mlm_use_span_masking is not None else config.mlm_use_span_masking,
        mlm_span_length=mlm_span_length if mlm_span_length is not None else config.mlm_span_length,
    )

    def _run_group(group_examples: List[Dict[str, Any]], label: str, backend: str) -> Dict[str, Any]:
        collated = collator(group_examples)
        input_ids = collated["input_ids"].to(device)
        labels = collated["labels"].to(device)
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        unwrapped.enable_efficient_inference(backend=backend, compile=False)
        _sync(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, aux_loss, sparsity_loss = model(input_ids=input_ids, attention_mask=attention_mask)
        _sync(device)
        elapsed = time.perf_counter() - t0

        # aux_loss/sparsity_loss are always 0 in eval mode (see HELMMultiViewRouter.forward),
        # kept here only for parity with the training validation loop's loss composition.
        loss = chunked_ce(logits, labels, config.vocab_size) + aux_loss + sparsity_loss
        metrics = _mlm_metrics(logits, labels, loss)
        head_usage = _head_usage_from_telemetry(unwrapped)

        print(
            f"  [{label}] time: {elapsed * 1000:.2f} ms | loss: {loss.item():.4f} "
            f"| ppl: {metrics['perplexity']:.2f} | acc: {metrics['accuracy']:.4f} "
            f"| top5_acc: {metrics['top5_accuracy']:.4f} | macro_f1: {metrics['macro_f1']:.4f} "
            f"| masked_tokens: {metrics['num_masked_tokens']}"
        )
        _print_head_usage(head_usage)
        return {"elapsed_sec": elapsed, "loss": loss.item(), **metrics, "head_usage": head_usage}

    all_results: Dict[int, Dict[str, Any]] = {}
    for seq_len in seq_lengths:
        print(f"\n{'=' * 60}\nSequence Length: {seq_len}\n{'=' * 60}")
        rows = _load_validation_pool(seq_len, pool_size, repo_id=data_repo_id, hf_token=hf_token, seed=seed)
        examples = _select_examples(rows, num_examples, balance_easiness, seed=seed)

        if not examples:
            print(f"No examples available for seq_len={seq_len} (pool had {len(rows)} rows, "
                  f"requested {num_examples}) -- skipping.")
            all_results[seq_len] = {"easiness_scores": []}
            continue

        easiness_scores = [r["easiness_score"] for r in examples]
        print(
            f"Selected {len(examples)}/{len(rows)}-pool example(s). Easiness scores: "
            f"{[f'{e:.3f}' for e in easiness_scores]} "
            f"(range: {min(easiness_scores):.3f} - {max(easiness_scores):.3f})"
        )

        seq_result: Dict[str, Any] = {"easiness_scores": easiness_scores}
        if batch and len(examples) > 1:
            seq_result["batch"] = _run_group(examples, f"batch of {len(examples)}", backend="flex")
        else:
            seq_result["single"] = [
                _run_group([ex], f"single #{i}", backend="gather") for i, ex in enumerate(examples)
            ]
        all_results[seq_len] = seq_result

    return all_results


# ============================================================
# CLI
# ============================================================

def _interactive_custom_inference(model, tokenizer, device):
    print("\nMLM inference REPL. Include [MASK] tokens to see fill-in predictions.")
    print("Type 'batch' to enter several sentences at once (blank line runs them together), or 'quit' to exit.\n")
    while True:
        line = input(">>> ").strip()
        if line.lower() in ("quit", "exit"):
            break
        if line.lower() == "batch":
            texts = []
            while True:
                sub = input("  sentence (blank to run): ")
                if not sub.strip():
                    break
                texts.append(sub)
            if texts:
                run_inference(model, tokenizer, texts, device=device)
        elif line:
            run_inference(model, tokenizer, line, device=device)


def main():
    parser = argparse.ArgumentParser(
        description="Load a HELM v1 checkpoint and run MLM inference / validation benchmarks."
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit path to a checkpoint .pt file.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint global step to load (e.g. 65000).")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_MODEL_REPO_ID)
    parser.add_argument("--data-repo-id", type=str, default=DEFAULT_DATA_REPO_ID)
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--mode", choices=["interactive", "text", "validation"], default="validation")
    parser.add_argument("--text", type=str, nargs="+", default=None,
                         help="One or more strings to run inference on directly (--mode text).")
    parser.add_argument("--seq-lengths", type=int, nargs="+", default=[1024, 2048, 4096])
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--no-batch", action="store_true", help="Run validation examples one at a time.")
    parser.add_argument("--balance-easiness", action="store_true",
                         help="Spread selected validation examples across the easiness-score range.")
    parser.add_argument("--pool-size", type=int, default=50,
                         help="How many validation rows to sample from before selecting examples.")
    args = parser.parse_args()

    model, tokenizer, config, device = load_helm_model(
        checkpoint_path=args.checkpoint, step=args.step, repo_id=args.repo_id, hf_token=args.hf_token,
    )

    if args.mode == "interactive":
        _interactive_custom_inference(model, tokenizer, device)
    elif args.mode == "text":
        if not args.text:
            raise ValueError('--mode text requires --text "..." ["..." ...]')
        inputs = args.text[0] if len(args.text) == 1 else args.text
        run_inference(model, tokenizer, inputs, device=device)
    else:
        run_validation_test(
            model, tokenizer, config,
            seq_lengths=args.seq_lengths,
            num_examples=args.num_examples,
            batch=not args.no_batch,
            balance_easiness=args.balance_easiness,
            pool_size=args.pool_size,
            device=device,
            data_repo_id=args.data_repo_id,
            hf_token=args.hf_token,
        )


if __name__ == "__main__":
    import sys
    # ModernBERT's byte-level BPE tokens (e.g. "ĠParis") can't be encoded by the
    # default Windows console codepage; fall back to safe substitution instead of crashing.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
