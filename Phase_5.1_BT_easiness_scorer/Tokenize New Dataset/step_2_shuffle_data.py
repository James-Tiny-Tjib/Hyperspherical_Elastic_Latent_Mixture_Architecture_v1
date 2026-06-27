######################################################
# Step 2: mix the per-source 512-block repos into ONE combined repo
# - validation: concatenate every source's validation split (already document-disjoint
#   from train thanks to step 1's hash routing), shuffle, upload as a SINGLE shard
#   (validation-00000.parquet) because step 3 packs that one file three ways (1024/2048/4096)
# - train: pull blocks from each source in proportion to its train block count, shard out
##################################################

from datasets import load_dataset
import datasets
import random
import multiprocessing
import math
from itertools import chain, islice
import json
import os
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import RepositoryNotFoundError

LENGTH = 512
SHARD_TOKEN_NUM = 1e8
NUM_EXAMPLES_PER_SHARD = int(-(-SHARD_TOKEN_NUM // LENGTH))   # ceil
COMBINED_REPO_ID = "JamesResearch1216/ModernBERT-512-Combined-v4"
SHUFFLE_SEED = 1216

SOURCE_REPOS = [
    "JamesResearch1216/ModernBERT-512-pubmed_simple",
    "JamesResearch1216/ModernBERT-512-arxiv-abstracts-large",
    "JamesResearch1216/ModernBERT-512-govreport-summarization",
    "JamesResearch1216/ModernBERT-512-TTCW-Based-Review",
    "JamesResearch1216/ModernBERT-512-fineweb",
    "JamesResearch1216/ModernBERT-512-wikihow-cleaned",
    "JamesResearch1216/ModernBERT-512-simple_wikipedia_LM",
    "JamesResearch1216/ModernBERT-512-reddit-title-body",
    "JamesResearch1216/ModernBERT-512-Amazon_Reviews_Binary_for_Sentiment_Analysis",
    "JamesResearch1216/ModernBERT-512-yelp_review_full",
    "JamesResearch1216/ModernBERT-512-review_corpus",
    "JamesResearch1216/ModernBERT-512-TinyStories",
    "JamesResearch1216/ModernBERT-512-SimpleStories",
]


def get_secret(key_name):
    try:
        from google.colab import userdata
        return userdata.get(key_name)
    except Exception:
        pass
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(key_name)
    except Exception:
        pass
    return os.getenv(key_name)


def get_progress(repo_id, filename="train_progress.json"):
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=hf_token)
        with open(path, "r") as f:
            return json.load(f)
    except RepositoryNotFoundError:
        print(f"Repo {repo_id} not found. Skipping.")
        return None
    except Exception as e:
        print(f"No progress for {repo_id}. (Notice: {e})")
        return None


def push_shard(shard_data, shard_index, repo_id, length, api, split):
    try:
        filename = f"{split}-{shard_index:05d}.parquet"
        repo_path = f"data/seq_{length}/{filename}"
        ds_shard = datasets.Dataset.from_list(shard_data)
        ds_shard.to_parquet(filename)
        api.upload_file(path_or_fileobj=filename, path_in_repo=repo_path,
                        repo_id=repo_id, repo_type="dataset")
        os.remove(filename)
        print(f"  {split} shard {shard_index:05d} -> {repo_path} ({len(shard_data)} rows)")
        return True
    except Exception as e:
        print(f"  {split} shard {shard_index} failed: {e}")
        return False


def build_validation(api):
    # Concatenate every source's validation split (each is document-disjoint from train),
    # shuffle so the single val parquet isn't clustered by source, push as index 0.
    print("\n=== Building validation split ===")
    val_rows = []
    for repo_id in SOURCE_REPOS:
        try:
            ds = load_dataset(repo_id, split="validation", streaming=True)
            n = 0
            for row in ds:
                val_rows.append({"input_ids": row["input_ids"]})
                n += 1
            print(f"  {repo_id}: +{n} val blocks")
        except Exception as e:
            print(f"  {repo_id}: no validation split ({e})")

    if not val_rows:
        print("  No validation rows found! Did step 1 run with VAL_FRACTION > 0?")
        return

    random.Random(SHUFFLE_SEED).shuffle(val_rows)
    push_shard(val_rows, 0, COMBINED_REPO_ID, LENGTH, api, "validation")
    print(f"  validation total: {len(val_rows)} blocks ({len(val_rows) * LENGTH:,} tokens)")


def build_train(api):
    print("\n=== Building train split ===")
    # Read each source's TRAIN block count (num_samples_created) to compute mixing ratios.
    counts = {}
    total = 0
    for repo_id in SOURCE_REPOS:
        prog = get_progress(repo_id)
        if prog is not None:
            c = prog.get("num_samples_created", 0)
            counts[repo_id] = c
            total += c
        else:
            counts[repo_id] = 0
            print(f"  {repo_id} has no progress (treated as 0)")

    if total == 0:
        print("  No train blocks found. Aborting train build.")
        return

    per_shard = {}
    actual = 0
    for repo_id, c in counts.items():
        ratio = c / total
        per_shard[repo_id] = int(ratio * NUM_EXAMPLES_PER_SHARD)
        actual += per_shard[repo_id]
        print(f"  {repo_id}: ratio {ratio:.4f} -> {per_shard[repo_id]} blocks/shard")
    print(f"  target/shard {NUM_EXAMPLES_PER_SHARD}, actual/shard {actual}, "
          f"total train tokens {total * LENGTH:,}")

    iters = {}
    for repo_id, c in per_shard.items():
        if c > 0:
            iters[repo_id] = iter(load_dataset(repo_id, split="train", streaming=True))

    shard_index = 0
    exhausted = False
    while not exhausted:
        mixed = []
        for repo_id, c in per_shard.items():
            if c == 0:
                continue
            chunk = list(islice(iters[repo_id], c))
            mixed.extend({"input_ids": r["input_ids"]} for r in chunk)
            if len(chunk) < c:
                print(f"  {repo_id} ran dry.")
                exhausted = True
        if not mixed:
            break
        # shuffle within the shard so step 3's consecutive-block packing isn't source-ordered
        random.Random(SHUFFLE_SEED + shard_index).shuffle(mixed)
        if not push_shard(mixed, shard_index, COMBINED_REPO_ID, LENGTH, api, "train"):
            print("  Halting (upload failure).")
            break
        shard_index += 1

    print(f"  train shards written: {shard_index}")


if __name__ == '__main__':
    hf_token = get_secret("HF_TOKEN")
    if hf_token is None:
        print("WARNING: HF_TOKEN could not be found! All uploads will fail.")
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=COMBINED_REPO_ID, repo_type="dataset", exist_ok=True)

    build_validation(api)   # one pass: build the held-out val shard first
    build_train(api)        # then mix the train shards

    print("\nStep 2 complete.")
