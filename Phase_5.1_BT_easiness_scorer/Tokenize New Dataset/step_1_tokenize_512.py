######################################################
# Step 1: tokenize raw sources into uniform 512-token blocks
# - ModernBERT tokenizer
# - DETERMINISTIC document-level train/val split (content hash)
#       * a whole document goes ENTIRELY to train OR validation, never both
#       * train docs and val docs are packed into SEPARATE 512-block streams
#       * identical text always routes to the same split (free exact-dup dedup)
# - per-source checkpointing via progress.json
##################################################

from transformers import AutoTokenizer
import datasets
from datasets import load_dataset
import huggingface_hub
import pyarrow
import multiprocessing
import math
import hashlib
from collections import deque
from itertools import chain, islice
import json
import os
from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi, HfFileSystem
from huggingface_hub.utils import RepositoryNotFoundError


# --- CONFIGURATION ---
LENGTH = 512
SHARD_TOKEN_NUM = 1e8

# ---- THE single source of truth for the train/val split ----
# Whole-document routing by content hash. Change VAL_FRACTION to resize validation;
# change HASH_SALT to reshuffle which documents are held out.
VAL_FRACTION = 0.002          # ~0.2% of tokens -> validation (~20M tokens on a 10B corpus)
HASH_SALT    = "helm-v3-split"

def doc_split(text):
    key = (HASH_SALT + (text or "").strip()).encode("utf-8")
    bucket = int(hashlib.md5(key).hexdigest()[:8], 16) / 0x100000000   # in [0, 1)
    return "validation" if bucket < VAL_FRACTION else "train"

SOURCES = [
    {"path": "HuggingFaceFW/fineweb", "name": "sample-10BT", "split": "train", "col_name": "text", "token_cap": 4e9},
    {"path": "roneneldan/TinyStories", "name": None, "split": "train", "col_name": "text", "token_cap": -1},
    {"path": "SimpleStories/SimpleStories", "name": None, "split": "train", "col_name": "story", "token_cap": -1},
    {"path": "UniverseTBD/arxiv-abstracts-large", "name": None, "split": "train", "col_name": "abstract", "token_cap": -1},
    {"path": "japhba/pubmed_simple", "name": None, "split": "train", "col_name": "abstract", "token_cap": -1},
    {"path": "ccdv/govreport-summarization", "name": None, "split": "train", "col_name": "report", "token_cap": -1},
    {"path": "VibrantVista/TTCW-Based-Review", "name": None, "split": "train", "col_name": "regenerated_story", "token_cap": 1e9},
    {"path": "pszemraj/simple_wikipedia_LM", "name": "default", "split": "train", "col_name": "text", "token_cap": -1},
    {"path": "sentence-transformers/reddit-title-body", "name": None, "split": "train", "col_name": "body", "token_cap": 2.5e9},
    {"path": "gursi26/wikihow-cleaned", "name": None, "split": "train", "col_name": "text", "token_cap": -1},
    {"path": "Yelp/yelp_review_full", "name": None, "split": "train", "col_name": "text", "token_cap": -1},
    {"path": "yassiracharki/Amazon_Reviews_Binary_for_Sentiment_Analysis", "name": None, "split": "train", "col_name": "review_text", "token_cap": -1},
    {"path": "yyu/review_corpus", "name": None, "split": "train", "col_name": "text", "token_cap": -1},
]
# --------------------------


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

hf_token = get_secret("HF_TOKEN")
if hf_token is None:
    print("WARNING: HF_TOKEN could not be found! All uploads will fail.")
api = HfApi(token=hf_token)


def fresh_progress():
    return {
        "last_train_shard": -1,
        "last_val_shard": -1,
        "num_rows_processed": 0,        # documents consumed (for resume skip)
        "num_train_tokens": 0,
        "num_val_tokens": 0,
        "num_samples_created": 0,       # TRAIN 512-blocks  (step 2 reads THIS for mixing ratios)
        "num_val_samples_created": 0,   # VALIDATION 512-blocks
    }


def retrieve_progress_from_hub(repo_id, filename="progress.json"):
    try:
        path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=hf_token)
        with open(path, "r") as f:
            progress = json.load(f)
        print(f"Resuming {repo_id}: {progress['num_rows_processed']} docs done, "
              f"train shard {progress['last_train_shard']}, val shard {progress['last_val_shard']}")
        # backfill any missing keys (forward-compat)
        for k, v in fresh_progress().items():
            progress.setdefault(k, v)
        return progress
    except RepositoryNotFoundError:
        print(f"Repo {repo_id} not found. Starting fresh.")
        return fresh_progress()
    except Exception as e:
        print(f"No progress file. Starting fresh. (Notice: {e})")
        return fresh_progress()


def push_progress_to_hub(progress, repo_id, filename="progress.json"):
    try:
        with open(filename, "w") as f:
            json.dump(progress, f)
        api.upload_file(path_or_fileobj=filename, path_in_repo=filename,
                        repo_id=repo_id, repo_type="dataset")
        return True
    except Exception as e:
        print(f"Failed to upload {filename}: {e}")
        return False


# Hash the document FIRST (in the main process), then tokenize in the worker.
def split_texts(data_stream, col_name):
    for row in data_stream:
        text = row[col_name]
        yield (doc_split(text), text)


# Worker fn: returns (split, tokens). Keeps the split tag glued to its document's tokens.
def tokenize_with_split(split_text):
    split, text = split_text
    if not text or not isinstance(text, str):
        return (split, [])
    toks = tokenizer(text, add_special_tokens=False)["input_ids"]
    return (split, toks + [tokenizer.sep_token_id])


# Packs a single split's token stream into uniform 512-blocks and uploads shards.
# Train and validation each get their own writer, so their streams never touch.
class ShardWriter:
    def __init__(self, repo_id, split, shard_size, cls_id, progress, progress_file,
                 shard_key, count_key, token_key):
        self.repo_id = repo_id
        self.split = split
        self.shard_size = shard_size
        self.cls_id = cls_id
        self.progress = progress
        self.progress_file = progress_file
        self.shard_key = shard_key
        self.count_key = count_key
        self.token_key = token_key
        self.buffer = deque()        # flat token buffer (carries across docs WITHIN this split)
        self.shard_seqs = []         # accumulated 512-blocks awaiting upload

    def add(self, toks):
        self.buffer.extend(toks)
        # emit full [CLS]+511 blocks; leftover stays in the buffer for the next doc
        while len(self.buffer) >= LENGTH - 1:
            block = [self.cls_id]
            for _ in range(LENGTH - 1):
                block.append(self.buffer.popleft())
            self.shard_seqs.append(block)
            self.progress[self.count_key] += 1
            if len(self.shard_seqs) >= self.shard_size:
                self._flush()

    def pending_tokens(self):
        return len(self.shard_seqs) * LENGTH

    def _flush(self):
        if not self.shard_seqs:
            return
        next_shard = self.progress[self.shard_key] + 1
        filename = f"{self.split}-{next_shard:05d}.parquet"
        repo_path = f"data/seq_{LENGTH}/{filename}"
        try:
            ds = datasets.Dataset.from_dict({"input_ids": self.shard_seqs})
            ds.to_parquet(filename)
            api.upload_file(path_or_fileobj=filename, path_in_repo=repo_path,
                            repo_id=self.repo_id, repo_type="dataset")
            os.remove(filename)
            self.progress[self.token_key] += len(self.shard_seqs) * LENGTH
            self.progress[self.shard_key] = next_shard
            push_progress_to_hub(self.progress, self.repo_id, self.progress_file)
            print(f"  {self.split} shard {next_shard:05d} -> {repo_path} ({len(self.shard_seqs)} seqs)")
            self.shard_seqs = []
        except Exception as e:
            raise Exception(f"{self.split} shard {next_shard} failed to upload: {e}")

    def finalize(self):
        # flush remaining FULL blocks (a final small shard); the partial token tail
        # is intentionally dropped so every row stays exactly 512 long (step 3 requires this).
        self._flush()


if __name__ == '__main__':

    tokenizer_path = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)

    num_seqs_in_shard = int(SHARD_TOKEN_NUM / LENGTH)
    num_cores = max(1, multiprocessing.cpu_count() - 1)

    for source in SOURCES:
        short = source["path"][source["path"].find('/') + 1:]
        repo_id = f"JamesResearch1216/ModernBERT-512-{short}"
        print(f"\n=== {repo_id} ===")
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

        PROGRESS_FILE = "train_progress.json"   # single progress file now (no separate val pass)
        progress = retrieve_progress_from_hub(repo_id, PROGRESS_FILE)

        full_dataset = load_dataset(
            path=source["path"], name=source["name"], split=source["split"], streaming=True
        )

        token_cap = source["token_cap"]

        with multiprocessing.Pool(processes=num_cores) as pool:
            train_writer = ShardWriter(repo_id, "train", num_seqs_in_shard, tokenizer.cls_token_id,
                                       progress, PROGRESS_FILE,
                                       "last_train_shard", "num_samples_created", "num_train_tokens")
            val_writer = ShardWriter(repo_id, "validation", num_seqs_in_shard, tokenizer.cls_token_id,
                                     progress, PROGRESS_FILE,
                                     "last_val_shard", "num_val_samples_created", "num_val_tokens")

            # Resume: skip documents we already consumed. Routing is deterministic per-doc,
            # so skipped docs already went to their correct split; no double-writes.
            stream = full_dataset.skip(progress["num_rows_processed"])
            routed = pool.imap(tokenize_with_split, split_texts(stream, source["col_name"]), chunksize=1000)

            for split, toks in routed:
                progress["num_rows_processed"] += 1
                if split == "train":
                    train_writer.add(toks)
                else:
                    val_writer.add(toks)

                # Train token cap (counts flushed + in-flight blocks). Overshoots by <=1 shard.
                if token_cap != -1 and (progress["num_train_tokens"] + train_writer.pending_tokens()) >= token_cap:
                    print(f"  token cap {token_cap:.2e} reached for {short}")
                    break

            train_writer.finalize()
            val_writer.finalize()
            push_progress_to_hub(progress, repo_id, PROGRESS_FILE)

        print(f"Done {short}: train_blocks={progress['num_samples_created']}, "
              f"val_blocks={progress['num_val_samples_created']}, "
              f"train_tokens={progress['num_train_tokens']:,}, val_tokens={progress['num_val_tokens']:,}")

    print("\nStep 1 complete.")
