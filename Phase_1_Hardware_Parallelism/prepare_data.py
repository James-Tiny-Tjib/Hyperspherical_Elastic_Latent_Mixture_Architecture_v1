##################################################
# Tokenizes all the text required to train HELM v1
# Uses ModernBert's Tokenizer
# Takes 10B Tokens from HuggingFaceFW/fineweb-edu
# Utilizes the sample-10BT subset
# Um yeah, here we go
##################################################

# ---
# configs:
#   - config_name: seq_1024
#     data_files:
#       - split: train
#         path: data/seq_1024/train-*.parquet
#       - split: validation
#         path: data/seq_1024/validation-*.parquet

#   - config_name: seq_2048
#     data_files:
#       - split: train
#         path: data/seq_2048/train-*.parquet
#       - split: validation
#         path: data/seq_2048/validation-*.parquet

#   - config_name: seq_4096
#     data_files:
#       - split: train
#         path: data/seq_4096/train-*.parquet
#       - split: validation
#         path: data/seq_4096/validation-*.parquet
# ---

# Note to Self:
# Generators exist, and they are like Iterators / classes inheriting Iterable in Java
# You can call iterators use next(), but lowkey just use a for loop

# Import Libraries
!pip install -U transformers datasets huggingface_hub tqdm pyarrow psutil
from transformers import AutoTokenizer
import datasets
from datasets import load_dataset
import torch
from tqdm import tqdm
import huggingface_hub
import pyarrow
import multiprocessing
import math
from itertools import chain, islice
import json
import os
from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi, HfFileSystem
from huggingface_hub.utils import RepositoryNotFoundError


# --- CONFIGURATION FLAG ---
VALIDATION_MODE = True  # Set to True for Run 1, False for Run 2
VAL_SIZE = 10000        # Number of docs for validation
# --------------------------


# Login to HF and create repo
hf_token = "" 
api = HfApi(token = hf_token)
repo_id = ""
api.create_repo(repo_id = repo_id, repo_type = "dataset", exist_ok = True)

progress = {
    "last_shard" : 0,
    "num_rows_processed": 0,
    "num_tokens_processed": 0
}


# progress.json retrieval
def retrieve_progress_from_hub(repo_id, filename = "progress.json"):
    try:
        path = hf_hub_download(repo_id=repo_id, filename = filename, repo_type = "dataset")
        with open(path, "r") as f:
            progress = json.load(f)
        print(f"Resuming from shard {progress['last_shard']}")
        return progress
    except RepositoryNotFoundError:
        print(f"Repo: \"{repo_id}\" was not found")
        return None
    except Exception as e:
        print(f"No progress file found or error occurred: {e}. Starting from token zero.")
        progress = {
            "last_shard" : 0,
            "num_rows_processed": 0,
            "num_tokens_processed": 0
        }
        return progress


# progress.json push
def push_progress_to_hub(progress, repo_id, filename = "progress.json"):
    try:
        with open(filename, "w") as f:
            json.dump(progress, f)
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=filename,
            repo_id=repo_id,
            repo_type="dataset"
        )
        return True
    except Exception as e:
        print("Error Occurred")
        return False


def push_shard_to_hub(shard_data, progress, repo_id, progress_file, length, api, data_split):
    
    next_shard = progress["last_shard"] + 1
    try:        
        # # Filename format exampele: data/seq_1024/train-00001.parquet
        # filename = f"{data_split}-{next_shard:05d}.parquet"
        # repo_path = f"data/seq_{length}/{filename}"

        # # Validation and Train will be subsets (name)
        # # Splits will be seq_1024, seq_2048, seq_4096 
        # split_name = f"seq_{length}"
        # filename = f"{split_name}-{next_shard:05d}.parquet"
        # repo_path = f"data/{data_split}/{filename}"

        # # seq_1024, seq_2048, seq_4096 will be subsets (name)
        # # Splits will be train and validation
        config_name = f"seq_{length}"
        filename = f"{data_split}-{next_shard:05d}.parquet"
        repo_path = f"data/{config_name}/{filename}"
        
        # Convert shard_data -> dictionary -> HF dataset -> .parquet
        ds_shard = datasets.Dataset.from_dict({"input_ids": shard_data})
        ds_shard.to_parquet(filename)
        
        # 2. Upload the raw file directly to the Hub
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset"
        )
        
        # 3. Clear local disk so Kaggle/Colab doesn't run out of storage
        os.remove(filename)
        

        # Update Progress and Push to Hub
        progress["last_shard"] = next_shard
        push_progress_to_hub(progress, repo_id, progress_file)
        
        # Success Message 
        print(f"✅ Shard {progress['last_shard']} uploaded successfully to {repo_path}.")
        return True

    except Exception as e:   

        # You done messed up
        print(f" ❌ Shard {progress['last_shard']} failed to upload")
        return False


# tokenization function (1 row)
def tokenize_single_row(row):
    tokenized_row = tokenizer(row, add_special_tokens=False)["input_ids"]
    return tokenized_row + [tokenizer.sep_token_id]


# Generator function to extract the text from the data_stream
def extract_text(data_stream):
    # Use for loop to extract the text from data_stream generator
    for row in data_stream:
        yield row["text"]


# Generator Function that returns the token list
def parallel_token_stream(data_stream, pool, progress, chunksize):

    # Define Generator Object 
    text_generator = extract_text(data_stream)

    # Use i(terable)map to iterate through and tokenize single rows
    results = pool.imap(tokenize_single_row, text_generator, chunksize=chunksize)

    # Use for loop to extract tokenized lists from results
    for token_list in results:
        # Increment documents processed
        progress["num_rows_processed"] += 1 
        # Yield / return 1 token list every time
        yield token_list


# Generator function that pack Tokens into correct lengths
def pack_tokens(data_chain, tokenizer, length = 1024):

    # Keep running until data_chain is empty
    while True:

        # Take slices of size "length"
        chunk = list(islice(data_chain, length-1))

        # If we run out, don't yield shit
        if (len(chunk) < length-1):
            break
        
        # Yield the list with [CLS] "EOS" token
        yield [tokenizer.cls_token_id] + chunk

# Generates a Shard 
def get_shard(packed_stream, shard_size, progress, repo_id, length, api, progress_file, split_name="train"):

    # Shard Data List
    shard_data = []

    # Use For Loop on Generator from pack_tokens
    for seq in packed_stream:
        shard_data.append(seq)

        # When shard reaches maximum size
        if (len(shard_data) >= shard_size):
            
            progress["num_tokens_processed"] += len(shard_data) * length

            if (push_shard_to_hub(shard_data, progress, repo_id, progress_file, length, api, split_name)):
                return True
            else:
                raise Exception("run_sharding_pipeline() done_messed up")
    
    return False
            

if __name__ == '__main__':

    PROGRESS_FILE = "val_progress.json" if VALIDATION_MODE else "train_progress.json"
    SPLIT_NAME = "validation" if VALIDATION_MODE else "train"

    # Import Tokenizer
    tokenizer_path = "answerdotai/ModernBERT-base" 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast = True)

    # Import Dataset
    dataset_path = "HuggingFaceFW/fineweb-edu"
    dataset_name_path = "sample-10BT"
    full_dataset = load_dataset(path = dataset_path, name = dataset_name_path, split = "train", streaming = True)

    progress = retrieve_progress_from_hub(repo_id, PROGRESS_FILE)

    if VALIDATION_MODE:

        print("Creating Validation Splits for 1024, 2048, and 4096")

        val_stream = full_dataset.take(VAL_SIZE)
        num_cores = max(1, multiprocessing.cpu_count()-1)

        # Apply Multiprocessing
        with multiprocessing.Pool(processes=num_cores) as pool:

            # Get Token Stream
            token_stream = parallel_token_stream(val_stream, pool, progress, chunksize = 1000)

            # CRITICAL FIX: Cast the generator to a list in memory.
            # We do this so we can reuse the exact same tokens for all 3 lengths.
            flat_chain_list = list(chain.from_iterable(token_stream))

            # Loop through the curriculum lengths
            for length in [1024, 2048, 4096]:
                print(f"Packing validation for seq_{length}...")
                
                # Pass an iterator of our list into pack_tokens
                packed_stream = pack_tokens(iter(flat_chain_list), tokenizer, length=length)

                # Convert to list and push
                val_data = list(packed_stream) 
                
                # Push to Hub
                push_shard_to_hub(val_data, progress, repo_id, PROGRESS_FILE, length=length, api=api, data_split=SPLIT_NAME)
            
            # Save final progress state
            push_progress_to_hub(progress, repo_id, PROGRESS_FILE)

        print("Validation Splits Completed")

    else:

        print("Creating Training Split")

        while progress["num_tokens_processed"] < 10e9:

            # Determine what length to do
            if (progress["num_tokens_processed"] >= 9e9):
                length = 4096
            elif (progress["num_tokens_processed"] >= 7e9):
                length = 2048
            else:
                length = 1024

            shard_token_num = 1e8
            num_sequences_in_shard = int(shard_token_num / length)

            # Calculate number of cores
            num_cores = max(1, multiprocessing.cpu_count()-1)
        
            # Apply Multiprocessing
            with multiprocessing.Pool(processes=num_cores) as pool:
                
                # Skip section we already processed
                train_dataset = full_dataset.skip(VAL_SIZE)
                skipped_dataset = train_dataset.skip(progress["num_rows_processed"])

                # Get Token Stream
                token_stream = parallel_token_stream(skipped_dataset, pool, progress, chunksize = 1000)

                # Get Long Chain of input_ids
                flat_chain = chain.from_iterable(token_stream)

                # Pack the Chain
                packed_stream = pack_tokens(flat_chain, tokenizer, length)

                # Process 1 Shard
                has_more_data = get_shard(packed_stream, num_sequences_in_shard, progress, repo_id, length, api, PROGRESS_FILE, SPLIT_NAME)

                # Save progress to hub
                push_progress_to_hub(progress, repo_id, PROGRESS_FILE)

                if not has_more_data:
                    print("Dataset stream exhausted")
                    print("Training Split Completed")
                    break