######################################################
# Tokenizes into 512 token-blocks & pack into parquets
# Uses ModernBert's Tokenizer
# Uses checkpointing features in case of a crash
# Takes 10 Billion Tokens from the sources below
##################################################

# Import Libraries
# !pip install -U transformers datasets huggingface_hub tqdm pyarrow psutil
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


# --- CONFIGURATION ---
LENGTH = 512
SHARD_TOKEN_NUM = 1e8
VALIDATION_MODE = False  # Set to True for Run 1, False for Run 2
CRASH_NUMBER = 0
SOURCES = [
    {
        "path" : "HuggingFaceFW/fineweb",
        "name" : "sample-10BT",
        "split" : "train",
        "col_name" : "text",
        "token_cap" : 4e9,
        "num_val_rows" : 200
    },
    {
        "path" : "roneneldan/TinyStories",
        "name" : None,
        "split" : "train",
        "col_name" : "text",
        "token_cap" : -1,
        "num_val_rows" : 200
    },
    {
        "path" : "SimpleStories/SimpleStories",
        "name" : None,
        "split" : "train",
        "col_name" : "story",
        "token_cap" : -1,
        "num_val_rows" : 200

    },
    {
        "path" : "UniverseTBD/arxiv-abstracts-large",
        "name" : None,
        "split" : "train",
        "col_name" : "abstract",
        "token_cap" : -1,
        "num_val_rows" : 200           
    },
    {
        "path" : "japhba/pubmed_simple",
        "name" : None,
        "split" : "train",
        "col_name" : "abstract",
        "token_cap" : -1,
        "num_val_rows" : 200      
    },
    {
        "path" : "ccdv/govreport-summarization",
        "name" : None,
        "split" : "train",
        "col_name" : "report",
        "token_cap" : -1,
        "num_val_rows" : 200       
    },
    {
        "path" : "VibrantVista/TTCW-Based-Review",
        "name" : None,
        "split" : "train",
        "col_name" : "regenerated_story",
        "token_cap" : 1e9,
        "num_val_rows" : 200        
    },
    {
        "path" : "pszemraj/simple_wikipedia_LM",
        "name" : "default",
        "split" : "train",
        "col_name" : "text",
        "token_cap" : -1,
        "num_val_rows" : 200    
    },
    {
        "path" : "sentence-transformers/reddit-title-body",
        "name" : None,
        "split" : "train",
        "col_name" : "body",
        "token_cap" : 2.5e9,
        "num_val_rows" : 200    
    },
    {
        "path" : "gursi26/wikihow-cleaned",
        "name" : None,
        "split" : "train",
        "col_name" : "text",
        "token_cap" : -1,
        "num_val_rows" : 200    
    },
]
# --------------------------


# Get HF_TOKEN and WANDB_API_KEY
def get_secret(key_name):

    # Try Colab
    try:
        from google.colab import userdata
        return userdata.get(key_name)
    except:
        pass

    # Try Kaggle
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret(key_name)
    except:
        pass

    # Local Env
    return os.getenv(key_name)

hf_token = get_secret("HF_TOKEN")
if hf_token is None:
    print("⚠️ WARNING: HF_TOKEN could not be found! All uploads will fail.")
api = HfApi(token = hf_token)

# repo_id = ""
# api.create_repo(repo_id = repo_id, repo_type = "dataset", exist_ok = True)

progress = {
    "last_shard" : -1,
    "num_rows_processed": 0,
    "num_tokens_processed": 0,
    "num_samples_created" : 0
}


# progress.json retrieval
def retrieve_progress_from_hub(repo_id, filename = "progress.json"):
    try:
        path = hf_hub_download(repo_id=repo_id, filename = filename, repo_type = "dataset", token=hf_token)
        with open(path, "r") as f:
            progress = json.load(f)
        print(f"Resuming from shard {progress['last_shard']}")
        return progress
    except RepositoryNotFoundError:
        print(f"Repo: \"{repo_id}\" was not found on the Hub (or token is missing). Starting fresh.")
        return {
            "last_shard" : -1,
            "num_rows_processed": 0,
            "num_tokens_processed": 0,
            "num_samples_created" : 0
        }
    except Exception as e:
        print(f"No progress file found. Starting fresh. (Notice: {e})")
        return {
            "last_shard" : -1,
            "num_rows_processed": 0,
            "num_tokens_processed": 0,
            "num_samples_created" : 0
        }


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
        print(f"❌ Failed to upload progress.json: {e}")
        return False


def push_shard_to_hub(shard_data, progress, repo_id, progress_file, length, api, data_split):
    
    next_shard = progress["last_shard"] + 1
    try:        

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
        print(f" ❌ Shard {progress['last_shard']} failed to upload: {e}")
        return False


# tokenization function (1 row)
def tokenize_single_row(row):
    if not row or not isinstance(row, str):
        return []
    tokenized_row = tokenizer(row, add_special_tokens=False)["input_ids"]
    return tokenized_row + [tokenizer.sep_token_id]


# Generator function to extract the text from the data_stream
def extract_text(data_stream, col_name):
    # Use for loop to extract the text from data_stream generator
    for row in data_stream:
        yield row[col_name]


# Generator Function that returns the token list
def parallel_token_stream(data_stream, pool, progress, chunksize, col_name):

    # Define Generator Object 
    text_generator = extract_text(data_stream, col_name)

    # Use i(terable)map to iterate through and tokenize single rows
    results = pool.imap(tokenize_single_row, text_generator, chunksize=chunksize)

    # Use for loop to extract tokenized lists from results
    for token_list in results:
        # Increment documents processed
        progress["num_rows_processed"] += 1 
        # Yield / return 1 token list every time
        yield token_list


# Generator function that pack Tokens into correct lengths
def pack_tokens(data_chain, tokenizer, progress, max_examples=-1, length=512):

    # Keep running until data_chain is empty
    while True:
        
        if (max_examples !=-1) and (progress["num_samples_created"] >= max_examples):
            break

        # Take slices of size "length"
        chunk = list(islice(data_chain, length-1))

        # If we run out, don't yield shit
        if (len(chunk) < length-1):
            break
        
        progress["num_samples_created"] += 1

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
    
    # Catch any leftover data if the stream runs out before filling a full shard
    if len(shard_data) > 0:
        progress["num_tokens_processed"] += len(shard_data) * length
        push_shard_to_hub(shard_data, progress, repo_id, progress_file, length, api, split_name)
                    
    return False
            

if __name__ == '__main__':

    # Import Tokenizer
    tokenizer_path = "answerdotai/ModernBERT-base" 
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast = True)

    for source in SOURCES:

        repo_id = f"JamesResearch1216/ModernBERT-512-{source["path"][source["path"].find('/')+1:]}"
        print(repo_id)
        
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)


        PROGRESS_FILE = "val_progress.json" if VALIDATION_MODE else "train_progress.json"
        SPLIT_NAME = "validation" if VALIDATION_MODE else "train"

        # Import Dataset
        dataset_path = source["path"]
        dataset_split_path = source["split"]
        dataset_name_path = source["name"]
        full_dataset = load_dataset(
            path = dataset_path, 
            name = dataset_name_path, 
            split = dataset_split_path, 
            streaming = True
        )


        progress = retrieve_progress_from_hub(repo_id, PROGRESS_FILE)

        if VALIDATION_MODE:

            
            max_examples = source["num_val_rows"]
            

            print("Creating Validation Rows")

            num_cores = max(1, multiprocessing.cpu_count()-1)

            # Apply Multiprocessing
            with multiprocessing.Pool(processes=num_cores) as pool:

                # Get Token Stream
                token_stream = parallel_token_stream(full_dataset, pool, progress, 1, source["col_name"] )

                required_tokens = max_examples * (LENGTH - 1)

                flat_chain_list = list(islice(chain.from_iterable(token_stream), required_tokens))
                
                # Pass an iterator of our list into pack_tokens
                packed_stream = pack_tokens(iter(flat_chain_list), tokenizer, progress, max_examples=max_examples, length = LENGTH)

                # Convert to list and push
                val_data = list(packed_stream) 
                
                # Push to Hub
                push_shard_to_hub(val_data, progress, repo_id, PROGRESS_FILE, length=LENGTH, api=api, data_split=SPLIT_NAME)
                
                # Save final progress state
                push_progress_to_hub(progress, repo_id, PROGRESS_FILE)

            print("Validation Splits Completed")

        else:

            print("Creating Training Split")

            token_cap = source["token_cap"]

            num_sequences_in_shard = int(SHARD_TOKEN_NUM / LENGTH)

            # Calculate number of cores
            num_cores = max(1, multiprocessing.cpu_count()-1)

            # Apply Multiprocessing
            with multiprocessing.Pool(processes=num_cores) as pool:
                
                # Get how many rows were used by val_progress
                val_progress = retrieve_progress_from_hub(repo_id, "val_progress.json")
                rows_to_skip = val_progress["num_rows_processed"] + progress["num_rows_processed"]
                skipped_dataset = full_dataset.skip(rows_to_skip)   # skip val rows + already-trained rows
                
                token_stream = parallel_token_stream(skipped_dataset, pool, progress, chunksize=1000, col_name=source["col_name"])
                flat_chain = chain.from_iterable(token_stream)

                # Pack the Chain
                packed_stream = pack_tokens(flat_chain, tokenizer, progress, max_examples = -1, length = LENGTH)

                while token_cap == -1 or progress["num_tokens_processed"] < token_cap:

                    # Process 1 Shard
                    has_more_data = get_shard(packed_stream, num_sequences_in_shard, progress, repo_id, LENGTH, api, PROGRESS_FILE, SPLIT_NAME)

                    # Save progress to hub
                    push_progress_to_hub(progress, repo_id, PROGRESS_FILE)

                    if not has_more_data:
                        print("Dataset stream exhausted")
                        print("Training Split Completed")
                        break
            print(f"Training Data for {source["path"]} completed!")