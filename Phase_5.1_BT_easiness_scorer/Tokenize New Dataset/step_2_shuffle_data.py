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

LENGTH = 512
SHARD_TOKEN_NUM = 1e8
NUM_EXAMPLES_PER_SHARD = float(-(-SHARD_TOKEN_NUM // LENGTH)) # Ceiling Trick I learned with AI
REAL_SHARD_TOKEN_COUNT = float(NUM_EXAMPLES_PER_SHARD * LENGTH)
COMBINED_REPO_ID = "JamesResearch1216/ModernBERT-512-Combined-v2" # Set your target repo here
VALIDATION_MODE = True
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

def get_progress(repo_id, filename = "train_progress.json"):
    try:
        path = hf_hub_download(repo_id=repo_id, filename = filename, repo_type = "dataset", token=hf_token)
        with open(path, "r") as f:
            progress = json.load(f)
        return progress
    except RepositoryNotFoundError:
        print(f"Repo: \"{repo_id}\" was not found on the Hub (or token is missing). Starting fresh.")
        return None
    except Exception as e:
        print(f"No progress file found. Starting fresh. (Notice: {e})")
        return None

# Simplified uploader for the mixed shards
def push_mixed_shard_to_hub(shard_data, shard_index, repo_id, length, api, split):
    try:        
        filename = f"{split}-{shard_index:05d}.parquet"
        repo_path = f"data/seq_{length}/{filename}"
        
        # Convert to HF dataset
        # We assume shard_data is a list of dictionaries: [{"input_ids": [...]}, ...]
        ds_shard = datasets.Dataset.from_list(shard_data)
        ds_shard.to_parquet(filename)
        
        # Upload
        api.upload_file(
            path_or_fileobj=filename,
            path_in_repo=repo_path,
            repo_id=repo_id,
            repo_type="dataset"
        )
        os.remove(filename)
        print(f"✅ Mixed Shard {shard_index} uploaded successfully to {repo_path}.")
        return True
    except Exception as e:   
        print(f"❌ Shard {shard_index} failed to upload: {e}")
        return False

hf_token = get_secret("HF_TOKEN")
if hf_token is None:
    print("⚠️ WARNING: HF_TOKEN could not be found! All uploads will fail.")
api = HfApi(token = hf_token)

dataset_progresses = {
    "JamesResearch1216/ModernBERT-512-pubmed_simple" : None,
    "JamesResearch1216/ModernBERT-512-arxiv-abstracts-large" : None,
    "JamesResearch1216/ModernBERT-512-govreport-summarization" : None,
    "JamesResearch1216/ModernBERT-512-TTCW-Based-Review" : None,
    "JamesResearch1216/ModernBERT-512-fineweb" : None,
    "JamesResearch1216/ModernBERT-512-wikihow-cleaned" : None,
    "JamesResearch1216/ModernBERT-512-simple_wikipedia_LM" : None,
    "JamesResearch1216/ModernBERT-512-reddit-title-body" : None,
    "JamesResearch1216/ModernBERT-512-TinyStories" : None,
    "JamesResearch1216/ModernBERT-512-SimpleStories" : None,
}

samples_per_shard = {
    "JamesResearch1216/ModernBERT-512-pubmed_simple" : 0.0,
    "JamesResearch1216/ModernBERT-512-arxiv-abstracts-large" : 0.0,
    "JamesResearch1216/ModernBERT-512-govreport-summarization" : 0.0,
    "JamesResearch1216/ModernBERT-512-TTCW-Based-Review" : 0.0,
    "JamesResearch1216/ModernBERT-512-fineweb" : 0.0,
    "JamesResearch1216/ModernBERT-512-wikihow-cleaned" : 0.0,
    "JamesResearch1216/ModernBERT-512-simple_wikipedia_LM" : 0.0,
    "JamesResearch1216/ModernBERT-512-reddit-title-body" : 0.0,
    "JamesResearch1216/ModernBERT-512-TinyStories" : 0.0,
    "JamesResearch1216/ModernBERT-512-SimpleStories" : 0.0,
}



if __name__ == '__main__':

    api.create_repo(repo_id=COMBINED_REPO_ID, repo_type="dataset", exist_ok=True)

    if (VALIDATION_MODE):
        
        validation_data = []

        for path in dataset_progresses.keys():

            print(f"Loading val. split for {path}...")
            dataset = load_dataset(path, split = "validation", streaming = True)
            for row in dataset:
                validation_data.append(row)



        success = push_mixed_shard_to_hub(
            shard_data=validation_data, 
            shard_index=0, 
            repo_id=COMBINED_REPO_ID, 
            length=LENGTH, 
            api=api,
            split = "validation"
        )
        
        if not success:
            print("Halting pipeline due to upload failure.")



    else:

        total_samples = 0

        for dataset in dataset_progresses.keys():
            dataset_progresses[dataset] = get_progress(dataset)
            if dataset_progresses[dataset] is not None:
                dataset_num_samples = dataset_progresses[dataset]["num_samples_created"]
                total_samples += dataset_num_samples
                samples_per_shard[dataset] = dataset_num_samples
            else:
                print(f"{dataset} is kinda cooked (REELY BAD)")
        
        actual_shard_total = 0

        for dataset, ds_samples in samples_per_shard.items():
            ratio = ds_samples / total_samples
            print(f"{dataset} Ratio: {ratio}")
            samples_per_shard[dataset] = int(ratio * NUM_EXAMPLES_PER_SHARD)
            actual_shard_total += samples_per_shard[dataset]    
        
        print(f"\nTarget examples per shard: {int(NUM_EXAMPLES_PER_SHARD)}")
        print(f"Actual examples per shard: {actual_shard_total}")
        print(f"Total tokens across all source datasets: {total_samples * LENGTH}\n")

        # 3. Initialize streaming iterators for all active datasets
        print("Initializing dataset streams...")
        dataset_iters = {}
        for dataset, count in samples_per_shard.items():
            if count > 0:
                # Load the tokenized repos we created in step 1
                stream = load_dataset(dataset, split="train", streaming=True)
                dataset_iters[dataset] = iter(stream)

        # 4. Pack, Shuffle, and Upload Loop
        shard_index = 0
        data_exhausted = False

        while not data_exhausted:
            mixed_shard_data = []

            # Pull exact required counts from each dataset
            for dataset, count in samples_per_shard.items():
                if count == 0:
                    continue
                    
                # islice acts exactly like .take(count) but for Python iterators
                chunk = list(islice(dataset_iters[dataset], count))
                mixed_shard_data.extend(chunk)

                # If a stream runs dry before we reach the requested count, we are out of data
                if len(chunk) < count:
                    print(f"Dataset {dataset} has run out of data!")
                    data_exhausted = True

            if len(mixed_shard_data) == 0:
                break # Completely empty, we are done

            # Upload the shard
            success = push_mixed_shard_to_hub(
                shard_data=mixed_shard_data, 
                shard_index=shard_index, 
                repo_id=COMBINED_REPO_ID, 
                length=LENGTH, 
                api=api,
                split = "train"
            )
            
            if not success:
                print("Halting pipeline due to upload failure.")
                break

            shard_index += 1

        print("\n🎉 Mixed pipeline completed!")









