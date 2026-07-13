%%writefile parallel_hardware_trainer.py
DOES_RESUME_FROM_WORK = False
TESTING_MODE = False

# Imports (Manifesting that my loss curve will look like this)
import io
import os
import re
import sys
import time
import json
import site
import math
import glob
import torch
import wandb
import warnings
import importlib
import numpy as np
import multiprocessing
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataclasses import dataclass, field
from datasets import load_dataset, Dataset
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, List, Union, ClassVar, Dict, Any
from huggingface_hub import hf_hub_download, create_repo, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError


# Disable Progress Bars
try:
    from huggingface_hub.utils import disable_progress_bars
    disable_progress_bars()
except ImportError:
    pass

# Ensure PJRT runtime gets selected, not XRT
for key in ["XRT_TPU_CONFIG", "PJRT_SELECT_DEVICE", "TPU_PROCESS_ADDRESSES"]:
    os.environ.pop(key, None)
os.environ["PJRT_DEVICE"] = "TPU"
# Add the framework quarantine just in case!
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Prevent C++ thread deadlocks during 10B token streaming
os.environ["OMP_NUM_THREADS"] = "1"
import pyarrow as pa
pa.set_cpu_count(1)
pa.set_io_thread_count(1)

# Tell everything to stay away from the TPU except PyTorch
os.environ["USE_TORCH"] = "1"
os.environ["USE_TF"] = "0"
os.environ["USE_JAX"] = "0"

# Gets modified to tell child processes to return cleanly
SHUTDOWN_FILE = "/tmp/SHUTDOWN_REQUESTED"
# This lets the restarting launching script whether user intentionally ended program or not
USER_STOP_MARKER = "/tmp/USER_STOPPED_TRAINING"

# Force Path Refresh
if 'site' in sys.modules:
    importlib.reload(site)

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

# Get the float value of something (mainly for losses)
def to_float(x):
    return x.item() if hasattr(x, 'item') else float(x)


# ==================================================
# HardwareConfig
# Keeps track of which device to use  what device-dependent values to use
# Not Static (Changes State)
# ==================================================

@dataclass
class HardwareConfig:

    HARDWARE_PROFILES = {
        "v5e-8": {
            "ws": 8, "target": 128, "dtype": torch.bfloat16,"use_scaler": False,
            0: {"mb": 8, "use_ckpt": False, "sl": 1024},
            1: {"mb": 2, "use_ckpt": False, "sl": 2048},
            2: {"mb": 1, "use_ckpt": True, "sl": 4096},
        },
        "v5e-1": {
            "ws": 1, "target": 128, "dtype": torch.bfloat16,"use_scaler": False,
            0: {"mb": 8, "use_ckpt": False, "sl": 1024},
            1: {"mb": 2, "use_ckpt": False, "sl": 2048},
            2: {"mb": 1, "use_ckpt": True, "sl": 4096},
        },
        "v6e-1": {
            "ws": 1, "target": 128, "dtype": torch.bfloat16,"use_scaler": False,
            0: {"mb": 16, "use_ckpt": False, "sl": 1024},
            1: {"mb": 4, "use_ckpt": False, "sl": 2048},
            2: {"mb": 2, "use_ckpt": False, "sl": 4096},
        },
        "t4*2": {
            "ws": 2, "target": 128, "dtype": torch.float16, "use_scaler": True,
            0: {"mb": 4, "use_ckpt": False, "sl": 1024},
            1: {"mb": 1, "use_ckpt": True, "sl": 2048},
            2: {"mb": 1, "use_ckpt": True, "sl": 4096},
        },
        "t4": {
            "ws": 1, "target": 128, "dtype": torch.float16, "use_scaler": True,
            0: {"mb": 4, "use_ckpt": False, "sl": 1024},
            1: {"mb": 1, "use_ckpt": True, "sl": 2048},
            2: {"mb": 1, "use_ckpt": True, "sl": 4096},
        },
        "g4": {
            "ws": 1, "target": 128, "dtype": torch.float16, "use_scaler": True,
            0: {"mb": 32, "use_ckpt": False, "sl": 1024},
            1: {"mb": 1, "use_ckpt": True, "sl": 2048},
            2: {"mb": 1, "use_ckpt": True, "sl": 4096},
        },
        "l4": {
            "ws": 1, "target": 128, "dtype": torch.float16, "use_scaler": True,
            0: {"mb": 8, "use_ckpt": False, "sl": 1024},
            1: {"mb": 2, "use_ckpt": True, "sl": 2048},
            2: {"mb": 1, "use_ckpt": True, "sl": 4096},
        },
        "p100": {
            "ws": 1, "target": 128, "dtype": torch.float16, "use_scaler": True,
            0: {"mb": 8, "use_ckpt": True, "sl": 1024},
            1: {"mb": 1, "use_ckpt": True, "sl": 2048},
            2: {"mb": 1, "use_ckpt": True, "sl": 4096},
        },
        "a100": {
            "ws": 1, "target": 128, "dtype": torch.bfloat16,"use_scaler": False,
            0: {"mb": 32, "use_ckpt": False, "sl": 1024},
            1: {"mb": 4, "use_ckpt": True, "sl": 2048},
            2: {"mb": 1, "use_ckpt": True, "sl": 4096},
        },
        "h100": {
            "ws": 1, "target": 128, "dtype": torch.bfloat16,"use_scaler": False,
            0: {"mb": 32, "use_ckpt": False, "sl": 1024},
            1: {"mb": 4, "use_ckpt": True, "sl": 2048},
            2: {"mb": 1, "use_ckpt": True, "sl": 4096},
        },
        "cpu": {
            "ws": 1, "target": 128, "dtype": torch.bfloat16,"use_scaler": False,
            0: {"mb": 1, "use_ckpt": False, "sl": 1024},
            1: {"mb": 1, "use_ckpt": False, "sl": 2048},
            2: {"mb": 1, "use_ckpt": False, "sl": 4096},
        }
    }

    hardware_string: str = "v6e-1 tpu"
    # hardware_string: str = "t4*2 gpu"
    hf_token: str = ""


    # These will be overwritten when we step thru the curriculum, but just place_holders for now
    world_size: int = 1
    target_gbs: int = 128
    dtype: torch.dtype = torch.float16
    use_scaler: bool = True
    batch_size: int = 16
    grad_accum_steps: int = 1
    hardware_profile: Dict[Union[str, int], Any] = field(
        default_factory=lambda: HardwareConfig.HARDWARE_PROFILES["cpu"]
    )
    device_type: str = "cpu"
    validation_step_num: int = 50 # or until exhaustion


# ==================================================
# MLMDataConfig
# Sets dataset repo, curriculum levels, and MLM parameters
# Static (Remains the same)
# ==================================================

@dataclass
class MLMDataConfig:
    data_repo_id: str = "JamesResearch1216/HELM-Easiness-Data-10B-Labeled-v6"
    curriculum: bool = True
    curriculum_subset_names: List[str] = field(
        default_factory=lambda: ["seq_1024", "seq_2048", "seq_4096"]
    )
    curriculum_parquet_start_index: List[int] = field(
        default_factory=lambda: [0, 72, 92]
    )

    train_split: str = "train"
    validation_split: str = "validation"
    tokenizer_name: str = "answerdotai/ModernBERT-base"
    mlm_probability: float = 0.3
    mlm_use_span_masking: bool = True
    mlm_span_length: int = 3
    # Format String without the f
    glob_pattern: str = "data/{subset_name}/{split}-*.parquet"
    # Will the model's architecture use the easiness labeler
    use_easiness: bool = True
    # Stopping index so the model will stop training when it hits this index
    parquet_stop_index: int = 10


# ==================================================
# CheckpointConfig
# Sets Checkpoint related fields
# Static (Remains the same)
# ==================================================

@dataclass
class CheckpointConfig:
    model_repo_id: str = "JamesResearch1216/phase06v7-no-perm"
    wandb_entity: str = "jhui16-university-of-maryland"
    wandb_project: str = "HELM-v1-10B-Run"
    wandb_name: str = "phase06v7"
    hf_token: str = ""
    wandb_key: str = ""
    use_wandb: bool = True

    # How to use interval_dict:
    #  - Let k_i be the ith key in interval_dict
    #  - Let v_i be the ith value in interval_dict
    #  - For step k_i to k_i+1, save a checkpoint every v_i steps
    #  - After the last k_i, save every v_i for the rest of the duration
    interval_dict: Dict[int, int] = field(
        default_factory=lambda: {0: 100, 1000: 200, 5000: 500}
    )
    # True: let step 0 = latest_step
    # False: let step 0 = 0
    start_from_global: bool = True



class MLMDataStrategy:

    # Initialize (Different for each hardware)
    def __init__(self, rank = 0, world_size = 1, is_tpu = False, config: Optional[MLMDataConfig] = None, hf_token=None):
        self.rank = rank
        self.world_size = world_size
        self.is_tpu = is_tpu
        self.config = config
        self.hf_token = hf_token

    # Input:
    # - index number i
    # - delete_prev_parquet_request flag
    # Output:
    # - local file_path name for dataset shard
    # - # of total rows in the shard
    # Load the ith parquet into runtime
    def download_parquet(self, is_train: bool, index = 0):

        # Finding Correct curriculum
        curriculum_level = 0
        for level in range(0, len(self.config.curriculum_parquet_start_index)):
            if (index >= self.config.curriculum_parquet_start_index[level]):
                curriculum_level = level

        # Set up dataset types
        dataset_type = "train" if is_train else "validation"

        # Fix indices for validation due to my weird validation parquet naming
        # the plus 1 because 1 indexed
        index = index if is_train else curriculum_level

        # Get parquet file path
        parquet_file_path = f"data/{self.config.curriculum_subset_names[curriculum_level]}/{dataset_type}-{index:05d}.parquet"

        local_storage_dir = "./local_parquet_shards"
        os.makedirs(local_storage_dir, exist_ok=True)

        # Form file path that potientially is already in file_path
        local_path = os.path.join(local_storage_dir, parquet_file_path)
        if os.path.exists(local_path):
            try:
                parquet_metadata = pq.read_metadata(parquet_file_path)
                num_rows = parquet_metadata.num_rows
                return local_path, num_rows, curriculum_level
            except Exception as e:
                if self.rank == 0:
                    print(f"Error when trying to access {local_path}: {e}. Redownloading instead")

        # Download
        try:
            parquet_file_path = hf_hub_download(
                repo_id = self.config.data_repo_id,
                filename = parquet_file_path,
                repo_type = "dataset",
                token = self.hf_token,
                local_dir = local_storage_dir,
                local_dir_use_symlinks = False
            )

            # Get # of rows
            parquet_metadata = pq.read_metadata(parquet_file_path)
            num_rows = parquet_metadata.num_rows

            return parquet_file_path, num_rows, curriculum_level

        except Exception as e:
            print(f"Failed to download {parquet_file_path}: {e}")
            return "", 0, 0

    # Delete Parquet
    def delete_parquet(self, parquet_file_path: str):
        try:
            if parquet_file_path and os.path.exists(parquet_file_path):
                os.remove(parquet_file_path)
            else:
                print(f"File not found for deletion: {parquet_file_path}")
                return -1
        except Exception as e:
            print(f"Failed to delete {parquet_file_path}: {e}")
            return -1



    # Create get_mlm_data_loader function
    # Note: This only bascially works with the dataset created by prepare_data.py
    def get_mlm_data_loader(
        self,
        parquet_file_path: str,
        collate_fn = None,
        skip_rows = 0,
        batch_size = 1,
        parquet_index = 1,
        is_train = True,
        ):

        # Get HF Dataset obj from parquet
        dataset = Dataset.from_parquet(path_or_paths = parquet_file_path, keep_in_memory = True)

        # Shuffle after you shard
        # Make sure you set a seed to ensure the I don't use the same data again
        # + index to keep this more random but predictable for reproductibility
        dataset = dataset.shuffle(seed = 67 + parquet_index)

        # Skip examples after you shuffle based on the specific seed:
        if skip_rows > 0 and is_train:

            # If skip rows is somehow over the dataset, return an empty dataset
            if skip_rows >= len(dataset):
                dataset = dataset.select(range(0))   # empty
            # Else skip skip_rows and grab remaining data
            else:
                dataset = dataset.select(range(skip_rows, len(dataset)))

        # If we're Parallel Processing, Shard the Data so that each device gets a different slice of data
        if self.world_size > 1 and is_train:
            dataset = dataset.shard(num_shards = self.world_size, index = self.rank)

        # Dataloader
        data_loader = DataLoader(
            dataset,
            batch_size = batch_size,
            num_workers = 0,
            drop_last = True,
            pin_memory = False,
            collate_fn = collate_fn
        )

        # Return data_loader
        return data_loader



class HardwareDriver:

    # Initialize Hardware
    def __init__(self, hw_config: HardwareConfig, data_config: MLMDataConfig, ckpt_config: CheckpointConfig):
        self.hw_config = hw_config
        self.data_config = data_config
        self.ckpt_config = ckpt_config
        # Call _parse_hardware here
        self.hw_config.hardware_profile = self._parse_hardware()


    # Parse hardware based on the curriculum level
    def _parse_hardware(self):
        # get hardware_string (with formatting)
        hardware_string = self.hw_config.hardware_string.lower().replace(" ", "")

        # Ensure that the hardware_string is valid
        profile = None
        for key in self.hw_config.HARDWARE_PROFILES:
            if (key in hardware_string):
                profile = self.hw_config.HARDWARE_PROFILES[key]
                break

        # Default to cpu if none were matching
        if not profile:
            profile = self.hw_config.HARDWARE_PROFILES["cpu"]
            warnings.warn("⚠️ hardware_string did not match any in HARDWARE_PROFILES. Using \"cpu\"", UserWarning)

        # Add attributes to config common to all curriculum levels
        #   - world_size
        #   - targt_gbs (global batch size)
        #   - dtype (data type)
        #   - use_scaler
        self.hw_config.world_size = profile["ws"]
        self.hw_config.target_gbs = profile["target"]
        self.hw_config.dtype = profile["dtype"]
        self.hw_config.use_scaler = profile["use_scaler"]

        # Get device type (save to hw_config)
        if "tpu" in hardware_string:
            self.hw_config.device_type = "tpu"
        elif any(x in hardware_string for x in ["gpu", "cuda", "a100", "p100", "h100", "t4", "l4"]):
            self.hw_config.device_type = "cuda"
        else:
            self.hw_config.device_type = "cpu"

        # Return the profile to use in train_worker
        return profile

    # Launch function: Spawn all the workers and make them run the worker_function
    def launch(self, worker_fn):

        # Define these for convience
        world_size = self.hw_config.world_size
        device = self.hw_config.device_type

        # If parallel processing
        if world_size > 1:
            if device == "tpu":
                import torch_xla.distributed.xla_multiprocessing as xmp
                xmp.spawn(worker_fn, args=(self.hw_config, self.data_config, self.ckpt_config), start_method='spawn')
            elif device == "cuda":
                import random
                import torch.multiprocessing as mp

                # Set up Multi-GPU network
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = str(random.randint(10001, 19999))


                mp.spawn(worker_fn, args=(self.hw_config, self.data_config, self.ckpt_config), nprocs=world_size)
        else:
            # Single Device Execution (Rank 0)
            worker_fn(0, self.hw_config, self.data_config, self.ckpt_config)



class CheckpointDriver:

    # Initialize Checkpoint Driver
    def __init__(self, hw_config: HardwareConfig, data_config: MLMDataConfig, ckpt_config: CheckpointConfig, rank: int, world_size: int):
        self.hw_config = hw_config
        self.data_config = data_config
        self.ckpt_config = ckpt_config
        self.rank = rank
        self.world_size = world_size
        self.api = HfApi(token=self.ckpt_config.hf_token)
        self.actual_resume_step = None
        self.total_rows_dict = None
        self.easiness_dict = None
        self.use_easiness = self.data_config.use_easiness

        # Just print to ensure shit is moving
        if rank == 0:
            print("⏳ Loading Checkpoint Driver...")

        self.training_state = self.get_training_state_from_hub()


    # Smart Barrier (Rendezvous) to prevent data races & ensure all devices make it to certain step
    def _smart_barrier(self, name="barrier"):
        if self.hw_config.world_size <= 1:
            return  # No synchronization needed for single device

        if self.hw_config.device_type == "tpu":
            import torch_xla.core.xla_model as xm
            xm.rendezvous(name)
        elif self.hw_config.device_type == "cuda":
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()

    # Get the number of rows
    def _get_total_rows(self):
        from huggingface_hub import HfFileSystem
        import pyarrow.parquet as pq

        # "all" = All curriculums
        #  0 = 0th level curriculum
        #  1 = 1st level curriculum
        # etc...
        total_rows_dict = {
            "all" : 0
        }

        # Initialize HfFileSystem Object
        fs = HfFileSystem(token = self.ckpt_config.hf_token)
        repo_id = self.data_config.data_repo_id

        # Get total num rows for lr_scheduler
        for level, subset_name in enumerate(self.data_config.curriculum_subset_names):
            total_rows_dict[str(level)] = 0

            try:
                pattern = self.data_config.glob_pattern.format(
                    subset_name = subset_name,
                    split = self.data_config.train_split
                )
                parquet_files = fs.glob(f"datasets/{repo_id}/{pattern}")

                for file_path in parquet_files:

                    # binary read mode to get metadata
                    with fs.open(file_path, "rb") as f:
                        # Get metadata
                        metadata = pq.read_metadata(f)
                        # Get num_rows attr.
                        num_rows = metadata.num_rows

                        total_rows_dict[str(level)] += num_rows
                        total_rows_dict["all"] += num_rows

            except Exception as e:
                if self.rank == 0:
                    print(f"💀 Error reading metadata for {subset_name}: {e}")

        if self.rank == 0:
            for k, v in total_rows_dict.items():
                label = "ALL" if k == "all" else self.data_config.curriculum_subset_names[int(k)]
                print(f"  📊 {label}: {v:,} rows")

        return total_rows_dict

    # Only use this for easiness
    def _compute_easiness_breakpoints(self, column="easiness_score", n_breakpoints=101, max_files=5):
        from huggingface_hub import HfFileSystem
        import numpy as np

        fs = HfFileSystem(token=self.ckpt_config.hf_token)
        repo_id = self.data_config.data_repo_id
        local_dir = "./local_parquet_shards"
        os.makedirs(local_dir, exist_ok=True)

        easiness_dict = None
        all_vals = []

        # For each curriculum:
        for level, subset_name in enumerate(self.data_config.curriculum_subset_names):

            try:

                # Take all the file_paths for the parquets used to calculate the break-points easiness distribution
                pattern = self.data_config.glob_pattern.format(
                    subset_name = subset_name,
                    split = self.data_config.train_split
                )

                parquet_files = sorted(fs.glob(f"datasets/{repo_id}/{pattern}"))

                parquets_sampled = parquet_files[:max_files]

                if self.rank == 0:
                    print(f"  📥 {subset_name}: sampling {len(parquets_sampled)}/{len(parquet_files)} ...")

                level_vals = []
                downloaded_paths = []

                # Go through all the curriculum's sampled parquets
                for hf_path in parquets_sampled:

                    # hf_path looks like "datasets/user/repo/data/seq_1024/train-00000.parquet"
                    # Extract the repo-relative filename for hf_hub_download
                    # Strip the "datasets/{repo_id}/" prefix
                    prefix = f"datasets/{repo_id}/"
                    filename = hf_path[len(prefix):] if hf_path.startswith(prefix) else hf_path

                    # Download parquet
                    try:
                        local_path = hf_hub_download(
                            repo_id=repo_id,
                            filename=filename,
                            repo_type="dataset",
                            token=self.ckpt_config.hf_token,
                            local_dir=local_dir,
                            local_dir_use_symlinks=False
                        )
                        downloaded_paths.append(local_path)

                        # Read ONLY the easiness column (fast, low memory)
                        col_data = pq.read_table(
                            local_path, columns=[column]
                        ).column(column).to_numpy(zero_copy_only=False)
                        level_vals.append(np.asarray(col_data, dtype=np.float64))

                    except Exception as e:
                        if self.rank == 0:
                            print(f"    ⚠️ Failed to read {filename}: {e}")

                    # Delete the parquet
                    for path in downloaded_paths:
                        try:
                            if os.path.exists(path):
                                os.remove(path)
                        except Exception:
                            pass

                    # Collect the values
                    if level_vals:
                        level_concat = np.concatenate(level_vals)
                        level_concat = level_concat[np.isfinite(level_concat)]
                        all_vals.append(level_concat)

            except Exception as e:
                if self.rank == 0:
                    print(f"  💀 Error processing {subset_name}: {e}")

        # Compute GLOBAL breakpoints (combining all subsets)
        if all_vals:

            global_concat = np.concatenate(all_vals)
            easiness_dict = self._compute_breakpoint_payload(
                global_concat, n_breakpoints, column
            )
            if self.rank == 0:
                g = easiness_dict
                print(f"  🌍 Global easiness: n={g['n']:,} median={g['median']:.3f} "
                    f"mean={g['mean']:.3f} frac>0.5={g['frac_above_0.5']:.2f}")
        else:
            if self.rank == 0:
                print("  ⚠️ No easiness data found — using logistic fallback in model")
            easiness_dict = None

        return easiness_dict

    @staticmethod
    def _compute_breakpoint_payload(values, n_breakpoints, column):
        """Helper: given a numpy array of easiness values, return the breakpoint dict."""
        import numpy as np
        breakpoints = np.quantile(values, np.linspace(0.0, 1.0, n_breakpoints)).tolist()
        return {
            "breakpoints": breakpoints,
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "frac_above_0.5": float(np.mean(values > 0.5)),
            "n": int(values.size),
            "column": column,
        }



    # Initialize new checkpoint dictionary
    def _init_new_training_state(self):

        if self.rank == 0:
            print("🔢 Computing total rows per curriculum level...")
        self.total_rows_dict = self._get_total_rows()

        training_state_dict = {
            "checkpoints": {},
            "session": 0,
            "total_rows_dict" : self.total_rows_dict,
        }

        if self.use_easiness:
            if self.rank == 0:
                print("📐 Computing easiness breakpoints from sample...")
            training_state_dict["easiness_dict"] = self._compute_easiness_breakpoints()

        formatted_json_str = json.dumps(training_state_dict, indent = 4)
        json_bytes = formatted_json_str.encode('utf-8')
        fileobj = io.BytesIO(json_bytes)
        try:
            self.api.upload_file(
                path_or_fileobj = fileobj,
                path_in_repo = "training_state.json",
                repo_id = self.ckpt_config.model_repo_id,
                repo_type = "model",
                token = self.ckpt_config.hf_token
            )
        except Exception as e:
            print(f"Failed to push Init State {e}")

        return training_state_dict


    # Ensure that if checkpoints are deleted but appear
    def _deletion_status_updates(self, training_state):

        # Try to take the repo file's file paths
        repo_files = None
        try:
            repo_files = list(self.api.list_repo_files(repo_id = self.ckpt_config.model_repo_id))
        except RepositoryNotFoundError:
            raise RepositoryNotFoundError(f"❌ Repo: \"{self.ckpt_config.model_repo_id}\" was not found when trying to update deletion status")


        # Loop Through every checkpoint and switch the status if necessary
        for ckpt_vals in training_state["checkpoints"].values():
            if ckpt_vals["file"] == "" or not ckpt_vals["file"] in repo_files:
                ckpt_vals["status"] = "deleted"
                ckpt_vals["file"] = ""


        # Return training_state
        return training_state


    # Get training_state.json from the HF repo
    def get_training_state_from_hub(self, filename = "training_state.json"):

        # Let only rank 0 to run this (prevent mass API calls)
        if self.world_size > 1:
            self._smart_barrier("state_fetch_start")

        # Let rank == 0 load the .json
        if self.rank == 0:
            # Attempt to pull the training_state.json from hub
            try:
                # Try Downlaoding
                path = hf_hub_download(
                    repo_id=self.ckpt_config.model_repo_id,
                    filename = filename,
                    repo_type = "model",
                    token = self.ckpt_config.hf_token
                )

                with open(path, "r") as f:
                    training_state = json.load(f)

                # Successfully loaded
                print(f"✅ {filename} loaded successfully from {self.ckpt_config.model_repo_id}")
                training_state = self._deletion_status_updates(training_state)

            # If the repo doesn't exist, make the repo
            except RepositoryNotFoundError:
                # Print Error Statements
                print(f"⚠️ Repo: \"{self.ckpt_config.model_repo_id}\" was not found")
                print(f"🏗️ Creating Repo: {self.ckpt_config.model_repo_id}")

                # Create Repo
                create_repo(
                    repo_id = self.ckpt_config.model_repo_id,
                    token = self.ckpt_config.hf_token,
                    repo_type = "model",
                    private = False,
                    exist_ok = False,
                )

                # Make new training_state dict
                training_state = self._init_new_training_state()

            # The repo exists, but it's empty or doesn't have the state file yet
            except EntryNotFoundError:
                # Print info
                print(f"⚠️ {self.ckpt_config.model_repo_id} exists, but no {filename} found. Starting fresh.")
                training_state = self._init_new_training_state()

            # Unknown Error
            except Exception as e:
                # Catch-all for network timeouts, corrupted JSON, etc.
                print(f"❌ An unexpected error occurred: {e}. Starting from token zero.")
                training_state = self._init_new_training_state()

            # Dump the training_state from rank 0 into .json
            with open("local_training_state.json", "w") as f:
                json.dump(training_state, f)

        # Once rank 0 finishes, end the barrier
        if self.world_size > 1:
            self._smart_barrier("state_fetch_end")

        # Then every rank (including) loads the dict from "local_training_state.json"
        with open("local_training_state.json", "r") as f:
            final_state_dict = json.load(f)

        # Wait for EVERY rank to finish reading the file
        if self.world_size > 1:
            self._smart_barrier("state_read_complete")

        # Then Delete
        if self.rank == 0:
            import os
            if os.path.exists("local_training_state.json"):
                os.remove("local_training_state.json")

        # All Return the same dict
        return final_state_dict

    # Check to see if a checkpoint should be uploaded
    def check_upload_condition(self, curr_global_step):
        # Subtract offset if start_from_global (it treated step 0 = last checkpoint's step value)
        if not self.ckpt_config.start_from_global:
            curr_global_step -= self.actual_resume_step
        if curr_global_step <=0:
            return False

        # Save which interval we will use to calculate if we need to upload
        active_interval = None

        # Iterate through the sorted keys
        for threshold in sorted(self.ckpt_config.interval_dict.keys()):
            # If our curr_global_step is bigger than threshold, save it's value
            if curr_global_step >= threshold:
                active_interval = self.ckpt_config.interval_dict[threshold]
            else:
                # else we break since we haven't to this threshold yet
                break

        # If dictionary was empty (no checkpointing)
        if active_interval is None:
            return False

        # return whether the current step is a perfect multiple of the active_interval
        return (curr_global_step % active_interval == 0)

    # Resume Training: resume from the correct checkpoint
    # Pass the model and optimizer by reference to be initialized
    # Returns:
    # Checkpoint Entry Dictionary Snapshot if available
    # Dicionary with a bunch of 0s if all checkpoints were deleted or starting fresh or the actual snapshot dictionary
    # the actual resume step and session number
    def resume_training(self, model, optimizer):
        # Lazy Load torch to get correct version
        import torch

        # Keep track of all the valid steps
        valid_steps = []
        for step, data in self.training_state["checkpoints"].items():
            if data["status"] != "deleted":
                valid_steps.append(int(step))

        # Print messege and return 0 if its brand new
        if not valid_steps:
            if self.rank == 0:
                print("According to the training_state, every single checkpoint is invalid or deleted. Starting from ground 0")
            # return 0,0
            self.actual_resume_step = 0
            return {
                "hardware": self.hw_config.hardware_string,
                "curriculum_level": 0,
                "rows_processed_at_curr_level": 0,
                "total_tokens_processed_global": 0,
                "total_rows_processed_global": 0,
                "run_id": wandb.util.generate_id() if self.ckpt_config.use_wandb else "",
                "parquet_index": 1,
                "total_rows_processed_parquet": 0
            }, 0, 0, None

        # Get Actual valid resume step (e.g. I deleted the most recent version but it still says otherwise)
        actual_resume_step = max(valid_steps)
        self.actual_resume_step = actual_resume_step
        ckpt_entry = self.training_state["checkpoints"][str(actual_resume_step)]
        filename = ckpt_entry["file"]

        # Barrier
        if self.world_size > 1:
            self._smart_barrier("weight_download_start")

        # Only let rank 0 start downloading (the others will download from the runtime local disk):
        if self.rank == 0:
            print(f"Downloading {filename} from Hub...")
            try:
                hf_hub_download(
                    repo_id = self.ckpt_config.model_repo_id,
                    filename = filename,
                    repo_type = "model",
                    token = self.ckpt_config.hf_token,
                    local_dir = "."
                )
            except Exception as e:
                raise RuntimeError(f"Critical HF Download Failure for {filename}: {e}")

        # Barrier
        if self.world_size > 1:
            self._smart_barrier("weight_download_end")

        # Now that the model has been downloaded onto the runtime local disk, let each device download it
        # All ranks load the weights from the local file
        try:

            # Load the checkpoint
            pt_path = os.path.join(".", filename)
            ckpt = torch.load(pt_path, map_location='cpu', weights_only=False)

            # Get the model state
            model_state = ckpt['model_state']

            # GPUs and TPUs might add module. or not have it at all
            # Add or subtract this to maintain hardware compatibility
            new_state_dict = {}
            for k, v in model_state.items():
                if k.startswith('module.') and not hasattr(model, 'module'):
                    new_state_dict[k[7:]] = v
                elif not k.startswith('module.') and hasattr(model, 'module'):
                    new_state_dict[f'module.{k}'] = v
                else:
                    new_state_dict[k] = v

            # Load the model into dictionary
            model.load_state_dict(new_state_dict, strict=False)

            # Load the optmizer
            if optimizer and 'optimizer_state' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state'])

            # Get scheduler state
            scheduler_state = ckpt.get('scheduler_state', None)

            # print success
            if self.rank == 0:
                print(f"Successfully loaded model and optimizer from Step {actual_resume_step}!")

            # Update the training_state session num
            self.training_state["session"] +=1

            # Sync up
            if self.world_size > 1:
                self._smart_barrier("model_optimizer_loaded")

            # Let rank = 0 delete the last checkpoint to resume
            if self.rank == 0:
                try:
                    os.remove(pt_path)
                except Exception as e:
                    printf("Loaded the model, but couldn't deleted intial checkpoint")

            # Just return the ckpt_entry; Extract the values later
            return ckpt_entry, actual_resume_step, self.training_state["session"], scheduler_state

        except Exception as e:
            raise RuntimeError(f"Critical Weight Loading Failure: {e}")

    # Saves the model to a .pt file
    # Makes checkpoint entry for training_state
    def save_checkpoint(self,
                        model,
                        optimizer,
                        scheduler,
                        global_step: int,
                        hardware_string: str,
                        metrics: dict,
                        is_tpu: bool,
                        curriculum_level: int,
                        total_tokens_processed_global: int,
                        total_rows_processed_global: int,
                        rows_processed_at_curr_level: int,
                        parquet_index: int,
                        total_rows_processed_parquet: int,
                        run_id = ""
                        ):

        # Lazy Load Torch
        import torch

        # Save filename
        step_str = str(global_step)
        filename = f"checkpoint-{global_step:06d}.pt"

        # If more than 1 worker, start barrier
        if self.world_size > 1:
            self._smart_barrier("save_start")

        if self.rank == 0:
            print(f"Saving model weights to {filename}...")

        # Ensure to use module or not to ensure compatibility
        save_dict = {
            "model_state" : model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            "optimizer_state" : optimizer.state_dict(),
            "scheduler_state" : scheduler.state_dict()
        }

        # Save using TPU or GPU .save()
        if is_tpu:
            import torch_xla.core.xla_model as xm
            xm.save(save_dict, filename)
        else:
            torch.save(save_dict, filename)

        # If more than 1 worker, end barrier
        if self.world_size > 1:
            self._smart_barrier("save_weights_end")

        # Update training_state (add checkpoint entry + update metadata)
        # Only let rank 0 change the state of the UPLOAD_REQUEST.json to ping the sidecar
        if self.rank == 0:

            # Ensure that the all latest tags get removed
            for step, data in self.training_state['checkpoints'].items():
                if data["status"] == "latest":
                    data["status"] = "history"

            # Create new checkpoint entry
            self.training_state["checkpoints"][step_str] = {
                "status": "latest",
                "file": filename,
                "hardware": hardware_string,
                "curriculum_level": curriculum_level,
                "rows_processed_at_curr_level": rows_processed_at_curr_level,
                "total_rows_processed_global": total_rows_processed_global,
                "total_tokens_processed_global": total_tokens_processed_global,
                "metrics": metrics,
                "run_id": run_id,
                "parquet_index": parquet_index,
                "total_rows_processed_parquet": total_rows_processed_parquet
            }

            # Ping sidecar by updating UPLOAD_REQUEST.json
            request_data = {
                "file_to_upload": filename,
                "step": global_step,
                "training_state_snapshot": self.training_state
            }

            with open(f"UPLOAD_REQUEST_{global_step}.json.tmp", "w") as f:
                json.dump(request_data, f)
            os.rename(f"UPLOAD_REQUEST_{global_step}.json.tmp", f"UPLOAD_REQUEST_{global_step}.json")

            # Print some bs idk lol
            print(f"Saved weights to local disk + updated training_state.json. Pinging Sidecar for Step {global_step}")

        # If more than 1 worker make sure other ranks wait for rank 0
        if self.world_size > 1:
            self._smart_barrier("save_training_state_end")



# Define Custom Learning Rate Scheduler
def get_curr_scheduler(optimizer, total_curr_level_steps, curr_max_lr, curr_min_lr, base_lr, warmup_steps = 0):

    # When given a lr_lambda function, the function multiplies this value by the base_lr
    # We want the real curr_max_lr / curr_min_lr, but we must divide before to cancel the multiplication
    max_mult = curr_max_lr / base_lr
    min_mult = curr_min_lr / base_lr

    def lr_lambda(current_step):
        # Warmup (only for first phase)
        if current_step < warmup_steps:
            return min_mult + (max_mult - min_mult) * (current_step / max(1, warmup_steps))

        # progress is a number between 0-1
        progress = (current_step - warmup_steps) / max (1, total_curr_level_steps - warmup_steps)

        # Make sure it doesn't go beyond 1
        progress = min (1.0, progress)

        # prog(0) = max_mult, prog(1) = min_mult
        return min_mult + 0.5 * (max_mult - min_mult) * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)



# Driver to Log Telemtry to WandB
class TelemetryDriver:

    # Initialize Hardware
    def __init__(self, rank, run_id, model_config, ckpt_config: CheckpointConfig, resume_step = 0, global_tokens_processed = 0):

        # Get Rank
        self.rank = rank

        # Get Checkpoint
        self.ckpt_config = ckpt_config

        # Get modeL_config
        self.model_config = model_config

        # Save new run object
        self.run = None

        # Save run type
        # "init"
        # "resume_from"
        # "fork_from"
        self.run_type = None

        # Initialize and resume data logging
        if self.rank == 0 and ckpt_config.use_wandb:

            # Starting Fresh
            if (resume_step == 0):
                self.run = wandb.init(
                    entity = ckpt_config.wandb_entity,
                    project = ckpt_config.wandb_project,
                    name = ckpt_config.wandb_name,
                    id = run_id,
                    config = vars(model_config)
                )
                self.run_type = "init"

            # If continuing and resume_from_work
            elif DOES_RESUME_FROM_WORK:
                self.run = wandb.init(
                        entity = ckpt_config.wandb_entity,
                        project = ckpt_config.wandb_project,
                        name = ckpt_config.wandb_name,
                        id = run_id,
                        resume_from = f"{run_id}?_step={resume_step}",
                        config = vars(model_config)
                )
                self.run_type = "resume_from"

            # Else use fork_from for better organization
            else:
                self.run = wandb.init(
                        entity = ckpt_config.wandb_entity,
                        project = ckpt_config.wandb_project,
                        name = ckpt_config.wandb_name,
                        fork_from = f"{run_id}?_step={resume_step}",
                        config = vars(model_config)
                )
                self.run_type = "fork_from"

    def _tele_key(self, k):
        # "layer_3_sqk_mean" -> "layer_3/sqk_mean" ; "lm_head_sz_mean" -> "lm_head/sz_mean"
        k = re.sub(r"^(layer_\d+)_", r"\1/", k)
        k = re.sub(r"^(lm_head)_",   r"\1/", k)
        return k

    def _make_heatmap(self, rows, label, cmap, global_step):
        matrix = np.stack(rows)
        fig, ax = plt.subplots(figsize=(10, 8))
        cax = ax.matshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0)
        fig.colorbar(cax, label=label)
        ax.set_xlabel("Elastic Head Index"); ax.set_ylabel("Layer")
        ax.set_title(f"{label} (Step {global_step})")
        ax.set_yticks(range(matrix.shape[0]))
        img = wandb.Image(fig); plt.close(fig)
        return img

    def log_step(self, telemetry_dict, ce_loss, aux_loss, sparsity_loss, total_loss, global_step, is_train=True, global_tokens_processed = None):
        if self.rank != 0 or not self.ckpt_config.use_wandb or self.run is None:
            return
        prefix = "train" if is_train else "validation"
        log_payload = {
            f"{prefix}/ce_loss": ce_loss,
            f"{prefix}/aux_loss": aux_loss,
            f"{prefix}/sparsity_loss": sparsity_loss,
            f"{prefix}/total_loss": total_loss,
        }

        activation_rows, confidence_rows = {}, {}   # optional router heatmap sources

        for key, val in telemetry_dict.items():
            # router heatmap sources: collect per-layer, don't log as plain histograms
            m_fm = re.match(r"layer_(\d+)_flat_mask$", key)
            m_ss = re.match(r"layer_(\d+)_sigmoid_scores$", key)
            if m_fm:
                activation_rows[int(m_fm.group(1))] = val.mean(dim=0).squeeze().numpy()
                continue
            if m_ss:
                confidence_rows[int(m_ss.group(1))] = val.mean(dim=0).squeeze().numpy()
                log_payload[self._tele_key(key) + "_hist"] = wandb.Histogram(val.numpy())
                continue

            # everything else: dispatch purely on type, so ANY architecture just works
            if isinstance(val, torch.Tensor):
                v = val.detach().cpu()
                log_payload[self._tele_key(key)] = (wandb.Histogram(v.numpy())
                                                    if v.numel() > 1 else v.item())
            elif isinstance(val, (int, float)):
                log_payload[self._tele_key(key)] = val
            # unknown types are silently skipped instead of crashing

        if activation_rows:
            rows = [activation_rows[i] for i in sorted(activation_rows)]
            log_payload["router/activation_heatmap"] = self._make_heatmap(rows, "Activation Frequency", "cool", global_step)
        if confidence_rows:
            rows = [confidence_rows[i] for i in sorted(confidence_rows)]
            log_payload["router/confidence_heatmap"] = self._make_heatmap(rows, "Mean Sigmoid Confidence", "winter", global_step)

        log_payload["global_tokens_processed"] = global_tokens_processed
        wandb.log(log_payload, step=global_step)


    # # Log the Data
    # def log_step (self, telemetry_dict, ce_loss, aux_loss, sparsity_loss, total_loss, global_step, is_train = True, global_tokens_processed = None):

    #     # Don't log if rank == 0 or not using wandb
    #     if self.rank != 0 or not self.ckpt_config.use_wandb or self.run is None:
    #         return

    #     if is_train:
    #         log_payload = {
    #             "train/ce_loss": ce_loss,
    #             "train/aux_loss": aux_loss,
    #             "train/sparsity_loss": sparsity_loss,
    #             "train/total_loss": total_loss
    #         }
    #     else:
    #         log_payload = {
    #             "validation/ce_loss": ce_loss,
    #             "validation/aux_loss": aux_loss,
    #             "validation/sparsity_loss": sparsity_loss,
    #             "validation/total_loss": total_loss
    #         }

    #     router_heatmap_data = []
    #     sigmoid_heatmap_data = []

    #     # Iterate through telemetry dict
    #     for i in range (self.model_config.num_hidden_layers):

    #         # Histograms (Full Tensors)
    #         log_payload[f"layer_{i}/sqk_hist"] = wandb.Histogram(telemetry_dict[f"layer_{i}_sqk_hist"].numpy())
    #         log_payload[f"layer_{i}/attn_alpha_hist"] = wandb.Histogram(telemetry_dict[f"layer_{i}_attn_alpha_hist"].numpy())
    #         log_payload[f"layer_{i}/mlp_alpha_hist"] = wandb.Histogram(telemetry_dict[f"layer_{i}_mlp_alpha_hist"].numpy())
    #         log_payload[f"layer_{i}/suv_hist"] = wandb.Histogram(telemetry_dict[f"layer_{i}_suv_hist"].numpy())
    #         log_payload[f"layer_{i}/sigmoid_scores_hist"] = wandb.Histogram(telemetry_dict[f"layer_{i}_sigmoid_scores"].numpy())

    #         # Attention SQK
    #         log_payload[f"layer_{i}/sqk_mean"] = telemetry_dict[f"layer_{i}_sqk_mean"]
    #         log_payload[f"layer_{i}/sqk_std"] = telemetry_dict[f"layer_{i}_sqk_std"]

    #         # Attention Alpha Eigen Learning Rate
    #         log_payload[f"layer_{i}/attn_alpha_mean"] = telemetry_dict[f"layer_{i}_attn_alpha_mean"]
    #         log_payload[f"layer_{i}/attn_alpha_std"] = telemetry_dict[f"layer_{i}_attn_alpha_std"]

    #         # MLP Alpha Eigen Learning Rate
    #         log_payload[f"layer_{i}/mlp_alpha_mean"] = telemetry_dict[f"layer_{i}_mlp_alpha_mean"]
    #         log_payload[f"layer_{i}/mlp_alpha_std"] = telemetry_dict[f"layer_{i}_mlp_alpha_std"]

    #         # Router Metrics
    #         log_payload[f"layer_{i}/elastic_active_ratio"] = telemetry_dict[f"layer_{i}_elastic_head_ratio"]
    #         log_payload[f"layer_{i}/tau"] = telemetry_dict[f"layer_{i}_tau"]
    #         log_payload[f"layer_{i}/l_i_weights"] = wandb.Histogram(telemetry_dict[f"layer_{i}_l_i_weights"].numpy())

    #         # Average the flat_mask across the batch
    #         # [b, num_attention_heads, 1, 1] -> [num_attention_heads]
    #         flat_mask = telemetry_dict[f"layer_{i}_flat_mask"].mean(dim=0).squeeze().numpy()
    #         router_heatmap_data.append(flat_mask)

    #         # Average the sigmoid_scores across the batch
    #         # [b, num_attention_heads, 1, 1] -> [num_attention_heads]
    #         sigmoid_scores = telemetry_dict[f"layer_{i}_sigmoid_scores"].mean(dim=0).squeeze().numpy()
    #         sigmoid_heatmap_data.append(sigmoid_scores)

    #     # LM_head
    #     log_payload["lm_head/sz_mean"] = telemetry_dict["lm_head_sz_mean"]
    #     log_payload["lm_head/sz_std"] = telemetry_dict["lm_head_sz_std"]

    #     log_payload["lm_head/sz_hist"] = wandb.Histogram(telemetry_dict["lm_head_sz_hist"].numpy())

    #     # Generate the 2D Router Heatmap
    #     heatmap_matrix = np.stack(router_heatmap_data)
    #     fig, ax = plt.subplots(figsize=(10, 8))
    #     cax = ax.matshow(heatmap_matrix, cmap="cool", vmin=0.0, vmax=1.0)
    #     fig.colorbar(cax, label="Activation Frequency")
    #     ax.set_xlabel("Elastic Head Index")
    #     ax.set_ylabel("Layer")
    #     ax.set_title(f"Router Head Activation Heatmap (Step {global_step})")
    #     ax.set_yticks(range(self.model_config.num_hidden_layers))

    #     log_payload["router/activation_heatmap"] = wandb.Image(fig)
    #     plt.close(fig)

    #     # Generate the 2D Sigmoid Score Heatmap
    #     sig_matrix = np.stack(sigmoid_heatmap_data)
    #     fig2, ax2 = plt.subplots(figsize=(10, 8))
    #     cax2 = ax2.matshow(sig_matrix, cmap="winter", vmin=0.0, vmax=1.0) # Different colormap to distinguish
    #     fig2.colorbar(cax2, label="Mean Sigmoid Confidence")
    #     ax2.set_xlabel("Elastic Head Index")
    #     ax2.set_ylabel("Layer")
    #     ax2.set_title(f"Router Sigmoid Confidence Heatmap (Step {global_step})")
    #     ax2.set_yticks(range(self.model_config.num_hidden_layers))

    #     log_payload["router/confidence_heatmap"] = wandb.Image(fig2)
    #     plt.close(fig2)

    #     # Add Total Tokens processed
    #     log_payload["global_tokens_processed"] = global_tokens_processed

    #     # Push to W&B
    #     wandb.log(log_payload, step=global_step)



# Function that all devices will run (ran from the launch function right above)
def train_worker(rank, hw_config, data_config, ckpt_config):

    # We need each TPU process to communicate to the main process
    # The best and cheapest way is to create a thread that write a file onto disk
    # This sits outside the training loop so we can detect whether the training loop is hanging
    import threading

    # Start Daemon Thread to write
    heartbeat_stop = threading.Event()

    # Fucntion to write the time
    # If the entire train_worker process dies, this thread dies and fails to write
    # This is how we will detect changes
    def heartbeat_report():
        path = f"/tmp/heartbeat_rank_{rank}.txt"
        # True if even got set, false if timeout expired
        while not heartbeat_stop.wait(timeout = 10):
            try:
                with open(path, "w") as f:
                    f.write(str(time.time()))
            except Exception:
                pass # Ensure training doesn't crash because of this john

    # Start the thread
    heartbeat_thread = threading.Thread(target=heartbeat_report, daemon = True)
    heartbeat_thread.start()

    # Smart Barrier (Rendezvous) to prevent data races & ensure all devices make it to certain step
    def _smart_barrier(self, name="barrier"):
        if hw_config.world_size <= 1:
            return  # No synchronization needed for single device

        if hw_config.device_type == "tpu":
            import torch_xla.core.xla_model as xm
            xm.rendezvous(name)
        elif hw_config.device_type == "cuda":
            import torch.distributed as dist
            if dist.is_initialized():
                dist.barrier()

    # Lazy Load
    import sys
    import traceback
    import os # Add os

    if hw_config.hf_token:
        os.environ["HF_TOKEN"] = hw_config.hf_token

    try:
        # Lazy Load
        import datasets
        datasets.config.TF_AVAILABLE = False
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from transformers import AutoTokenizer
        import SpanMLMCollatorWithEasiness
        from model import HELMConfig, HELMForMaskedLM


        # Default for Data Collator
        is_tpu = False

        # Load correct packages and get device
        if hw_config.device_type == "tpu":
            # Lazy Load even more for TPU
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
            import torch_xla.runtime as xr

            device = xm.xla_device()
            is_tpu = True

            # get real world size just in case
            real_world_size = xr.world_size()
            if hw_config.world_size != real_world_size:
                if rank == 0:
                    print(f"⚠️ CONFIG MISMATCH: Adjusting world size to {real_world_size}")
                hw_config.world_size = real_world_size

        elif hw_config.device_type == "cuda":
            # Set cuda device to torch
            torch.cuda.set_device(rank)
            device = torch.device(f"cuda:{rank}")
            # Initialize Distributed comm framework
            # acts like a rendezvous
            # NVIDIA Collective Communications Library (nccl)
            if hw_config.world_size > 1:
                import torch.distributed as dist
                dist.init_process_group("nccl", rank=rank, world_size=hw_config.world_size)

        else:
            # Default to CPU just in case
            device = torch.device("cpu")


        if rank == 0:
            print(f"Rank 0 is online: {device}.")

        # Define Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            data_config.tokenizer_name, token=hw_config.hf_token
        )

        # Define MLMDataStrategy
        data_strat = MLMDataStrategy(
            rank = rank, world_size = hw_config.world_size, is_tpu = is_tpu,config = data_config, hf_token=hw_config.hf_token
        )

        # Define Checkpoint Driver
        checkpoint_driver = CheckpointDriver(
            hw_config = hw_config, data_config = data_config, ckpt_config = ckpt_config,
            rank = rank, world_size = hw_config.world_size
        )

        # Get the total steps of the data
        num_rows_dict = checkpoint_driver.training_state["total_rows_dict"]
        dataset_total_steps = num_rows_dict["all"] // hw_config.target_gbs

        # Pull easiness breakpoints only if this run uses the easiness labeler.
        # The no-router / no-easiness baseline never populates easiness_dict.
        easiness_kwargs = {}
        if data_config.use_easiness:
            easiness_dict = checkpoint_driver.training_state.get("easiness_dict", None)
            if easiness_dict:
                easiness_kwargs["easiness_cdf_breakpoints"] = easiness_dict["breakpoints"]

        # Initialize Config
        helm_config = HELMConfig(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
            dataset_total_steps = dataset_total_steps,
            **easiness_kwargs
        )

        # Create model and attach to device
        model = HELMForMaskedLM(helm_config).to(device)

        # Require DDP to Wrap the model if using cuda
        if hw_config.device_type == "cuda" and hw_config.world_size > 1:
            from torch.nn.parallel import DistributedDataParallel as DDP
            # model = DDP(model, device_ids=[rank], find_unused_parameters=True)
            # There shouldn't be extra args
            model = DDP(model, device_ids=[rank])

        # Define Optimizer
        optimizer = optim.AdamW(model.parameters(), lr = helm_config.base_lr, weight_decay = helm_config.weight_decay)

        # Zero the gradient
        optimizer.zero_grad()

        # Define CE Loss
        loss_fct = nn.CrossEntropyLoss()
        loss_fct_sum = nn.CrossEntropyLoss(reduction="sum")  # ignore_index defaults to -100
        def chunked_ce(logits, labels, vocab_size, n_chunks=8):
            flat_logits = logits.reshape(-1, vocab_size)
            flat_labels = labels.reshape(-1)
            total = flat_logits.size(0)
            chunk = (total + n_chunks - 1) // n_chunks
            valid = (flat_labels != loss_fct.ignore_index).sum().clamp_min(1)
            loss_sum = flat_logits.new_zeros(())
            for i in range(0, total, chunk):
                loss_sum = loss_sum + loss_fct_sum(
                    flat_logits[i:i + chunk].float(),
                    flat_labels[i:i + chunk],
                )
            return loss_sum / valid

        # Set data type that will be used
        dtype = hw_config.dtype

        # Allow Scaler
        use_scaler = hw_config.use_scaler
        scaler = torch.amp.GradScaler('cuda') if hw_config.device_type == "cuda" and use_scaler else None

        # ========== CHECKPOINT TECHNOLOGICA ==========

        # Loading the model/optimizer returns the most recent, valid / undeleted checkpoint
        ckpt_snapshot, actual_resume_step, session_number, scheduler_state = checkpoint_driver.resume_training(model, optimizer)

        # Extract the values from the ckpt_snapshot
        start_curr_level = ckpt_snapshot["curriculum_level"]
        rows_processed_at_curr_level = ckpt_snapshot["rows_processed_at_curr_level"]
        total_rows_processed_global = ckpt_snapshot["total_rows_processed_global"]
        total_tokens_processed_global = ckpt_snapshot["total_tokens_processed_global"]
        parquet_index = ckpt_snapshot["parquet_index"]
        total_rows_processed_parquet = ckpt_snapshot["total_rows_processed_parquet"]

        # Set the global step to where we left off from the previous checkpoint
        global_step = actual_resume_step

        # If starting fresh initialize weights
        if actual_resume_step == 0:
            # Use .module to access the original HELMForMaskedLM if wrapped in DDP
            unwrapped_model = model.module if hasattr(model, "module") else model
            unwrapped_model.apply(unwrapped_model._init_weights)

        # Extract run_id (for wandb logging)
        run_id = ckpt_snapshot["run_id"]

        # If we aren't using resume_from, have incremental session number names
        if not DOES_RESUME_FROM_WORK:
            ckpt_config.wandb_name = f"{ckpt_config.wandb_name}-{session_number:05}"

        # Initialize TelemetryDriver
        telemetry_driver = TelemetryDriver(
            rank = rank,
            run_id = run_id,
            ckpt_config = ckpt_config,
            model_config = helm_config,
            resume_step = actual_resume_step
        )

        # ========== CURRICULUM LOOP ==========
        # Curriculum Outer Loop (starting from the current curriculum):
        for level in range(start_curr_level,len(data_config.curriculum_subset_names)):

            # --- ADDED PARQUET STOP LOGIC ---
            if parquet_index >= data_config.parquet_stop_index:
                if rank == 0:
                    print(f"🛑 Reached parquet stop index ({data_config.parquet_stop_index}). Stopping curriculum.")
                break
            # --------------------------------

            total_curr_level_steps = checkpoint_driver.training_state["total_rows_dict"][str(level)] // hw_config.target_gbs

            # Reset optimizer's internal LR (each curr_level turns it -> min_lr, so reset is required)
            if not (scheduler_state is not None and level == start_curr_level):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = helm_config.base_lr

            if level == 0:
                warmup_steps = int(total_curr_level_steps * 0.01)
                scheduler = get_curr_scheduler(
                    optimizer, total_curr_level_steps, helm_config.base_lr,
                    helm_config.min_lr, helm_config.base_lr, warmup_steps
                )
            elif level == 1:
                scheduler = get_curr_scheduler(
                    optimizer, total_curr_level_steps, helm_config.base_lr * 0.35,
                    helm_config.min_lr, helm_config.base_lr, 0
                )
            elif level == 2:
                scheduler = get_curr_scheduler(
                    optimizer, total_curr_level_steps, helm_config.base_lr * 0.18,
                    helm_config.min_lr, helm_config.base_lr, 0
                )

            # If resuming, overwrite the scheduler's internal state
            # (restores step counter so cosine decay continues from where it left off)
            if scheduler_state is not None and level == start_curr_level:
                scheduler.load_state_dict(scheduler_state)
                scheduler_state = None


            # Sync up devices
            if hw_config.world_size > 1:
                _smart_barrier("load_scheduler")

            # Set Model to Training Mode
            model.train()

            # Get Profile Level
            level_profile = hw_config.hardware_profile[level]

            # Get micro batch size (mb) and use gradient checkpointing (use_ckpt)
            hw_config.batch_size = level_profile["mb"]

            # CRITICAL: Unwrap the model first to handle DDP (GPUs) vs Raw (TPUs)
            unwrap_model = model.module if hasattr(model, "module") else model
            # Change the Model Configs using the safely unwrapped model
            unwrap_model.config.use_ckpt = level_profile["use_ckpt"]
            unwrap_model.model.use_ckpt = level_profile["use_ckpt"] # HELMModel caches this

            # Save seq_len somewhere just in case if we need to use it
            seq_len = level_profile["sl"]

            # Calculate grad_accum_steps
            hw_config.grad_accum_steps = max(1, hw_config.target_gbs // (hw_config.batch_size * hw_config.world_size))

            # Define the Collator
            collator = SpanMLMCollatorWithEasiness.SpanMLMCollatorWithEasiness(
                config = data_config, tokenizer = tokenizer
            )

            validation_file_path = ""
            if rank == 0:
                # Load Validation parquet for current curriculum level
                validation_file_path, val_parquet_num_rows, parquet_curr_level = data_strat.download_parquet(is_train = False, index = parquet_index)

            if hw_config.world_size > 1:
                _smart_barrier("start_download_validation")

            if rank !=0:
                # Load Validation parquet for current curriculum level
                validation_file_path, val_parquet_num_rows, parquet_curr_level = data_strat.download_parquet(is_train = False, index = parquet_index) # , loaded_parquet_file_path = validation_file_path)



            # Define validation_dataloader
            validation_loader = data_strat.get_mlm_data_loader(
                parquet_file_path = validation_file_path,
                collate_fn = collator,
                batch_size = hw_config.batch_size,
                parquet_index = parquet_index,
                is_train = False,
            )

            # var holding a new train_file_path so it can be preloaded without training hiccups
            # This shouldn't affect the curriculum level, I'm just storing it for consistency
            new_train_file_path = ""
            new_train_parquet_num_rows = 0
            new_parquet_curr_level = 0

            # if TPU is being used, apply the ParallelLoader().per_device_loader()
            if is_tpu:
                validation_loader = pl.ParallelLoader(validation_loader, [device]).per_device_loader(device)



            # ========== TRAIN_LOADER PREPPER LOOP ==========

            while True:

                # --- ADDED PARQUET STOP LOGIC ---
                if parquet_index >= data_config.parquet_stop_index:
                    if rank == 0:
                        print(f"🛑 Reached parquet stop index ({data_config.parquet_stop_index}). Stopping data loader loop.")
                    break
                # --------------------------------

                # Load Validation parquet for current curriculum level unless it's been preloaded
                if new_train_file_path == "":
                    train_file_path = ""
                    if rank == 0:
                        train_file_path, train_parquet_num_rows, parquet_curr_level = data_strat.download_parquet(is_train = True, index = parquet_index)

                    if hw_config.world_size > 1:
                        _smart_barrier("start_download_training")

                    if rank !=0:
                        train_file_path, train_parquet_num_rows, parquet_curr_level = data_strat.download_parquet(is_train = True, index = parquet_index) # , loaded_parquet_file_path = train_file_path)

                else:
                    train_file_path = new_train_file_path
                    train_parquet_num_rows = new_train_parquet_num_rows
                    parquet_curr_level = new_parquet_curr_level
                    new_train_file_path = ""
                    new_train_parquet_num_rows = 0
                    new_parquet_curr_level = 0

                # If we were on the last curriculum level's parquet and downloaded the next, break out
                if (parquet_curr_level != level):
                    break

                # Define train dataloader
                train_loader = data_strat.get_mlm_data_loader(
                    parquet_file_path = train_file_path,
                    collate_fn = collator,
                    skip_rows =  total_rows_processed_parquet,
                    batch_size = hw_config.batch_size,
                    parquet_index = parquet_index,
                    is_train = True,
                )

                # if TPU is being used, apply the ParallelLoader().per_device_loader()
                if is_tpu:
                    train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)

                # ========== TRAINING LOOP ==========
                # Loop through each batch
                for step, batch in enumerate(train_loader):

                    # If SHUTDOWN_FILE exists, set the break boolean and break
                    # Claude recommends to call the checkpoint driver, but then we might train on the same information
                    # Therefore, we will just set the flag and dip
                    if os.path.exists(SHUTDOWN_FILE):
                        if rank == 0:
                            print("SHUTDOWN_FILE is up. Ending...")
                        break

                    # Get Batch's input ids, labels, and attn_mask (we don't have one but just in case) and attach it to device
                    input_ids = batch["input_ids"].to(device)
                    labels = batch["labels"].to(device)
                    attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

                    # If model architecture uses easiness
                    if data_config.use_easiness:

                        easiness_score = batch["easiness_score"].to(device)

                        # GPUs require Autocast for Mixed Precision. TPUs handle it natively via Env Variables.
                        if hw_config.device_type == "cuda":
                            with torch.autocast(device_type="cuda", dtype=dtype):
                                logits, aux_loss, sparsity_loss = model(input_ids=input_ids, attention_mask=attention_mask, easiness_score = easiness_score, current_step=global_step)
                                # logits: [mb, seq_len, vocab_size] -> [mb*seq_len, vocab_size]
                                # labels: [mb, seq_len] -> [mb*seq_len]
                                ce_loss = chunked_ce(logits, labels, helm_config.vocab_size)
                                total_loss = (ce_loss + aux_loss + sparsity_loss) / hw_config.grad_accum_steps
                        else:
                            with torch.autocast(device_type="xla", dtype=torch.bfloat16):
                                logits, aux_loss, sparsity_loss = model(input_ids=input_ids, attention_mask=attention_mask, easiness_score = easiness_score, current_step=global_step)
                                # logits: [mb, seq_len, vocab_size] -> [mb*seq_len, vocab_size]
                                # labels: [mb, seq_len] -> [mb*seq_len]
                                ce_loss = chunked_ce(logits, labels, helm_config.vocab_size)
                                total_loss = (ce_loss + aux_loss + sparsity_loss) / hw_config.grad_accum_steps

                    else:
                        # GPUs require Autocast for Mixed Precision. TPUs handle it natively via Env Variables.
                        if hw_config.device_type == "cuda":
                            with torch.autocast(device_type="cuda", dtype=dtype):
                                logits, aux_loss, sparsity_loss = model(input_ids=input_ids, attention_mask=attention_mask, current_step=global_step)
                                # logits: [mb, seq_len, vocab_size] -> [mb*seq_len, vocab_size]
                                # labels: [mb, seq_len] -> [mb*seq_len]
                                ce_loss = chunked_ce(logits, labels, helm_config.vocab_size)
                                total_loss = (ce_loss + aux_loss + sparsity_loss) / hw_config.grad_accum_steps
                        else:
                            with torch.autocast(device_type="xla", dtype=torch.bfloat16):
                                logits, aux_loss, sparsity_loss = model(input_ids=input_ids, attention_mask=attention_mask, current_step=global_step)
                                # logits: [mb, seq_len, vocab_size] -> [mb*seq_len, vocab_size]
                                # labels: [mb, seq_len] -> [mb*seq_len]
                                ce_loss = chunked_ce(logits, labels, helm_config.vocab_size)
                                total_loss = (ce_loss + aux_loss + sparsity_loss) / hw_config.grad_accum_steps


                    if scaler is not None:
                        scaler.scale(total_loss).backward()
                    else:
                        total_loss.backward()
                    # Mark every micro batch to prevent accumulating the entire graph
                    if is_tpu:
                        xm.mark_step()

                    # Once Gradient has been accumulated, step the model and the optimizer
                    if (step + 1) % hw_config.grad_accum_steps == 0:

                        # Apply Gradient Clipping Here instead of inside the model
                        # Start by unwrapping model form DDP or not
                        unwrapped_model = model.module if hasattr(model, "module") else model

                        # Now apply gradient clipping here to multi-view router's learnable params
                        # (skip cleanly for the no-router baseline, which has no mlt_vw_rtr)
                        if hw_config.device_type == "tpu" or hw_config.device_type == "cuda":
                            for block in unwrapped_model.model.blocks:
                                if hasattr(block, "mlt_vw_rtr"):
                                    torch.nn.utils.clip_grad_value_(
                                        block.mlt_vw_rtr.parameters(),
                                        clip_value = helm_config.router_grad_clip
                                    )

                        if is_tpu:
                            xm.optimizer_step(optimizer)
                            xm.mark_step()
                        elif scaler is not None:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                        scheduler.step()

                        # Normalize the model's weights
                        unwrapped_model.normalize_ngpt_matrices()

                        # Zero the gradient
                        optimizer.zero_grad()
                        global_step += 1


                        # Calculating the values for the save_checkpoint
                        # Should just be the target gbs, but just in case
                        rows_this_step = hw_config.batch_size * hw_config.grad_accum_steps * hw_config.world_size
                        tokens_this_step = rows_this_step * seq_len

                        # Increment the total amount of rows processed in parquet
                        total_rows_processed_parquet += rows_this_step

                        # Preload the next parquet and save vars if the current parquet is 95% done
                        # Maybe make this asynchronous ???
                        if (((float) (total_rows_processed_parquet) / train_parquet_num_rows) >= .95) and new_train_file_path == "":
                            new_train_file_path, new_train_parquet_num_rows, new_parquet_curr_level = data_strat.download_parquet(is_train = True, index = parquet_index + 1)


                        rows_processed_at_curr_level +=  rows_this_step
                        total_rows_processed_global += rows_this_step
                        total_tokens_processed_global += tokens_this_step

                        report_total_loss = to_float(ce_loss) + to_float(aux_loss) + to_float(sparsity_loss)

                        # Log Data to Wandb
                        if global_step < 100 or global_step % 10 == 0:

                            # Save telemetry_dict
                            telemetry_dict = unwrapped_model.get_telemetry()

                            telemetry_driver.log_step(
                                telemetry_dict = telemetry_dict,
                                ce_loss = to_float(ce_loss),
                                aux_loss = to_float(aux_loss),
                                sparsity_loss = to_float(sparsity_loss),
                                total_loss = report_total_loss,
                                global_step = global_step,
                                is_train = True,
                                global_tokens_processed = total_tokens_processed_global
                            )


                        # Use 1 device (rank = 0) to calculate the real loss
                        if rank == 0:
                            print(f"Step {global_step} | Total Loss: {report_total_loss:.4f} | Avg CE: {to_float(ce_loss):.4f} | Avg Aux: {to_float(aux_loss):.4f} | Avg Sparsity: {to_float(sparsity_loss):.4f}")


                        # Save the model if the time is right (based on interval_dict from CheckpoingConfig)
                        if checkpoint_driver.check_upload_condition(global_step):

                            # Ensure correct run_id is saved (should only change when using fork_from)
                            if telemetry_driver.run_type == "fork_from":
                                run_id = telemetry_driver.run.id

                            checkpoint_driver.save_checkpoint(
                                model = model,
                                optimizer = optimizer,
                                scheduler = scheduler,
                                global_step = global_step,
                                hardware_string = hw_config.hardware_string,
                                metrics = {
                                "Total Loss": round(report_total_loss, 5),
                                    "CE Loss" : round(to_float(ce_loss),5),
                                    "AUX Loss":   round(to_float(aux_loss), 5),
                                    "Sparsity":   round(to_float(sparsity_loss), 5)

                                },
                                is_tpu = is_tpu,
                                curriculum_level = level,
                                total_tokens_processed_global = total_tokens_processed_global,
                                total_rows_processed_global = total_rows_processed_global,
                                rows_processed_at_curr_level = rows_processed_at_curr_level,
                                parquet_index = parquet_index,
                                total_rows_processed_parquet = total_rows_processed_parquet,
                                run_id = run_id,
                            )

                        # ========== VALIDATION LOOP ==========
                        # Log Valdiation every 500 steps
                        if global_step % (500 if not TESTING_MODE else 50) == 0:

                            if rank == 0:
                                print("⏳ Calculating Validation...")

                            # zero the gradient again just in case
                            optimizer.zero_grad()

                            # Put model into eval mode
                            model.eval()

                            # Initialize accumulators
                            total_val_loss = 0.0
                            total_ce_loss = 0.0
                            total_aux_loss = 0.0
                            total_sparsity_loss = 0.0

                            # Loop through the validation_loader
                            for step, batch in enumerate(validation_loader):

                                if step > hw_config.validation_step_num:
                                    break

                                # Get Batch's input ids, labels, and attn_mask and attach it to device
                                input_ids = batch["input_ids"].to(device)
                                labels = batch["labels"].to(device)
                                attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

                                # Use no_grad to prevent OOM during validation
                                with torch.no_grad():
                                    # GPUs require Autocast for Mixed Precision. TPUs handle it natively via Env Variables.
                                    if hw_config.device_type == "cuda":
                                        with torch.autocast(device_type="cuda", dtype=dtype):
                                            logits, aux_loss, sparsity_loss = model(input_ids=input_ids, attention_mask=attention_mask)
                                            ce_loss = chunked_ce(logits, labels, helm_config.vocab_size)
                                            val_loss = ce_loss + aux_loss + sparsity_loss
                                    else:
                                        logits, aux_loss, sparsity_loss = model(input_ids=input_ids, attention_mask=attention_mask)
                                        ce_loss = chunked_ce(logits, labels, helm_config.vocab_size)
                                        val_loss = ce_loss + aux_loss + sparsity_loss

                                # Add Loss Values
                                total_val_loss += to_float(val_loss)
                                total_ce_loss += to_float(ce_loss)
                                total_aux_loss += to_float(aux_loss)
                                total_sparsity_loss += to_float(sparsity_loss)

                                if rank == 0 and step % 10 == 0:
                                    print(f"Completed Validation Step {step}/{hw_config.validation_step_num} - we are alive")


                            # Calculate and print the final averages once the loop naturally finishes
                            if hw_config.validation_step_num > 0:
                                avg_val_loss = total_val_loss / hw_config.validation_step_num
                                avg_ce_loss = total_ce_loss / hw_config.validation_step_num
                                avg_aux_loss = total_aux_loss / hw_config.validation_step_num
                                avg_sparsity_loss = total_sparsity_loss / hw_config.validation_step_num

                                # Normalize the model's weights
                                unwrapped_model = model.module if hasattr(model, "module") else model
                                unwrapped_model.normalize_ngpt_matrices()

                                # Save telemetry_dict
                                telemetry_dict = unwrapped_model.get_telemetry()

                                # Log the Data to WandB
                                telemetry_driver.log_step(
                                    telemetry_dict = telemetry_dict,
                                    ce_loss = avg_ce_loss,
                                    aux_loss = avg_aux_loss,
                                    sparsity_loss = avg_sparsity_loss,
                                    total_loss = avg_val_loss,
                                    global_step = global_step,
                                    is_train = False
                                )

                                if rank == 0:
                                    print(f"Total Loss: {avg_val_loss:.4f} | CE: {avg_ce_loss:.4f} | Aux: {avg_aux_loss:.4f} | Sparsity: {avg_sparsity_loss:.4f}")

                            # Call model.train
                            model.train()

                # break out of parquet loop
                if os.path.exists(SHUTDOWN_FILE):
                    break

                if rank == 0:
                    print(f"📦 Finished parquet {parquet_index} (level {level}). Advancing.")

                # Delete the consumed parquet so disk doesn't fill up
                data_strat.delete_parquet(train_file_path)

                # Advance to the next parquet and reset within-parquet row counter
                parquet_index += 1
                total_rows_processed_parquet = 0

            # Delete valdiation parquet once the curriculum is over
            data_strat.delete_parquet(validation_file_path)

            # # break out of curriculum loop
            if os.path.exists(SHUTDOWN_FILE):
                break

        # Destroy once all of these johns are done
        if hw_config.device_type == "cuda" and hw_config.world_size > 1:
            dist.destroy_process_group()

        # Stop the heartbeat and delete (so when rerun / revive occurs, it doesn't use the old file)
        heartbeat_stop.set()

    except Exception as e:
        print(f"\n❌ FATAL WORKER ERROR ON RANK {rank}:")
        traceback.print_exc()
        return



def sidecar_uploader_loop(hf_token, repo_id):
    # LAZY LOAD
    import os
    import json
    import time
    import signal
    from datetime import datetime, timezone
    from huggingface_hub import HfApi

    # Ignore stop signals
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    # Get HF API Token to upload
    api = HfApi(token=hf_token)

    # Forever Loop to constantly check
    while True:

        # Check to see if any valid upload requests exist & take the step size
        upload_requests = []
        for file in os.listdir("."):
            if file.startswith("UPLOAD_REQUEST_") and file.endswith(".json"):
                upload_requests.append(int(file.replace("UPLOAD_REQUEST_", "").replace(".json", "")))

        # Sort the list and take the first request
        if upload_requests:

            # Get the next upload_request and process that first
            next_upload = sorted(upload_requests)[0]
            upload_request_filename = f"UPLOAD_REQUEST_{next_upload}.json"

            # Try to upload the model and training_state.json to HF HUB
            try:

                # Open the UPLOADER_REQUEST.json
                with open(upload_request_filename, "r") as f:
                    UPLOAD_REQUEST = json.load(f)

                # Get filename and the step
                model_filename = UPLOAD_REQUEST["file_to_upload"]
                step = UPLOAD_REQUEST["step"]
                training_state_snapshot = UPLOAD_REQUEST["training_state_snapshot"]

                # Print Messeage
                print(f"⏳ Attempting to upload {model_filename} to {repo_id}")

                # Upload the model first (most unstable action to do before uplaoding the .json)
                if os.path.exists(model_filename):
                    api.upload_file(
                        path_or_fileobj=model_filename,
                        path_in_repo=model_filename,
                        repo_id=repo_id,
                        repo_type="model"
                    )
                else:
                    print(f"❌ Failed to Upload. {upload_request_filename} was pinged, but {model_filename} does not exist")

                # Format the .json to include whitespace
                formatted_json_str = json.dumps(training_state_snapshot, indent=4)
                json_bytes = formatted_json_str.encode('utf-8')
                fileobj = io.BytesIO(json_bytes)

                # Upload the training_state.json
                api.upload_file(
                    path_or_fileobj=fileobj,
                    path_in_repo="training_state.json",
                    repo_id=repo_id,
                    repo_type="model"
                )


                # Delete the big .pt file immediately to free disk space
                os.remove(model_filename)
                os.remove(upload_request_filename)

                # Squash history in background — don't block the next upload
                try:
                    api.super_squash_history(repo_id=repo_id)
                except Exception:
                    pass  # Non-critical, repo just gets bigger

                print(f"✅ Successfully uploaded {model_filename} @ step {step} to {repo_id}")

            except Exception as e:
                print(f"❌ Failed to upload to HF: {e}")
                print("Trying again in 5 seconds...")

        # Pause 5 seconds before rechecking if UPLOAD_REQUEST.json exists
        time.sleep(5)



if __name__ == "__main__":

    # Allow to kill all processes / end them correctly --------
    import signal
    import sys
    import threading
    import shutil

    # Shutdown Event (Essentially Thread-safe boolean)
    shutdown_event = threading.Event()
    shutdown_count = {"n": 0}

    # Shutdown manager
    # This is called once automatically. Press again if the shutdown is
    # completely cooked, skipping all cleanup.
    def graceful_shutdown(signum, frame):
        shutdown_count["n"] +=1
        if shutdown_count["n"] >=2:
            # Hard termination (x2 hits)
            print(f"2nd signal {signum} received. Hard Exit")
            try:
                with open(USER_STOP_MARKER, "w") as f:
                    f.write("hard")
            except Exception:
                pass
            os._exit(1)
        else:
            # Graceful termination
            print(f"Termination Signal ({signum}). Sending shutdown file to workers... (Ctrl+C again to hard-exit)")
            try:
                with open(SHUTDOWN_FILE, "w") as f:
                    f.write("user")
                with open(USER_STOP_MARKER, "w") as f:
                    f.write("graceful")
            except Exception:
                pass

            # Mark the sutdown event to gracefully shutdown
            shutdown_event.set()
            # raise KeyboardInterrupt


    # Catch Kaggle's Stop button (SIGTERM) and Keyboard Interrupts (SIGINT)
    signal.signal(signal.SIGTERM, graceful_shutdown)
    signal.signal(signal.SIGINT, graceful_shutdown)
    # ---------------------------------------------------------

    # Delete all existing SHUTDOWN_FILE and USER_STOP_MARKER that could've been from previous runs
    for f in [SHUTDOWN_FILE, USER_STOP_MARKER]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    # Ensure environment is primed for TPU PJRT
    for key in ["XRT_TPU_CONFIG", "PJRT_SELECT_DEVICE", "TPU_PROCESS_ADDRESSES"]:
        os.environ.pop(key, None)
    os.environ["PJRT_DEVICE"] = "TPU"

    # Set wandb key to None
    wandb_key = None

    # Get dummy ckpt_config to check if use_wandb is true:
    dummy_ckpt_cfg = CheckpointConfig()

    # Get wandb api key if wandb is being used
    if dummy_ckpt_cfg.use_wandb:
        wandb_key = get_secret("WANDB_API_KEY")

    # HF Token (must)
    hf_token = get_secret("HF_TOKEN")

    # Initialize all configs
    HW_CFG = HardwareConfig(hf_token = hf_token)
    DATA_CFG = MLMDataConfig()
    CKPT_CFG = None

    if not TESTING_MODE:
        CKPT_CFG = CheckpointConfig(hf_token = hf_token, wandb_key = wandb_key if wandb_key is not None else "")
    else:
        CKPT_CFG = CheckpointConfig(hf_token = hf_token, wandb_key = wandb_key if wandb_key is not None else "", interval_dict = {0:10})


    # Log into wandb if we are logging in:
    if wandb_key and CKPT_CFG.use_wandb:
        os.environ["WANDB_API_KEY"] = wandb_key
        wandb.login()
    elif not CKPT_CFG.use_wandb:
        print("wandb disabled. Change use_wandb in the CheckpointConfig to True if you intended to log")
    else:
        print("⚠️ wandb_key was NULL. Make sure you allow secrets on Colab or Kaggle. Continuing Anonymous Logging")


    # Use the 'spawn' context to prevent C++ state corruption
    ctx = multiprocessing.get_context('spawn')

    # Update: Sidecar respawn. Insteaf spawning one time, create respawning thread

    # Create holder so watchdog can swap the sidecar
    sidecar_holder = {"proc": None}

    def spawn_sidecar():
        # Loading Sidecar via isolated CPU thread to upload
        uploader_process = ctx.Process(
            target = sidecar_uploader_loop,
            args = (CKPT_CFG.hf_token, CKPT_CFG.model_repo_id),
            daemon = False
        )
        uploader_process.start()
        return uploader_process

    sidecar_holder["proc"] = spawn_sidecar()

    # Watchdog worker function to respawn sidecar
    def sidecar_watchdog():
        crash_count = 0
        MAX_CRASHES = 5
        # Checks every 30 seconds
        while not shutdown_event.wait(timeout = 30):
            proc = sidecar_holder["proc"]
            # is_alive() is really waitpid(pid, WNOHANG)
            # True when running
            # False when zombie
            if not proc.is_alive():
                # Reap the zombie
                proc.join(timeout = 5)
                if (crash_count >= MAX_CRASHES):
                    print(f"Sidecar crashed {crash_count} times. No reboot card for you anymore...")
                    return
                crash_count +=1
                exitcode = proc.exitcode
                print(f"⚠️ Sidecar died (exitcode={exitcode}, crash #{crash_count}). Respawning...")
                sidecar_holder["proc"] = spawn_sidecar()

    # Create Watchdog
    sidecar_watchdog_thread = threading.Thread(target=sidecar_watchdog, daemon = True)
    sidecar_watchdog_thread.start()

    # Simple training watchdog — warn on stale heartbeats, that's it.
    # With JAX_PLATFORMS=cpu, the C++ runtime handles cleanup.
    # If a worker hangs, press Stop twice → os._exit(1).
    def training_watchdog():
        STALE_WARN = 120
        warned = set()
        while not shutdown_event.wait(timeout=30):
            now = time.time()
            for r in range(HW_CFG.world_size):
                path = f"/tmp/heartbeat_rank_{r}.txt"
                if not os.path.exists(path):
                    continue
                try:
                    with open(path) as f:
                        last = float(f.read().strip())
                except Exception:
                    continue
                age = now - last
                if age > STALE_WARN and r not in warned:
                    print(f"⚠️ WATCHDOG: Rank {r} heartbeat {age:.0f}s old. Possible hang.")
                    warned.add(r)
                elif age <= STALE_WARN and r in warned:
                    print(f"✅ WATCHDOG: Rank {r} recovered.")
                    warned.discard(r)

    # Start trainer watchdog for all threads
    training_watchdog_thread = threading.Thread(target=training_watchdog, daemon=True)
    training_watchdog_thread.start()


    # Prepare Hardware Driver
    driver = HardwareDriver(HW_CFG, DATA_CFG, CKPT_CFG)

    # Try to launch training process
    try:
        driver.launch(train_worker)
    except KeyboardInterrupt:
        print("⚠️ Training interrupted by signal or pause button")
    except Exception as e:
        print(f"💀 Summ done messed up cuh {e}")
        import traceback
        traceback.print_exc()


    finally:
        print("🧹 Cleanup starting...")
        shutdown_event.set()

        # Wait for sidecar to finish any in-flight upload
        time_count = 0
        MAX_DRAIN_SEC = 600
        while time_count < MAX_DRAIN_SEC:
            still_uploading = any(
                file.startswith("UPLOAD_REQUEST_") and file.endswith(".json")
                for file in os.listdir(".")
            )
            if not still_uploading:
                break
            if not sidecar_holder["proc"].is_alive():
                print("⚠️ Sidecar died before finishing uploads.")
                break
            if time_count % 30 == 0:
                print(f"⏳ Sidecar uploading... ({time_count}s elapsed)")
            time.sleep(5)
            time_count += 5
        if time_count >= MAX_DRAIN_SEC:
            print(f"⚠️ Sidecar drain timed out after {MAX_DRAIN_SEC}s.")

        # Kill sidecar
        try:
            sidecar_holder["proc"].kill()
            sidecar_holder["proc"].join(timeout=10)
        except Exception:
            pass

        # Clean up TPU lockfile (safety net)
        if HW_CFG.device_type == "tpu":
            if os.path.exists("/tmp/libtpu_lockfile"):
                try:
                    os.remove("/tmp/libtpu_lockfile")
                    print("🧹 Removed stale libtpu_lockfile")
                except Exception:
                    pass

        # Clean heartbeat files
        for i in range(HW_CFG.world_size):
            try:
                os.remove(f"/tmp/heartbeat_rank_{i}.txt")
            except Exception:
                pass

        # Clean shutdown files
        for f in [SHUTDOWN_FILE, USER_STOP_MARKER]:
            try:
                os.remove(f)
            except Exception:
                pass

        print("💅 Okay girl... shutdown is  ✨✨COMPLETE✨✨")