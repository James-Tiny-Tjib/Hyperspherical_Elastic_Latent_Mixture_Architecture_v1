%%writefile parallel_hardware_trainer.py
import os
import sys
import importlib
import time
import json
import site
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download, create_repo, HfApi
from dataclasses import dataclass, field # Added field here
import multiprocessing
from typing import Optional, List, Union, ClassVar, Dict, Any
import warnings
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
os.environ["XLA_USE_BF16"] = "1"
os.environ["XLA_DOWNCAST_BF16"] = "1"

# Force Path Refresh
if 'site' in sys.modules:
    importlib.reload(site)

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
            0: {"mb": 16, "use_ckpt": False, "sl": 1024},
            1: {"mb": 4, "use_ckpt": False, "sl": 2048},
            2: {"mb": 1, "use_ckpt": False, "sl": 4096},
        },
        "v5e-1": {
            "ws": 1, "target": 128, "dtype": torch.bfloat16,"use_scaler": False,
            0: {"mb": 16, "use_ckpt": False, "sl": 1024},
            1: {"mb": 4, "use_ckpt": False, "sl": 2048},
            2: {"mb": 1, "use_ckpt": False, "sl": 4096},
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

    # hardware_string: str = "v6e-1 tpu"
    hardware_string: str = "a100 gpu"
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


# ==================================================
# MLMDataConfig
# Sets dataset repo, curriculum levels, and MLM parameters
# Static (Remains the same)
# ==================================================

@dataclass
class MLMDataConfig:
    data_repo_id: str = "JamesResearch1216/HELM-Processed-Data-10B"
    curriculum: bool = True
    curriculum_subset_names: List[str] = field(
        default_factory=lambda: ["seq_1024", "seq_2048", "seq_4096"]
    )
    # curriculum_subset_lens: List[int] = field(
    #     default_factory=lambda: [1024, 2048, 4096]
    # )
    train_split: str = "train"
    validation_split: str = "validation"
    tokenizer_name: str = "answerdotai/ModernBERT-base"
    mlm_probability: float = 0.3
    mlm_use_span_masking: bool = True
    mlm_span_length: int = 3


# ==================================================
# CheckpointConfig
# Sets Checkpoint related fields
# Static (Remains the same)
# ==================================================

@dataclass
class CheckpointConfig:
    model_repo_id: str = "JamesResearch1216/HELM-v1-Architecture"
    # repo_ver_override: Optional[int] = None
    hf_token: str = ""

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

    # This must be updated by calling the resume_training_step
    latest_step: int = -1



class MLMDataStrategy:

    # Initialize (Different for each hardware)
    def __init__(self, rank = 0, world_size = 1, is_tpu = False, config: Optional[MLMDataConfig] = None, hf_token=None):
        self.rank = rank
        self.world_size = world_size
        self.is_tpu = is_tpu
        self.config = config
        self.hf_token = hf_token

    # Create get_mlm_data_loader function
    # Note: This only bascially works with the dataset created by prepare_data.py

    def get_mlm_data_loader(self, collate_fn = None, skip_rows = 0, batch_size = 1, curriculum_level = 0, is_train = True):

        # Set up Dataset Differently for curriculum
        if (self.config.curriculum):
            dataset = load_dataset(
                path = self.config.data_repo_id,
                name = self.config.curriculum_subset_names[curriculum_level],
                split = self.config.train_split if is_train else self.config.validation_split,
                streaming = True,
                token=self.hf_token
            )
        else:
            # Else just load normally
            dataset = load_dataset(
                path = self.config.data_repo_id,
                name = self.config.dataset_subset,
                split = self.config.train_split if is_train else self.config.validation_split,
                streaming = True,
                token=self.hf_token
            )

        # IF we're Parallel Processing, Shard the Data so that each device gets a different slice of data
        if self.world_size > 1 and is_train:
            dataset = dataset.shard(num_shards = self.world_size, index = self.rank)

        # Shuffle after you shard
        # Make sure you set a seed to ensure the I don't use the same data again
        dataset = dataset.shuffle(buffer_size = 10000, seed = 67)

        # Skip examples after you shuffle based on the specific seed:
        dataset = dataset.skip(skip_rows)

        # set workers = 0 for TPUs else actually the real number of CPU cores
        # this was in the 1.x_model_trainer series of scripts
        # workers = 0 if self.is_tpu else max(1, multiprocessing.cpu_count()-1)
        # Just set workers to 0. Don't wanna crash our CPU, right?

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
    def __init__(self, hw_config: HardwareConfig, ckpt_config: CheckpointConfig, rank: int, world_size: int):
        self.hw_config = hw_config
        self.checkpoint_config = ckpt_config
        self.rank = rank
        self.world_size = world_size
        self.api = HfApi(token=self.checkpoint_config.hf_token)
        self.actual_resume_step = None

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


    # Initialize new checkpoint dictionary
    def _init_new_training_state(self):
        training_state_dict = {
            "checkpoints": {}
        }
        return training_state_dict


    # Ensure that if checkpoints are deleted but appear
    def _deletion_status_updates(self, training_state):

        # Try to take the repo file's file paths
        repo_files = None
        try:
            repo_files = list(self.api.list_repo_files(repo_id = self.checkpoint_config.model_repo_id))
        except RepositoryNotFoundError:
            raise RepositoryNotFoundError(f"❌ Repo: \"{self.checkpoint_config.model_repo_id}\" was not found when trying to update deletion status")


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
                path = hf_hub_download(repo_id=self.checkpoint_config.model_repo_id, filename = filename, repo_type = "model")
                with open(path, "r") as f:
                    training_state = json.load(f)

                # Successfully loaded
                print(f"✅ {filename} loaded successfully from {self.checkpoint_config.model_repo_id}")
                training_state = self._deletion_status_updates(training_state)

            # If the repo doesn't exist, make the repo
            except RepositoryNotFoundError:
                # Print Error Statements
                print(f"⚠️ Repo: \"{self.checkpoint_config.model_repo_id}\" was not found")
                print(f"🏗️ Creating Repo: {self.checkpoint_config.model_repo_id}")

                # Create Repo
                create_repo(
                    repo_id = self.checkpoint_config.model_repo_id,
                    token = self.checkpoint_config.hf_token,
                    repo_type = "model",
                    private = False,
                    exist_ok = False
                )

                # Make new training_state dict
                training_state = self._init_new_training_state()

            # The repo exists, but it's empty or doesn't have the state file yet
            except EntryNotFoundError:
                # Print info
                print(f"⚠️ {self.checkpoint_config.model_repo_id} exists, but no {filename} found. Starting fresh.")
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
        if not self.checkpoint_config.start_from_global:
            curr_global_step -= self.actual_resume_step
        if curr_global_step <=0:
            return False

        # Save which interval we will use to calculate if we need to upload
        active_interval = None

        # Iterate through the sorted keys
        for threshold in sorted(self.checkpoint_config.interval_dict.keys()):
            # If our curr_global_step is bigger than threshold, save it's value
            if curr_global_step >= threshold:
                active_interval = self.checkpoint_config.interval_dict[threshold]
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
    # Dicionary with a bunch of 0s if all checkpoints were deleted or starting fresh
    # None if other errors (fast failing)
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
                "curriculum_level": 0,
                "rows_processed_at_curr_level": 0,
                "total_tokens_processed_global": 0,
                "total_rows_processed_global": 0,
            }, 0

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
                    repo_id = self.checkpoint_config.model_repo_id,
                    filename = filename,
                    repo_type = "model",
                    token = self.checkpoint_config.hf_token,
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

            # print success
            if self.rank == 0:
                print(f"Successfully loaded model and optimizer from Step {actual_resume_step}!")

            # Just return the ckpt_entry; Extract the values later
            return ckpt_entry, actual_resume_step

        except Exception as e:
            raise RuntimeError(f"Critical Weight Loading Failure: {e}")

    # Saves the model to a .pt file
    # Makes checkpoint entry for training_state
    def save_checkpoint(self, model, optimizer, global_step: int, hardware_string: str, metrics: dict, is_tpu: bool, curriculum_level: int, total_tokens_processed_global: int, total_rows_processed_global: int, rows_processed_at_curr_level: int):
        # Lazy Load Torch
        import torch

        # Save filename
        step_str = str(global_step)
        filename = f"checkpoint-{step_str}.pt"

        # If more than 1 worker, start barrier
        if self.world_size > 1:
            self._smart_barrier("save_start")

        if self.rank == 0:
            print(f"Saving model weights to {filename}...")

        # Ensure to use module or not to ensure compatibility
        save_dict = {
            "model_state": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            "optimizer_state": optimizer.state_dict()
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
            }

            # Dump new training_state into training_state.json
            with open("training_state.json", "w") as f:
                json.dump(self.training_state, f, indent = 4)

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
            print(f"Saved weights to local disk + updated training_state.json. Pinging Sidecar for Step{global_step}")

        # If more than 1 worker make sure other ranks wait for rank 0
        if self.world_size > 1:
            self._smart_barrier("save_training_state_end")



# Function that all devices will run (ran from the launch function right above)
def train_worker(rank, hw_config, data_config, ckpt_config):

    # Lazy Load
    import sys
    import traceback
    import os # Add os

    os.environ["HF_TOKEN"] = hw_config.hf_token


    try:
        # Lazy Load
        import datasets
        datasets.config.TF_AVAILABLE = False
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from transformers import AutoTokenizer
        import SpanMLMCollator
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

        # Initialize Config
        helm_config = HELMConfig(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
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
        optimizer = optim.AdamW(model.parameters(), lr = helm_config.lr)

        # Zero the gradient
        optimizer.zero_grad()

        # Define CE Loss
        loss_fct = nn.CrossEntropyLoss()

        # Set data type that will be used
        dtype = hw_config.dtype

        # Allow Scaler
        use_scaler = hw_config.use_scaler
        scaler = torch.amp.GradScaler('cuda') if hw_config.device_type == "cuda" and use_scaler else None

        # ========== CHECKPOINT TECHNOLOGICA ==========

        # Define Checkpoint Driver
        checkpoint_driver = CheckpointDriver(
            hw_config = hw_config, ckpt_config = ckpt_config, rank = rank, world_size = hw_config.world_size
        )

        # Loading the model/optimizer returns the most recent, valid / undeleted checkpoint
        ckpt_snapshot, actual_resume_step = checkpoint_driver.resume_training(model, optimizer)

        # Extract the values from the ckpt_snapshot
        start_curr_level = ckpt_snapshot["curriculum_level"]
        rows_processed_at_curr_level = ckpt_snapshot["rows_processed_at_curr_level"]
        total_rows_processed_global = ckpt_snapshot["total_rows_processed_global"]
        total_tokens_processed_global = ckpt_snapshot["total_tokens_processed_global"]

        # Set the global step to where we left off from the previous checkpoint
        global_step = actual_resume_step

        # If starting fresh initialize weights
        if actual_resume_step == 0:
            # Use .module to access the original HELMForMaskedLM if wrapped in DDP
            unwrapped_model = model.module if hasattr(model, "module") else model
            unwrapped_model.apply(unwrapped_model._init_weights)


        # Curriculum Outer Loop (starting from the current curriculum):
        for level in range(start_curr_level,len(data_config.curriculum_subset_names)):

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
            collator = SpanMLMCollator.SpanMLMCollator(
                config = data_config, tokenizer = tokenizer
            )

            # Define train_dataloader
            train_loader = data_strat.get_mlm_data_loader(
                collate_fn = collator, skip_rows = rows_processed_at_curr_level, batch_size = hw_config.batch_size, curriculum_level = level, is_train = True
            )

            # Define validation_dataloader
            validation_loader = data_strat.get_mlm_data_loader(
                collate_fn = collator, batch_size = hw_config.batch_size, curriculum_level = level, is_train = False
            )

            # if TPU is being used, apply the ParallelLoader().per_device_loader()
            if is_tpu:
                train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
                validation_loader = pl.ParallelLoader(validation_loader, [device]).per_device_loader(device)

            # ========== TRAINING LOOP ==========
            # Loop through each batch
            for step, batch in enumerate(train_loader):

                # Get Batch's input ids, labels, and attn_mask (we don't have one but just in case) and attach it to device
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = (input_ids != tokenizer.pad_token_id).long().to(device)

                # GPUs require Autocast for Mixed Precision. TPUs handle it natively via Env Variables.
                if hw_config.device_type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        logits, aux_loss, sparsity_loss = model(input_ids=input_ids, attention_mask=attention_mask, current_step=global_step)
                        # logits: [mb, seq_len, vocab_size] -> [mb*seq_len, vocab_size]
                        # labels: [mb, seq_len] -> [mb*seq_len]
                        ce_loss = loss_fct(logits.view(-1, helm_config.vocab_size), labels.view(-1))
                        total_loss = (ce_loss + aux_loss + sparsity_loss) / hw_config.grad_accum_steps
                else:
                    logits, aux_loss, sparsity_loss = model(input_ids=input_ids, attention_mask=attention_mask, current_step=global_step)
                    # logits: [mb, seq_len, vocab_size] -> [mb*seq_len, vocab_size]
                    # labels: [mb, seq_len] -> [mb*seq_len]
                    ce_loss = loss_fct(logits.view(-1, helm_config.vocab_size), labels.view(-1))
                    total_loss = (ce_loss + aux_loss + sparsity_loss) / hw_config.grad_accum_steps

                if scaler is not None:
                    scaler.scale(total_loss).backward()
                else:
                    total_loss.backward()

                # Once Gradient has been accumulated, step the model and the optimizer
                if (step + 1) % hw_config.grad_accum_steps == 0:
                    if is_tpu:
                        xm.optimizer_step(optimizer)
                        xm.mark_step()
                    elif scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    # Normalize the model's weights
                    unwrapped_model = model.module if hasattr(model, "module") else model
                    unwrapped_model.normalize_ngpt_matrices()

                    # Zero the gradient
                    optimizer.zero_grad()
                    global_step += 1

                    # Calculating the values for the save_checkpoint
                    # Should just be the target gbs, but just in case
                    rows_this_step = hw_config.batch_size * hw_config.grad_accum_steps * hw_config.world_size
                    tokens_this_step = rows_this_step * seq_len

                    rows_processed_at_curr_level +=  rows_this_step
                    total_rows_processed_global += rows_this_step
                    total_tokens_processed_global += tokens_this_step

                    # Use 1 device (rank = 0) to calculate the real loss
                    if rank == 0:
                        print(f"Step {global_step} | Total Loss: {total_loss.item()*hw_config.grad_accum_steps:.4f} | CE: {ce_loss.item():.4f} | Aux: {aux_loss if isinstance(aux_loss, float) else aux_loss.item():.4f} | Sparsity: {sparsity_loss if isinstance(sparsity_loss, float) else sparsity_loss.item():.4f}")

                    # Save the model if the time is right (based on interval_dict from CheckpoingConfig)
                    if checkpoint_driver.check_upload_condition(global_step):
                        checkpoint_driver.save_checkpoint(
                            model = model,
                            optimizer = optimizer,
                            global_step = global_step,
                            hardware_string = hw_config.hardware_string,
                            metrics = {
                                "Total Loss" :  round(total_loss.item()*hw_config.grad_accum_steps,5),
                                "CE Loss" : round(ce_loss.item(),5),
                                "AUX Loss" : round(aux_loss.item(),5),
                                "Sparsity" : round(sparsity_loss.item(),5)

                            },
                            is_tpu = is_tpu,
                            curriculum_level = level,
                            total_tokens_processed_global = total_tokens_processed_global,
                            total_rows_processed_global = total_rows_processed_global,
                            rows_processed_at_curr_level = rows_processed_at_curr_level
                        )

            # zero the gradient again just in case
            optimizer.zero_grad()



            # ========== VALIDATION LOOP ==========

            # Put model into eval mode
            model.eval()

            # Initialize accumulators
            total_val_loss = 0.0
            total_ce_loss = 0.0
            total_aux_loss = 0.0
            total_sparsity_loss = 0.0
            step_count = 0

            # Loop through the validation_loader
            for step, batch in enumerate(validation_loader):
                step_count += 1

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
                            ce_loss = loss_fct(logits.view(-1, helm_config.vocab_size), labels.view(-1))
                            val_loss = ce_loss + aux_loss + sparsity_loss
                    else:
                        logits, aux_loss, sparsity_loss = model(input_ids=input_ids, attention_mask=attention_mask)
                        ce_loss = loss_fct(logits.view(-1, helm_config.vocab_size), labels.view(-1))
                        val_loss = ce_loss + aux_loss + sparsity_loss

                # Get aux and sparsity values
                current_aux = aux_loss if isinstance(aux_loss, float) else aux_loss.item()
                current_sparsity = sparsity_loss if isinstance(sparsity_loss, float) else sparsity_loss.item()

                # Add Loss Values
                total_val_loss += val_loss.item()
                total_ce_loss += ce_loss.item()
                total_aux_loss += current_aux
                total_sparsity_loss += current_sparsity

                # Print Validation every so ofter
                if rank == 0 and step_count % 100 == 0:
                    print(f"Validation Step {step_count} | Batch Loss: {val_loss.item():.4f}")

            # Calculate and print the final averages once the loop naturally finishes
            if step_count > 0:
                avg_val_loss = total_val_loss / step_count
                avg_ce_loss = total_ce_loss / step_count
                avg_aux_loss = total_aux_loss / step_count
                avg_sparsity_loss = total_sparsity_loss / step_count

                if rank == 0:
                    print(f"Average Total Loss: {avg_val_loss:.4f} | Avg CE: {avg_ce_loss:.4f} | Avg Aux: {avg_aux_loss:.4f} | Avg Sparsity: {avg_sparsity_loss:.4f}\n")

        # Destroy once all of these johns are done
        if hw_config.device_type == "cuda" and hw_config.world_size > 1:
            dist.destroy_process_group()

    except Exception as e:
        print(f"\n❌ FATAL WORKER ERROR ON RANK {rank}:")
        traceback.print_exc()
        return



def sidecar_uploader_loop(hf_token, repo_id):
    # LAZY LOAD
    import os
    import json
    import time
    from datetime import datetime, timezone
    from huggingface_hub import HfApi

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

                # Dump the snapshot's training state to a temporary .json
                with open(f"uploading_training_state.json", "w") as f:
                    json.dump(training_state_snapshot, f)

                # Upload the training_state.json
                api.upload_file(
                    path_or_fileobj="uploading_training_state.json",
                    path_in_repo="training_state.json",
                    repo_id=repo_id,
                    repo_type="model"
                )


                # Squash History to ensure that the repo doesn't hold onto the archives
                api.super_squash_history(repo_id=repo_id)

                # Remove the current checkpoint and UPLOAD_REQUEST_XXXXXX.json, and uploading_training_state.json
                os.remove(model_filename)
                os.remove(upload_request_filename)
                os.remove("uploading_training_state.json")

                print(f"✅ Successfully uploaded {model_filename} @ step {step} to {repo_id}")

            except Exception as e:
                print(f"❌ Failed to upload to HF: {e}")
                print("Trying again in 5 seconds...")

        # Pause 5 seconds before rechecking if UPLOAD_REQUEST.json exists
        time.sleep(5)


if __name__ == "__main__":
    # Ensure environment is primed for TPU PJRT
    for key in ["XRT_TPU_CONFIG", "PJRT_SELECT_DEVICE", "TPU_PROCESS_ADDRESSES"]:
        os.environ.pop(key, None)
    os.environ["PJRT_DEVICE"] = "TPU"

    # Initialize all configs
    HW_CFG = HardwareConfig()
    DATA_CFG = MLMDataConfig()
    CKPT_CFG = CheckpointConfig()

    # Loading Sidecar via isolated CPU thread to upload
    uploader_process = multiprocessing.Process(
        target = sidecar_uploader_loop,
        args = (CKPT_CFG.hf_token, CKPT_CFG.model_repo_id)
    )
    uploader_process.start()

    # Prepare Hardware Driver
    driver = HardwareDriver(HW_CFG, DATA_CFG, CKPT_CFG)

    # Try to launch training process
    try:
        driver.launch(train_worker)
    except Exception as e:
        print(f"💀 Summ done messed up cuh {e}")

    finally:

        print("🔪 Initiating Termination Sequence...")

        # Ensure Sidecar finishes before termiantion
        time_count = 0
        while True:
            still_uploading = False
            for file in os.listdir("."):
                if file.startswith("UPLOAD_REQUEST_") and file.endswith(".json"):

                    if time_count % 60 == 0:
                        print(f"⏳ Sidecar is currently processing a final upload. Waiting for it to finish...")
                        print(f"Elapsed Time: {time_count//60} minute(s)")

                    still_uploading = True
                    break

            if not still_uploading:
                break

            time.sleep(5)
            time_count +=5

        # Terminate Sidecars
        uploader_process.terminate()
        uploader_process.join()
        print("Shutdown complete.")