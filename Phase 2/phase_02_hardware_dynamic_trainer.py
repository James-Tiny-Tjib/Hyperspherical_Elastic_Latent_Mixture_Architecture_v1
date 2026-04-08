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
from dataclasses import dataclass, field # Added field here
import multiprocessing
from typing import Optional, List, Union, ClassVar, Dict, Any
import warnings

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


@dataclass
class HardwareConfig:

    HARDWARE_PROFILES = {
        "v5e-8": {
            "ws": 8, "target": 128, "dtype": torch.bfloat16,"use_scaler": False,
            0: {"mb": 8, "use_ckpt": False, "sl": 1024},
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
            0: {"mb": 32, "use_ckpt": False, "sl": 1024},
            1: {"mb": 16, "use_ckpt": False, "sl": 2048},
            2: {"mb": 4, "use_ckpt": False, "sl": 4096},
        },
        "t4*2": {
            "ws": 2, "target": 128, "dtype": torch.float16, "use_scaler": True,
            0: {"mb": 4, "use_ckpt": True, "sl": 1024},
            1: {"mb": 1, "use_ckpt": True, "sl": 2048},
            2: {"mb": 1, "use_ckpt": True, "sl": 4096},
        },
        "t4": {
            "ws": 1, "target": 128, "dtype": torch.float16, "use_scaler": True,
            0: {"mb": 4, "use_ckpt": True, "sl": 1024},
            1: {"mb": 1, "use_ckpt": True, "sl": 2048},
            2: {"mb": 1, "use_ckpt": True, "sl": 4096},
        },
        "g4": {
            "ws": 1, "target": 128, "dtype": torch.float16, "use_scaler": True,
            0: {"mb": 4, "use_ckpt": False, "sl": 1024},
            1: {"mb": 1, "use_ckpt": True, "sl": 2048},
            2: {"mb": 1, "use_ckpt": True, "sl": 4096},
        },
        "l4": {
            "ws": 1, "target": 128, "dtype": torch.float16, "use_scaler": True,
            0: {"mb": 4, "use_ckpt": False, "sl": 1024},
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
            0: {"mb": 16, "use_ckpt": False, "sl": 1024},
            1: {"mb": 4, "use_ckpt": True, "sl": 2048},
            2: {"mb": 1, "use_ckpt": True, "sl": 4096},
        },
        "h100": {
            "ws": 1, "target": 128, "dtype": torch.bfloat16,"use_scaler": False,
            0: {"mb": 16, "use_ckpt": False, "sl": 1024},
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

    hardware_string: str = "p100 gpu"
    hf_token: str = "" 
    base_repo_id = "JamesResearch1216/HELM-Architecture"
    repo_ver_override: Optional[int] = None


    # These will be overwritten when we step thru the curriculum, but just place_holders for now
    world_size: int = 1
    target_gbs: int = 128
    dtype: torch.dtype = torch.float16
    use_scaler: bool = True
    batch_size: int = 16
    use_ckpt: bool = False
    grad_accum_steps: int = 1
    hardware_profile: Dict[Union[str, int], Any] = field(
        default_factory=lambda: HardwareConfig.HARDWARE_PROFILES["cpu"]
    )
    device_type: str = "cpu"
    



@dataclass
# This MLMDataConfig Class 
class MLMDataConfig:
    repo_id: str = "JamesResearch1216/HELM-Processed-Data-10B"
    curriculum: bool = True
    curriculum_subset_names: List[str] = field(
        default_factory=lambda: ["seq_1024", "seq_2048", "seq_4096"]
    )
    train_split: str = "train"
    validation_split: str = "validation" 
    tokenizer_name: str = "answerdotai/ModernBERT-base"
    mlm_probability: float = 0.3
    mlm_use_span_masking: bool = True
    mlm_span_length: int = 3

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

    def get_mlm_data_loader(self, collate_fn = None, batch_size = 1, curriculum_level = 0, is_train = True):

        # Set up Dataset Differently for curriculum
        if (self.config.curriculum):
            dataset = load_dataset(
                path = self.config.repo_id, 
                name = self.config.curriculum_subset_names[curriculum_level], 
                split = self.config.train_split if is_train else self.config.validation_split,
                streaming = True,
                token=self.hf_token
            )
        else:
            # Else just load normally
            dataset = load_dataset(
                path = self.config.repo_id, 
                name = self.config.dataset_subset, 
                split = self.config.train_split if is_train else self.config.validation_split,
                streaming = True,
                token=self.hf_token
            )

        # IF we're Parallel Processing, Shard the Data so that each device gets a different slice of data
        if self.world_size > 1 and is_train:
            dataset = dataset.shard(num_shards = self.world_size, index = self.rank)

        # Shuffle after you shard
        dataset = dataset.shuffle(buffer_size = 10000)

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
    def __init__(self, hw_config: HardwareConfig, data_config: MLMDataConfig):
        self.hw_config = hw_config
        self.data_config = data_config
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
            warnings.warn("hardware_string did not match any in HARDWARE_PROFILES. Using \"cpu\"", UserWarning)

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
                xmp.spawn(worker_fn, args=(self.hw_config, self.data_config), start_method='spawn')
            elif device == "cuda":
                import random
                import torch.multiprocessing as mp
                
                # Set up Multi-GPU network
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = str(random.randint(10001, 19999))


                mp.spawn(worker_fn, args=(self.hw_config, self.data_config), nprocs=world_size)
        else:
            # Single Device Execution (Rank 0)
            worker_fn(0, self.hw_config, self.data_config)


# Function that all devices will run (ran from the launch function right above)
def train_worker(rank, hw_config, data_config):

    import sys
    import traceback
    import os # Add os
    
    os.environ["HF_TOKEN"] = hw_config.hf_token
    
    
    try:
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


        # Define data_strat
        data_strat = MLMDataStrategy(
            rank = rank, world_size = hw_config.world_size, is_tpu = is_tpu,config = data_config, hf_token=hw_config.hf_token
        ) 
        
        # Initialize Config
        helm_config = HELMConfig(
            vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id,
        )

        # Initialize model and attach to device
        model = HELMForMaskedLM(helm_config).to(device)
        
        # Initialize Weights
        model.apply(model._init_weights)

        # Change the Model Config to allow gradient checkpointing or not
        model.config.use_ckpt = hw_config.use_ckpt

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

        # Global step count
        global_step = 0
        
        # Curriculum Outer Loop:
        for level, subset_name in enumerate(data_config.curriculum_subset_names):

            # Set Model to Training Mode
            model.train()

            # Get Profile Level
            level_profile = hw_config.hardware_profile[level] 

            # Get micro batch size (mb) and use gradient checkpointing (use_ckpt)
            hw_config.batch_size = level_profile["mb"]
            hw_config.use_ckpt = level_profile["use_ckpt"]
            # CRITICAL: Push the flag into the model and its internal HELMModel
            unwrap_model = model.module if hasattr(model, "module") else model
            unwrap_model.config.use_ckpt = hw_config.use_ckpt
            unwrap_model.model.use_ckpt = hw_config.use_ckpt # HELMModel caches this

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
                collate_fn = collator, batch_size = hw_config.batch_size, curriculum_level = level, is_train = True
            )

            # Define validation_dataloader
            validation_loader = data_strat.get_mlm_data_loader(
                collate_fn = collator, batch_size = hw_config.batch_size, curriculum_level = level, is_train = False
            )

            # if TPU is being used, apply the ParallelLoader().per_device_loader()
            if is_tpu:
                train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
                validation_loader = pl.ParallelLoader(validation_loader, [device]).per_device_loader(device)
                

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
                        ce_loss = loss_fct(logits.view(-1, helm_config.vocab_size), labels.view(-1))
                        total_loss = (ce_loss + aux_loss + sparsity_loss) / hw_config.grad_accum_steps
                else:
                    logits, aux_loss, sparsity_loss = model(input_ids=input_ids, attention_mask=attention_mask, current_step=global_step)
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

                    # Zero the gradient
                    optimizer.zero_grad()
                    global_step += 1
                    
                    # Use 1 device (rank = 0) to calculate the real loss
                    if rank == 0:
                        print(f"Step {global_step} | Total Loss: {total_loss.item()*hw_config.grad_accum_steps:.4f} | CE: {ce_loss.item():.4f} | Aux: {aux_loss if isinstance(aux_loss, float) else aux_loss.item():.4f} | Sparsity: {sparsity_loss if isinstance(sparsity_loss, float) else sparsity_loss.item():.4f}")        
            
            # zero the gradient again just in case
            optimizer.zero_grad()

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


if __name__ == "__main__":
    # Ensure environment is primed for TPU PJRT
    for key in ["XRT_TPU_CONFIG", "PJRT_SELECT_DEVICE", "TPU_PROCESS_ADDRESSES"]:
        os.environ.pop(key, None)
    os.environ["PJRT_DEVICE"] = "TPU"

    import dataclasses

    HW_CFG = HardwareConfig()
    DATA_CFG = MLMDataConfig()
    
    driver = HardwareDriver(HW_CFG, DATA_CFG)
    driver.launch(train_worker)