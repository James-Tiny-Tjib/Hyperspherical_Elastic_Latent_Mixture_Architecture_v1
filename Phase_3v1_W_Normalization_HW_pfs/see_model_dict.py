import torch
from huggingface_hub import hf_hub_download, create_repo, HfApi



try:
    hf_hub_download(
        repo_id = "JamesResearch1216/HELM-v1-Architecture",
        filename = "JamesResearch1216/HELM-v1-Architecture/blob/main/checkpoint-100.pt",
        repo_type = "model",
        token = "" ,
        local_dir = "."
    )
except Exception as e:
    raise RuntimeError(f"Critical HF Download Failure for {filename}: {e}")

# Load the .pt file
checkpoint = torch.load('checkpoint-100.pt')

# Print all keys (common keys: 'model', 'state_dict', 'optimizer', etc.)
print(checkpoint.keys())

# If the file is a state_dict itself, list parameter names
if isinstance(checkpoint, dict):                
    for key in checkpoint.keys():
        print(key)
