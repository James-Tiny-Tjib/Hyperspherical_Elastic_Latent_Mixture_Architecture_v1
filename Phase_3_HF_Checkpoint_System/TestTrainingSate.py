import os
import json
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class TrainingState:
    model_repo_id: str = "JamesResearch1216/HELM-Architecture"
    total_tokens_processed_global: int = 0
    current_curriculum_level: int = 0
    latest_step: int = 0
    best_step: int = 0

    # The Ledger: Tracks all checkpoints for the Sidecar
    ckpt_history: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, filepath: str):
        with open(filepath, "w") as f:
            json.dump(dataclasses.asdict(self), f, indent=4)

    @classmethod
    def from_json(cls, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls(**data)

# =====================================================================
# SIMULATED TRAINING LOOP
# Run this multiple times to watch the ledger grow!
# =====================================================================
if __name__ == "__main__":
    filepath = "Hyperspherical Elastic Language Model or Hyperspherical Elastic Latent Mixture/Phase 3/training_state.json"
    
    # ---------------------------------------------------------
    # Phase 1: The Boot-Up (What your CheckpointDriver will do)
    # ---------------------------------------------------------
    if os.path.exists(filepath):
        print(f"📄 Found existing {filepath}. Loading state...")
        state = TrainingState.from_json(filepath)
        print(f"   -> Resumed at Step {state.latest_step}. Tokens processed: {state.total_tokens_processed_global}")
    else:
        print(f"✨ No existing {filepath} found. Creating a fresh training state...")
        state = TrainingState()
        
    # ---------------------------------------------------------
    # Phase 2: Simulating Training Progress
    # ---------------------------------------------------------
    # Let's pretend the model just trained for 500 steps
    step_increment = 500
    new_step = state.latest_step + step_increment
    state.latest_step = new_step
    
    # We also processed a bunch of tokens
    tokens_per_step = 1024 * 16 # Just an example: seq_len * batch_size
    state.total_tokens_processed_global += (tokens_per_step * step_increment)
    
    # Let's fake a descending loss so it looks like it's learning
    fake_loss = max(1.5, 8.0 - (new_step * 0.001)) 
    
    # ---------------------------------------------------------
    # Phase 3: Writing the Checkpoint to the Ledger
    # ---------------------------------------------------------
    step_key = str(new_step)
    print(f"\n💾 Saving new checkpoint data for Step {step_key}...")
    
    # Update all older "latest" statuses to "deleted" (simulating local cleanup)
    for existing_step, data in state.ckpt_history.items():
        if data["status"] == "latest":
            data["status"] = "deleted"
    
    # Add the new checkpoint
    state.ckpt_history[step_key] = {
        "status": "latest",
        "file_path": f"checkpoint-{step_key}.pt",
        "hardware": "v5e-8", 
        "total_loss": round(fake_loss, 4),
        "ce_loss": round(fake_loss - 0.2, 4),
        "aux_loss": 0.15,
        "sparsity_loss": 0.05
    }
    
    # ---------------------------------------------------------
    # Phase 4: Save to Disk
    # ---------------------------------------------------------
    state.to_json(filepath)
    print(f"✅ Successfully updated {filepath}!")
    
    # Print out the ledger so you can see it working
    print("\n--- Current Checkpoint Ledger ---")
    for step, data in state.ckpt_history.items():
        print(f"Step {step.ljust(5)} | Status: {data['status'].ljust(8)} | Loss: {data['total_loss']}")
    print("---------------------------------")