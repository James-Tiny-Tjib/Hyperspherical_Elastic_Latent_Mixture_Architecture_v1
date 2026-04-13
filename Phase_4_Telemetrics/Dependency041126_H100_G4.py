import os
import sys
import subprocess

def repair_next_gen_gpu():
    print("🚀 Initializing Isolated Next-Gen GPU Setup (H100 & Blackwell)...")

    # 1. The "Clean Slate" Wipe
    # We must wipe huggingface_hub to kill that 'response' keyword error
    print("🧹 Wiping conflicting libraries...")
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                    "torch", "torchvision", "torchaudio", "huggingface_hub", "fsspec"],
                    capture_output=True)

    try:
        # 2. Optimized 2026 Blackwell Stack
        # We use --extra-index-url so pip can fallback to PyPI for non-CUDA dependencies
        # This resolves the 'exit status 1' by preventing dependency resolution locks
        print("📥 Part 1: Installing PyTorch Stack (sm_90/sm_120 support)...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "--no-warn-script-location",
            "torch", "torchvision", "torchaudio",
            "huggingface_hub>=0.28.0", # Fixes the HfHubHTTPError bug
            "--extra-index-url", "https://download.pytorch.org/whl/cu128"
        ])

        print("📥 Part 2: Finalizing Training dependencies...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "-U",
            "datasets", "transformers", "wandb", "pandas<3.0.0"
        ])

        # 3. Connection Stability for 10B streaming
        os.environ["HF_HUB_READ_TIMEOUT"] = "120"

        print("\n✅ H100/BLACKWELL READY.")
        print("⚠️ RESTART SESSION: You MUST click 'Run' -> 'Restart Session' now.")

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed. The index might be under maintenance.")
        print("💡 TRY THIS: Change 'cu128' to 'cu124' and run again for PTX fallback.")

if __name__ == "__main__":
    repair_next_gen_gpu()