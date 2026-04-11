import os
import sys
import subprocess
import glob

# 🎛️ SET THIS TO TRUE FOR TPU, FALSE FOR GPU
FORCE_TPU = False

def repair_environment():

    if FORCE_TPU:
        print("🔍 Starting High-Speed TPU Repair...")

        # 1. Faster Uninstallation
        print("🧹 Wiping libraries...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                        "torch", "torch_xla", "torchvision", "numpy", "tensorflow", "huggingface_hub"],
                       capture_output=True)

        # 2. Parallel/Bulk Installation
        print("📥 Installing Synced TPU Stack...")
        common_args = ["install", "-q", "--no-warn-script-location"]

        if FORCE_TPU or glob.glob("/dev/accel*"):
            cmd = [
                sys.executable, "-m", "pip", *common_args,
                "torch==2.8.0",
                "torchvision==0.23.0",
                "torch_xla[tpu]==2.8.0",
                # 🩹 THE FIX: Unpinned numpy, pyarrow, and fsspec to allow Numpy 2.0+ compatibility
                "numpy", "pyarrow", "fsspec",
                "datasets", "transformers", "huggingface_hub>=0.28.0", "wandb",
                "cloud-tpu-client", "scikit-learn", "pandas<3.0.0",
                "-f", "https://storage.googleapis.com/libtpu-releases/index.html",
                "--extra-index-url", "https://download.pytorch.org/whl/cpu"
            ]
            subprocess.check_call(cmd)
        else:
            # Fallback
            cmd = [
                sys.executable, "-m", "pip", *common_args, "-U",
                "torch", "datasets", "pyarrow", "transformers", "huggingface_hub>=0.28.0", "fsspec", "wandb", "scipy", "numpy", "pandas<3.0.0"
            ]
            subprocess.check_call(cmd)

        print("\n✅ TPU REPAIR COMPLETE.")
        print("⚠️ Click 'Run' -> 'Restart Session' NOW.")

    else:
        print("🔍 Starting Robust GPU Repair...")

        # 1. Clean Wipe
        print("🧹 Wiping conflicting libraries...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q",
                        "torch", "torchvision", "torchaudio", "huggingface_hub"],
                       capture_output=True)

        # 2. Setup Arguments
        common_args = ["install", "-q", "--no-warn-script-location"]

        try:
            print("📥 Installing GPU/CUDA Stack...")

            print("   ⚡ Part 1: PyTorch Core...")
            subprocess.check_call([
                sys.executable, "-m", "pip", *common_args,
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ])

            print("   ⚡ Part 2: Transformers & Data...")
            subprocess.check_call([
                sys.executable, "-m", "pip", *common_args, "-U",
                "datasets", "transformers", "huggingface_hub>=0.28.0",
                "wandb", "pandas<3.0.0"
            ])

            print("\n✅ GPU REPAIR COMPLETE.")
            print("⚠️ MANDATORY: Click 'Run' -> 'Restart Session' NOW.")

        except subprocess.CalledProcessError as e:
            print(f"\n❌ Installation failed. Error: {e}")
            print("💡 Try manually restarting the session and running this cell again.")

if __name__ == "__main__":
    repair_environment()