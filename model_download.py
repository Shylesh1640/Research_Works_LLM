from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-8B"

# Store the model inside the current research project
MODEL_DIR = Path(__file__).resolve().parent / "models" / "Qwen3-8B"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading {MODEL_ID}")
print(f"Destination: {MODEL_DIR}")

snapshot_download(
    repo_id=MODEL_ID,
    local_dir=str(MODEL_DIR),
    local_dir_use_symlinks=False,
)

print("\n✅ Model downloaded successfully!")
print(f"📁 Location: {MODEL_DIR}")