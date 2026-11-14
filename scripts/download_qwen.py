"""Download the baseline Qwen checkpoint to the project models directory."""
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DESTINATION = ROOT / "models/Qwen2.5-0.5B-Instruct"


def main():
    # Resolve once and pin the snapshot, so all files come from the same revision.
    revision = HfApi().model_info(MODEL_ID).sha
    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=revision,
        local_dir=DESTINATION,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.md", "*.jinja"],
        max_workers=4,
    )
    model_dir = Path(path)
    if not (model_dir / "config.json").is_file() or not list(model_dir.glob("*.safetensors")):
        raise RuntimeError("Download did not produce a config and safetensors weights.")
    manifest = {
        "model_id": MODEL_ID,
        "revision": revision,
        "local_directory": str(model_dir),
        "files": [
            {"name": item.name, "bytes": item.stat().st_size}
            for item in sorted(model_dir.iterdir()) if item.is_file()
        ],
    }
    (ROOT / "docs/qwen_download.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
