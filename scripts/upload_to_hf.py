"""Upload a local model directory to the Hugging Face Hub.

Reads ``HF_TOKEN`` from the environment or from a local ``.env`` file. Never pass
a token on the command line -- it ends up in your shell history.

Usage:
    python scripts/upload_to_hf.py \\
        --model-dir whisper-large-v3-quantized-bf16 \\
        --repo-id your-username/whisper-large-v3-quantized-bf16

    python scripts/upload_to_hf.py \\
        --model-dir whisper-large-v3-quantized-bf16 \\
        --repo-id your-username/whisper-large-v3-quantized-bf16 \\
        --private
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a model directory to Hugging Face Hub")
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to the local model directory",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HF repo ID, e.g. username/model-name",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repo as private",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Add it to your environment or to a .env file.")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        exist_ok=True,
        private=args.private,
    )
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
