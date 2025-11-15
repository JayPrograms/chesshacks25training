from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi

from rl_chess.models import PolicyValueNet


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload a trained checkpoint to the Hugging Face Hub."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target repository in the form <user>/<model>",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/policy_step_100000.pt"),
        help="Path to the checkpoint file to upload.",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="Repository type to create if it does not exist.",
    )
    parser.add_argument(
        "--commit-message",
        default="Add chess policy checkpoint",
        help="Commit message to use on the Hub.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create or update the repo as private.",
    )
    return parser.parse_args()


def ensure_checkpoint(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        sys.exit(f"[error] checkpoint not found: {resolved}")
    if not resolved.is_file():
        sys.exit(f"[error] checkpoint is not a file: {resolved}")
    return resolved


def require_token() -> str:
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or _read_stored_token()
    )
    if not token:
        sys.exit(
            "[error] No Hugging Face token found. "
            "Run `hf auth login` first or set "
            "HF_TOKEN / HUGGINGFACEHUB_API_TOKEN."
        )
    return token.strip()


def _read_stored_token() -> str | None:
    legacy_path = Path.home() / ".huggingface" / "token"
    cache_path = Path.home() / ".cache" / "huggingface" / "token"
    for path in (legacy_path, cache_path):
        if path.exists():
            return path.read_text().strip()
    return None


def load_policy(checkpoint: Path) -> PolicyValueNet:
    print(f"[info] loading checkpoint from {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    policy = PolicyValueNet()
    missing, unexpected = policy.load_state_dict(state, strict=False)
    if missing or unexpected:
        missing_msg = f"Missing keys: {missing or '[]'}"
        unexpected_msg = f"unexpected keys: {unexpected or '[]'}"
        print("[warn] Non-strict load.", missing_msg, unexpected_msg)
    policy.eval()
    return policy


def ensure_repo(repo_id: str, repo_type: str, token: str, private: bool):
    api = HfApi(token=token)
    print(
        f"[info] ensuring repo `{repo_id}` ({repo_type}) exists "
        f"(private={private})..."
    )
    api.create_repo(
        repo_id,
        repo_type=repo_type,
        exist_ok=True,
        private=private,
    )


def main():
    args = parse_args()
    checkpoint = ensure_checkpoint(args.checkpoint)
    token = require_token()

    ensure_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        token=token,
        private=args.private,
    )
    policy = load_policy(checkpoint)
    print(f"[info] pushing model to {args.repo_id} ...")
    policy.push_to_hub(
        args.repo_id,
        token=token,
        commit_message=args.commit_message,
        private=args.private,
    )
    print("[info] upload complete.")


if __name__ == "__main__":
    main()
