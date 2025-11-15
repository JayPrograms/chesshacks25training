"""
Minimal PPO training loop for the custom chess environment.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from rl_chess import ChessEnv
from rl_chess.models import PolicyValueNet, make_optimizer


@dataclass
class PPOConfig:
    total_steps: int = 10_000
    rollout_length: int = 512
    mini_batch_size: int = 256
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    checkpoint_dir: Path = Path("checkpoints")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hf_save_dir: Optional[Path] = None
    hf_repo_id: Optional[str] = None
    hf_push_to_hub: bool = False
    hf_private: bool = False
    hf_commit_message: str = "Add chess policy checkpoint"
    hf_token: Optional[str] = None


class RolloutBuffer:
    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[float] = []
        self.values: List[float] = []
        self.masks: List[np.ndarray] = []

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        done: bool,
        value: float,
        mask: np.ndarray,
    ):
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(float(done))
        self.values.append(value)
        self.masks.append(mask)

    def clear(self):
        self.__init__()


def to_tensor(array, device):
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def collect_rollout(env: ChessEnv, policy: PolicyValueNet, buffer: RolloutBuffer, rollout_len: int, device):
    obs, info = env.reset()
    mask = info["legal_moves_mask"]
    for _ in range(rollout_len):
        obs_tensor = to_tensor(obs, device).unsqueeze(0)
        mask_tensor = to_tensor(mask, device).unsqueeze(0)
        action, log_prob, value = policy.act(obs_tensor, mask_tensor)
        next_obs, reward, terminated, truncated, info = env.step(action.item())
        done = terminated or truncated
        buffer.add(obs, action.item(), log_prob.item(), reward, done, value.item(), mask)
        if done:
            next_obs, info = env.reset()
        obs, mask = next_obs, info["legal_moves_mask"]


def compute_gae(buffer: RolloutBuffer, last_value: float, last_done: float, cfg: PPOConfig):
    advantages = []
    gae = 0.0
    rewards = buffer.rewards + [last_value]
    values = buffer.values + [last_value]
    dones = buffer.dones + [last_done]
    for step in reversed(range(len(buffer.rewards))):
        delta = rewards[step] + cfg.gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + cfg.gamma * cfg.gae_lambda * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, buffer.values)]
    return np.array(advantages, dtype=np.float32), np.array(returns, dtype=np.float32)


def ppo_update(policy: PolicyValueNet, optimizer, buffer: RolloutBuffer, cfg: PPOConfig):
    device = cfg.device
    obs = to_tensor(np.stack(buffer.obs), device)
    actions = torch.as_tensor(buffer.actions, dtype=torch.int64, device=device)
    masks = to_tensor(np.stack(buffer.masks), device)
    old_log_probs = to_tensor(buffer.log_probs, device)
    advantages, returns = compute_gae(buffer, last_value=0.0, last_done=0.0, cfg=cfg)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = to_tensor(advantages, device)
    returns = to_tensor(returns, device)

    total_batch = obs.size(0)
    for _ in range(cfg.update_epochs):
        indices = torch.randperm(total_batch)
        for start in range(0, total_batch, cfg.mini_batch_size):
            end = start + cfg.mini_batch_size
            mb_idx = indices[start:end]

            logits, values = policy(obs[mb_idx], masks[mb_idx])
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions[mb_idx])
            entropy = dist.entropy().mean()

            ratios = torch.exp(log_probs - old_log_probs[mb_idx])
            surr1 = ratios * advantages[mb_idx]
            surr2 = torch.clamp(ratios, 1 - cfg.clip_coef, 1 + cfg.clip_coef) * advantages[mb_idx]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns[mb_idx])
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()


def save_checkpoint(policy: PolicyValueNet, step: int, cfg: PPOConfig):
    cfg.checkpoint_dir.mkdir(exist_ok=True, parents=True)
    ckpt_path = cfg.checkpoint_dir / f"policy_step_{step}.pt"
    torch.save(policy.state_dict(), ckpt_path)
    print(f"[checkpoint] saved {ckpt_path}")


def maybe_export_to_hub(
    policy: PolicyValueNet,
    cfg: PPOConfig,
    step_label: str,
):
    if cfg.hf_save_dir is not None:
        export_dir = cfg.hf_save_dir.expanduser()
        export_dir.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(export_dir)
        print(f"[hf] saved pretrained bundle to {export_dir}")
    if cfg.hf_push_to_hub:
        if not cfg.hf_repo_id:
            raise ValueError("--hf-repo-id must be set when --hf-push-to-hub is used.")
        push_kwargs = {
            "commit_message": cfg.hf_commit_message.format(step=step_label),
            "private": cfg.hf_private,
        }
        if cfg.hf_token:
            push_kwargs["token"] = cfg.hf_token
        print(
            f"[hf] pushing weights to {cfg.hf_repo_id} "
            f"(private={cfg.hf_private})..."
        )
        policy.push_to_hub(cfg.hf_repo_id, **push_kwargs)
        print("[hf] push complete.")


def parse_args() -> PPOConfig:
    parser = argparse.ArgumentParser(description="Train a chessbot with PPO.")
    parser.add_argument("--total-steps", type=int, default=25_000)
    parser.add_argument("--rollout-length", type=int, default=512)
    parser.add_argument("--mini-batch-size", type=int, default=256)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument(
        "--hf-save-dir",
        type=str,
        default=None,
        help="Optional directory to store a Hugging Face save_pretrained bundle.",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default=None,
        help="Hugging Face repo id (e.g. username/chessbot-ppo).",
    )
    parser.add_argument(
        "--hf-push-to-hub",
        action="store_true",
        help="Push the final trained weights to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Mark the Hugging Face repo as private when pushing.",
    )
    parser.add_argument(
        "--hf-commit-message",
        type=str,
        default="Add chess policy checkpoint",
        help="Commit message to use when pushing to the Hub.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token to use for push_to_hub.",
    )
    args = parser.parse_args()

    cfg = PPOConfig(
        total_steps=args.total_steps,
        rollout_length=args.rollout_length,
        mini_batch_size=args.mini_batch_size,
        update_epochs=args.update_epochs,
        learning_rate=args.lr,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
        checkpoint_dir=Path(args.checkpoint_dir),
        hf_save_dir=Path(args.hf_save_dir).expanduser()
        if args.hf_save_dir
        else None,
        hf_repo_id=args.hf_repo_id,
        hf_push_to_hub=args.hf_push_to_hub,
        hf_private=args.hf_private,
        hf_commit_message=args.hf_commit_message,
        hf_token=args.hf_token,
    )
    return cfg


def main():
    cfg = parse_args()
    env = ChessEnv()
    policy = PolicyValueNet().to(cfg.device)
    optimizer = make_optimizer(policy, lr=cfg.learning_rate)

    total_steps = 0
    buffer = RolloutBuffer()
    start_time = time.time()

    while total_steps < cfg.total_steps:
        collect_rollout(env, policy, buffer, cfg.rollout_length, cfg.device)
        total_steps += cfg.rollout_length
        ppo_update(policy, optimizer, buffer, cfg)
        buffer.clear()

        if total_steps % (cfg.rollout_length * 10) == 0:
            save_checkpoint(policy, total_steps, cfg)
            elapsed = time.time() - start_time
            print(f"[info] steps={total_steps} elapsed={elapsed:.1f}s")

    save_checkpoint(policy, total_steps, cfg)
    if cfg.hf_save_dir or cfg.hf_push_to_hub:
        maybe_export_to_hub(policy, cfg, step_label=f"step-{total_steps}")
    print("Training complete.")


if __name__ == "__main__":
    main()

