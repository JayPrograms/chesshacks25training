"""
PyTorch policy/value network for PPO-style training.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from rl_chess.move_encoding import MAX_MOVES


def _masked_logits(
    logits: torch.Tensor, mask: Optional[torch.Tensor]
) -> torch.Tensor:
    if mask is None:
        return logits
    # Use a large negative number to zero-out invalid moves post-softmax.
    illegal = (1.0 - mask) * -1e9
    return logits + illegal


class PolicyValueNet(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        board_channels: int = 12,
        board_size: int = 8,
        extra_feats: int = 7,
        conv_channels: int = 64,
        linear_dim: int = 512,
        action_dim: int = MAX_MOVES,
    ):
        super().__init__()
        self.board_channels = board_channels
        self.board_size = board_size
        self.extra_feats = extra_feats
        self.board_flat = board_channels * board_size * board_size

        self.conv = nn.Sequential(
            nn.Conv2d(board_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
        )
        conv_output_dim = conv_channels * board_size * board_size
        self.fc = nn.Sequential(
            nn.Linear(conv_output_dim + extra_feats, linear_dim),
            nn.LayerNorm(linear_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(linear_dim, action_dim)
        self.value_head = nn.Linear(linear_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        legal_moves_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        board = obs[..., : self.board_flat].reshape(
            -1,
            self.board_channels,
            self.board_size,
            self.board_size,
        )
        extra = obs[..., self.board_flat:]
        conv_out = self.conv(board).reshape(board.shape[0], -1)
        features = torch.cat([conv_out, extra], dim=-1)
        features = self.fc(features)
        logits = self.policy_head(features)
        if legal_moves_mask is not None:
            logits = _masked_logits(logits, legal_moves_mask)
        value = self.value_head(features).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(
        self,
        obs: torch.Tensor,
        legal_moves_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs, legal_moves_mask)
        dist = torch.distributions.Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value


def make_optimizer(
    model: nn.Module, lr: float = 3e-4, weight_decay: float = 0.0
):
    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
