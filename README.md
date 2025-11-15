# RL Chessbot Training Starter

This repo contains boilerplate to train a chessbot with reinforcement learning (PPO) using `gym`, `python-chess`, and PyTorch.

## 1. Environment Setup

1. Create/activate a Python 3.10+ virtualenv.
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) For GPU training install the CUDA-enabled PyTorch wheel matching your system from [pytorch.org](https://pytorch.org/get-started/locally/).

Project layout:

- `rl_chess/chess_env.py` – custom `gym.Env` that wraps `python-chess`
- `rl_chess/move_encoding.py` – action-index ↔ move helpers
- `rl_chess/models/policy_net.py` – shared encoder with policy/value heads
- `train.py` – PPO-style loop wiring the pieces together

## 2. Core Concepts to Customize

- **Observations** (`ChessEnv._get_obs`) – 12×8×8 bitboards + metadata. Add new scalar channels if you want more context (e.g., repetition counts, captured pieces).
- **Action space** – Discrete index over all pseudo-legal moves; legal-move masks from `info["legal_moves_mask"]` keep the policy valid.
- **Rewards** – Default is +1/-1/0 for win/loss/draw and -1 for illegal moves. Modify `RewardConfig` or plug a custom function when instantiating `ChessEnv`.
- **Opponent policy** – By default `ChessEnv` uses `opponents.greedy_material_policy`, a simple capture-first heuristic. Swap in other helpers from `rl_chess/opponents.py` or supply your own callable:
  ```python
  from rl_chess import ChessEnv, opponents

  env = ChessEnv(opponent_policy=opponents.greedy_material_policy)

  def my_custom_policy(board):
      # inspect python-chess board state and return a chess.Move
      return opponents.random_policy(board)

  env = ChessEnv(opponent_policy=my_custom_policy)
  ```
  Swap in scripted openings, Stockfish via UCI, or your model for self-play.
- **Network** – `PolicyValueNet` is a CNN encoder (Conv2d layers over the 12×8×8 board planes plus scalar extras) feeding policy/value heads. You can swap in deeper CNNs, transformers, or other inductive biases as you iterate.

## 3. Running a Training Session

```bash
python train.py --total-steps 100000 --rollout-length 1024 --mini-batch-size 512
```

During training:

- Rollouts are collected via `collect_rollout`, storing observations, masks, log probs, etc.
- `ppo_update` computes GAE advantages, applies clipping, and performs multiple epochs of SGD per batch.
- Checkpoints land in `checkpoints/policy_step_*.pt`. Load them back with `PolicyValueNet.load_state_dict`.

## 4. Experiment Tips

- **Curriculum** – Reduce `max_moves` or restrict legal moves early on to stabilize learning.
- **Reward shaping** – Incorporate material balance, center control, or check bonuses for denser feedback.
- **Evaluation** – Write a small script that plays the current model against baseline heuristics to gauge improvement.
- **Logging** – Integrate TensorBoard or Weights & Biases by recording episode returns, entropy, and value loss inside the main loop.

## 5. Next Steps

- Implement self-play by alternating colors and sharing the network weights.
- Add dataset bootstrapping: pretrain policy head using PG from master games to shorten exploration.
- Deploy the trained agent by wrapping `ChessEnv` inside a web or CLI interface that feeds UCI moves.

## 6. Publishing Checkpoints to Hugging Face

`PolicyValueNet` mixes in Hugging Face’s `PyTorchModelHubMixin`, so you can call the standard `save_pretrained` / `push_to_hub` helpers exactly like the Transformers examples. Make sure you have `huggingface-hub` installed (`pip install huggingface-hub`) and authenticate once via `hf auth login` (stores a token that both `train.py` and `hg.py` can reuse).

### Option A – Export directly from `train.py`

Run training with the new flags to produce a Hub-ready artifact automatically and (optionally) push it after the final checkpoint:

```bash
python train.py \
  --total-steps 100000 \
  --hf-save-dir artifacts/chessbot \
  --hf-repo-id your-username/chessbot-ppo \
  --hf-push-to-hub \
  --hf-commit-message "Upload step {step} weights"
```

Flags overview:

- `--hf-save-dir`: calls `policy.save_pretrained` so the repo contains `pytorch_model.bin` + `config.json`.
- `--hf-repo-id`: target Hub repo (`username/model-name`).
- `--hf-push-to-hub`: push right after training finishes (token picked up from `HF_TOKEN`, `HUGGINGFACEHUB_API_TOKEN`, or `huggingface-cli login`).
- `--hf-private`, `--hf-token`, `--hf-commit-message`: match the `hg.py` helper semantics.

### Option B – Use the standalone helper

If you prefer a manual step, the helper script still works:

```bash
python hg.py \
  --repo-id your-username/chessbot-ppo \
  --checkpoint checkpoints/policy_step_100000.pt
```

The script verifies the checkpoint, ensures the repo exists, and then calls `push_to_hub` under the hood. Use `--private` or `--commit-message` to customize the upload.

### Using the model in your competition bot

In any downstream app (e.g., your ACC chess bot), install this repo or copy `rl_chess` into your project, then load the uploaded weights:

```python
from rl_chess.models import PolicyValueNet

MODEL_ID = "your-username/chessbot-ppo"

policy = PolicyValueNet.from_pretrained(
    MODEL_ID,
    cache_dir="./.model_cache",
)
policy.eval()

def get_move(board_tensor, legal_mask):
    action, _, _ = policy.act(board_tensor, legal_mask, deterministic=True)
    return action.item()
```

The Hub cache means your bot always grabs the latest published weights without bloating the contest repository. Document the training config (steps, reward shaping, opponent policy) in the Hugging Face model card so others can reproduce results.