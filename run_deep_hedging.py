"""
Phase 6.3 — Deep Hedging Backtest Runner
==========================================
Usage:
    python run_deep_hedging.py

Pipeline:
  1. Load real data → train/test temporal split (2015-2024 / 2025-2026)
  2. Build offline synthetic pool (load pre-generated PI-LCDM surfaces,
     bootstrap to 5000 via sampling-with-replacement)
  3. Train Agent A on real training data         (25 batches × 200 episodes)
  4. Train Agent B on synthetic pool             (25 batches × 200 episodes)
  5. Backtest both on hold-out test set          (252 real paths, 2025–2026)
  6. Compare vs. Black-Scholes delta hedge baseline
  7. Print final report + save results
"""

import os
import torch
import numpy as np

from validation.deep_hedging import (
    RealDataGenerator,
    SynthDataGenerator,
    DeepHedgingEnv,
    PPOTrainer,
    run_backtest,
    print_backtest_report,
)

# ── Paths ─────────────────────────────────────────────────────────────
PROCESSED_DATA   = "data/processed_tensors.pt"
SYNTH_POOL_PATH  = "results/validation_generation/synthetic_1000.pt"
RESULTS_DIR      = "results/deep_hedging"
AGENT_A_PATH     = os.path.join(RESULTS_DIR, "agent_a_policy.pth")
AGENT_B_PATH     = os.path.join(RESULTS_DIR, "agent_b_policy.pth")

os.makedirs(RESULTS_DIR, exist_ok=True)

N_TRAIN_BATCHES  = 25    # 25 × 200 rollouts = 5000 training episodes per agent
N_ROLLOUTS       = 200   # episodes per PPO batch
SYNTH_POOL_SIZE  = 5000  # bootstrap target size for the offline pool
TEST_SPLIT       = 252   # last 252 trading days as hold-out

DEVICE = torch.device('cpu')   # pure CPU (no CUDA needed for MLP)


# ══════════════════════════════════════════════════════════════════════
# Step 1: Load & Split Data
# ══════════════════════════════════════════════════════════════════════
print("=" * 65)
print("  PHASE 6.3 — DEEP HEDGING BACKTEST (TSTR)")
print("=" * 65)

print("\n[1/5] Loading real data...")
data       = torch.load(PROCESSED_DATA)
ivs_all    = data['iv_surface']          # (2667, 1, 32, 32)
cond_all   = data['conditions']          # (2667, 3)
dates_all  = data['dates']

N_total    = ivs_all.shape[0]
N_test     = TEST_SPLIT
N_train    = N_total - N_test

train_ivs  = ivs_all[:N_train]          # (2415, 1, 32, 32)
train_cond = cond_all[:N_train]         # (2415, 3)
test_ivs   = ivs_all[N_train:]          # (252, 1, 32, 32)
test_cond  = cond_all[N_train:]         # (252, 3)

print(f"  Train: {N_train} surfaces  [{dates_all[0]} → {dates_all[N_train-1]}]")
print(f"  Test:  {N_test} surfaces   [{dates_all[N_train]} → {dates_all[-1]}]")


# ══════════════════════════════════════════════════════════════════════
# Step 2: Build Offline Synthetic Pool
# ══════════════════════════════════════════════════════════════════════
print(f"\n[2/5] Building offline synthetic pool (target: {SYNTH_POOL_SIZE} surfaces)...")

if not os.path.exists(SYNTH_POOL_PATH):
    raise FileNotFoundError(
        f"Synthetic surfaces not found at {SYNTH_POOL_PATH}.\n"
        "Run `python run_validation.py` first to generate them."
    )

synth_base = torch.load(SYNTH_POOL_PATH)            # (1000, 1, 32, 32)
print(f"  Loaded {synth_base.shape[0]} PI-LCDM surfaces from disk.")

# Bootstrap to SYNTH_POOL_SIZE via sampling-with-replacement
if synth_base.shape[0] < SYNTH_POOL_SIZE:
    n_base = synth_base.shape[0]
    extra_idx = torch.randint(0, n_base, (SYNTH_POOL_SIZE - n_base,))
    synth_pool = torch.cat([synth_base, synth_base[extra_idx]], dim=0)
else:
    synth_pool = synth_base[:SYNTH_POOL_SIZE]

print(f"  Synthetic pool size: {synth_pool.shape[0]} surfaces  (bootstrap ×{SYNTH_POOL_SIZE//synth_base.shape[0]:.1f})")


# ══════════════════════════════════════════════════════════════════════
# Step 3: Train Agent A — Real Data
# ══════════════════════════════════════════════════════════════════════
print(f"\n[3/5] Training Agent A (Real Data) — {N_TRAIN_BATCHES} batches × {N_ROLLOUTS} episodes...")

real_gen_train = RealDataGenerator(train_ivs, train_cond)

def make_env_real():
    return DeepHedgingEnv(real_gen_train, device=DEVICE)

trainer_a = PPOTrainer(
    env_factory  = make_env_real,
    device       = DEVICE,
    n_rollout    = N_ROLLOUTS,
    n_epoch      = 5,
)
trainer_a.train(n_batches=N_TRAIN_BATCHES, verbose=True)
torch.save(trainer_a.policy.state_dict(), AGENT_A_PATH)
print(f"  Agent A saved → {AGENT_A_PATH}")


# ══════════════════════════════════════════════════════════════════════
# Step 4: Train Agent B — Synthetic Data
# ══════════════════════════════════════════════════════════════════════
print(f"\n[4/5] Training Agent B (Synthetic Data) — {N_TRAIN_BATCHES} batches × {N_ROLLOUTS} episodes...")

synth_gen = SynthDataGenerator(synth_pool)

def make_env_synth():
    return DeepHedgingEnv(synth_gen, device=DEVICE)

trainer_b = PPOTrainer(
    env_factory  = make_env_synth,
    device       = DEVICE,
    n_rollout    = N_ROLLOUTS,
    n_epoch      = 5,
)
trainer_b.train(n_batches=N_TRAIN_BATCHES, verbose=True)
torch.save(trainer_b.policy.state_dict(), AGENT_B_PATH)
print(f"  Agent B saved → {AGENT_B_PATH}")


# ══════════════════════════════════════════════════════════════════════
# Step 5: Backtest on Hold-Out Test Set
# ══════════════════════════════════════════════════════════════════════
print(f"\n[5/5] Running backtest on {N_test} hold-out paths...")

results_a = run_backtest(
    agent_policy  = trainer_a.policy,
    test_real_ivs = test_ivs,
    test_real_cond= test_cond,
    label         = 'Agent A (Real)',
)

results_b = run_backtest(
    agent_policy  = trainer_b.policy,
    test_real_ivs = test_ivs,
    test_real_cond= test_cond,
    label         = 'Agent B (Synthetic)',
)

print_backtest_report(results_a, results_b)

# ── Save raw PnL arrays ───────────────────────────────────────────────
np.save(os.path.join(RESULTS_DIR, "pnl_agent_a.npy"), results_a['terminal_pnls'])
np.save(os.path.join(RESULTS_DIR, "pnl_agent_b.npy"), results_b['terminal_pnls'])
np.save(os.path.join(RESULTS_DIR, "pnl_bs.npy"),      results_a['bs_pnls'])
print(f"PnL arrays saved to {RESULTS_DIR}/")
