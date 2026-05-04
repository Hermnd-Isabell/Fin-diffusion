# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**PI-LCDM** (Physics-Informed Latent Conditional Diffusion Model) generates arbitrage-free implied volatility surfaces (IVS) for Chinese 50ETF options. The model is a Latent Diffusion Transformer (DiT) operating in a VAE-compressed latent space, with no-arbitrage constraints injected at both training time (Dynamic-SNR-weighted physics loss) and inference time (a 5-stage no-arbitrage projection pipeline). The downstream goal is **TSTR** (Train on Synthetic, Test on Real) for deep-hedging policies — see `PINA_CORRECTOR_REPORT.md` and `data_factory/SVI_EVOLUTION.md` for the engineering history of two of the most subtle subsystems.

## Environment

- Python 3.12 in a local `venv/` (Windows). Activate with `venv\Scripts\activate`.
- Pinned heavy deps: `torch==2.2.0+cpu`, `torchvision==0.17.0+cpu`, `numpy==1.26.4`, `scipy==1.17.0`, `pandas==3.0.1`, `matplotlib`, `openpyxl`, `tqdm`.
- The codebase auto-detects CUDA (`torch.cuda.is_available()`) but the bundled venv is CPU-only. The deep-hedging script forces `cpu` explicitly.

## Common Commands

End-to-end pipeline (run in this order on a fresh checkout):

```bash
# 1. Build the (32×32) IVS tensor dataset from 50etf_options.xlsx (slow, ~minutes)
python main_train.py --process_data
#    --limit_data N   restricts to last N trading dates for quick smoke runs.

# 2. Pre-train the VAE (writes checkpoints/vae_pretrained.pth)
python main_train.py --pretrain_vae

# 3. Train the full PI-LCDM (DiT + Conditioner + physics loss, 100 epochs)
python main_train.py --train_full

# 4. Generate 1000 synthetic IVS + run statistical/arbitrage validation
python run_validation.py

# 5. Phase-6.3 Deep-Hedging TSTR backtest
#    Requires step 4 to have produced results/validation_generation/synthetic_1000.pt
python run_deep_hedging.py
```

Module-level smoke tests (each is a `__main__`-runnable script, not pytest):

```bash
python data_factory/test_solver.py     # SVI on a synthetic smile, writes test_svi_solver.png
python models/test_networks.py         # VAE / Conditioner / DiT shape checks
python physics/test_physics.py         # BS pricing + arbitrage gradient flow
python engine/test_engine.py           # End-to-end train_step + PINA sampler smoke test
python data_factory/run_pipeline.py --date YYYYMMDD   # SVI + grid mapping for one day
```

Useful one-off scripts at the root:

- `debug_violations.py`, `debug_pina.py`, `debug_projection.py` — diagnostic scripts that produced the metrics in `PINA_CORRECTOR_REPORT.md` (kept for reproducibility, can be deleted).
- `analyze_data.py`, `extract_pdf_text.py`, `convert_output.py`, `read_output.py`, `verify_tensor.py` — ad-hoc data exploration helpers.

## Architecture

### Data flow

```
50etf_options.xlsx
  → RawLoader.clean_data       (per-date filter, IV inversion via Brent)
  → SVICalibrator              (one fit per expiry slice)
  → GridMapper.interpolate_surface  (SVI params → 32×32 IVS in log-moneyness × tenor)
  → torch.save → data/processed_tensors.pt   {iv_surface (N,1,32,32), conditions (N,3), dates}
  → MarketConditioner / VAE / LatentDiT (training)
  → PILCDMSampler (inference)
  → results/validation_generation/synthetic_1000.pt
  → DeepHedgingEnv / PPOTrainer (TSTR backtest)
```

### Tensor / grid conventions (these are load-bearing)

- IVS tensor shape is **(B, 1, T_dim=32, K_dim=32)**. Time-to-maturity varies along dim=2 (rows), strike along dim=3 (cols).
- Default normalised grid: `k = log(K/S) ∈ linspace(-0.3, 0.3, 32)`, `T ∈ linspace(0.05, 1.0, 32) years`, `S=1.0`, `r=0.0`. Both training (`main_train.py`) and inference (`run_validation.py`, `run_deep_hedging.py`) use this exact grid.
- Conditions are a 3-vector `(Spot, ATM_Vol, Slope)`, computed in `main_train.py::process_data` as `surface[0,16]` and `surface[-1,16]-surface[0,16]`. They are **z-score normalised** (per-feature mean/std over the training set) before being fed to `MarketConditioner`.

### Models (`models/`)

- `VAE`: 3-stride conv encoder/decoder, compresses (B,1,32,32) ↔ latent (B,4,4,4). Decoder ends with **Softplus** to guarantee σ>0. Trained with `recons_loss + 0.00025 * KLD`.
- `MarketConditioner`: 2-layer SiLU MLP that lifts `(B,3)` → `(B,1,128)` for cross-attention (sequence length 1 — global context).
- `LatentDiT`: ViT-style stack on the 4×4=16 latent patches, embed_dim=128, depth=4, 4 heads. Each `DiTBlock` is `self-attn → cross-attn(context) → MLP` with residuals.

### Diffusion + physics (`physics/`, `engine/`)

- `LogSpaceGBM` (`physics/sde_solver.py`): Variance-Exploding SDE with geometric noise schedule, `std(t)=σ_min·(σ_max/σ_min)^t` (`σ_min=0.01, σ_max=2.0`). Mean is identity (additive noise). The "GBM-Guided" name refers to the schedule shape, not a literal asset-price simulation.
- `BSModule` (`physics/market_ops.py`): fully differentiable Black-Scholes call price, delta, vega. Used by both `ArbitrageLoss` (training) and `_prices_to_iv_newton` (sampler stage 5).
- `ArbitrageLoss` (`physics/arbitrage_loss.py`): three ReLU-penalty terms — Butterfly (conv2d with kernel `[1,-2,1]` along K), Calendar (forward diff along T), Vertical (negative diff along K). Default weights `{fly:1, cal:1, vert:1}`; the sampler overrides to `{fly:5, cal:1, vert:1}` because butterfly is the dominant failure mode.
- `PILCDMTrainer` (`engine/trainer.py`): `total = MSE(noise_pred, noise) + snr_weight · arb_loss`, where `snr_weight = 1/(1+std_t²)` (Dynamic SNR — focuses physics loss on the low-noise regime). Estimates `z_0_hat = z_t − std_t · noise_pred`, decodes to IVS, sanitises with `nan_to_num` + `clamp(1e-4, 5.0)` before pricing.

### **Sampler — the most important file: `engine/sampler.py`**

A 5-stage pipeline (read `PINA_CORRECTOR_REPORT.md` before changing anything here — earlier gradient-only versions only achieved ~42% arbitrage-free rate; the price-space projection brought it to 92%):

1. Reverse diffusion (50 DDIM-style steps) **with latent-space PINA correction** for `t<0.4` (single step) and `t<0.2` (4 inner steps), dynamic LR `pina_lr·(1+γ·loss_arb)`.
2. VAE decode → Gaussian smoothing (σ=0.6) on the IVS to suppress VAE-induced HF noise (this single step cut max Gamma spike from 223 → 90).
3. BS forward pricing IVS → call-price surface.
4. **`_project_prices_to_convex` — direct algebraic projection in PRICE space** (not IV space, not latent space). Multi-pass `torch.maximum` enforces K-monotonicity (running max right-to-left), T-monotonicity (running max left-to-right), and K-convexity (12 bidirectional sweeps lifting interior points to the chord midpoint). **Critical correctness invariant: use `maximum`, never `minimum` — `minimum` enforces concavity and inverts the constraint** (this bug raised butterfly violation rate from 43% to 80% during development).
5. Newton-Raphson BS inversion (`_prices_to_iv_newton`, 10 iters, Brenner-Subrahmanyam initial guess) recovers the corrected IVS.

When tweaking the sampler, the load-bearing parameters in production are `pina_lr=0.05, pina_inner_steps=4, gamma_scale=5.0, num_steps=50`.

### Validation (`validation/`)

- `statistics.py`: Wasserstein distances on ATM vol / skew / global; ACF and squared-diff ACF for vol-clustering; spot-IV correlation for leverage effect.
- `financial.py`: per-grid-cell violation rates for the three arbitrage types, Gamma smoothness via finite differences (autograd-based Gamma is also implemented but used only as a sanity check).
- `deep_hedging.py`: standalone Phase-6.3 module containing `RealDataGenerator`, `SynthDataGenerator`, `DeepHedgingEnv` (30-day delta-hedging episode with mean-reverting OU vol, ρ=−0.5 spot-vol correlation, terminal squared-error reward, transaction cost 3 bps), a minimal pure-PyTorch PPO `ActorCritic` + `PPOTrainer`, and `run_backtest` against a Black-Scholes delta baseline.

## Repository-specific gotchas

- **`MarketConditioner` weights are not persisted by `main_train.py`.** The training loop only saves `dit_epoch_*.pth` and `vae_epoch_*.pth` (see `run_validation.py::load_models` for the relevant lament). Inference therefore uses a freshly-initialised conditioner — generated samples are **not** truly conditional on the requested market state. If you fix this, also update `run_validation.py` and `run_deep_hedging.py`.
- The Excel data file (`50etf_options.xlsx`, 65 MB) is read-only (`-r--r--r--`); do not attempt to overwrite it.
- `data_factory/run_pipeline.py` uses **relative imports** (`from raw_loader import ...`) and must be invoked from the `data_factory/` directory or with `sys.path` set up — `main_train.py` uses absolute imports (`from data_factory.raw_loader import ...`) and is the canonical entry point.
- The `latest_cp` selector in `run_validation.py::load_models` parses filenames like `dit_epoch_100.pth`. If you rename checkpoints, update the `int(x.split('_')[2].split('.')[0])` parser.
- `run_deep_hedging.py` requires `results/validation_generation/synthetic_1000.pt` to exist — run `run_validation.py` first.
- The SVI calibrator (`data_factory/svi_calibrator.py`) is the v2.0 robust version: Huber loss + soft penalty constraints + RANSAC-style 2σ outlier filter + smart `(a, m)` initialisation anchored to `argmin(σ²_market)`. Reverting any of these will reintroduce the "floating curve" / "flat line" failures documented in `data_factory/SVI_EVOLUTION.md`. Calibration returns `(None, inf)` whenever `b<1e-4` or `σ>2.0` — callers must handle this.
- `GridMapper` interpolates **total variance** `w = σ²·T` (not σ directly) and falls back to constant-vol extrapolation `w_target = w_ref · (t_target / t_ref)` outside the calibrated tenor range.
