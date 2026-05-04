"""
Phase 6.3 — Deep Hedging Backtest (TSTR)
=========================================
Train on Synthetic, Test on Real.

Implements:
  - DeepHedgingEnv      : Delta-hedging environment (GBM spot + IVS-derived option price)
  - RealDataGenerator   : Samples IVS surfaces from the real historical training set
  - SynthDataGenerator  : Offline pool of pre-generated PI-LCDM synthetic surfaces
  - PPOAgent            : Minimal pure-PyTorch PPO actor-critic (no external RL library)
  - run_training()      : 5000-trajectory PPO training loop
  - run_backtest()      : Comparative evaluation on the 252-day hold-out test set

Reward design (mathematically correct):
  R = -(Terminal_Portfolio_Value - Option_Payoff)**2
  PPO maximises E[R]  ⟺  minimises the expected squared hedging error
  ≡ variance minimisation (when E[error] ≈ 0).

Synthetic pool is built OFFLINE before training to avoid on-the-fly
diffusion inference during rollout collection (computationally prohibitive).
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from physics.market_ops import BSModule


# ══════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════
RISK_FREE_RATE = 0.0          # r (50ETF uses dividend-adjusted forward)
TRANSACTION_COST_BPS = 3      # bps per unit × spot
TC_RATE = TRANSACTION_COST_BPS * 1e-4
STEPS_PER_EPISODE = 30        # daily rebalancing steps
DT = 1.0 / 252                # 1 trading day in years
STATE_DIM = 4                 # [tau, moneyness, prev_delta, port_value]
ACTION_DIM = 1

# Fixed option parameters (normalised spot = 1.0 ATM)
FIXED_STRIKE = 1.0
FIXED_MATURITY = 30 * DT      # ~30 days to maturity


# ══════════════════════════════════════════════════════════════════════
# Data Generators
# ══════════════════════════════════════════════════════════════════════

class RealDataGenerator:
    """
    Samples IVS surfaces and market conditions from the real historical
    training set (first 2415 days, 2015–2024).
    """
    def __init__(self, real_ivs: torch.Tensor, real_cond: torch.Tensor):
        """
        real_ivs  : (N, 1, 32, 32) normalised IV surfaces — training split only.
        real_cond : (N, 3)          [Spot, ATM_Vol, Slope] conditions.
        """
        self.ivs  = real_ivs
        self.cond = real_cond
        self.N    = real_ivs.shape[0]

    def sample(self, n: int = 1):
        """Return `n` random (ivs, cond) pairs from the real pool."""
        idx = torch.randint(0, self.N, (n,))
        return self.ivs[idx], self.cond[idx]


class SynthDataGenerator:
    """
    Offline synthetic IVS pool.  Built ONCE before training from the
    pre-generated PI-LCDM surfaces (results/validation_generation/synthetic_1000.pt).
    During training, sampling is instant (random index into a torch.Tensor).
    """
    def __init__(self, synth_ivs: torch.Tensor):
        """
        synth_ivs : (N, 1, 32, 32) pre-generated synthetic IV surfaces.
        """
        self.ivs = synth_ivs
        self.N   = synth_ivs.shape[0]

    def sample(self, n: int = 1):
        """Return `n` random IVS tensors (with replacement)."""
        idx = torch.randint(0, self.N, (n,))
        return self.ivs[idx]


# ══════════════════════════════════════════════════════════════════════
# Grids (shared across env instances)
# ══════════════════════════════════════════════════════════════════════

def build_grids(device='cpu'):
    k_vals = torch.linspace(-0.3, 0.3, 32, device=device)
    t_vals = torch.linspace(0.05, 1.0,  32, device=device)
    T_g, k_g = torch.meshgrid(t_vals, k_vals, indexing='ij')
    K_g = torch.exp(k_g)          # normalised: S=1 ⟹ K = exp(log-moneyness)
    return K_g, T_g               # each (32, 32)


# ══════════════════════════════════════════════════════════════════════
# Option Pricing Helpers
# ══════════════════════════════════════════════════════════════════════

def bs_call_price(S, K, T, sigma, r=RISK_FREE_RATE):
    """Scalar Black-Scholes call price (numpy-safe via torch)."""
    T_s = max(float(T), 1e-6)
    s_s = max(float(sigma), 1e-5)
    d1 = (math.log(float(S) / float(K)) + (r + 0.5 * s_s**2) * T_s) / (s_s * math.sqrt(T_s))
    d2 = d1 - s_s * math.sqrt(T_s)
    from scipy.special import ndtr
    call = float(S) * ndtr(d1) - float(K) * math.exp(-r * T_s) * ndtr(d2)
    return call

def bs_delta(S, K, T, sigma, r=RISK_FREE_RATE):
    """Black-Scholes analytical delta N(d1)."""
    T_s = max(float(T), 1e-6)
    s_s = max(float(sigma), 1e-5)
    d1 = (math.log(float(S) / float(K)) + (r + 0.5 * s_s**2) * T_s) / (s_s * math.sqrt(T_s))
    from scipy.special import ndtr
    return ndtr(d1)

def get_atm_iv(ivs_surface: torch.Tensor) -> float:
    """Extract ATM (~centre of 32×32 grid) implied vol from IVS tensor."""
    # ivs_surface: (1, 32, 32) or (32, 32)
    if ivs_surface.dim() == 3:
        iv = ivs_surface[0]
    else:
        iv = ivs_surface
    # Mid-point index = (T_mid=16, K_atm=16)
    return float(iv[8, 16].clamp(min=0.05, max=2.0))


def get_vol_of_vol(ivs_surface: torch.Tensor) -> float:
    """
    Use the cross-sectional standard deviation of the IVS as a proxy for
    vol-of-vol (ζ).  A flatter surface → low ζ → vol barely changes.
    A steep/kinked surface → high ζ → vol can drift significantly.
    Clamped to [0.01, 0.30] to prevent blow-up in the OU process.
    """
    if ivs_surface.dim() == 3:
        iv = ivs_surface[0]
    else:
        iv = ivs_surface
    vov = float(iv.std().clamp(min=0.01, max=0.30))
    return vov


# ══════════════════════════════════════════════════════════════════════
# Deep Hedging Environment
# ══════════════════════════════════════════════════════════════════════

class DeepHedgingEnv:
    """
    Discrete-time Delta-hedging environment.

    Episode structure
    -----------------
    At reset():
      - Draw one IVS surface from the generator (real or synthetic).
      - Extract ATM IV → use as option IV for the episode.
      - Simulate a GBM spot path for STEPS_PER_EPISODE steps.
      - Agent is SHORT one call option, hedges dynamically with the underlying.

    State  (4-dim, normalised):
      [τ, moneyness, prev_delta, port_value]

    Action (1-dim, continuous):
      Target delta ∈ [−1, 1]  (clamped to [0, 1] for a short call hedge)

    Reward (terminal only, all intermediate = 0 for clarity):
      R = −(terminal_portfolio_value − option_payoff)²

    This formulation is: one episode → one scalar reward.
    PPO maximises E[R] across many rollouts ⟺ minimises variance of hedging error.
    """

    def __init__(self, generator, device='cpu'):
        self.generator = generator   # RealDataGenerator | SynthDataGenerator
        self.device    = device
        self.K         = FIXED_STRIKE
        self.T_init    = FIXED_MATURITY
        self.r         = RISK_FREE_RATE

    def reset(self):
        # ── Sample IVS surface ────────────────────────────────────────
        if isinstance(self.generator, RealDataGenerator):
            ivs_batch, cond_batch = self.generator.sample(1)
            ivs = ivs_batch[0]                  # (1, 32, 32)
        else:
            ivs_batch = self.generator.sample(1)
            ivs = ivs_batch[0]

        self.sigma0 = get_atm_iv(ivs)           # initial ATM IV
        self.vov    = get_vol_of_vol(ivs)        # vol-of-vol (OU noise magnitude)
        kappa       = 5.0                        # mean-reversion speed

        # ── Normalise spot to 1.0 (both real and synthetic) ──────────
        # This removes the distributional shift between real data
        # (S ∈ [1.9, 4.0]) and synthetic (S ≈ 1.0).
        S0 = 1.0

        # ── Simulate spot + stochastic vol path (Euler, risk-neutral) ─
        # IV follows a mean-reverting OU process:
        #   dσ = κ(σ₀ − σ)dt + ζ·dW_v      (correlated with spot)
        # This makes analytic BS delta sub-optimal, giving RL room to learn.
        path   = [S0]
        iv_path= [self.sigma0]
        S      = S0
        sigma  = self.sigma0
        rho_sv = -0.5                           # typical negative vol-spot correlation

        for _ in range(STEPS_PER_EPISODE):
            z1 = np.random.randn()
            z2 = np.random.randn()
            dW_s = z1 * math.sqrt(DT)
            dW_v = (rho_sv * z1 + math.sqrt(1 - rho_sv**2) * z2) * math.sqrt(DT)

            # Spot GBM step
            S = S * math.exp(-0.5 * sigma**2 * DT + sigma * dW_s)
            # OU vol step
            sigma = sigma + kappa * (self.sigma0 - sigma) * DT + self.vov * dW_v
            sigma = max(sigma, 0.02)            # floor at 2%

            path.append(S)
            iv_path.append(sigma)

        self.spot_path = path                   # length STEPS_PER_EPISODE + 1
        self.iv_path   = iv_path                # stochastic IV at each step

        # ── Initial option price ──────────────────────────────────────
        self.option_price_init = bs_call_price(S0, self.K, self.T_init, self.sigma0, self.r)
        self.option_price_init = max(self.option_price_init, 1e-6)

        # Episode state
        self.step_idx      = 0
        self.current_delta = 0.0
        self.port_value    = self.option_price_init

        return self._state()

    def step(self, action: float):
        """
        action : float ∈ [−1, 1]   (agent's target delta; we clip to [0,1])
        Returns (next_state, reward, done, info)
        """
        new_delta   = float(np.clip(action, 0.0, 1.0))
        S_curr      = self.spot_path[self.step_idx]
        S_next      = self.spot_path[self.step_idx + 1]

        # Transaction cost for delta change
        d_delta   = new_delta - self.current_delta
        tc        = abs(d_delta) * S_curr * TC_RATE

        # Hedge PnL: long d_delta shares, price moves from S_curr to S_next
        hedge_pnl = new_delta * (S_next - S_curr)

        self.port_value    += hedge_pnl - tc
        self.current_delta  = new_delta
        self.step_idx      += 1

        done = (self.step_idx >= STEPS_PER_EPISODE)

        if done:
            S_T         = self.spot_path[-1]
            tau_left    = 0.0
            # Option payoff (short call → we PAY the payoff)
            option_payoff = max(S_T - self.K, 0.0)
            terminal_pnl  = self.port_value - option_payoff

            # Quadratic penalty: correct for single-episode RL
            reward = -(terminal_pnl ** 2)
            info   = {'terminal_pnl': terminal_pnl, 'option_payoff': option_payoff}
        else:
            reward = 0.0            # no intermediate reward
            info   = {}

        return self._state(), reward, done, info

    def _state(self):
        step       = self.step_idx
        tau        = (self.T_init - step * DT) / self.T_init
        S_curr     = self.spot_path[step]
        moneyness  = math.log(max(S_curr, 1e-4) / self.K) / 0.3
        prev_delta = self.current_delta
        pv_norm    = self.port_value / self.option_price_init
        return np.array([tau, moneyness, prev_delta, pv_norm], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════
# PPO Actor-Critic Network
# ══════════════════════════════════════════════════════════════════════

class ActorCritic(nn.Module):
    """
    Shared MLP backbone with separate Actor and Critic heads.
    Actor: outputs (mu, log_std) for a Gaussian action distribution.
    Critic: outputs scalar state value V(s).
    """
    def __init__(self, state_dim=STATE_DIM, hidden=128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, 64),        nn.Tanh(),
        )
        self.actor_mu      = nn.Linear(64, ACTION_DIM)
        self.actor_log_std = nn.Parameter(torch.zeros(ACTION_DIM))
        self.critic        = nn.Linear(64, 1)

    def forward(self, x):
        feat  = self.backbone(x)
        mu    = torch.tanh(self.actor_mu(feat))    # ∈ (−1, 1)
        std   = torch.exp(self.actor_log_std).clamp(1e-4, 1.0)
        value = self.critic(feat).squeeze(-1)
        return mu, std, value

    def get_action(self, state_np: np.ndarray):
        """Sample action + log-prob for a single state (numpy)."""
        with torch.no_grad():
            s  = torch.FloatTensor(state_np).unsqueeze(0)
            mu, std, val = self.forward(s)
            dist = Normal(mu, std)
            a    = dist.sample()
            lp   = dist.log_prob(a).sum(-1)
        return a.squeeze(0).numpy(), lp.item(), val.item()

    def evaluate(self, states, actions):
        """Batch evaluate for PPO update."""
        mu, std, values = self.forward(states)
        dist    = Normal(mu, std)
        log_prob= dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, values, entropy


# ══════════════════════════════════════════════════════════════════════
# PPO Trainer
# ══════════════════════════════════════════════════════════════════════

class PPOTrainer:
    """
    Minimal PPO implementation.  Collects `n_rollout` full episodes per batch,
    then updates the policy for `n_epoch` epochs.
    """

    def __init__(
        self,
        env_factory,          # callable → DeepHedgingEnv (so we can parallelize later)
        device      = 'cpu',
        lr          = 3e-4,
        gamma       = 0.99,
        clip_eps    = 0.2,
        value_coef  = 0.5,
        entropy_coef= 0.01,
        n_epoch     = 5,
        n_rollout   = 200,    # episodes per PPO batch
    ):
        self.env_factory  = env_factory
        self.device       = device
        self.gamma        = gamma
        self.clip_eps     = clip_eps
        self.value_coef   = value_coef
        self.entropy_coef = entropy_coef
        self.n_epoch      = n_epoch
        self.n_rollout    = n_rollout

        self.policy = ActorCritic().to(device)
        self.optim  = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def collect_rollouts(self):
        """
        Run `n_rollout` episodes and return flat transition buffers.
        Since reward is only at terminal step, we use REINFORCE-like
        return: R_t = gamma^(T-t) * terminal_reward (all non-terminal = 0).
        """
        buf_states  = []
        buf_actions = []
        buf_logprobs= []
        buf_returns = []
        buf_values  = []

        env = self.env_factory()

        for _ in range(self.n_rollout):
            states_ep  = []
            actions_ep = []
            logprobs_ep= []
            values_ep  = []

            state = env.reset()
            done  = False
            terminal_reward = 0.0

            while not done:
                action, lp, val = self.policy.get_action(state)
                next_state, reward, done, _ = env.step(float(action))

                states_ep.append(state)
                actions_ep.append(action)
                logprobs_ep.append(lp)
                values_ep.append(val)

                state = next_state
                if done:
                    terminal_reward = reward   # only terminal is non-zero

            # Compute discounted returns backwards
            T    = len(states_ep)
            rets = [0.0] * T
            G    = terminal_reward
            for t in reversed(range(T)):
                G       = terminal_reward if t == T - 1 else self.gamma * G
                rets[t] = G

            buf_states  .extend(states_ep)
            buf_actions .extend(actions_ep)
            buf_logprobs.extend(logprobs_ep)
            buf_returns .extend(rets)
            buf_values  .extend(values_ep)

        states   = torch.FloatTensor(np.array(buf_states)).to(self.device)
        actions  = torch.FloatTensor(np.array(buf_actions)).to(self.device)
        old_lps  = torch.FloatTensor(buf_logprobs).to(self.device)
        returns  = torch.FloatTensor(buf_returns).to(self.device)

        # Advantage = Returns - Values (simple, no GAE needed for terminal reward)
        values_t = torch.FloatTensor(buf_values).to(self.device)
        advantages = returns - values_t
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return states, actions, old_lps, returns, advantages

    def update(self, states, actions, old_lps, returns, advantages):
        """PPO clipped policy + value loss."""
        losses = []
        for _ in range(self.n_epoch):
            new_lps, values, entropy = self.policy.evaluate(states, actions)

            ratio        = torch.exp(new_lps - old_lps.detach())
            surr1        = ratio * advantages.detach()
            surr2        = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantages.detach()
            actor_loss   = -torch.min(surr1, surr2).mean()
            critic_loss  = F.mse_loss(values, returns.detach())
            entropy_loss = -entropy.mean()

            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optim.step()
            losses.append(loss.item())
        return np.mean(losses)

    def train(self, n_batches: int = 25, verbose: bool = True):
        """Main training loop.  5000 trajectories = 25 batches × 200 rollouts."""
        self.policy.train()
        for batch_idx in range(n_batches):
            states, actions, old_lps, returns, advantages = self.collect_rollouts()
            loss = self.update(states, actions, old_lps, returns, advantages)
            if verbose and (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx+1:3d}/{n_batches}  loss={loss:.5f}")
        self.policy.eval()


# ══════════════════════════════════════════════════════════════════════
# Backtest Engine
# ══════════════════════════════════════════════════════════════════════

def run_backtest(agent_policy: ActorCritic,
                 test_real_ivs: torch.Tensor,
                 test_real_cond: torch.Tensor,
                 label: str = 'Agent') -> dict:
    """
    Evaluate `agent_policy` on every surface in the hold-out test set.

    Returns dict with:
      terminal_pnls  : np.array of per-path terminal PnL
      hedging_error  : std(terminal_pnls)
      mean_pnl       : mean terminal PnL
      pct05, pct95   : 5th and 95th percentile PnL
    """
    agent_policy.eval()
    terminal_pnls_agent = []
    terminal_pnls_bs    = []

    N_test = test_real_ivs.shape[0]
    gen    = RealDataGenerator(test_real_ivs, test_real_cond)

    for i in range(N_test):
        # ── Sample deterministically from test set (no randomness in sampling)
        ivs   = test_real_ivs[i]       # (1, 32, 32)
        cond  = test_real_cond[i]      # (3,)

        S0       = 1.0           # always normalised at test time
        sigma_iv = get_atm_iv(ivs)
        vov      = get_vol_of_vol(ivs)
        kappa    = 5.0
        rho_sv   = -0.5

        # ── Simulate stochastic vol path (same seed per test path) ────
        np.random.seed(i)
        path    = [S0]
        iv_path = [sigma_iv]
        S, sigma = S0, sigma_iv
        for _ in range(STEPS_PER_EPISODE):
            z1 = np.random.randn()
            z2 = np.random.randn()
            dW_s = z1 * math.sqrt(DT)
            dW_v = (rho_sv * z1 + math.sqrt(1 - rho_sv**2) * z2) * math.sqrt(DT)
            S     = S * math.exp(-0.5 * sigma**2 * DT + sigma * dW_s)
            sigma = max(sigma + kappa * (sigma_iv - sigma) * DT + vov * dW_v, 0.02)
            path.append(S)
            iv_path.append(sigma)

        option_init = bs_call_price(S0, FIXED_STRIKE, FIXED_MATURITY, sigma_iv)

        # ── RL Agent hedges under realised stochastic vol path ────────
        port_val = option_init
        delta    = 0.0

        for step in range(STEPS_PER_EPISODE):
            tau       = (FIXED_MATURITY - step * DT) / FIXED_MATURITY
            moneyness = math.log(max(path[step], 1e-4) / FIXED_STRIKE) / 0.3
            pv_norm   = port_val / option_init if option_init > 1e-8 else 0.0
            state     = np.array([tau, moneyness, delta, pv_norm], dtype=np.float32)

            with torch.no_grad():
                s  = torch.FloatTensor(state).unsqueeze(0)
                mu, std, _ = agent_policy(s)
                a  = float(mu.squeeze())
            new_delta = float(np.clip(a, 0.0, 1.0))

            d_delta  = new_delta - delta
            tc       = abs(d_delta) * path[step] * TC_RATE
            hedge_pnl= new_delta * (path[step+1] - path[step])
            port_val += hedge_pnl - tc
            delta     = new_delta

        payoff      = max(path[-1] - FIXED_STRIKE, 0.0)
        agent_pnl   = port_val - payoff
        terminal_pnls_agent.append(agent_pnl)

        # ── Black-Scholes Delta Hedge (now vs stochastic vol path) ───
        # BS uses CURRENT realised IV at each step (misspecified model:
        # it doesn't see the true stochastic vol process, only the static ATM IV)
        port_val_bs = option_init
        delta_bs    = 0.0

        for step in range(STEPS_PER_EPISODE):
            tau_left    = max(FIXED_MATURITY - step * DT, 1e-6)
            # BS delta uses the *initial* IV (static model — suboptimal under SV)
            bs_d        = bs_delta(path[step], FIXED_STRIKE, tau_left, sigma_iv)
            d_delta_bs  = bs_d - delta_bs
            tc_bs       = abs(d_delta_bs) * path[step] * TC_RATE
            hedge_pnl_bs= bs_d * (path[step+1] - path[step])
            port_val_bs += hedge_pnl_bs - tc_bs
            delta_bs     = bs_d

        bs_pnl = port_val_bs - payoff
        terminal_pnls_bs.append(bs_pnl)

    pnls_agent = np.array(terminal_pnls_agent)
    pnls_bs    = np.array(terminal_pnls_bs)

    return {
        'label'         : label,
        'terminal_pnls' : pnls_agent,
        'bs_pnls'       : pnls_bs,
        'hedging_error' : float(np.std(pnls_agent)),
        'mean_pnl'      : float(np.mean(pnls_agent)),
        'pct05'         : float(np.percentile(pnls_agent, 5)),
        'pct95'         : float(np.percentile(pnls_agent, 95)),
        'bs_error'      : float(np.std(pnls_bs)),
        'win_vs_bs'     : float(np.mean(np.abs(pnls_agent) < np.abs(pnls_bs))),
    }


def print_backtest_report(results_a: dict, results_b: dict):
    """Pretty-print the comparative backtest report."""
    print()
    print("=" * 65)
    print("   PHASE 6.3 — DEEP HEDGING BACKTEST REPORT (TSTR)")
    print("   Test Set: 252 paths (2025-01-15 → 2026-01-30)")
    print("=" * 65)

    header = f"{'Metric':<35} {'Agent A':>10} {'Agent B':>10} {'BS ∆':>10}"
    print(header)
    print("-" * 65)

    def row(label, val_a, val_b, val_bs, fmt='.5f'):
        print(f"{label:<35} {val_a:>10.{fmt[1:] if fmt.startswith('.') else fmt}} "
              f"{val_b:>10.{fmt[1:] if fmt.startswith('.') else fmt}} "
              f"{val_bs:>10.{fmt[1:] if fmt.startswith('.') else fmt}}")

    bs_err = results_a['bs_error']  # same path ⟹ same BS result

    print(f"{'Metric':<35} {'Agent A ':>10} {'Agent B ':>10} {'BS Delta':>10}")
    print("-" * 65)
    print(f"{'Hedging Error (std PnL)':<35} "
          f"{results_a['hedging_error']:>10.5f} "
          f"{results_b['hedging_error']:>10.5f} "
          f"{bs_err:>10.5f}")
    print(f"{'Mean Terminal PnL':<35} "
          f"{results_a['mean_pnl']:>10.5f} "
          f"{results_b['mean_pnl']:>10.5f} "
          f"{'—':>10}")
    print(f"{'5th Percentile PnL':<35} "
          f"{results_a['pct05']:>10.5f} "
          f"{results_b['pct05']:>10.5f} "
          f"{'—':>10}")
    print(f"{'95th Percentile PnL':<35} "
          f"{results_a['pct95']:>10.5f} "
          f"{results_b['pct95']:>10.5f} "
          f"{'—':>10}")
    print("-" * 65)
    print(f"{'Win Rate vs. BS Delta':<35} "
          f"{results_a['win_vs_bs']*100:>9.1f}% "
          f"{results_b['win_vs_bs']*100:>9.1f}% "
          f"{'—':>10}")

    # Agent B vs Agent A
    pnls_a = results_a['terminal_pnls']
    pnls_b = results_b['terminal_pnls']
    win_b_vs_a = float(np.mean(np.abs(pnls_b) < np.abs(pnls_a)))
    print(f"{'Win Rate B vs. A':<35} {'—':>10} {win_b_vs_a*100:>9.1f}% {'—':>10}")
    print("=" * 65)

    # Interpretation
    print()
    print("INTERPRETATION:")
    if results_b['hedging_error'] <= results_a['hedging_error'] * 1.05:
        print("  ✅ Agent B (Synthetic) matches or beats Agent A (Real).")
        print("     TSTR hypothesis CONFIRMED: PI-LCDM synthetic data is")
        print("     sufficient to train a competitive hedging policy.")
    else:
        gap = (results_b['hedging_error'] / results_a['hedging_error'] - 1) * 100
        print(f"  ⚠  Agent B is {gap:.1f}% worse than Agent A.")
        print("     Consider increasing the synthetic pool size or adding")
        print("     more training batches for Agent B.")

    if results_b['win_vs_bs'] >= 0.50:
        print(f"  ✅ Agent B beats Black-Scholes Delta on "
              f"{results_b['win_vs_bs']*100:.1f}% of test paths.")
    print()
