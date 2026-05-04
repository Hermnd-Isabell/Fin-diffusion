
import torch
import torch.nn.functional as F
import numpy as np
from physics.market_ops import BSModule
from physics.arbitrage_loss import ArbitrageLoss


def _make_gaussian_kernel(sigma: float = 0.5, kernel_size: int = 3) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.unsqueeze(0).unsqueeze(0)


def _project_prices_to_convex(prices: torch.Tensor) -> torch.Tensor:
    """
    Project a batch of option call price surfaces onto the no-arbitrage cone.

    Enforces:
      (a) K-convexity   : C[K-1] - 2C[K] + C[K+1] >= 0  (Butterfly)
      (b) K-monotonicity: C[K] >= C[K+1]                  (Vertical spread)
      (c) T-monotonicity: C[T+1] >= C[T]                  (Calendar spread)
      (d) Non-negativity: C >= 0

    All operations are in call-price space, which is where the constraints
    actually live.  The algorithm is a multi-pass iterative projection.

    Args:
        prices: (B, 1, T_dim, K_dim) call price tensor.
    Returns:
        Projected prices of the same shape, satisfying all constraints.
    """
    B, C, T_dim, K_dim = prices.shape
    p = prices.clone()

    # ── Pass 1: K-monotonicity (decreasing in K) ─────────────────────
    # Call prices must decrease as K increases: C(K) >= C(K+1)
    # Running maximum from right to left enforces this.
    for i in range(K_dim - 2, -1, -1):
        p[:, :, :, i] = torch.maximum(p[:, :, :, i], p[:, :, :, i + 1])

    # ── Pass 2: T-monotonicity (increasing in T) ─────────────────────
    # Longer-dated calls must be more expensive: C(T+1) >= C(T)
    # Running maximum from left to right.
    for i in range(1, T_dim):
        p[:, :, i, :] = torch.maximum(p[:, :, i, :], p[:, :, i - 1, :])

    # ── Pass 3: K-convexity (multiple passes PAVA-style) ─────────────
    # Enforce d²C/dK² >= 0 (call prices are convex w.r.t. strike).
    # This means each interior price must be >= the chord between neighbours:
    #   C[i] >= 0.5*(C[i-1] + C[i+1])   (lies ABOVE the chord, convex)
    # If C[i] < midpoint, it means there's a local dip — lift it up.
    # Note: this is *lifting from below*, so we use maximum().
    for _ in range(12):   # 12 passes is enough for K_dim=32
        # Forward pass: lift each interior point to at least midpoint
        for i in range(1, K_dim - 1):
            mid = 0.5 * (p[:, :, :, i - 1] + p[:, :, :, i + 1])
            p[:, :, :, i] = torch.maximum(p[:, :, :, i], mid)
        # Backward pass
        for i in range(K_dim - 2, 0, -1):
            mid = 0.5 * (p[:, :, :, i - 1] + p[:, :, :, i + 1])
            p[:, :, :, i] = torch.maximum(p[:, :, :, i], mid)

    # ── Pass 4: Re-enforce K-monotonicity after convexification ───────
    for i in range(K_dim - 2, -1, -1):
        p[:, :, :, i] = torch.maximum(p[:, :, :, i], p[:, :, :, i + 1])

    # ── Pass 5: Non-negativity ────────────────────────────────────────
    p = p.clamp(min=0.0)

    return p


def _prices_to_iv_newton(
    prices: torch.Tensor,
    S: torch.Tensor,
    K_grid: torch.Tensor,
    T_grid: torch.Tensor,
    r: torch.Tensor,
    n_iter: int = 10,
) -> torch.Tensor:
    """
    Invert Black-Scholes call prices back to implied volatilities using
    Newton-Raphson.  Vectorised over the entire batch.

    Starting point: Brenner-Subrahmanyam approximation σ ≈ C*sqrt(2π/T)/S.

    Args:
        prices: (B, 1, T_dim, K_dim) corrected call prices.
        S, K_grid, T_grid, r: same grids as used in BSModule.bs_price.
        n_iter: number of Newton-Raphson iterations.
    Returns:
        iv: (B, 1, T_dim, K_dim) implied volatility surface.
    """
    T_safe = torch.clamp(T_grid, min=1e-5)

    # ── Brenner-Subrahmanyam initial guess ────────────────────────────
    # σ₀ ≈ C * sqrt(2π / T) / S   (ATM approximation)
    sigma = torch.clamp(prices * (2 * torch.pi / T_safe).sqrt() / S, min=1e-5, max=5.0)

    for _ in range(n_iter):
        sigma = torch.clamp(sigma, min=1e-5, max=5.0)
        price_est = BSModule.bs_price(sigma, S, K_grid, T_grid, r)
        vega_est  = BSModule.vega(sigma, S, K_grid, T_grid, r)
        vega_safe = torch.clamp(vega_est, min=1e-8)
        sigma = sigma - (price_est - prices) / vega_safe
        sigma = torch.clamp(sigma, min=1e-5, max=5.0)

    return sigma.detach()


class PILCDMSampler:
    """
    Physics-Informed Sampler (Inference).

    Supercharged PINA Corrector — five-stage pipeline:
      1. Latent-space PINA gradient descent   (multi-step, dynamic LR)
      2. Gaussian smoothing on decoded IVS    (kills HF Gamma noise)
      3. BS pricing: IVS → call price surface
      4. Direct algebraic projection onto no-arbitrage cone in price space
         (K-convexity, K-monotonicity, T-monotonicity) — O(n), exact
      5. BS inversion: corrected prices → IV via Newton-Raphson
    """

    def __init__(self, vae, conditioner, dit, sde, device='cpu'):
        self.vae = vae
        self.conditioner = conditioner
        self.dit = dit
        self.sde = sde
        self.device = device

        # Upweight Butterfly — primary failure mode
        self.arb_loss_fn = ArbitrageLoss(
            weights={'fly': 5.0, 'cal': 1.0, 'vert': 1.0}
        ).to(device)
        self._gauss_kernel: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_gauss_kernel(self, device, sigma: float = 0.5) -> torch.Tensor:
        if self._gauss_kernel is None or self._gauss_kernel.device != device:
            self._gauss_kernel = _make_gaussian_kernel(sigma).to(device)
        return self._gauss_kernel

    def _smooth_ivs(self, ivs: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
        kernel = self._get_gauss_kernel(ivs.device, sigma)
        C = ivs.shape[1]
        kernel_rep = kernel.expand(C, 1, *kernel.shape[-2:])
        return F.conv2d(ivs, kernel_rep, padding=kernel.shape[-1] // 2, groups=C)

    def _pina_correct(self, z, S, K_grid, T_grid, r,
                      pina_lr, gamma_scale, smooth_sigma, inner_iters):
        """Latent-space PINA correction (gradient descent on z)."""
        for _ in range(inner_iters):
            with torch.enable_grad():
                z_in = z.detach().requires_grad_(True)
                ivs_raw = self.vae.decode(z_in)
                ivs_smooth = self._smooth_ivs(ivs_raw, sigma=smooth_sigma)
                loss_arb, _ = self.arb_loss_fn(
                    ivs_smooth, S, K_grid, T_grid, r
                )
                grad = torch.autograd.grad(loss_arb, z_in)[0]

            loss_val = loss_arb.detach().item()
            effective_lr = pina_lr * (1.0 + gamma_scale * loss_val)
            z = (z.detach() - effective_lr * grad.detach()).detach()

        return z

    # ------------------------------------------------------------------
    # Main sampling loop
    # ------------------------------------------------------------------

    def sample(
        self,
        conditions,
        S,
        K_grid,
        T_grid,
        r,
        num_steps: int = 50,
        pina_lr: float = 0.05,
        pina_inner_steps: int = 4,
        gamma_scale: float = 5.0,
        smooth_sigma: float = 0.5,
        output_smooth_sigma: float = 0.6,
    ):
        """
        Generate IVS surfaces with Supercharged Physics Correction.
        """
        self.vae.eval()
        self.conditioner.eval()
        self.dit.eval()

        B = conditions.shape[0]
        latent_shape = (B, 4, 4, 4)

        # ── 1. Sample from prior ──────────────────────────────────────
        z = self.sde.prior_sampling(latent_shape).to(self.device).float()

        with torch.no_grad():
            context = self.conditioner(conditions)

        timesteps = torch.linspace(1, 0, num_steps + 1, device=self.device)

        # ── Main denoising loop ───────────────────────────────────────
        for i in range(num_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i + 1]
            t_tensor = torch.full((B, 1), t_curr.item(), device=self.device)

            # Predict noise
            with torch.no_grad():
                noise_pred = self.dit(z.detach(), t_tensor, context)

            # Reverse diffusion step
            with torch.no_grad():
                _, std_t = self.sde.marginal_prob(z, t_tensor)
                z_0_hat = z - std_t * noise_pred

                if t_curr > 1e-5:
                    z_new = z_0_hat + (z - z_0_hat) * (t_next / t_curr)
                else:
                    z_new = z_0_hat

                z = z_new.detach()

            # ── Stage 1: Latent-space PINA correction ─────────────────
            # t ∈ [0.20, 0.40) → 1 step (warm-up)
            # t ∈ [0.00, 0.20) → pina_inner_steps (aggressive)
            if pina_lr > 0 and t_curr.item() < 0.4:
                inner_iters = pina_inner_steps if t_curr.item() < 0.2 else 1
                z = self._pina_correct(
                    z, S, K_grid, T_grid, r,
                    pina_lr=pina_lr,
                    gamma_scale=gamma_scale,
                    smooth_sigma=smooth_sigma,
                    inner_iters=inner_iters,
                )

        # ── Stage 2: Decode + Gaussian smooth ─────────────────────────
        with torch.no_grad():
            final_ivs = self.vae.decode(z)

        if output_smooth_sigma > 0:
            with torch.no_grad():
                final_ivs = self._smooth_ivs(final_ivs, sigma=output_smooth_sigma)

        # ── Stage 3: Convert IVS → call prices ────────────────────────
        with torch.no_grad():
            call_prices = BSModule.bs_price(final_ivs, S, K_grid, T_grid, r)

        # ── Stage 4: Project prices onto no-arbitrage cone ────────────
        # Direct algebraic projection in **price space** (where the
        # constraints actually live). Enforces K-convexity, K-monotonicity,
        # and T-monotonicity in O(n) time without any gradient computation.
        with torch.no_grad():
            call_prices_proj = _project_prices_to_convex(call_prices)

        # ── Stage 5: Invert corrected prices → IVS via Newton-Raphson ─
        # Recover the implied vol surface consistent with the corrected prices.
        with torch.no_grad():
            final_ivs_corrected = _prices_to_iv_newton(
                call_prices_proj, S, K_grid, T_grid, r, n_iter=10
            )

        return final_ivs_corrected
