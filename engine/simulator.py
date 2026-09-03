import numpy as np


class GBMSimulator:
    def __init__(self, s0: float, r: float, sigma: float, t: float, n_steps: int):
        self.s0 = s0
        self.r = r
        self.sigma = sigma
        self.t = t
        self.n_steps = n_steps
        self.dt = t / n_steps

    def simulate(
        self, n_paths: int, antithetic: bool = True, seed: int | None = None
    ) -> np.ndarray:
        rng = np.random.default_rng(seed)

        if antithetic:
            half_paths = (n_paths + 1) // 2
            z_half = rng.standard_normal((self.n_steps, half_paths))
            z = np.concatenate([z_half, -z_half], axis=1)[:, :n_paths]
        else:
            z = rng.standard_normal((self.n_steps, n_paths))

        drift = (self.r - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt) * z
        log_returns = drift + diffusion

        log_paths = np.vstack([np.zeros((1, n_paths)), np.cumsum(log_returns, axis=0)])
        return self.s0 * np.exp(log_paths)
