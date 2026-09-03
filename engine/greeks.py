import numpy as np

from engine.options import AsianOption, EuropeanOption
from engine.pricer import MonteCarloPricer
from engine.simulator import GBMSimulator


class GreeksEngine:
    def __init__(self, base_sim: GBMSimulator, option: EuropeanOption | AsianOption):
        self.sim = base_sim
        self.option = option

    def delta_gamma(
        self, bump: float = 0.5, n_paths: int = 100_000, seed: int = 42
    ) -> tuple[float, float]:
        paths = self.sim.simulate(n_paths=n_paths, antithetic=True, seed=seed)
        df = np.exp(-self.sim.r * self.sim.t)

        # Rescale existing simulated paths without re-sampling normals
        paths_up = paths * ((self.sim.s0 + bump) / self.sim.s0)
        paths_down = paths * ((self.sim.s0 - bump) / self.sim.s0)

        p_mid = float(np.mean(self.option.payoff(paths)) * df)
        p_up = float(np.mean(self.option.payoff(paths_up)) * df)
        p_down = float(np.mean(self.option.payoff(paths_down)) * df)

        delta = (p_up - p_down) / (2.0 * bump)
        gamma = (p_up - 2.0 * p_mid + p_down) / (bump**2)
        return delta, gamma

    def vega(self, bump: float = 0.01, n_paths: int = 100_000, seed: int = 42) -> float:
        sim_bumped = GBMSimulator(
            self.sim.s0,
            self.sim.r,
            self.sim.sigma + bump,
            self.sim.t,
            self.sim.n_steps,
        )

        p_base, _, _ = MonteCarloPricer(self.sim, self.option).price(
            n_paths=n_paths, antithetic=True, seed=seed
        )
        p_up, _, _ = MonteCarloPricer(sim_bumped, self.option).price(
            n_paths=n_paths, antithetic=True, seed=seed
        )

        return (p_up - p_base) / bump
