import numpy as np

from engine.options import AsianOption, EuropeanOption
from engine.simulator import GBMSimulator


class MonteCarloPricer:
    def __init__(self, simulator: GBMSimulator, option: EuropeanOption | AsianOption):
        self.simulator = simulator
        self.option = option

    def price(
        self,
        n_paths: int = 100_000,
        antithetic: bool = True,
        seed: int | None = None,
    ) -> tuple[float, float, tuple[float, float]]:
        paths = self.simulator.simulate(
            n_paths=n_paths, antithetic=antithetic, seed=seed
        )
        payoffs = self.option.payoff(paths)

        discount_factor = np.exp(-self.simulator.r * self.simulator.t)
        discounted = payoffs * discount_factor

        price = float(np.mean(discounted))
        std_err = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
        ci = (price - 1.96 * std_err, price + 1.96 * std_err)

        return price, std_err, ci
