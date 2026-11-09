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

        if antithetic:
            half = len(payoffs) // 2
            paired = (payoffs[:half] + payoffs[half : 2 * half]) / 2.0
            if len(payoffs) % 2:
                paired = np.append(paired, payoffs[-1])
            discounted = paired * discount_factor
        else:
            discounted = payoffs * discount_factor

        price = float(np.mean(discounted))
        if len(discounted) > 1:
            std_err = float(np.std(discounted, ddof=1) / np.sqrt(len(discounted)))
        else:
            std_err = 0.0
        ci = (price - 1.96 * std_err, price + 1.96 * std_err)

        return price, std_err, ci
