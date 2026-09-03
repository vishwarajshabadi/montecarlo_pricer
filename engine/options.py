from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EuropeanOption:
    strike: float
    is_call: bool = True

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        s_t = paths[-1]
        return np.maximum(s_t - self.strike if self.is_call else self.strike - s_t, 0.0)


@dataclass(frozen=True)
class AsianOption:
    strike: float
    is_call: bool = True

    def payoff(self, paths: np.ndarray) -> np.ndarray:
        s_mean = np.mean(paths[1:], axis=0)
        return np.maximum(
            s_mean - self.strike if self.is_call else self.strike - s_mean, 0.0
        )
