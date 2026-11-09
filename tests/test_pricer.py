import numpy as np
import pytest
from scipy.stats import norm

from engine.greeks import GreeksEngine
from engine.options import EuropeanOption
from engine.pricer import MonteCarloPricer
from engine.simulator import GBMSimulator


def black_scholes_price(
    s0: float, k: float, r: float, sigma: float, t: float, is_call: bool = True
) -> float:
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if is_call:
        return float(s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))
    return float(k * np.exp(-r * t) * norm.cdf(-d2) - s0 * norm.cdf(-d1))


def test_european_call_matches_black_scholes():
    s0, k, r, sigma, t = 100.0, 100.0, 0.05, 0.2, 1.0
    analytical = black_scholes_price(s0, k, r, sigma, t, is_call=True)

    sim = GBMSimulator(s0=s0, r=r, sigma=sigma, t=t, n_steps=252)
    option = EuropeanOption(strike=k, is_call=True)
    price, std_err, _ = MonteCarloPricer(sim, option).price(
        n_paths=150_000, antithetic=True, seed=42
    )

    assert abs(price - analytical) < 2.58 * std_err


def test_put_call_parity_holds():
    s0, k, r, sigma, t = 100.0, 105.0, 0.03, 0.20, 1.0
    sim = GBMSimulator(s0=s0, r=r, sigma=sigma, t=t, n_steps=252)

    call = EuropeanOption(strike=k, is_call=True)
    put = EuropeanOption(strike=k, is_call=False)

    c_price, _, _ = MonteCarloPricer(sim, call).price(
        n_paths=100_000, antithetic=True, seed=42
    )
    p_price, _, _ = MonteCarloPricer(sim, put).price(
        n_paths=100_000, antithetic=True, seed=42
    )

    parity_lhs = c_price - p_price
    parity_rhs = s0 - k * np.exp(-r * t)

    assert abs(parity_lhs - parity_rhs) < 0.15


def test_antithetic_variance_reduction():
    sim = GBMSimulator(s0=100.0, r=0.05, sigma=0.2, t=1.0, n_steps=100)
    option = EuropeanOption(strike=100.0, is_call=True)

    _, se_standard, _ = MonteCarloPricer(sim, option).price(
        n_paths=50_000, antithetic=False, seed=42
    )
    _, se_antithetic, _ = MonteCarloPricer(sim, option).price(
        n_paths=50_000, antithetic=True, seed=42
    )

    assert se_antithetic < se_standard


def test_greeks_match_black_scholes():
    s0, k, r, sigma, t = 100.0, 105.0, 0.03, 0.20, 1.0
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    bs_delta = norm.cdf(d1)
    bs_gamma = norm.pdf(d1) / (s0 * sigma * np.sqrt(t))
    bs_vega = s0 * norm.pdf(d1) * np.sqrt(t)

    sim = GBMSimulator(s0=s0, r=r, sigma=sigma, t=t, n_steps=252)
    option = EuropeanOption(strike=k, is_call=True)

    delta, gamma = GreeksEngine(sim, option).delta_gamma(
        bump=0.5, n_paths=200_000, seed=42
    )
    vega = GreeksEngine(sim, option).vega(bump=0.01, n_paths=200_000, seed=42)

    assert abs(delta - bs_delta) < 0.01
    assert abs(gamma - bs_gamma) < 0.005
    assert abs(vega - bs_vega) < 1.0
