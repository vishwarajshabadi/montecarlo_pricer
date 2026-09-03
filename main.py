from engine.greeks import GreeksEngine
from engine.options import AsianOption, EuropeanOption
from engine.pricer import MonteCarloPricer
from engine.simulator import GBMSimulator


def main():
    s0, k, r, sigma, t = 100.0, 105.0, 0.03, 0.20, 1.0
    sim = GBMSimulator(s0=s0, r=r, sigma=sigma, t=t, n_steps=252)

    euro = EuropeanOption(strike=k, is_call=True)
    asian = AsianOption(strike=k, is_call=True)

    euro_price, euro_se, _ = MonteCarloPricer(sim, euro).price(n_paths=100_000, seed=42)
    asian_price, asian_se, _ = MonteCarloPricer(sim, asian).price(
        n_paths=100_000, seed=42
    )

    greeks = GreeksEngine(sim, euro)
    delta, gamma = greeks.delta_gamma(bump=0.5, n_paths=100_000, seed=42)
    vega = greeks.vega(bump=0.01, n_paths=100_000, seed=42)

    print(f"Pricing Parameters: S0={s0}, K={k}, r={r}, vol={sigma}, T={t}y")
    print(f"  European Call : {euro_price:.4f}  (SE: {euro_se:.4f})")
    print(f"  Asian Call    : {asian_price:.4f}  (SE: {asian_se:.4f})")
    print(
        f"  Sensitivities : Delta={delta:.4f} | Gamma={gamma:.5f} | Vega(1%)={vega / 100.0:.4f}"
    )


if __name__ == "__main__":
    main()
