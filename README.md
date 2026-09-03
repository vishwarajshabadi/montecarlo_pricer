# Monte Carlo Option Pricing & Greeks Engine

A Python Monte Carlo pricing tool I built to learn stochastic simulation concepts (GBM path generation, variance reduction, numerical Greeks) and add a derivatives project alongside my backtester for internship applications.

## Why Monte Carlo

Formulas like Black-Scholes work well for standard European options because there is a known mathematical equation for the final price. But for options that depend on the entire path—like Asian options, which depend on the average stock price over time—there is no simple formula.

To price them, you have to simulate thousands of possible price paths and average the outcomes. I built this project to understand how to simulate stock paths correctly without bias, how to speed up convergence using variance reduction, and how to compute risk sensitivities (Greeks) without random noise messing up the results.

## Architecture

The engine separates generating stock paths from calculating payoffs and computing Greeks:

For each run:

1. `GBMSimulator.simulate(n_paths)` → generates a 2D matrix of stock prices over time using exact log-returns.
2. `Option.payoff(paths)` → takes the price matrix and calculates the payoff for each path.
3. `MonteCarloPricer.price(...)` → discounts the payoffs back to today at rate $r$, calculates the average price, and finds the standard error.
4. `GreeksEngine.delta_gamma(...)` → scales the already simulated paths up and down by $(S_0 \pm \epsilon) / S_0$ to get Delta and Gamma without generating new random numbers.
5. `GreeksEngine.vega(...)` → runs bumped volatility runs using the exact same random seed (Common Random Numbers) to isolate the volatility effect.

**Invariant:** In step 4, the Delta and Gamma calculations reuse the exact same Brownian paths ($Z$ stays identical). This removes simulation noise from the calculation.

## Key Design Decisions

- **Log-space simulation:** Instead of adding price changes directly ($S_{t+1} = S_t + \dots$), the simulator tracks returns in log space. This prevents stock prices from ever turning negative and keeps the math exact without step-size errors.
- **Antithetic sampling:** For every random shock $Z$, the code also runs a path with $-Z$. Because higher shocks generally mean higher option payouts, pairing opposite paths cancels out a large portion of the random variance, giving a more accurate price with fewer runs.

```python
half_paths = (n_paths + 1) // 2
z_half = rng.standard_normal((n_steps, half_paths))
z = np.concatenate([z_half, -z_half], axis=1)[:, :n_paths]

drift = (r - 0.5 * sigma**2) * dt
diffusion = sigma * np.sqrt(dt) * z
paths = s0 * np.exp(
    np.vstack([np.zeros((1, n_paths)), np.cumsum(drift + diffusion, axis=0)])
)

```

* **Reusing paths for Delta and Gamma:** Because stock prices scale proportionally with $S_0$ under GBM, a path starting at $S_0 + \epsilon$ is just the original path multiplied by $(S_0 + \epsilon) / S_0$. Scaling the existing matrix makes calculating Delta and Gamma about $3\times$ faster and stops random simulation noise from breaking the finite-difference math.


* **Fixed random seeds for Vega:** Changing volatility $\sigma$ changes the shape of the diffusion, so new paths must be generated. By locking the random seed (Common Random Numbers), the price difference reflects only the change in volatility rather than random sampling luck.


* **Lightweight option classes:** Payoffs are structured as simple `@dataclass(frozen=True)` classes that take in path arrays and return 1D payoff arrays.



## Mathematical Formulation & Conventions

| Concept / Metric | Formulation | Implementation Detail |
| :--- | :--- | :--- |
| **GBM Step** | $S_{t+\Delta t} = S_t \exp\left(\left(r - \frac{1}{2}\sigma^2\right)\Delta t + \sigma \sqrt{\Delta t} Z\right)$ | Vectorized log-returns via `np.cumsum(..., axis=0)` |
| **European Payoff** | $h(S_T) = \max(\phi(S_T - K), 0)$ | Terminal slice: `np.maximum(phi * (paths[-1] - K), 0)` |
| **Asian Payoff (Arithmetic)** | $h(\bar{S}) = \max\left(\phi\left(\frac{1}{M}\sum_{i=1}^M S_{t_i} - K\right), 0\right)$ | Path mean: `np.maximum(phi * (paths[1:].mean(axis=0) - K), 0)` |
| **Discounted Price ($\hat{V}$)** | $\hat{V} = e^{-r T} \frac{1}{N}\sum_{n=1}^N h(S^{(n)})$ | `np.exp(-r * T) * payoffs.mean()` |
| **Confidence Interval (95%)** | $\hat{V} \pm 1.96 \cdot \frac{\hat{\sigma}_{\text{discounted}}}{\sqrt{N}}$ | Standard error using sample variance of discounted payoffs |
| **Delta ($\Delta$)** | $\frac{\hat{V}(S_0 + \epsilon) - \hat{V}(S_0 - \epsilon)}{2\epsilon}$ | Central difference reusing scaled paths: $S_t \cdot (1 \pm \epsilon / S_0)$ |
| **Gamma ($\Gamma$)** | $\frac{\hat{V}(S_0 + \epsilon) - 2\hat{V}(S_0) + \hat{V}(S_0 - \epsilon)}{\epsilon^2}$ | Second-order central difference reusing scaled paths |
| **Vega ($\nu$)** | $\frac{\hat{V}(\sigma + \Delta\sigma) - \hat{V}(\sigma - \Delta\sigma)}{2\Delta\sigma}$ | Central difference freezing Common Random Numbers (CRN) |

*(where $\phi = 1$ for Calls, $-1$ for Puts; $Z \sim \mathcal{N}(0, 1)$)*

## Project Structure

```text
montecarlo_pricer/
├── engine/
│   ├── __init__.py
│   ├── simulator.py     # Vectorized GBM paths and antithetic draws
│   ├── options.py       # European & Asian payoff dataclasses
│   ├── pricer.py        # Discounting, mean price, and standard error
│   └── greeks.py        # Fast Delta/Gamma (rescaled) and Vega (CRN)
├── tests/
│   ├── __init__.py
│   └── test_pricer.py   # Pytest: BS checks, Put-Call Parity, Variance test
├── main.py              # Runner script
├── requirements.txt     # Dependencies
└── README.md

```

## Quickstart

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
python main.py
pytest tests/

```

## Sample Output

```text
$ python main.py
Pricing Parameters: S0=100.0, K=105.0, r=0.03, vol=0.2, T=1.0y
  European Call : 7.1350  (SE: 0.0396)
  Asian Call    : 3.1489  (SE: 0.0193)
  Sensitivities : Delta=0.5023 | Gamma=0.01971 | Vega(1%)=0.3993

```

**Interpretation:** The Asian call is around 56% cheaper than the European call ($3.1489$ vs $7.1350$). Because the Asian payoff uses an average of 252 daily prices, extreme high and low price spikes get smoothed out. Less volatility means less option value, which lowers the price and also cuts the simulation standard error ($0.0193$ vs $0.0396$). Delta is near $0.50$ because the strike ($105$) is close to the forward price ($100 \cdot e^{0.03} \approx 103.05$).

## What I Learned

**Why simulate in log space.** If you simulate prices by adding changes directly, big negative shocks can accidentally push the stock price below zero. Simulating the logarithm of the stock price guarantees prices stay positive ($S_t > 0$) and keeps the steps mathematically accurate over any time gap.

**Why numerical Greeks need careful handling.** If you recalculate prices with new random numbers to find Delta or Gamma, the random difference between the two runs can be larger than the bump itself. For Gamma, dividing by $\epsilon^2$ makes that random noise explode. Re-scaling the exact same simulated paths completely removes the simulation noise, making the Greeks clean and fast.

**Where antithetic sampling helps (and where it doesn't).** Pairing opposite random shocks works well for standard options because higher market returns reliably mean higher option payoffs. But for more complex contracts where the payoff jumps up and down unpredictably, pairing $+Z$ with $-Z$ does not reduce variance as much, meaning other methods like Control Variates are needed.

**How to test simulation code.** You cannot test Monte Carlo with exact number matches because every run has slight random variance. Instead, I wrote tests that verify mathematical properties: checking that the simulated European price stays inside the 99% confidence window of the Black-Scholes formula, verifying that Put-Call Parity holds, and checking that antithetic sampling actually lowers the standard error compared to regular sampling.

