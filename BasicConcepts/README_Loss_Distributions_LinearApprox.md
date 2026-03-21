# Loss Distribution and Linear Approximation for a Stock Portfolio

# Disclaimer : The base framework is developed via personal study The main source were the slides of Associate Professor of Analytics and OR @ Imperial Business School: Martin Haugh

This project implements a simple **Stock Portfolio risk model** in Python for estimating the **Loss Distribution** of a stock portfolio and computing key market risk measures such as:

- **Value at Risk (VaR)**
- **Expected Shortfall (ES)**

The script combines an introductory format of the following methods:

- **Monte Carlo simulation** for exact portfolio loss estimation
- **First-order Taylor series (linear approximation)** for an analytical approximation of portfolio loss

The implementation is centered around a `StockPortfolioLossModel` class that simulates correlated log-returns, computes exact and linearized losses, and visualizes the resulting loss distributions.

---

## Mathematical Idea

- At first we know that loss of value in a portfolio for a time horizon `Δ`:
- `L_{t+1}` := - (`V_{t+1} - V_{t}`) where `L_{t+1}` is a positive quantity.
-  Now given a set of risk factors : `Z_t = (Z_{t,1},...Z_{t,d})` 
-  Then value of portfolio becomes a function of those ; `V_t = f(t,Z_t)`
-  The above risk factors in our Stock Portfolio can be the stock prices (weights).
- `S_t[i]` =  price of asset `i` at time `t`
- `lambda_i` = number of shares held in asset `i` 
-  Now a key aspect is that these risk factors (e.g Stocks) can change over time
-  We denote that change of risk as : `X_t = Z_t - Z_{t-1}`

-  At the portfolio of d stocks we compute the log-prices by assuming small changes 
-  of the prices and a small time horizon.
- `X_i = log(S_{t+1,i}) - log(S_{t,i})` = log-return of asset `i`


### Exact Monte Carlo portfolio loss

The exact portfolio loss over one horizon is:

```math
L_{t+1} = -\sum_i \lambda_i S_t[i] \left(e^{X_i} - 1\right)
```

### Linear approximation

Using a first-order Taylor approximation `e^{X_i} - 1 \approx X_i`, the loss becomes:

```math
\hat{L}_{t+1} = -\sum_i \lambda_i S_t[i] X_i
```

This gives a simpler approximation that is useful for fast risk estimation and theoretical analysis.

---

## Features

- Defines a reusable `StockPortfolioLossModel` class
- Validates portfolio and covariance-matrix inputs
- Simulates correlated asset log-returns using a multivariate normal model
- Computes:
  - exact Monte Carlo losses
  - linearized losses
  - theoretical mean of the linearized loss
  - theoretical variance and standard deviation of the linearized loss
  - VaR at a chosen confidence level
  - Expected Shortfall beyond VaR
- Plots exact and linearized loss distributions using histograms and KDE curves
- Includes an example run with a 3-asset portfolio in the main block

---

## File

```text
Loss_Distributions_LinearApprox.py
```

---

## Requirements

Install the following Python packages:

```bash
pip install numpy matplotlib scipy
```

---

## How to Run

Run the script directly:

```bash
python Loss_Distributions_LinearApprox.py
```

The script will:

1. Define sample portfolio prices, shares, expected returns, and covariance matrix
2. Simulate risk factors
3. Compute exact and linearized losses
4. Print portfolio statistics, theoretical linearized moments, VaR, and Expected Shortfall
5. Display the portfolio loss distribution plot

---

## Example Portfolio Used

The example included in the script uses:

- Prices: `[185.0, 410.0, 92.0]`
- Shares: `[30, 20, 50]`
- Mean returns: `[0.00035, 0.00025, 0.00045]`
- Covariance matrix:

```python
[
    [0.000324, 0.0001386, 0.0001386],
    [0.0001386, 0.000196, 0.0001232],
    [0.0001386, 0.0001232, 0.000484]
]
```

with `50000` Monte Carlo simulations in the example execution block.

---

## Example Output

The exact numerical values will vary slightly because they depend on simulation, but the script prints results in this format:

```text
Portfolio value Vt: ...
Weights: ...
Linearized theoretical mean: ...
Linearized theoretical variance: ...
Linearized theoretical std: ...
Exact VaR 99%: ...
Exact ES 99%: ...
```

---

## What This Project Demonstrates

This project is plain vanilla but was developed to introduce :

- portfolio loss modeling
- Monte Carlo simulation in risk management
- linear approximation of nonlinear portfolio losses
- VaR and Expected Shortfall estimation
- visualization of portfolio risk distributions

---

## Possible Extensions

Some natural next steps for improving this project are:

- Setting up a data set of actual stocks > e.g "MSFT","TSLA","NVDIA" etc
- Adding closed-form parametric VaR and ES under the normal approximation (Var-Covar)
- Trying with Historical Data
- Comparing exact Monte Carlo risk measures against Linearized risk measures
- Adding Cholesky Decomposition explicitly for Return simulation
- stress testing the portfolio under shocked scenarios
- extending the model to nonlinear instruments such as options

-- Some of the above will be done sequentialy.

---

## Notes

- The model assumes that asset log-returns are jointly normally distributed.
- Time horizon is kept at 1 day where the market is not too volatile.
- The linear approximation is most accurate for small return moves.
- The exact Monte Carlo loss captures the exponential price dynamics more faithfully than the first-order approximation.
