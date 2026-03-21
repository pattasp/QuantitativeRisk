# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:21:28 2026

@author: Panos
"""

# Creating a module for computing Loss_Distribution of a Stock Portfolio and 
# trying to compute VaR and ES for that Loss Distribution with Linear Approximation
# Here the method == First order Taylor Series for Loss Distribution 


# Setting the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


# Setting a class for Stock Portfolio Loss Model


class StockPortfolioLossModel:
    
    """
    Compute the Loss Distribution of a stock portfolio L_{t+1,i with a time horizon scale

    Portfolio definition:
        - S_t[i] = current price of stock i
        - lambda_[i] = number of shares held in stock i

    Risk factors:
        X_{t+1,i} = log(S_{t+1,i}) - log(S_{t,i})

    MC portfolio loss:
        L_{t+1} = -sum_i lambda_i * S_t[i] * (exp(X_i) - 1)

    Linear approximation:
        L_hat_{t+1} = -sum_i lambda_i * S_t[i] * X_i
                    = -V_t * sum_i w_i * X_i
    """
    
    def __init__(self,prices,shares,mu,sigma,dt=1.0):
        
        # Setting our parameters in the form of arrays
        self.prices = np.asarray(prices, dtype=float)
        self.shares = np.asarray(shares, dtype=float)
        self.mu = np.asarray(mu, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.dt = float(dt)
        
        # Validating inputs
        
        self._validate_inputs()
        
        # Defining attributes of position values/Value of Portfolio/Weights
        self.position_values = self.prices * self.shares
        self.Vt = np.sum(self.position_values)
        self.weights = self.position_values / self.Vt
        
    
    def _validate_inputs(self):
        
        n = len(self.prices)

        if len(self.shares) != n:
            raise ValueError("shares must have the same length as prices")
        if len(self.mu) != n:
            raise ValueError("mu must have the same length as prices")
        if self.sigma.shape != (n, n):
            raise ValueError("sigma must be an (n x n) covariance matrix")

        if np.any(self.prices <= 0):
            raise ValueError("All prices must be positive")
        if np.any(self.shares < 0):
            raise ValueError("Shares must be non-negative")
            
        if self.dt <=0:
            raise ValueError("Time Horizon must be positive")
            
    @property
    def mu_dt(self):
        return self.dt * self.mu
    
    @property 
    def sigma_dt(self):
        return self.dt * self.sigma
    
    # Simulating the possible risk factors
        
    def simulate_risk_factors(self,n_sims = 100000,random_seed =42):
        
        rng = np.random.default_rng(random_seed)
        
        # Creating via the prices distributions per stock the X risk factors
        x = rng.multivariate_normal(
            mean = self.mu_dt,
            cov = self.sigma_dt,
            size = n_sims
            )
        
        return x
    
    # Getting the Exact Losses
    
    def exact_loss(self,x):
        
        """
        Exact Loss => L = -sum_i [lambda_i * S_i * (exp(X_i) -1))]
        
        Parameters ------
        x : ndarray of Shape  (n_sims,n_assets) or (n_assets,)
        
        Returns ---------
        losses : ndarray 
        
        """
        
        x = np.asarray(x,dtype=float)
        
        if x.ndim == 1:
            if len(x) != len(self.prices):
                raise ValueError("x must have the same length as number of assets")
            
            return -np.sum(self.position_values * (np.exp(x)-1.0))
        
        elif x.ndim == 2:
            if x.shape[1] != len(self.prices):
                raise ValueError("x must have shape (n_sims,n_assets")
                
            return -np.sum((np.exp(x)-1.0) * self.position_values,axis=1 )
        
        else :
            raise ValueError("x must be 1D or 2D")
            
    # Setting the Linearized Loss
    def linearized_loss(self,x):
        """
        Linearized Loss is obtained over the dt horizon as:
            L_hat = -Σ_i shares_i * prices_i * x_i
        
        """
        if x.ndim == 1:
            return -np.sum(self.position_values * x)
        elif x.ndim == 2:
            return -np.sum(self.position_values * x, axis=1)       
            
    
    # Taking the linearized version of mean closed form formula
    
    def linearized_mean(self):
        """
        
        E[L_hat] = -V_t * w^T * E[Rt] if Normal then E[Rt] == mu
         """
        
        return -self.Vt * (self.weights @ self.mu)
    
    # Same for linearized version of var for Loss Distribution
    
    def linearized_variance(self):
        
        """
        Var(L_hat) = V_t^2 * w^T * sigma * w
        """
        
        return self.Vt**2 * (self.weights @ self.sigma @ self.weights)
    
    def linearized_std(self):
        return np.sqrt(self.linearized_variance())
    
    # Setting up our Monte Carlo Simulation by following the Variance-Covariance Method
    
    def monte_carlo_losses(self, n_sims=100000, random_seed=42):
        """
        Simulate x and return both exact and linearized losses.
        """
        x = self.simulate_risk_factors(n_sims=n_sims, random_seed=random_seed)
        exact = self.exact_loss(x)
        linear = self.linearized_loss(x)
        return x, exact, linear
    
    # Now following plain BASEL-II we compute the Vanilla VaR for a quantile a
    # Where a is the confidence interval.VaR acts like a threshold/cutoff point
    
    
    
    def var(self, losses, alpha=0.99):
        """
        Value at Risk at level alpha.
        Since loss is positive for bad outcomes, VaR is the alpha-quantile.
        
        x here ~N(mu,sigma) over a Δ time horizon thus 
        VaR_a = mu + sigma*Phi^-1(a) where Phi^-1 is the CDF of N
        """
        losses = np.asarray(losses, dtype=float)
        return np.quantile(losses, alpha)
    
    
    # Now we set up the Conditional VaR or else the Expected Shortfall
    
    def expected_shortfall(self, losses, alpha=0.99):
        """
        Expected Shortfall = average loss beyond VaR.
        """
        losses = np.asarray(losses, dtype=float)
        var_alpha = self.var(losses, alpha=alpha)
        tail_losses = losses[losses >= var_alpha]
        return np.mean(tail_losses)

    # Finally we create a plot
    
    def plot_loss_distribution(self, exact_losses, linear_losses=None, bins=150):
        """
        Plot histogram (and optional KDE) of exact and linearized losses.
        """
        plt.figure(figsize=(10, 6))

        plt.hist(exact_losses, bins=bins, density=True, alpha=0.5, label="Exact MC Loss")

        grid_min = np.min(exact_losses)
        grid_max = np.max(exact_losses)

        if linear_losses is not None:
            plt.hist(linear_losses, bins=bins, density=True, alpha=0.5, label="Linearized Loss")
            grid_min = min(grid_min, np.min(linear_losses))
            grid_max = max(grid_max, np.max(linear_losses))

        grid = np.linspace(grid_min, grid_max, 500)

        kde_exact = gaussian_kde(exact_losses)
        plt.plot(grid, kde_exact(grid), linewidth=2, label="Exact KDE")

        if linear_losses is not None:
            kde_linear = gaussian_kde(linear_losses)
            plt.plot(grid, kde_linear(grid), linewidth=2, label="Linear KDE")

        plt.xlabel("Portfolio Loss")
        plt.ylabel("Density")
        plt.title("Portfolio Loss Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        

# Go ahead and exercise it for arbitary values
    
if __name__ == "__main__":
    
    prices = [185.0, 410.0, 92.0]
    shares = [30, 20, 50]
    mu = [0.00035, 0.00025, 0.00045]

    sigma = [
    [0.000324, 0.0001386, 0.0001386],
    [0.0001386, 0.000196, 0.0001232],
    [0.0001386, 0.0001232, 0.000484]
            ]

    model = StockPortfolioLossModel(prices, shares, mu, sigma)

    x, exact_losses, linear_losses = model.monte_carlo_losses(n_sims=50000)

    print("Portfolio value Vt:", model.Vt)
    print("Weights:", model.weights)

    print("Linearized theoretical mean:", model.linearized_mean())
    print("Linearized theoretical variance:", model.linearized_variance())
    print("Linearized theoretical std:", model.linearized_std())
 
    print("Exact VaR 99%:", model.var(exact_losses, alpha=0.99))
    print("Exact ES 99%:", model.expected_shortfall(exact_losses, alpha=0.99))

    model.plot_loss_distribution(exact_losses, linear_losses)