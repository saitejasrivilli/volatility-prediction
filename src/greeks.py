"""Black-Scholes Greeks calculation for options pricing and risk management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import newton


@dataclass
class GreeksSnapshot:
    """Greeks snapshot for a single option."""

    S: float  # Spot price
    K: float  # Strike price
    T: float  # Time to expiration (years)
    r: float  # Risk-free rate
    sigma: float  # Volatility (annualized)
    option_type: str  # 'call' or 'put'

    delta: float = 0.0
    gamma: float = 0.0
    vega: float = 0.0
    theta: float = 0.0
    rho: float = 0.0
    price: float = 0.0
    iv: float = 0.0


def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d1 for Black-Scholes formula."""
    if T <= 0 or sigma <= 0:
        return 0.0
    return (
        (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    )


def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute d2 for Black-Scholes formula."""
    d1_val = d1(S, K, T, r, sigma)
    return d1_val - sigma * np.sqrt(T)


def black_scholes_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """Compute option price using Black-Scholes.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'

    Returns:
        Option price
    """
    if T <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)

    if option_type == "call":
        price = S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)

    return float(max(price, 0.0))


def delta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """Compute delta (sensitivity to spot price changes).

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'

    Returns:
        Delta value [-1, 1]
    """
    if T <= 0:
        return 1.0 if option_type == "call" and S > K else (0.0 if option_type == "call" else -1.0)

    d1_val = d1(S, K, T, r, sigma)

    if option_type == "call":
        return float(norm.cdf(d1_val))
    else:
        return float(norm.cdf(d1_val) - 1.0)


def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute gamma (delta sensitivity to spot price).

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility (annualized)

    Returns:
        Gamma value
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1_val = d1(S, K, T, r, sigma)
    return float(norm.pdf(d1_val) / (S * sigma * np.sqrt(T)))


def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Compute vega (sensitivity to volatility changes).

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility (annualized)

    Returns:
        Vega per 1% change in volatility (divide by 100 for per-basis-point)
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d1_val = d1(S, K, T, r, sigma)
    return float(S * norm.pdf(d1_val) * np.sqrt(T) / 100.0)


def theta(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """Compute theta (time decay per day).

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'

    Returns:
        Theta per day (in same units as price)
    """
    if T <= 0:
        return 0.0

    d1_val = d1(S, K, T, r, sigma)
    d2_val = d2(S, K, T, r, sigma)

    if option_type == "call":
        theta_val = (
            -S * norm.pdf(d1_val) * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2_val)
        )
    else:
        theta_val = (
            -S * norm.pdf(d1_val) * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2_val)
        )

    return float(theta_val / 365.0)  # Convert to per-day


def rho(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """Compute rho (sensitivity to interest rate changes).

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Volatility (annualized)
        option_type: 'call' or 'put'

    Returns:
        Rho per 1% change in rates
    """
    if T <= 0 or sigma <= 0:
        return 0.0

    d2_val = d2(S, K, T, r, sigma)

    if option_type == "call":
        rho_val = K * T * np.exp(-r * T) * norm.cdf(d2_val)
    else:
        rho_val = -K * T * np.exp(-r * T) * norm.cdf(-d2_val)

    return float(rho_val / 100.0)


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal["call", "put"] = "call",
    initial_guess: float = 0.2,
    tol: float = 1e-6,
) -> float:
    """Compute implied volatility using Newton-Raphson.

    Args:
        market_price: Observed market option price
        S: Spot price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        option_type: 'call' or 'put'
        initial_guess: Initial volatility guess for solver
        tol: Convergence tolerance

    Returns:
        Implied volatility
    """

    def objective(sigma):
        theo_price = black_scholes_price(S, K, T, r, sigma, option_type)
        return theo_price - market_price

    def derivative(sigma):
        return vega(S, K, T, r, sigma)

    try:
        iv = newton(objective, initial_guess, fprime=derivative, tol=tol, maxiter=100)
        return float(max(iv, 0.001))  # Floor at 0.1%
    except Exception:
        return initial_guess


class GreeksCalculator:
    """Batch Greeks computation for options chains."""

    @staticmethod
    def compute_snapshot(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: Literal["call", "put"] = "call",
    ) -> GreeksSnapshot:
        """Compute all Greeks for a single option.

        Returns:
            GreeksSnapshot with all Greeks
        """
        price = black_scholes_price(S, K, T, r, sigma, option_type)

        return GreeksSnapshot(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            option_type=option_type,
            delta=delta(S, K, T, r, sigma, option_type),
            gamma=gamma(S, K, T, r, sigma),
            vega=vega(S, K, T, r, sigma),
            theta=theta(S, K, T, r, sigma, option_type),
            rho=rho(S, K, T, r, sigma, option_type),
            price=price,
            iv=sigma,
        )

    @staticmethod
    def compute_chain(
        chain_df: pd.DataFrame,
        S: float,
        T: float,
        r: float,
        sigma: float,
    ) -> pd.DataFrame:
        """Compute Greeks for options chain.

        Args:
            chain_df: DataFrame with columns: Strike, OptionType (call/put), ...
            S: Current spot price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annualized)

        Returns:
            DataFrame with added columns: Delta, Gamma, Vega, Theta, Price
        """
        results = []

        for _, row in chain_df.iterrows():
            K = row["Strike"]
            opt_type = row.get("OptionType", "call").lower()

            snapshot = GreeksCalculator.compute_snapshot(
                S, K, T, r, sigma, opt_type
            )

            results.append({
                **row.to_dict(),
                "Delta": snapshot.delta,
                "Gamma": snapshot.gamma,
                "Vega": snapshot.vega,
                "Theta": snapshot.theta,
                "Price": snapshot.price,
            })

        return pd.DataFrame(results)
