"""Unit tests for Black-Scholes Greeks."""

import numpy as np
import pytest

try:
    from src.greeks import (
        black_scholes_price,
        delta,
        gamma,
        vega,
        theta,
        GreeksCalculator,
    )
except ImportError:
    from greeks import (
        black_scholes_price,
        delta,
        gamma,
        vega,
        theta,
        GreeksCalculator,
    )


class TestBlackScholesPrice:
    """Test Black-Scholes pricing."""

    def test_atm_call_price_positive(self):
        """ATM call should have positive price."""
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
        price = black_scholes_price(S, K, T, r, sigma, "call")
        assert price > 0

    def test_atm_put_price_positive(self):
        """ATM put should have positive price."""
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
        price = black_scholes_price(S, K, T, r, sigma, "put")
        assert price > 0

    def test_itm_call_price(self):
        """ITM call should be worth at least intrinsic."""
        S, K, T, r, sigma = 110, 100, 0.25, 0.05, 0.20
        price = black_scholes_price(S, K, T, r, sigma, "call")
        assert price >= S - K

    def test_otm_put_price(self):
        """OTM put should have positive time value."""
        S, K, T, r, sigma = 110, 100, 0.25, 0.05, 0.20
        price = black_scholes_price(S, K, T, r, sigma, "put")
        assert price > 0  # OTM puts have time value


class TestDelta:
    """Test delta calculation."""

    def test_call_delta_range(self):
        """Call delta should be in [0, 1]."""
        for strike_mult in [0.8, 1.0, 1.2]:
            S, K, T, r, sigma = 100, 100 * strike_mult, 0.25, 0.05, 0.20
            d = delta(S, K, T, r, sigma, "call")
            assert 0 <= d <= 1

    def test_put_delta_range(self):
        """Put delta should be in [-1, 0]."""
        for strike_mult in [0.8, 1.0, 1.2]:
            S, K, T, r, sigma = 100, 100 * strike_mult, 0.25, 0.05, 0.20
            d = delta(S, K, T, r, sigma, "put")
            assert -1 <= d <= 0



class TestGamma:
    """Test gamma calculation."""

    def test_gamma_positive(self):
        """Gamma should always be positive."""
        for strike_mult in [0.8, 1.0, 1.2]:
            S, K, T, r, sigma = 100, 100 * strike_mult, 0.25, 0.05, 0.20
            g = gamma(S, K, T, r, sigma)
            assert g >= 0

    def test_gamma_zero_at_expiry(self):
        """Gamma should approach infinity as T->0 (deep ITM/OTM)."""
        S, K, r, sigma = 100, 100, 0.05, 0.20
        g_short = gamma(S, K, 0.01, r, sigma)
        g_long = gamma(S, K, 1.0, r, sigma)
        # Shorter expiry has higher gamma
        assert g_short > g_long


class TestVega:
    """Test vega calculation."""

    def test_vega_positive(self):
        """Vega should be positive."""
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
        v = vega(S, K, T, r, sigma)
        assert v > 0


class TestTheta:
    """Test theta calculation."""

    def test_call_theta_negative(self):
        """Call theta is typically negative (time decay)."""
        S, K, T, r, sigma = 100, 100, 0.25, 0.05, 0.20
        th = theta(S, K, T, r, sigma, "call")
        # Long calls lose value as time passes
        assert th <= 0

    def test_theta_approaches_zero_at_expiry(self):
        """Theta becomes zero at expiration."""
        S, K, r, sigma = 100, 100, 0.05, 0.20
        th = theta(S, K, 0.0, r, sigma)
        assert abs(th) < 1e-6


class TestGreeksCalculator:
    """Test batch Greeks computation."""

    def test_compute_snapshot(self):
        """Test snapshot computation."""
        snapshot = GreeksCalculator.compute_snapshot(
            S=100, K=100, T=0.25, r=0.05, sigma=0.20, option_type="call"
        )
        assert snapshot.delta > 0
        assert snapshot.gamma > 0
        assert snapshot.vega > 0
        assert snapshot.price > 0

    def test_compute_chain(self):
        """Test chain computation."""
        import pandas as pd

        chain = pd.DataFrame({
            "Strike": [90, 100, 110],
            "OptionType": ["call", "call", "call"],
        })

        result = GreeksCalculator.compute_chain(
            chain, S=100, T=0.25, r=0.05, sigma=0.20
        )

        assert len(result) == 3
        assert "Delta" in result.columns
        assert "Gamma" in result.columns
        assert "Price" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
