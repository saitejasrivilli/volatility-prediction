"""Unit tests for agent modules."""

import pandas as pd
import pytest

try:
    from src.agents.portfolio_agent import PortfolioDecisionAgent
    from src.agents.research_agent import ResearchOrchestrationAgent
except ImportError:
    from agents.portfolio_agent import PortfolioDecisionAgent
    from agents.research_agent import ResearchOrchestrationAgent


class TestPortfolioDecisionAgent:
    """Test portfolio decision agent."""

    def test_evaluate_alerts_empty(self):
        """Test with empty DataFrame."""
        agent = PortfolioDecisionAgent()
        df = pd.DataFrame(columns=["Date", "Ticker", "Score"])
        result = agent.evaluate_alerts(df)
        assert len(result) == 0

    def test_evaluate_alerts_below_threshold(self):
        """Test that alerts below threshold are rejected."""
        agent = PortfolioDecisionAgent()
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01"]),
            "Ticker": ["AAPL"],
            "Score": [0.40],  # Below MIN_SIGNAL_STRENGTH (~0.55)
        })
        result = agent.evaluate_alerts(df)
        assert not result["approved"].iloc[0]

    def test_evaluate_alerts_above_threshold(self):
        """Test that alerts above threshold are approved."""
        agent = PortfolioDecisionAgent()
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01"]),
            "Ticker": ["AAPL"],
            "Score": [0.70],
        })
        result = agent.evaluate_alerts(df)
        assert result["approved"].iloc[0]

    def test_portfolio_capacity_limit(self):
        """Test portfolio capacity constraint."""
        agent = PortfolioDecisionAgent()
        # Create more alerts than portfolio capacity
        n = agent.config.PORTFOLIO_CAPACITY + 5
        df = pd.DataFrame({
            "Date": pd.to_datetime(["2024-01-01"] * n),
            "Ticker": [f"TICK{i}" for i in range(n)],
            "Score": [0.80] * n,
        })
        result = agent.evaluate_alerts(df)
        approved_count = result["approved"].sum()
        assert approved_count == agent.config.PORTFOLIO_CAPACITY


class TestResearchOrchestrationAgent:
    """Test research agent."""

    def test_generate_allocation_weights_empty(self):
        """Test weight generation with no rankings."""
        agent = ResearchOrchestrationAgent()
        weights = agent.generate_allocation_weights()
        assert weights == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
