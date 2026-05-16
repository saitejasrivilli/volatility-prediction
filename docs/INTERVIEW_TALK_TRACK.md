# Interview Talk Track

## 60-second version

I built an equity-volatility research platform focused on predicting transitions into higher-volatility regimes. I started with a single-name AAPL setup, but the positive class was too sparse to support a useful classifier, so rather than overfit it, I reframed the problem as pooled rare-event ranking across liquid equities. The platform uses leakage-safe walk-forward validation, calibrated models, hand-built rule baselines, coefficient-stability reporting, drift diagnostics, and deployable artifacts. In the pooled setting, the strongest fold achieved 13.9x lift in the top 1% of alerts, while the later fold degraded meaningfully, so I present it as evidence of regime-dependent learnable structure rather than production-ready alpha. I also built a provider-pluggable downstream options layer, with a synthetic demo mode so the architecture can be reviewed without licensed historical options data.

## If they ask what you learned

- Rare-event financial ML should not be judged by headline accuracy.
- A bad target framing can make a technically correct model commercially useless.
- Baselines and drift diagnostics matter as much as model choice.
- Honest negative results improve the research process; the AAPL-only failure led to a better pooled framing.

## If they ask what you would do next

1. Add licensed historical options data and test whether alert lift converts into option PnL.
2. Expand the universe and evaluate sector / market-regime sensitivity.
3. Add model monitoring around calibration drift and feature drift.
4. Explore richer event labels, but only after preserving the current baseline discipline.

## If they ask why this is relevant to an AI Quant Engineer role

- It combines ML with financial problem framing rather than treating finance as a generic classification task.
- It produces reproducible artifacts, diagnostics, and an inference surface instead of remaining a notebook experiment.
- It shows ownership of the full lifecycle: data quality, modeling, evaluation, deployment shape, and honest communication of limitations.
