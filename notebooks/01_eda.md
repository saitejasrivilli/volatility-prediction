# Exploratory Data Analysis: AAPL Volatility

This notebook demonstrates the data and features used in the system.

## Step 1: Load Data

```python
from src.config import DATA_CONFIG
from src.data_pipeline import load_data

df = load_data(DATA_CONFIG)
print(f"Shape: {df.shape}")
print(f"Date range: {df.index.min()} to {df.index.max()}")
df.head()
```

## Step 2: Examine Raw Data

```python
# Summary statistics
print(df.describe())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Price range
print(f"\nPrice range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
print(f"Average daily return: {df['Close'].pct_change().mean()*100:.3f}%")
print(f"Daily volatility: {df['Close'].pct_change().std()*100:.3f}%")
```

## Step 3: Build Features

```python
from src.feature_engineering import build_features

features = build_features(df)
print(f"Features shape: {features.shape}")
print(f"\nFeature columns:\n{features.columns.tolist()}")
```

## Step 4: Feature Correlation with Target

```python
from src.feature_engineering import TargetBuilder

close = df['Close']
returns = close.pct_change()
target = TargetBuilder.volatility_spike_target(close, returns)

# Correlation
correlations = features.corrwith(target).abs().sort_values(ascending=False)
print("Top features correlated with volatility spikes:")
print(correlations.head(10))
```

## Step 5: Volatility Analysis

```python
# Calculate rolling volatility
vol_20 = returns.rolling(20).std()

# Show volatility statistics
print(f"Mean volatility (20-day): {vol_20.mean()*100:.3f}%")
print(f"Std of volatility: {vol_20.std()*100:.3f}%")
print(f"Min volatility: {vol_20.min()*100:.3f}%")
print(f"Max volatility: {vol_20.max()*100:.3f}%")

# Percentile analysis
print(f"\n25th percentile: {vol_20.quantile(0.25)*100:.3f}%")
print(f"50th percentile: {vol_20.quantile(0.50)*100:.3f}%")
print(f"75th percentile: {vol_20.quantile(0.75)*100:.3f}%")
```

## Step 6: Feature Distribution

```python
# RSI distribution
print("RSI Statistics:")
print(features['RSI'].describe())

# Bollinger Band Width
print("\nBollinger Band Width:")
print(features['BB_Width'].describe())

# Volume ratio
print("\nVolume Ratio:")
print(features['Volume_Ratio'].describe())
```

## Insights

1. **Volatility clustering**: High volatility days are followed by more high volatility days
2. **Mean reversion**: Extreme RSI readings tend to revert
3. **Trend persistence**: Price tends to follow moving averages
4. **Seasonality**: Certain periods (earnings, Fed announcements) have higher vol spikes

See README.md for full methodology explanation.
