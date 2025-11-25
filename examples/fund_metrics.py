

import numpy as np
import pandas as pd
from math import sqrt



TRADING_DAYS_PER_YEAR: int = 252


def create_dummy_time_series() -> tuple[pd.Series, pd.Series, pd.Series]:
    """Build synthetic portfolio, benchmark and risk free time series suitable for examples and tests."""
    # Create a simple range of business days to represent trading dates.
    index = pd.date_range("2020-01-01", periods=504, freq="B")
    random_generator = np.random.default_rng(42)

    # Simulate a random walk for the benchmark and an excess return for the fund.
    daily_benchmark_returns = random_generator.normal(loc=0.0003, scale=0.01, size=len(index))
    daily_fund_excess_returns = random_generator.normal(loc=0.0001, scale=0.005, size=len(index))
    daily_fund_returns = daily_benchmark_returns + daily_fund_excess_returns

    # Turn daily returns into level series starting at 100.
    benchmark_levels = 100.0 * np.cumprod(1.0 + daily_benchmark_returns)
    portfolio_levels = 100.0 * np.cumprod(1.0 + daily_fund_returns)

    # Expected: pandas Series of portfolio market values indexed by business dates with dtype float64.
    portfolio_series = pd.Series(portfolio_levels, index=index, name="portfolio_value")

    # Expected: pandas Series of benchmark index levels indexed by the same business dates with dtype float64.
    benchmark_series = pd.Series(benchmark_levels, index=index, name="benchmark_value")

    # Assume a flat annual risk free rate and convert it to an equivalent daily rate.
    annual_risk_free_rate = 0.02
    daily_risk_free_return = (1.0 + annual_risk_free_rate) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

    # Expected: pandas Series of daily risk free returns (already in decimal form) indexed by the same dates.
    risk_free_series = pd.Series(daily_risk_free_return, index=index, name="risk_free_rate")

    return portfolio_series, benchmark_series, risk_free_series


def compute_daily_returns(level_series: pd.Series) -> pd.Series:
    """Percentage change between consecutive observations in a value series, used as base input for all return based metrics."""
    # Use pct_change from pandas and drop the first missing value.
    daily_returns = level_series.pct_change().dropna()
    return daily_returns


def align_series(*series: pd.Series) -> list[pd.Series]:
    """Align several series on their common dates so that all metrics use identical observations."""
    if not series:
        return []
    # Concatenate along columns and keep only rows where all series are present.
    combined = pd.concat(series, axis=1, join="inner").dropna()
    if combined.empty:
        raise ValueError("Alignment removed all observations. Check that the input series share dates.")
    # Split the aligned DataFrame back into individual Series with the same index.
    aligned = [combined.iloc[:, index] for index in range(combined.shape[1])]
    return aligned


def compute_cumulative_return_pct(daily_returns: pd.Series) -> float:
    """Total growth of capital over the full period, expressed as a decimal, used to answer how much the fund has gained or lost in absolute terms."""
    if daily_returns.empty:
        return 0.0
    # Compound all daily returns into a single growth factor.
    cumulative_growth = float((1.0 + daily_returns).prod())
    cumulative_return = cumulative_growth - 1.0
    return cumulative_return


def compute_annualized_return_pct(daily_returns: pd.Series) -> float:
    """Compounded yearly growth rate implied by the daily returns, used to compare funds with different track record lengths on a common annual basis."""
    if daily_returns.empty:
        return 0.0
    # Compute total growth over the sample.
    cumulative_growth = float((1.0 + daily_returns).prod())
    # Scale to an annual horizon assuming a fixed number of trading days per year.
    periods_per_year = TRADING_DAYS_PER_YEAR / len(daily_returns)
    annualized_return = cumulative_growth ** periods_per_year - 1.0
    return float(annualized_return)


def compute_annualized_volatility_pct(daily_returns: pd.Series) -> float:
    """Standard deviation of daily returns scaled to a yearly horizon, used to quantify how noisy or risky the fund path is in absolute terms."""
    if daily_returns.empty:
        return 0.0
    # Use population standard deviation and annualize using the square root of time rule.
    daily_standard_deviation = float(daily_returns.std(ddof=0))
    annualized_volatility = daily_standard_deviation * sqrt(TRADING_DAYS_PER_YEAR)
    return float(annualized_volatility)


def compute_max_drawdown_pct(level_series: pd.Series) -> float:
    """Largest peak to trough loss observed in the history, expressed as a decimal, used to capture worst historical capital loss and compare downside protection across funds."""
    if level_series.empty:
        return 0.0
    # Track the running maximum and compute percentage distance from that peak.
    running_max = level_series.cummax()
    drawdown_series = level_series / running_max - 1.0
    max_drawdown = float(drawdown_series.min())
    return max_drawdown


def compute_downside_deviation_pct(fund_returns: pd.Series, risk_free_returns: pd.Series) -> float:
    """Standard deviation of negative excess returns relative to the risk free rate, annualized, used to focus on bad volatility that hurts investors instead of all volatility."""
    aligned_fund_returns, aligned_risk_free_returns = align_series(fund_returns, risk_free_returns)
    # Work with excess returns over the risk free rate.
    excess_returns = aligned_fund_returns - aligned_risk_free_returns
    negative_excess_returns = excess_returns[excess_returns < 0.0]
    if negative_excess_returns.empty:
        return 0.0
    # Compute dispersion of bad outcomes only and annualize.
    daily_downside_standard_deviation = float(negative_excess_returns.std(ddof=0))
    annualized_downside_deviation = daily_downside_standard_deviation * sqrt(TRADING_DAYS_PER_YEAR)
    return float(annualized_downside_deviation)


def compute_sharpe_ratio(fund_returns: pd.Series, risk_free_returns: pd.Series) -> float | None:
    """Risk adjusted performance ratio defined as annualized excess return divided by annualized volatility, used to compare efficiency of different funds in turning risk into return."""
    aligned_fund_returns, aligned_risk_free_returns = align_series(fund_returns, risk_free_returns)
    # Compute daily excess returns relative to the risk free rate.
    excess_returns = aligned_fund_returns - aligned_risk_free_returns
    if excess_returns.empty:
        return None
    daily_mean_excess = float(excess_returns.mean())
    daily_standard_deviation = float(excess_returns.std(ddof=0))
    if daily_standard_deviation == 0.0:
        return None
    # Annualize both mean and volatility and form the ratio.
    annualized_mean_excess = daily_mean_excess * TRADING_DAYS_PER_YEAR
    annualized_volatility = daily_standard_deviation * sqrt(TRADING_DAYS_PER_YEAR)
    sharpe_ratio = annualized_mean_excess / annualized_volatility
    return float(sharpe_ratio)


def compute_sortino_ratio(fund_returns: pd.Series, risk_free_returns: pd.Series) -> float | None:
    """Risk adjusted performance ratio that only penalizes downside volatility, used when investors care more about harmful drawdowns than about upside fluctuations."""
    aligned_fund_returns, aligned_risk_free_returns = align_series(fund_returns, risk_free_returns)
    # Start from daily excess returns over the risk free rate.
    excess_returns = aligned_fund_returns - aligned_risk_free_returns
    if excess_returns.empty:
        return None
    daily_mean_excess = float(excess_returns.mean())
    annualized_mean_excess = daily_mean_excess * TRADING_DAYS_PER_YEAR
    # Use downside deviation as the risk measure instead of total volatility.
    annualized_downside_deviation = compute_downside_deviation_pct(fund_returns, risk_free_returns)
    if annualized_downside_deviation == 0.0:
        return None
    sortino_ratio = annualized_mean_excess / annualized_downside_deviation
    return float(sortino_ratio)


def compute_tracking_error_pct(fund_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Annualized standard deviation of active returns of the fund versus the benchmark, used to quantify how tightly the fund hugs or deviates from its reference index."""
    aligned_fund_returns, aligned_benchmark_returns = align_series(fund_returns, benchmark_returns)
    # Compute the daily active return series.
    active_returns = aligned_fund_returns - aligned_benchmark_returns
    if active_returns.empty:
        return 0.0
    # Turn daily tracking volatility into an annual figure.
    daily_tracking_error = float(active_returns.std(ddof=0))
    annualized_tracking_error = daily_tracking_error * sqrt(TRADING_DAYS_PER_YEAR)
    return float(annualized_tracking_error)


def compute_information_ratio(fund_returns: pd.Series, benchmark_returns: pd.Series) -> float | None:
    """Risk adjusted skill metric defined as annualized active return divided by annualized tracking error, used to measure how efficiently the manager adds value versus the benchmark."""
    aligned_fund_returns, aligned_benchmark_returns = align_series(fund_returns, benchmark_returns)
    # Active returns capture outperformance versus the benchmark.
    active_returns = aligned_fund_returns - aligned_benchmark_returns
    if active_returns.empty:
        return None
    daily_mean_active = float(active_returns.mean())
    daily_tracking_error = float(active_returns.std(ddof=0))
    if daily_tracking_error == 0.0:
        return None
    # Annualize both active return and tracking error before forming the ratio.
    annualized_mean_active = daily_mean_active * TRADING_DAYS_PER_YEAR
    annualized_tracking_error = daily_tracking_error * sqrt(TRADING_DAYS_PER_YEAR)
    information_ratio = annualized_mean_active / annualized_tracking_error
    return float(information_ratio)


def compute_beta_and_alpha_pct(fund_returns: pd.Series, benchmark_returns: pd.Series, risk_free_returns: pd.Series) -> tuple[float, float]:
    """Systematic risk and skill metrics where beta measures sensitivity to benchmark moves and alpha measures annualized value added beyond what beta would explain."""
    aligned_fund_returns, aligned_benchmark_returns, aligned_risk_free_returns = align_series(fund_returns, benchmark_returns, risk_free_returns)
    # Work with excess returns for a proper capital asset pricing interpretation.
    excess_fund_returns = aligned_fund_returns - aligned_risk_free_returns
    excess_benchmark_returns = aligned_benchmark_returns - aligned_risk_free_returns

    benchmark_variance = float(excess_benchmark_returns.var(ddof=0))
    if benchmark_variance == 0.0:
        beta_value = 0.0
    else:
        # Estimate covariance and divide by benchmark variance.
        covariance_matrix = np.cov(excess_fund_returns, excess_benchmark_returns, ddof=0)
        covariance = float(covariance_matrix[0, 1])
        beta_value = covariance / benchmark_variance

    # Alpha is the part of excess return not explained by beta times benchmark excess.
    daily_alpha = float(excess_fund_returns.mean() - beta_value * excess_benchmark_returns.mean())
    annualized_alpha = daily_alpha * TRADING_DAYS_PER_YEAR
    return float(beta_value), float(annualized_alpha)


def compute_hit_rate_pct(daily_returns: pd.Series) -> float:
    """Proportion of periods with a positive return, expressed as a decimal, used as an intuitive measure of how often the fund makes money."""
    # Remove missing observations to avoid distortions.
    valid_returns = daily_returns.dropna()
    if valid_returns.empty:
        return 0.0
    # Count how many days have strictly positive performance.
    wins = int((valid_returns > 0.0).sum())
    hit_rate = wins / len(valid_returns)
    return float(hit_rate)


def compute_best_day_return_pct(daily_returns: pd.Series) -> float:
    """Maximum single period return, expressed as a decimal, used to understand upside extremes that may indicate tail events or high convexity."""
    valid_returns = daily_returns.dropna()
    if valid_returns.empty:
        return 0.0
    # Take the largest observed point return.
    best_return = float(valid_returns.max())
    return best_return


def compute_worst_day_return_pct(daily_returns: pd.Series) -> float:
    """Minimum single period return, expressed as a decimal, used to quantify worst day losses that drive stress scenarios and drawdown risk."""
    valid_returns = daily_returns.dropna()
    if valid_returns.empty:
        return 0.0
    # Take the smallest observed point return.
    worst_return = float(valid_returns.min())
    return worst_return


def compute_fund_metrics(portfolio_levels: pd.Series, benchmark_levels: pd.Series, risk_free_returns: pd.Series) -> dict[str, dict[str, float | int | None]]:
    """High level aggregator that computes a comprehensive set of fund metrics grouped by theme from standard pandas Series inputs."""
    # Expected: all inputs are pandas Series indexed by the same business dates with float values.
    fund_returns = compute_daily_returns(portfolio_levels)
    benchmark_returns = compute_daily_returns(benchmark_levels)

    # Pure performance metrics.
    cumulative_return_pct = compute_cumulative_return_pct(fund_returns)
    annualized_return_pct = compute_annualized_return_pct(fund_returns)
    benchmark_annualized_return_pct = compute_annualized_return_pct(benchmark_returns)
    excess_annualized_return_pct = annualized_return_pct - benchmark_annualized_return_pct

    # Absolute risk metrics.
    volatility_pct = compute_annualized_volatility_pct(fund_returns)
    max_drawdown_pct = compute_max_drawdown_pct(portfolio_levels)
    downside_deviation_pct = compute_downside_deviation_pct(fund_returns, risk_free_returns)

    # Risk adjusted performance metrics.
    sharpe_ratio = compute_sharpe_ratio(fund_returns, risk_free_returns)
    sortino_ratio = compute_sortino_ratio(fund_returns, risk_free_returns)

    # Metrics relative to the benchmark.
    tracking_error_pct = compute_tracking_error_pct(fund_returns, benchmark_returns)
    information_ratio = compute_information_ratio(fund_returns, benchmark_returns)
    beta_value, alpha_annualized_pct = compute_beta_and_alpha_pct(fund_returns, benchmark_returns, risk_free_returns)

    # Simple trading profile from the daily return stream.
    hit_rate_pct = compute_hit_rate_pct(fund_returns)
    best_day_return_pct = compute_best_day_return_pct(fund_returns)
    worst_day_return_pct = compute_worst_day_return_pct(fund_returns)

    days_in_sample = int(len(fund_returns))

    metrics: dict[str, dict[str, float | int | None]] = {
        "performance": {
            "cumulative_return_pct": cumulative_return_pct,
            "annualized_return_pct": annualized_return_pct,
            "benchmark_annualized_return_pct": benchmark_annualized_return_pct,
            "excess_annualized_return_pct": excess_annualized_return_pct,
            "days_in_sample": days_in_sample,
        },
        "risk": {
            "annualized_volatility_pct": volatility_pct,
            "max_drawdown_pct": max_drawdown_pct,
            "downside_deviation_pct": downside_deviation_pct,
        },
        "risk_adjusted": {
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
        },
        "relative_to_benchmark": {
            "beta": beta_value,
            "alpha_annualized_pct": alpha_annualized_pct,
            "tracking_error_pct": tracking_error_pct,
            "information_ratio": information_ratio,
        },
        "trading_profile": {
            "hit_rate_pct": hit_rate_pct,
            "best_day_return_pct": best_day_return_pct,
            "worst_day_return_pct": worst_day_return_pct,
        },
    }
    return metrics


def format_metric_value(metric_name: str, value: float | int | None) -> str:
    """Render a metric value as a human friendly string, adding percent signs only when the name ends with the pct suffix."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if metric_name.endswith("_pct"):
            # Metrics with the pct suffix are stored as decimals and rendered as percentages.
            return f"{value * 100.0:.2f} %"
        return f"{value:.4f}"
    return str(value)


def pretty_print_metrics(metrics: dict[str, dict[str, float | int | None]]) -> None:
    """Console printer that produces a compact, readable report grouped by logical metric sections."""
    for section_name, section_metrics in metrics.items():
        title = section_name.replace("_", " ").title()
        print()
        print(title)
        print("-" * len(title))
        for metric_name, metric_value in section_metrics.items():
            # Turn snake_case names into more readable labels.
            label = metric_name.replace("_", " ")
            formatted_value = format_metric_value(metric_name, metric_value)
            print(f"{label:35s}{formatted_value}")


def print_stats() -> None:
    """Main entry point used when the module is executed as a script to build dummy data, compute metrics and print them."""
    portfolio_levels, benchmark_levels, risk_free_returns = create_dummy_time_series()
    metrics = compute_fund_metrics(portfolio_levels, benchmark_levels, risk_free_returns)
    pretty_print_metrics(metrics)


if __name__ == "__main__":
    print_stats()
