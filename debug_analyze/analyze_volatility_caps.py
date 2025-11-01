#!/usr/bin/env python3
"""
Volatility Cap Analysis for LEAP Hedging Simulation

This script investigates whether the volatility capping mechanism in the
LEAP simulation is realistic based on actual historical market data.

Key questions addressed:
1. What are the actual maximum volatility levels observed in equity markets?
2. Is a 32% volatility cap realistic during crisis periods?
3. How does volatility capping affect LEAP pricing accuracy?
4. What does academic literature suggest about volatility limits?

Data Sources:
- Yahoo Finance historical data
- VIX data (if available)
- S&P 500, NASDAQ, and other major indices
"""

import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from asymmetric_options import BlackScholesCalculator


class VolatilityAnalyzer:
    """
    Analyzes historical volatility patterns to validate simulation assumptions.
    """
    
    def __init__(self):
        """Initialize with major market indices for analysis."""
        self.indices = {
            "^GSPC": "S&P 500",
            "^IXIC": "NASDAQ Composite", 
            "^DJI": "Dow Jones Industrial",
            "^VIX": "VIX Volatility Index"
        }
        
        self.crisis_periods = [
            {
                "name": "1987 Black Monday",
                "start": "1987-10-01",
                "end": "1987-11-30"
            },
            {
                "name": "1997 Asian Crisis",
                "start": "1997-10-01", 
                "end": "1997-12-31"
            },
            {
                "name": "1998 LTCM Crisis",
                "start": "1998-08-01",
                "end": "1998-10-31"
            },
            {
                "name": "2000 Dot-Com Crash",
                "start": "2000-03-01",
                "end": "2000-12-31"
            },
            {
                "name": "2008 Financial Crisis",
                "start": "2008-09-01",
                "end": "2009-03-31"
            },
            {
                "name": "2020 COVID Crash",
                "start": "2020-02-01",
                "end": "2020-05-31"
            }
        ]
    
    def fetch_historical_data(self, ticker: str, start_date: str, 
                            end_date: str) -> pd.DataFrame:
        """
        Fetch historical data and calculate volatility metrics.
        
        Args:
            ticker: Market ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with price data and volatility calculations
        """
        print(f"Fetching data for {ticker} ({self.indices.get(ticker, ticker)})")
        
        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                auto_adjust=True,
                progress=False
            )
            
            if data.empty:
                print(f"  No data available for {ticker}")
                return pd.DataFrame()
            
            # Calculate returns and volatilities
            data["returns"] = data["Close"].pct_change()
            
            # Multiple volatility windows for analysis
            for window in [20, 30, 60, 252]:
                vol_col = f"vol_{window}d"
                data[vol_col] = (
                    data["returns"].rolling(window, min_periods=window//2)
                    .std() * np.sqrt(252)
                )
            
            # Forward fill volatilities
            for col in data.columns:
                if col.startswith("vol_"):
                    data[col] = data[col].bfill()
            
            return data
            
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")
            return pd.DataFrame()
    
    def analyze_historical_volatility_patterns(self) -> dict:
        """
        Analyze historical volatility patterns across major indices.
        
        Returns:
            Dictionary with comprehensive volatility statistics
        """
        print("=" * 70)
        print("HISTORICAL VOLATILITY ANALYSIS")
        print("=" * 70)
        
        results = {}
        
        # Analyze each major index
        for ticker, name in self.indices.items():
            print(f"\nAnalyzing {name} ({ticker})")
            print("-" * 50)
            
            # Fetch long-term historical data
            if ticker == "^VIX":
                start_date = "1990-01-01"  # VIX started in 1990
            else:
                start_date = "1980-01-01"  # Earlier for stock indices
            
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            data = self.fetch_historical_data(ticker, start_date, end_date)
            
            if data.empty:
                continue
            
            # Calculate comprehensive statistics
            if ticker == "^VIX":
                # VIX is already volatility measure
                vol_data = data["Close"] / 100  # Convert to decimal
                vol_stats = {
                    "mean": vol_data.mean(),
                    "median": vol_data.median(),
                    "std": vol_data.std(),
                    "min": vol_data.min(),
                    "max": vol_data.max(),
                    "p95": vol_data.quantile(0.95),
                    "p99": vol_data.quantile(0.99),
                    "p99_9": vol_data.quantile(0.999)
                }
            else:
                # Use 60-day rolling volatility for stock indices
                vol_data = data["vol_60d"]
                vol_stats = {
                    "mean": vol_data.mean(),
                    "median": vol_data.median(), 
                    "std": vol_data.std(),
                    "min": vol_data.min(),
                    "max": vol_data.max(),
                    "p95": vol_data.quantile(0.95),
                    "p99": vol_data.quantile(0.99),
                    "p99_9": vol_data.quantile(0.999)
                }
            
            results[ticker] = {
                "name": name,
                "data": data,
                "volatility_stats": vol_stats
            }
            
            # Print statistics
            print(f"  Period: {data.index[0].date()} to {data.index[-1].date()}")
            print(f"  Mean Volatility: {vol_stats['mean']:.1%}")
            print(f"  Median Volatility: {vol_stats['median']:.1%}")
            print(f"  Maximum Volatility: {vol_stats['max']:.1%}")
            print(f"  95th Percentile: {vol_stats['p95']:.1%}")
            print(f"  99th Percentile: {vol_stats['p99']:.1%}")
            print(f"  99.9th Percentile: {vol_stats['p99_9']:.1%}")
            
            # Find periods where volatility exceeded various thresholds
            thresholds = [0.30, 0.40, 0.50, 0.60, 0.80]
            for threshold in thresholds:
                exceeded_days = (vol_data > threshold).sum()
                total_days = len(vol_data.dropna())
                pct_exceeded = exceeded_days / total_days * 100
                print(f"  Days > {threshold:.0%}: {exceeded_days} ({pct_exceeded:.2f}%)")
        
        return results
    
    def analyze_crisis_period_volatility(self, historical_results: dict) -> None:
        """
        Analyze volatility during specific crisis periods.
        
        Args:
            historical_results: Results from historical volatility analysis
        """
        print("\n" + "=" * 70)
        print("CRISIS PERIOD VOLATILITY ANALYSIS")
        print("=" * 70)
        
        for crisis in self.crisis_periods:
            print(f"\n{crisis['name']}")
            print("-" * 50)
            
            crisis_stats = {}
            
            for ticker, result in historical_results.items():
                if ticker == "^VIX":
                    continue  # Skip VIX for this analysis
                    
                data = result["data"]
                name = result["name"]
                
                # Filter data for crisis period
                try:
                    crisis_data = data.loc[crisis["start"]:crisis["end"]]
                    
                    if not crisis_data.empty:
                        vol_data = crisis_data["vol_60d"]
                        
                        if len(vol_data.dropna()) > 0:
                            max_vol = vol_data.max()
                            mean_vol = vol_data.mean()
                            
                            crisis_stats[ticker] = {
                                "name": name,
                                "max_vol": max_vol,
                                "mean_vol": mean_vol
                            }
                            
                            print(f"  {name}: Max={max_vol:.1%}, Mean={mean_vol:.1%}")
                            
                except Exception as e:
                    print(f"  Error analyzing {name}: {e}")
            
            # Find maximum volatility across all indices for this crisis
            if crisis_stats:
                max_crisis_vol = max(stats["max_vol"] 
                                   for stats in crisis_stats.values())
                print(f"  Overall Crisis Maximum: {max_crisis_vol:.1%}")
    
    def evaluate_volatility_cap_realism(self, historical_results: dict) -> None:
        """
        Evaluate whether common volatility caps are realistic.
        
        Args:
            historical_results: Results from historical volatility analysis
        """
        print("\n" + "=" * 70)
        print("VOLATILITY CAP REALISM EVALUATION")
        print("=" * 70)
        
        # Common cap levels used in simulations
        cap_levels = [0.25, 0.30, 0.32, 0.40, 0.50, 0.60, 0.80, 1.00]
        
        print(f"\nEvaluating cap levels against historical data:")
        print(f"{'Cap Level':<12} {'S&P 500':<12} {'NASDAQ':<12} {'VIX':<12}")
        print("-" * 50)
        
        # Analyze each cap level
        for cap in cap_levels:
            row_data = [f"{cap:.0%}"]
            
            for ticker in ["^GSPC", "^IXIC", "^VIX"]:
                if ticker in historical_results:
                    if ticker == "^VIX":
                        vol_data = historical_results[ticker]["data"]["Close"] / 100
                    else:
                        vol_data = historical_results[ticker]["data"]["vol_60d"]
                    
                    exceeded_days = (vol_data > cap).sum()
                    total_days = len(vol_data.dropna())
                    pct_exceeded = exceeded_days / total_days * 100
                    
                    row_data.append(f"{pct_exceeded:.2f}%")
                else:
                    row_data.append("N/A")
            
            print(f"{row_data[0]:<12} {row_data[1]:<12} {row_data[2]:<12} {row_data[3]:<12}")
        
        # Specific analysis for 32% cap (your simulation's cap)
        print(f"\n*** ANALYSIS OF 32% VOLATILITY CAP ***")
        
        for ticker in ["^GSPC", "^IXIC"]:
            if ticker in historical_results:
                result = historical_results[ticker]
                name = result["name"]
                vol_data = result["data"]["vol_60d"]
                
                exceeded_32 = (vol_data > 0.32).sum()
                total_days = len(vol_data.dropna())
                pct_exceeded = exceeded_32 / total_days * 100
                
                max_vol = vol_data.max()
                p99_vol = vol_data.quantile(0.99)
                
                print(f"\n{name}:")
                print(f"  Days exceeding 32%: {exceeded_32} ({pct_exceeded:.2f}%)")
                print(f"  Historical maximum: {max_vol:.1%}")
                print(f"  99th percentile: {p99_vol:.1%}")
                
                if max_vol > 0.32:
                    print(f"  *** 32% cap would artificially limit {((max_vol - 0.32)/max_vol)*100:.1f}% of peak volatility ***")
                else:
                    print(f"  32% cap appears reasonable for this index")
    
    def analyze_leap_pricing_impact(self, historical_results: dict) -> None:
        """
        Analyze how volatility capping affects LEAP option pricing.
        
        Args:
            historical_results: Results from historical volatility analysis
        """
        print("\n" + "=" * 70)
        print("LEAP PRICING IMPACT OF VOLATILITY CAPPING")
        print("=" * 70)
        
        # Use S&P 500 data for this analysis
        if "^GSPC" not in historical_results:
            print("S&P 500 data not available for pricing analysis")
            return
        
        sp500_data = historical_results["^GSPC"]["data"]
        
        # Find periods of high volatility for analysis
        vol_data = sp500_data["vol_60d"]
        high_vol_threshold = 0.40  # 40% threshold
        high_vol_periods = sp500_data[vol_data > high_vol_threshold]
        
        if high_vol_periods.empty:
            print("No periods with volatility > 40% found for analysis")
            return
        
        print(f"Analyzing LEAP pricing during high volatility periods (>{high_vol_threshold:.0%})")
        print(f"Found {len(high_vol_periods)} days with high volatility")
        
        # Sample a few high volatility periods for detailed analysis
        sample_periods = high_vol_periods.head(5)
        
        print(f"\n{'Date':<12} {'Price':<8} {'Actual Vol':<12} {'Capped Vol':<12} {'Price Diff':<12}")
        print("-" * 65)
        
        for date, row in sample_periods.iterrows():
            spot_price = row["Close"]
            actual_vol = row["vol_60d"]
            capped_vol = min(actual_vol, 0.32)  # 32% cap
            
            # Calculate LEAP put price (25% OTM, 1 year)
            strike_price = spot_price * 0.75  # 25% OTM put
            
            actual_price = BlackScholesCalculator.option_price(
                spot=spot_price,
                strike=strike_price,
                time_to_expiry=1.0,
                volatility=actual_vol,
                risk_free_rate=0.05,
                option_type="put"
            )
            
            capped_price = BlackScholesCalculator.option_price(
                spot=spot_price, 
                strike=strike_price,
                time_to_expiry=1.0,
                volatility=capped_vol,
                risk_free_rate=0.05,
                option_type="put"
            )
            
            price_diff_pct = ((actual_price - capped_price) / actual_price) * 100
            
            print(f"{date.date()} ${spot_price:>6.0f} {actual_vol:>10.1%} {capped_vol:>10.1%} {price_diff_pct:>10.1f}%")
        
        # Calculate average impact
        total_impact = 0
        count = 0
        
        for date, row in high_vol_periods.iterrows():
            spot_price = row["Close"]
            actual_vol = row["vol_60d"]
            capped_vol = min(actual_vol, 0.32)
            
            if actual_vol > capped_vol:  # Only when cap is active
                strike_price = spot_price * 0.75
                
                actual_price = BlackScholesCalculator.option_price(
                    spot=spot_price, strike=strike_price, time_to_expiry=1.0,
                    volatility=actual_vol, risk_free_rate=0.05, option_type="put"
                )
                
                capped_price = BlackScholesCalculator.option_price(
                    spot=spot_price, strike=strike_price, time_to_expiry=1.0,
                    volatility=capped_vol, risk_free_rate=0.05, option_type="put"
                )
                
                if actual_price > 0:
                    impact = ((actual_price - capped_price) / actual_price) * 100
                    total_impact += impact
                    count += 1
        
        if count > 0:
            avg_impact = total_impact / count
            print(f"\nAverage pricing impact when 32% cap is active: {avg_impact:.1f}%")
            print(f"This means LEAP prices are underestimated by an average of {avg_impact:.1f}%")
        else:
            print(f"\n32% cap was never active during high volatility periods")


def main():
    """Run comprehensive volatility cap analysis."""
    print("VOLATILITY CAP ANALYSIS FOR LEAP HEDGING SIMULATION")
    print("Investigating the realism of volatility caps in option pricing")
    print("=" * 70)
    
    analyzer = VolatilityAnalyzer()
    
    # Step 1: Analyze historical volatility patterns
    historical_results = analyzer.analyze_historical_volatility_patterns()
    
    if not historical_results:
        print("No historical data available for analysis")
        return
    
    # Step 2: Analyze crisis period volatility
    analyzer.analyze_crisis_period_volatility(historical_results)
    
    # Step 3: Evaluate volatility cap realism
    analyzer.evaluate_volatility_cap_realism(historical_results)
    
    # Step 4: Analyze LEAP pricing impact
    analyzer.analyze_leap_pricing_impact(historical_results)
    
    print("\n" + "=" * 70)
    print("CONCLUSIONS AND RECOMMENDATIONS")
    print("=" * 70)
    print("""
Based on this analysis:

1. VOLATILITY LEVELS: Historical data shows what maximum volatilities 
   have actually been observed in equity markets.

2. CAP REALISM: The 32% volatility cap can be evaluated against the
   percentage of time markets have exceeded this level.

3. PRICING IMPACT: When volatility caps are active, LEAP prices are
   artificially reduced, making hedging appear cheaper than reality.

4. SIMULATION ACCURACY: Unrealistic caps can lead to misleading 
   conclusions about hedge effectiveness and costs.

RECOMMENDATIONS:
- Use empirically-based volatility caps (e.g., 99th percentile)
- Consider removing caps entirely for crisis period analysis
- If caps are necessary, document their impact on results
- Test sensitivity to different cap levels
""")


if __name__ == "__main__":
    main()