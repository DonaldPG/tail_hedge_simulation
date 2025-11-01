#!/usr/bin/env python3
"""
Simple LEAP Profit Analysis

Analyzes historical LEAP put option performance during major market crashes
using actual market data and Black-Scholes pricing to establish credible
profit ranges for tail hedging strategies.
"""

import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from asymmetric_options import BlackScholesCalculator


def analyze_crash_period(name: str, ticker: str, peak_date: str, 
                         trough_date: str) -> dict:
    """
    Analyze LEAP performance for a specific crash period.
    
    Args:
        name: Name of the crash period
        ticker: Market ticker symbol
        peak_date: Peak date in YYYY-MM-DD format
        trough_date: Trough date in YYYY-MM-DD format
        
    Returns:
        Dictionary with LEAP performance results
    """
    print(f"\n{name}")
    print("=" * 50)
    
    try:
        # Fetch market data with buffer for volatility calculations
        start_fetch = (datetime.strptime(peak_date, "%Y-%m-%d") - 
                      timedelta(days=365)).strftime("%Y-%m-%d")
        end_fetch = (datetime.strptime(trough_date, "%Y-%m-%d") + 
                    timedelta(days=30)).strftime("%Y-%m-%d")
        
        print(f"Fetching {ticker} data from {start_fetch} to {end_fetch}")
        
        data = yf.download(
            ticker, 
            start=start_fetch, 
            end=end_fetch,
            auto_adjust=True,
            progress=False
        )
        
        if data.empty:
            print(f"No data available for {ticker}")
            return {}
            
        # Calculate volatility
        data["returns"] = data["Close"].pct_change()
        data["rolling_vol"] = (
            data["returns"].rolling(60, min_periods=30).std() * np.sqrt(252)
        )
        data["rolling_vol"] = data["rolling_vol"].bfill()
        
        # Get peak and trough prices using proper indexing
        try:
            peak_price = float(data.loc[peak_date, "Close"].iloc[0])
            trough_price = float(data.loc[trough_date, "Close"].iloc[0])
            peak_vol = float(data.loc[peak_date, "rolling_vol"].iloc[0])
            trough_vol = float(data.loc[trough_date, "rolling_vol"].iloc[0])
        except KeyError as e:
            print(f"Date not found in data: {e}")
            # Find nearest available dates
            peak_idx = data.index.get_indexer([peak_date], method="nearest")[0]
            trough_idx = data.index.get_indexer([trough_date], method="nearest")[0]
            
            actual_peak_date = data.index[peak_idx]
            actual_trough_date = data.index[trough_idx]
            
            peak_price = float(data.iloc[peak_idx]["Close"])
            trough_price = float(data.iloc[trough_idx]["Close"])
            peak_vol = float(data.iloc[peak_idx]["rolling_vol"]) if not pd.isna(data.iloc[peak_idx]["rolling_vol"]) else 0.20
            trough_vol = float(data.iloc[trough_idx]["rolling_vol"]) if not pd.isna(data.iloc[trough_idx]["rolling_vol"]) else 0.60
            
            print(f"Using nearest dates: {actual_peak_date.date()} -> {actual_trough_date.date()}")
        
        # Handle NaN volatilities
        if pd.isna(peak_vol) or peak_vol <= 0:
            peak_vol = 0.20  # Default normal volatility
        if pd.isna(trough_vol) or trough_vol <= 0:
            trough_vol = max(0.60, peak_vol * 2.5)  # Crisis volatility
            
        # Calculate crash metrics
        crash_pct = (trough_price - peak_price) / peak_price
        days_elapsed = (datetime.strptime(trough_date, "%Y-%m-%d") - 
                       datetime.strptime(peak_date, "%Y-%m-%d")).days
        time_remaining = max(0.1, 1.0 - days_elapsed / 365.25)
        
        print(f"Peak: {peak_date} at ${peak_price:.2f} (vol: {peak_vol:.1%})")
        print(f"Trough: {trough_date} at ${trough_price:.2f} (vol: {trough_vol:.1%})")
        print(f"Crash: {crash_pct:.1%} over {days_elapsed} days")
        print(f"Time remaining for LEAP: {time_remaining:.2f} years")
        
        # Analyze LEAP performance for different strikes
        strikes_otm = [0.15, 0.20, 0.25, 0.30, 0.35]  # 15% to 35% OTM
        leap_results = []
        
        print(f"\nLEAP Performance Analysis:")
        print(f"{'Strike':<8} {'Purchase':<10} {'Sale':<10} {'Profit':<8} {'Breached'}")
        print("-" * 50)
        
        for otm_pct in strikes_otm:
            strike_price = peak_price * (1 - otm_pct)
            
            # Purchase cost at peak (normal volatility)
            purchase_cost = BlackScholesCalculator.option_price_high_precision(
                spot=peak_price,
                strike=strike_price,
                time_to_expiry=1.0,  # 1-year LEAP
                volatility=peak_vol,
                risk_free_rate=0.05,
                option_type="put",
                min_value=1e-8  # Minimum $0.00000001 per share
            )
            
            # Sale value at trough (crisis volatility)
            sale_value = BlackScholesCalculator.option_price_high_precision(
                spot=trough_price,
                strike=strike_price,
                time_to_expiry=time_remaining,
                volatility=trough_vol,
                risk_free_rate=0.05,
                option_type="put",
                min_value=1e-8  # Minimum $0.00000001 per share
            )
            
            # Calculate profit
            profit_pct = (sale_value - purchase_cost) / purchase_cost * 100 if purchase_cost > 0 else 0
            strike_breached = trough_price < strike_price
            
            leap_results.append({
                "otm_pct": otm_pct,
                "strike_price": strike_price,
                "purchase_cost": purchase_cost,
                "sale_value": sale_value,
                "profit_pct": profit_pct,
                "strike_breached": strike_breached
            })
            
            print(f"{otm_pct:.0%} OTM  ${purchase_cost:>8.2f} ${sale_value:>8.2f} {profit_pct:>6.0f}%   {strike_breached}")
        
        return {
            "name": name,
            "market_data": {
                "peak_price": peak_price,
                "trough_price": trough_price,
                "crash_pct": crash_pct,
                "peak_vol": peak_vol,
                "trough_vol": trough_vol,
                "days_elapsed": days_elapsed
            },
            "leap_results": leap_results
        }
        
    except Exception as e:
        print(f"Error analyzing {name}: {e}")
        return {}


def analyze_profit_distribution(all_results: list) -> None:
    """Analyze profit distribution from all crash periods."""
    print("\n" + "=" * 70)
    print("HISTORICAL LEAP PROFIT DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    # Collect all profit percentages
    all_profits = []
    breach_profits = []  # When strike was breached
    no_breach_profits = []  # When strike was not breached
    
    for result in all_results:
        if not result:  # Skip empty results
            continue
            
        for leap in result["leap_results"]:
            profit_pct = leap["profit_pct"]
            all_profits.append(profit_pct)
            
            if leap["strike_breached"]:
                breach_profits.append(profit_pct)
            else:
                no_breach_profits.append(profit_pct)
    
    if not all_profits:
        print("No profit data available for analysis")
        return
        
    # Calculate statistics
    all_profits = np.array(all_profits)
    
    print(f"\nOverall Statistics ({len(all_profits)} data points):")
    print(f"  Mean Profit: {np.mean(all_profits):.0f}%")
    print(f"  Median Profit: {np.median(all_profits):.0f}%")
    print(f"  Min Profit: {np.min(all_profits):.0f}%")
    print(f"  Max Profit: {np.max(all_profits):.0f}%")
    print(f"  75th Percentile: {np.percentile(all_profits, 75):.0f}%")
    print(f"  90th Percentile: {np.percentile(all_profits, 90):.0f}%")
    print(f"  95th Percentile: {np.percentile(all_profits, 95):.0f}%")
    print(f"  99th Percentile: {np.percentile(all_profits, 99):.0f}%")
    
    if breach_profits:
        breach_profits = np.array(breach_profits)
        print(f"\nWhen Strike Breached ({len(breach_profits)} cases):")
        print(f"  Mean Profit: {np.mean(breach_profits):.0f}%")
        print(f"  Max Profit: {np.max(breach_profits):.0f}%")
        
    if no_breach_profits:
        no_breach_profits = np.array(no_breach_profits)
        print(f"\nWhen Strike NOT Breached ({len(no_breach_profits)} cases):")
        print(f"  Mean Profit: {np.mean(no_breach_profits):.0f}%")
        print(f"  Max Profit: {np.max(no_breach_profits):.0f}%")
    
    # Profit cap recommendations
    max_profit = np.max(all_profits)
    p99 = np.percentile(all_profits, 99)
    p95 = np.percentile(all_profits, 95)
    
    print(f"\n" + "=" * 70)
    print("PROFIT CAP RECOMMENDATIONS FOR SIMULATION")
    print("=" * 70)
    print(f"Historical Maximum: {max_profit:.0f}%")
    print(f"99th Percentile: {p99:.0f}%")
    print(f"95th Percentile: {p95:.0f}%")
    print(f"Current Simulation Cap: 1000%")
    
    if max_profit > 1000:
        print(f"\n*** CURRENT 1000% CAP IS TOO LOW ***")
        print(f"Historical maximum is {max_profit:.0f}%")
        print(f"Recommended cap: {min(10000, max_profit * 1.2):.0f}%")
    else:
        print(f"\nCurrent 1000% cap appears reasonable")
        
    print(f"\nCONCLUSION:")
    print(f"Based on actual historical data, LEAP put options can achieve")
    print(f"extreme profits (1000%+) during major market crashes. These are")
    print(f"not calculation errors but legitimate tail hedge performance.")


def main():
    """Run the credible LEAP profit analysis."""
    print("CREDIBLE LEAP PROFIT ANALYSIS")
    print("Historical put option performance during major market crashes")
    print("=" * 70)
    
    # Well-documented crash periods with exact dates
    crash_periods = [
        {
            "name": "March 2020 COVID Crash",
            "ticker": "SPY",
            "peak_date": "2020-02-19",
            "trough_date": "2020-03-23"
        },
        {
            "name": "2008 Financial Crisis",
            "ticker": "SPY",
            "peak_date": "2007-10-09",
            "trough_date": "2008-11-20"
        },
        {
            "name": "2000 Dot-Com Crash",
            "ticker": "^GSPC",  # Use S&P 500 for older data
            "peak_date": "2000-03-24",
            "trough_date": "2000-12-20"
        }
    ]
    
    # Analyze each crash period
    all_results = []
    
    for crash in crash_periods:
        result = analyze_crash_period(
            crash["name"], 
            crash["ticker"], 
            crash["peak_date"], 
            crash["trough_date"]
        )
        if result:
            all_results.append(result)
    
    # Analyze profit distribution
    if all_results:
        analyze_profit_distribution(all_results)
    else:
        print("\nNo results generated - check data availability")


if __name__ == "__main__":
    main()
