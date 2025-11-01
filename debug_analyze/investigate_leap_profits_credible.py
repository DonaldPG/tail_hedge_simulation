#!/usr/bin/env python3
"""
Credible Investigation of Extreme LEAP Profits.

This script analyzes actual historical market data to determine realistic
ranges for LEAP put option profits during major market crashes. Rather than
relying on anecdotal claims, we calculate theoretical profits using:

1. Actual historical price data from Yahoo Finance
2. Standard Black-Scholes option pricing model
3. Historical volatility calculations from market data
4. Documented market crash periods with precise dates

The goal is to establish credible benchmarks for what constitutes realistic
vs unrealistic LEAP profits in our simulation.
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname%s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from asymmetric_options import BlackScholesCalculator


class HistoricalLEAPAnalyzer:
    """
    Analyzes historical LEAP performance using actual market data and
    standard option pricing models.
    """
    
    def __init__(self):
        """Initialize the analyzer with documented crash periods."""
        # Well-documented market crash periods with approximate dates
        # We'll find the actual peak/trough dates from the data
        self.crash_periods = [
            {
                "name": "March 2020 COVID Crash",
                "ticker": "SPY",
                "peak_year_month": "2020-02",
                "trough_year_month": "2020-03",
                "description": "Fastest bear market in history"
            },
            {
                "name": "2008 Financial Crisis",
                "ticker": "^GSPC",  # Use S&P 500 for older data
                "peak_year_month": "2007-10", 
                "trough_year_month": "2008-11",
                "description": "Lehman collapse period"
            },
            {
                "name": "2000 Dot-Com Crash", 
                "ticker": "^GSPC",  # Use S&P 500 for consistent comparison
                "peak_year_month": "2000-03",
                "trough_year_month": "2000-09",  # Earlier trough date
                "description": "Technology bubble burst"
            },
            {
                "name": "1987 Black Monday", 
                "ticker": "^GSPC",  # S&P 500 has data back to 1950s
                "peak_year_month": "1987-08",
                "trough_year_month": "1987-10",
                "description": "Single day crash"
            }
        ]
        
        # Standard LEAP parameters for analysis
        self.otm_strikes = [0.15, 0.20, 0.25, 0.30, 0.35]  # 15% to 35% OTM
        self.leap_duration = 1.0  # 1-year LEAPs
        self.risk_free_rate = 0.05  # 5% risk-free rate
        
    def fetch_market_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical market data for analysis.
        
        Args:
            ticker: Market ticker symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with price and volatility data
        """
        try:
            # Fetch data with some buffer for volatility calculations
            buffer_start = (datetime.strptime(start_date, "%Y-%m-%d") - 
                          timedelta(days=365)).strftime("%Y-%m-%d")
            
            logger.info(f"Fetching {ticker} data from {buffer_start} to {end_date}")
            
            data = yf.download(
                ticker, 
                start=buffer_start,
                end=end_date,
                auto_adjust=True,
                progress=False
            )
            
            if data.empty:
                logger.error(f"No data found for {ticker}")
                return pd.DataFrame()
                
            # Calculate returns and rolling volatility
            data["returns"] = data["Close"].pct_change()
            data["rolling_vol_30d"] = (
                data["returns"].rolling(30, min_periods=15).std() * np.sqrt(252)
            )
            data["rolling_vol_60d"] = (
                data["returns"].rolling(60, min_periods=30).std() * np.sqrt(252)
            )
            
            # Fix deprecated fillna method
            data["rolling_vol_30d"] = data["rolling_vol_30d"].bfill()
            data["rolling_vol_60d"] = data["rolling_vol_60d"].bfill()
            
            return data.loc[start_date:end_date]
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
    
    def find_peak_and_trough(self, data: pd.DataFrame, peak_month: str, 
                           trough_month: str) -> tuple:
        """
        Find actual peak and trough dates within specified months.
        
        Args:
            data: Market data DataFrame
            peak_month: Month to search for peak (YYYY-MM format)
            trough_month: Month to search for trough (YYYY-MM format)
            
        Returns:
            Tuple of (peak_date, trough_date, peak_price, trough_price)
        """
        try:
            # Create proper date ranges for peak month
            peak_year, peak_month_num = peak_month.split("-")
            peak_start = f"{peak_year}-{peak_month_num}-01"
            
            # Calculate last day of peak month properly
            if peak_month_num in ["01", "03", "05", "07", "08", "10", "12"]:
                peak_end_day = "31"
            elif peak_month_num in ["04", "06", "09", "11"]:
                peak_end_day = "30"
            else:  # February
                # Simple leap year check
                year_int = int(peak_year)
                if year_int % 4 == 0 and (year_int % 100 != 0 or year_int % 400 == 0):
                    peak_end_day = "29"
                else:
                    peak_end_day = "28"
            
            peak_end = f"{peak_year}-{peak_month_num}-{peak_end_day}"
            
            # Find peak in specified month
            peak_mask = (data.index >= peak_start) & (data.index <= peak_end)
            peak_data = data[peak_mask]
            
            if peak_data.empty:
                # Expand search to 3-month window around target month
                prev_month = int(peak_month_num) - 1
                next_month = int(peak_month_num) + 1
                
                if prev_month <= 0:
                    prev_month = 12
                    prev_year = str(int(peak_year) - 1)
                else:
                    prev_year = peak_year
                    
                if next_month > 12:
                    next_month = 1
                    next_year = str(int(peak_year) + 1)
                else:
                    next_year = peak_year
                
                peak_start_expanded = f"{prev_year}-{prev_month:02d}-01"
                peak_end_expanded = f"{next_year}-{next_month:02d}-28"
                peak_mask = (data.index >= peak_start_expanded) & (data.index <= peak_end_expanded)
                peak_data = data[peak_mask]
            
            if peak_data.empty:
                logger.warning(f"No peak data found for {peak_month}")
                return None, None, None, None
                
            peak_idx = peak_data["Close"].idxmax()
            peak_price = float(peak_data.loc[peak_idx, "Close"])
            
            # Create proper date ranges for trough month
            trough_year, trough_month_num = trough_month.split("-")
            trough_start = f"{trough_year}-{trough_month_num}-01"
            
            # Calculate last day of trough month properly
            if trough_month_num in ["01", "03", "05", "07", "08", "10", "12"]:
                trough_end_day = "31"
            elif trough_month_num in ["04", "06", "09", "11"]:
                trough_end_day = "30"
            else:  # February
                year_int = int(trough_year)
                if year_int % 4 == 0 and (year_int % 100 != 0 or year_int % 400 == 0):
                    trough_end_day = "29"
                else:
                    trough_end_day = "28"
            
            trough_end = f"{trough_year}-{trough_month_num}-{trough_end_day}"
            
            # Find trough in specified month
            trough_mask = (data.index >= trough_start) & (data.index <= trough_end)
            trough_data = data[trough_mask]
            
            if trough_data.empty:
                # Expand search to 3-month window around target month
                prev_month = int(trough_month_num) - 1
                next_month = int(trough_month_num) + 1
                
                if prev_month <= 0:
                    prev_month = 12
                    prev_year = str(int(trough_year) - 1)
                else:
                    prev_year = trough_year
                    
                if next_month > 12:
                    next_month = 1
                    next_year = str(int(trough_year) + 1)
                else:
                    next_year = trough_year
                
                trough_start_expanded = f"{prev_year}-{prev_month:02d}-01"
                trough_end_expanded = f"{next_year}-{next_month:02d}-28"
                trough_mask = (data.index >= trough_start_expanded) & (data.index <= trough_end_expanded)
                trough_data = data[trough_mask]
            
            if trough_data.empty:
                logger.warning(f"No trough data found for {trough_month}")
                return None, None, None, None
                
            trough_idx = trough_data["Close"].idxmin()
            trough_price = float(trough_data.loc[trough_idx, "Close"])
            
            return peak_idx, trough_idx, peak_price, trough_price
            
        except Exception as e:
            logger.error(f"Error finding peak/trough: {e}")
            return None, None, None, None

    def calculate_leap_performance(self, crash_period: dict) -> dict:
        """
        Calculate theoretical LEAP performance for a specific crash period.
        
        Args:
            crash_period: Dictionary containing crash period information
            
        Returns:
            Dictionary with detailed LEAP performance analysis
        """
        logger.info(f"Analyzing {crash_period['name']}")
        
        # Determine date range for data fetch
        peak_year = int(crash_period["peak_year_month"].split("-")[0])
        trough_year = int(crash_period["trough_year_month"].split("-")[0])
        
        start_date = f"{peak_year-1}-01-01"
        end_date = f"{trough_year+1}-12-31"
        
        # Fetch market data
        data = self.fetch_market_data(crash_period["ticker"], start_date, end_date)
        
        if data.empty:
            logger.warning(f"No data available for {crash_period['name']}")
            return {}
            
        # Find actual peak and trough dates
        peak_date, trough_date, peak_price, trough_price = self.find_peak_and_trough(
            data, crash_period["peak_year_month"], crash_period["trough_year_month"]
        )
        
        # Check if we got valid results
        if peak_date is None or trough_date is None:
            logger.warning(f"Could not find peak/trough for {crash_period['name']}")
            return {}
            
        # Ensure peak_date and trough_date are Timestamps, not Series
        if isinstance(peak_date, pd.Series):
            peak_date = peak_date.iloc[0] if len(peak_date) > 0 else None
        if isinstance(trough_date, pd.Series):
            trough_date = trough_date.iloc[0] if len(trough_date) > 0 else None
            
        if peak_date is None or trough_date is None:
            logger.warning(f"Invalid peak/trough dates for {crash_period['name']}")
            return {}
            
        # Calculate time elapsed
        days_elapsed = (trough_date - peak_date).days
        time_to_expiry_at_sale = max(0.1, self.leap_duration - days_elapsed / 365.25)
        
        # Get volatilities with proper error handling
        try:
            peak_vol = float(data.loc[peak_date, "rolling_vol_60d"])
        except (KeyError, TypeError):
            peak_vol = 0.20  # Default normal volatility
            
        try:
            trough_vol = float(data.loc[trough_date, "rolling_vol_60d"])
        except (KeyError, TypeError):
            trough_vol = min(0.80, peak_vol * 3.0)  # Crisis volatility estimate
        
        # Handle NaN volatilities
        if pd.isna(peak_vol):
            peak_vol = 0.20
        if pd.isna(trough_vol):
            trough_vol = min(0.80, peak_vol * 3.0)
            
        # Calculate crash magnitude
        crash_magnitude = (trough_price - peak_price) / peak_price
        
        print(f"\n{crash_period['name']} Analysis:")
        print(f"  Period: {peak_date.date()} to {trough_date.date()} ({days_elapsed} days)")
        print(f"  Price: ${peak_price:.2f} → ${trough_price:.2f}")
        print(f"  Crash: {crash_magnitude:.1%}")
        print(f"  Volatility: {peak_vol:.1%} → {trough_vol:.1%}")
        print(f"  Time Remaining: {time_to_expiry_at_sale:.2f} years")
        
        # Calculate LEAP performance for each strike
        leap_results = []
        
        for otm_pct in self.otm_strikes:
            strike_price = peak_price * (1 - otm_pct)
            
            # Purchase cost at peak (normal volatility)
            purchase_cost = BlackScholesCalculator.option_price(
                spot=peak_price,
                strike=strike_price,
                time_to_expiry=self.leap_duration,
                volatility=peak_vol,
                risk_free_rate=self.risk_free_rate,
                option_type="put"
            )
            
            # Sale value at trough (crisis volatility)
            sale_value = BlackScholesCalculator.option_price(
                spot=trough_price,
                strike=strike_price,
                time_to_expiry=time_to_expiry_at_sale,
                volatility=trough_vol,
                risk_free_rate=self.risk_free_rate,
                option_type="put"
            )
            
            # Calculate intrinsic value for validation
            intrinsic_value = max(0, strike_price - trough_price)
            
            # Calculate performance metrics
            if purchase_cost > 0:
                profit = sale_value - purchase_cost
                profit_pct = profit / purchase_cost * 100
                
                # Check if strike was breached
                strike_breached = trough_price < strike_price
                
                leap_result = {
                    "otm_pct": otm_pct,
                    "strike_price": strike_price,
                    "purchase_cost": purchase_cost,
                    "sale_value": sale_value,
                    "intrinsic_value": intrinsic_value,
                    "profit": profit,
                    "profit_pct": profit_pct,
                    "strike_breached": strike_breached
                }
                
                leap_results.append(leap_result)
                
                print(f"    {otm_pct:.0%} OTM (${strike_price:.2f}):")
                print(f"      Purchase: ${purchase_cost:.2f}")
                print(f"      Sale: ${sale_value:.2f}")
                print(f"      Intrinsic: ${intrinsic_value:.2f}")
                print(f"      Profit: {profit_pct:.0f}%")
                print(f"      Strike Breached: {strike_breached}")
                
                # Flag extreme profits for investigation
                if profit_pct > 1000:
                    print(f"      *** EXTREME PROFIT: {profit_pct:.0f}% ***")
        
        return {
            "crash_period": crash_period,
            "market_data": {
                "peak_date": peak_date,
                "trough_date": trough_date,
                "peak_price": peak_price,
                "trough_price": trough_price,
                "crash_magnitude": crash_magnitude,
                "peak_vol": peak_vol,
                "trough_vol": trough_vol,
                "days_elapsed": days_elapsed
            },
            "leap_results": leap_results
        }
    
    def analyze_all_crashes(self) -> list:
        """Analyze LEAP performance across all documented crash periods."""
        logger.info("Starting comprehensive historical crash analysis")
        
        all_results = []
        
        for crash_period in self.crash_periods:
            try:
                result = self.calculate_leap_performance(crash_period)
                if result:
                    all_results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing {crash_period['name']}: {e}")
        
        return all_results
    
    def generate_profit_distribution(self, all_results: list) -> None:
        """Generate statistical distribution of LEAP profits from historical data."""
        print("\n" + "=" * 60)
        print("HISTORICAL LEAP PROFIT DISTRIBUTION")
        print("=" * 60)
        
        # Collect all profit percentages
        all_profits = []
        breach_profits = []  # Profits when strike was actually breached
        no_breach_profits = []  # Profits when strike was not breached
        
        for result in all_results:
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
        breach_profits = np.array(breach_profits) if breach_profits else np.array([])
        no_breach_profits = np.array(no_breach_profits) if no_breach_profits else np.array([])
        
        print(f"\nOverall Statistics ({len(all_profits)} data points):")
        print(f"  Mean Profit: {np.mean(all_profits):.0f}%")
        print(f"  Median Profit: {np.median(all_profits):.0f}%")
        print(f"  Max Profit: {np.max(all_profits):.0f}%")
        print(f"  Min Profit: {np.min(all_profits):.0f}%")
        print(f"  90th Percentile: {np.percentile(all_profits, 90):.0f}%")
        print(f"  95th Percentile: {np.percentile(all_profits, 95):.0f}%")
        print(f"  99th Percentile: {np.percentile(all_profits, 99):.0f}%")
        
        if len(breach_profits) > 0:
            print(f"\nWhen Strike Breached ({len(breach_profits)} cases):")
            print(f"  Mean Profit: {np.mean(breach_profits):.0f}%")
            print(f"  Median Profit: {np.median(breach_profits):.0f}%")
            print(f"  Max Profit: {np.max(breach_profits):.0f}%")
            
        if len(no_breach_profits) > 0:
            print(f"\nWhen Strike NOT Breached ({len(no_breach_profits)} cases):")
            print(f"  Mean Profit: {np.mean(no_breach_profits):.0f}%")
            print(f"  Median Profit: {np.median(no_breach_profits):.0f}%")
            print(f"  Max Profit: {np.max(no_breach_profits):.0f}%")
        
        # Determine reasonable profit caps based on data
        p99 = np.percentile(all_profits, 99)
        p95 = np.percentile(all_profits, 95)
        max_profit = np.max(all_profits)
        
        print(f"\nProfit Cap Recommendations:")
        print(f"  Conservative (95th percentile): {p95:.0f}%")
        print(f"  Moderate (99th percentile): {p99:.0f}%")
        print(f"  Liberal (Historical maximum): {max_profit:.0f}%")
        print(f"  Current simulation cap: 1000%")
        
        if max_profit > 1000:
            print(f"  *** CURRENT CAP TOO LOW: Historical max is {max_profit:.0f}% ***")
        else:
            print(f"  Current cap appears reasonable")


def main():
    """Run the credible historical LEAP profit analysis."""
    print("CREDIBLE HISTORICAL LEAP PROFIT ANALYSIS")
    print("Using actual market data and standard option pricing models")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = HistoricalLEAPAnalyzer()
    
    # Analyze all documented crash periods
    results = analyzer.analyze_all_crashes()
    
    if not results:
        logger.error("No results generated - check data availability")
        return
        
    # Generate statistical distribution
    analyzer.generate_profit_distribution(results)
    
    print("\n" + "=" * 60)
    print("CREDIBILITY ASSESSMENT")
    print("=" * 60)
    print("""
Data Sources:
- Yahoo Finance (publicly available historical data)
- Standard Black-Scholes option pricing model
- Well-documented market crash dates from financial history

Methodology:
- Uses actual historical price movements
- Applies standard volatility calculations
- No speculative or anecdotal profit claims
- Transparent calculation methods

Limitations:
- Black-Scholes assumes constant volatility (reality varies)
- Transaction costs not included
- Liquidity constraints not modeled
- Assumes perfect execution at theoretical prices

Conclusion: Any profit caps should be based on this empirical analysis
rather than arbitrary limits.
""")


if __name__ == "__main__":
    main()