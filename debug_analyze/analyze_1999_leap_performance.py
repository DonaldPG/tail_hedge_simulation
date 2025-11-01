#!/usr/bin/env python3
"""
1999 LEAP Performance Analysis Through Dot-Com Crash

This script analyzes the performance of LEAPs purchased in June 1999
through the dot-com crash of 2000-2002, showing when they would have
been profitable vs. when they expired worthless.
"""

import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.asymmetric_options import BlackScholesCalculator


def analyze_1999_leaps():
    """Analyze LEAPs purchased in June 1999 through the crash."""
    
    # LEAP purchase parameters
    leap_date = datetime(1999, 6, 1)
    expiry_date = leap_date + relativedelta(years=1)  # June 2000 expiry
    
    print("1999 LEAP Performance Through Dot-Com Crash")
    print("=" * 60)
    print(f"Purchase Date: {leap_date.strftime('%Y-%m-%d')}")
    print(f"Expiry Date: {expiry_date.strftime('%Y-%m-%d')}")
    print()
    
    # Fetch NASDAQ data
    start_date = leap_date - timedelta(days=180)  # 6 months prior for vol
    end_date = datetime(2002, 12, 31)  # Through the crash
    
    print("Fetching NASDAQ data...")
    data = yf.download("^IXIC", start=start_date.strftime("%Y-%m-%d"), 
                       end=end_date.strftime("%Y-%m-%d"), progress=False)
    
    # Calculate volatility
    data["returns"] = data["Close"].pct_change()
    data["rolling_vol"] = (data["returns"].rolling(window=60, min_periods=30).std() 
                          * np.sqrt(252))
    
    # Get initial conditions
    initial_price = float(data.loc[leap_date]["Close"])
    initial_vol = float(data.loc[leap_date]["rolling_vol"])
    
    print(f"Initial NASDAQ Price: ${initial_price:,.2f}")
    print(f"Initial Volatility: {initial_vol:.1%}")
    print()
    
    # Define strikes
    strikes_pct = [-0.25, -0.30, -0.35]
    strikes = [initial_price * (1 + pct) for pct in strikes_pct]
    
    # Calculate initial LEAP costs
    initial_costs = []
    for i, strike in enumerate(strikes):
        cost_per_share = BlackScholesCalculator.option_price(
            spot=initial_price, strike=strike, time_to_expiry=1.0,
            volatility=initial_vol, risk_free_rate=0.05, option_type="put"
        )
        initial_costs.append(cost_per_share * 100)
        print(f"Strike {strikes_pct[i]:.0%} OTM (${strike:,.0f}): "
              f"${cost_per_share * 100:.2f} per contract")
    
    print()
    
    # Track performance through time
    results = []
    analysis_end = min(expiry_date, data.index[-1])
    
    for date in data.loc[leap_date:analysis_end].index:
        current_price = float(data.loc[date]["Close"])
        current_vol = float(data.loc[date]["rolling_vol"])
        time_to_expiry = max(0, (expiry_date - date).days / 365.25)
        
        if time_to_expiry <= 0:
            # At expiry, value is intrinsic value only
            leap_values = [max(0, strike - current_price) * 100 
                          for strike in strikes]
        else:
            # Calculate Black-Scholes value
            leap_values = []
            for strike in strikes:
                value_per_share = BlackScholesCalculator.option_price(
                    spot=current_price, strike=strike, 
                    time_to_expiry=time_to_expiry,
                    volatility=current_vol, risk_free_rate=0.05, 
                    option_type="put"
                )
                leap_values.append(value_per_share * 100)
        
        # Calculate P&L
        pnl = [value - cost for value, cost in zip(leap_values, initial_costs)]
        pnl_pct = [p/c * 100 for p, c in zip(pnl, initial_costs)]
        
        results.append({
            "date": date,
            "price": current_price,
            "vol": current_vol,
            "time_to_expiry": time_to_expiry,
            "25_value": leap_values[0],
            "30_value": leap_values[1], 
            "35_value": leap_values[2],
            "25_pnl": pnl[0],
            "30_pnl": pnl[1],
            "35_pnl": pnl[2],
            "25_pnl_pct": pnl_pct[0],
            "30_pnl_pct": pnl_pct[1],
            "35_pnl_pct": pnl_pct[2]
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    df.set_index("date", inplace=True)
    
    # Find key moments
    max_pnl_dates = {}
    for strike_pct in ["25", "30", "35"]:
        max_idx = df[f"{strike_pct}_pnl"].idxmax()
        max_pnl_dates[strike_pct] = {
            "date": max_idx,
            "price": df.loc[max_idx, "price"],
            "pnl": df.loc[max_idx, f"{strike_pct}_pnl"],
            "pnl_pct": df.loc[max_idx, f"{strike_pct}_pnl_pct"],
            "value": df.loc[max_idx, f"{strike_pct}_value"]
        }
    
    # Print key results
    print("Key Performance Moments:")
    print("-" * 40)
    
    for strike_pct, strike_val in zip(["25", "30", "35"], strikes):
        data_point = max_pnl_dates[strike_pct]
        print(f"\n{strike_pct}% OTM Strike (${strike_val:,.0f}):")
        print(f"  Max P&L Date: {data_point['date'].strftime('%Y-%m-%d')}")
        print(f"  NASDAQ Price: ${data_point['price']:,.2f}")
        print(f"  LEAP Value: ${data_point['value']:.2f}")
        print(f"  P&L: ${data_point['pnl']:.2f} ({data_point['pnl_pct']:.1f}%)")
        
        # Check if strike was breached
        min_price = df["price"].min()
        if min_price < strike_val:
            breach_date = df[df["price"] < strike_val].index[0]
            print(f"  Strike BREACHED on {breach_date.strftime('%Y-%m-%d')} "
                  f"(Price: ${min_price:,.2f})")
        else:
            print(f"  Strike NEVER breached (Min price: ${min_price:,.2f})")
    
    # Final expiry values
    print(f"\nAt Expiry ({expiry_date.strftime('%Y-%m-%d')}):")
    print("-" * 40)
    expiry_price = df.iloc[-1]["price"]
    print(f"NASDAQ Price: ${expiry_price:,.2f}")
    
    for i, (strike_pct, strike_val) in enumerate(zip(["25", "30", "35"], strikes)):
        expiry_value = df.iloc[-1][f"{strike_pct}_value"]
        expiry_pnl = df.iloc[-1][f"{strike_pct}_pnl"]
        expiry_pnl_pct = df.iloc[-1][f"{strike_pct}_pnl_pct"]
        
        print(f"{strike_pct}% OTM: Value ${expiry_value:.2f}, "
              f"P&L ${expiry_pnl:.2f} ({expiry_pnl_pct:.1f}%)")
    
    # Create visualization
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # Plot 1: NASDAQ price vs strikes
    ax1.plot(df.index, df["price"], label="NASDAQ", color="blue", linewidth=2)
    colors = ["red", "orange", "green"]
    for i, (strike_pct, strike_val) in enumerate(zip(["25", "30", "35"], strikes)):
        ax1.axhline(y=strike_val, color=colors[i], linestyle="--", 
                   label=f"{strike_pct}% OTM Strike (${strike_val:,.0f})")
    
    ax1.set_ylabel("Price ($)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title("NASDAQ Price vs LEAP Strike Levels")
    
    # Plot 2: LEAP values
    for i, strike_pct in enumerate(["25", "30", "35"]):
        ax2.plot(df.index, df[f"{strike_pct}_value"], 
                label=f"{strike_pct}% OTM Value", color=colors[i])
    
    ax2.set_ylabel("LEAP Value ($)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_title("LEAP Values Over Time")
    
    # Plot 3: P&L percentage
    for i, strike_pct in enumerate(["25", "30", "35"]):
        ax3.plot(df.index, df[f"{strike_pct}_pnl_pct"], 
                label=f"{strike_pct}% OTM P&L", color=colors[i])
    
    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax3.set_ylabel("P&L (%)")
    ax3.set_xlabel("Date")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title("LEAP P&L Percentage")
    
    plt.tight_layout()
    plt.savefig("1999_leap_crash_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    return df


if __name__ == "__main__":
    results_df = analyze_1999_leaps()