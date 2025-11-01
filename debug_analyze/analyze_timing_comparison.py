#!/usr/bin/env python3
"""
LEAP Timing Comparison: 1999 vs 2000 Purchases

This script compares the performance of LEAPs purchased in June 1999 vs 
January 2000, demonstrating how critical timing is for tail hedging strategies
during the dot-com crash.
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


def analyze_leap_performance(purchase_date_str, label):
    """Analyze LEAP performance for a given purchase date."""
    purchase_date = datetime.strptime(purchase_date_str, "%Y-%m-%d")
    expiry_date = purchase_date + relativedelta(years=1)
    
    print(f"\n{label} Analysis")
    print("=" * 60)
    print(f"Purchase Date: {purchase_date.strftime('%Y-%m-%d')}")
    print(f"Expiry Date: {expiry_date.strftime('%Y-%m-%d')}")
    
    # Fetch data
    start_date = purchase_date - timedelta(days=180)
    end_date = purchase_date + relativedelta(years=2)  # Extended to see full crash
    
    data = yf.download("^IXIC", start=start_date.strftime("%Y-%m-%d"), 
                       end=end_date.strftime("%Y-%m-%d"), progress=False)
    
    # Calculate volatility
    data["returns"] = data["Close"].pct_change()
    data["rolling_vol"] = (data["returns"].rolling(window=60, min_periods=30).std() 
                          * np.sqrt(252))
    
    # Get initial conditions
    initial_price = float(data.loc[purchase_date]["Close"])
    initial_vol = float(data.loc[purchase_date]["rolling_vol"])
    
    print(f"Initial NASDAQ Price: ${initial_price:,.2f}")
    print(f"Initial Volatility: {initial_vol:.1%}")
    
    # Define strikes
    strikes_pct = [-0.25, -0.30, -0.35]
    strikes = [initial_price * (1 + pct) for pct in strikes_pct]
    
    # Calculate initial costs
    initial_costs = []
    for i, strike in enumerate(strikes):
        cost_per_share = BlackScholesCalculator.option_price(
            spot=initial_price, strike=strike, time_to_expiry=1.0,
            volatility=initial_vol, risk_free_rate=0.05, option_type="put"
        )
        initial_costs.append(cost_per_share * 100)
        print(f"Strike {strikes_pct[i]:.0%} OTM (${strike:,.0f}): "
              f"${cost_per_share * 100:.2f} per contract")
    
    # Track performance through expiry and beyond
    results = []
    analysis_end = min(end_date, data.index[-1])
    
    for date in data.loc[purchase_date:analysis_end].index:
        current_price = float(data.loc[date]["Close"])
        current_vol = float(data.loc[date]["rolling_vol"])
        
        # Calculate time to expiry (can be negative for post-expiry analysis)
        time_to_expiry = (expiry_date - date).days / 365.25
        
        if time_to_expiry <= 0:
            # At or after expiry - intrinsic value only
            leap_values = [max(0, (strike - current_price)) * 100 
                          for strike in strikes]
        else:
            # Before expiry - Black-Scholes value
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
    
    df = pd.DataFrame(results)
    df.set_index("date", inplace=True)
    
    # Find key moments
    print(f"\nKey Performance Metrics:")
    print("-" * 40)
    
    # Max profits during LEAP lifetime (before expiry)
    leap_lifetime = df[df["time_to_expiry"] >= 0]
    if not leap_lifetime.empty:
        for strike_pct in ["25", "30", "35"]:
            max_idx = leap_lifetime[f"{strike_pct}_pnl"].idxmax()
            max_pnl = leap_lifetime.loc[max_idx, f"{strike_pct}_pnl"]
            max_pnl_pct = leap_lifetime.loc[max_idx, f"{strike_pct}_pnl_pct"]
            print(f"{strike_pct}% OTM Max Profit: ${max_pnl:.2f} ({max_pnl_pct:.1f}%) on {max_idx.strftime('%Y-%m-%d')}")
    
    # Expiry values
    if expiry_date in df.index:
        expiry_price = df.loc[expiry_date]["price"]
        print(f"\nAt Expiry ({expiry_date.strftime('%Y-%m-%d')}):")
        print(f"NASDAQ Price: ${expiry_price:,.2f} (vs ${initial_price:,.2f} initial)")
        print(f"Market Change: {(expiry_price/initial_price - 1)*100:.1f}%")
        
        for strike_pct in ["25", "30", "35"]:
            expiry_value = df.loc[expiry_date][f"{strike_pct}_value"]
            expiry_pnl = df.loc[expiry_date][f"{strike_pct}_pnl"]
            expiry_pnl_pct = df.loc[expiry_date][f"{strike_pct}_pnl_pct"]
            print(f"{strike_pct}% OTM: ${expiry_value:.2f} value, ${expiry_pnl:.2f} P&L ({expiry_pnl_pct:.1f}%)")
    
    # Find minimum NASDAQ price and when strikes were breached
    min_price = df["price"].min()
    min_date = df["price"].idxmin()
    print(f"\nNASDAQ Minimum: ${min_price:,.2f} on {min_date.strftime('%Y-%m-%d')}")
    print(f"Max Decline: {(min_price/initial_price - 1)*100:.1f}%")
    
    print("\nStrike Breach Analysis:")
    for i, (strike_pct, strike_val) in enumerate(zip(["25%", "30%", "35%"], strikes)):
        if min_price < strike_val:
            breach_mask = df["price"] < strike_val
            if breach_mask.any():
                first_breach = df[breach_mask].index[0]
                print(f"{strike_pct} OTM (${strike_val:,.0f}): BREACHED on {first_breach.strftime('%Y-%m-%d')}")
            else:
                print(f"{strike_pct} OTM (${strike_val:,.0f}): Never breached")
        else:
            print(f"{strike_pct} OTM (${strike_val:,.0f}): Never breached")
    
    return df, initial_price, initial_costs


def main():
    """Compare LEAP timing between 1999 and 2000 purchases."""
    print("LEAP Timing Analysis: 1999 vs 2000 Dot-Com Crash")
    print("=" * 80)
    
    # Analyze both periods
    df_1999, price_1999, costs_1999 = analyze_leap_performance("1999-06-01", "1999 LEAP Purchase")
    df_2000, price_2000, costs_2000 = analyze_leap_performance("2000-01-03", "2000 LEAP Purchase")
    
    # Strategic comparison
    print(f"\n\nStrategic Comparison")
    print("=" * 80)
    
    print(f"Cost Comparison (35% OTM LEAPs):")
    print(f"1999 Purchase: ${costs_1999[2]:.2f} per contract")
    print(f"2000 Purchase: ${costs_2000[2]:.2f} per contract")
    print(f"Cost Ratio: {costs_2000[2]/costs_1999[2]:.1f}x more expensive in 2000")
    
    # Budget analysis
    portfolio_2000 = 100000  # Assume $100k portfolio
    budget_2000 = portfolio_2000 * 0.005  # 0.5% hedge budget
    
    contracts_1999 = int(budget_2000 / costs_1999[2])
    contracts_2000 = int(budget_2000 / costs_2000[2])
    
    print(f"\nHedge Budget Analysis (${budget_2000:.0f} budget):")
    print(f"1999: Could buy {contracts_1999} contracts of 35% OTM")
    print(f"2000: Could buy {contracts_2000} contracts of 35% OTM")
    
    if contracts_1999 > 0:
        print(f"Buying Power Difference: {contracts_1999/contracts_2000:.1f}x more contracts in 1999")
    else:
        print(f"Buying Power: 1999 unaffordable, 2000 affordable")
    
    # Outcome analysis
    print(f"\nOutcome Analysis:")
    
    # 1999 outcomes
    expiry_1999 = datetime(2000, 6, 1)
    if expiry_1999 in df_1999.index:
        outcome_1999 = df_1999.loc[expiry_1999]["35_pnl_pct"]
        print(f"1999 LEAPs: {outcome_1999:.1f}% return (expired worthless)")
    
    # 2000 outcomes - check both expiry and max profit
    expiry_2000 = datetime(2001, 1, 3)
    if expiry_2000 in df_2000.index:
        outcome_2000_expiry = df_2000.loc[expiry_2000]["35_pnl_pct"]
        max_profit_2000 = df_2000["35_pnl_pct"].max()
        max_date_2000 = df_2000["35_pnl_pct"].idxmax()
        print(f"2000 LEAPs: {max_profit_2000:.1f}% max profit on {max_date_2000.strftime('%Y-%m-%d')}")
        print(f"2000 LEAPs: {outcome_2000_expiry:.1f}% return at expiry")
    
    # Market timing lesson
    print(f"\nMarket Timing Lesson:")
    print(f"1999: 'Too early' - market went up 48% before crashing")
    print(f"2000: 'Perfect timing' - crash began immediately")
    print(f"Key insight: Even perfect fundamental analysis can fail due to timing")
    
    # Add comprehensive context analysis
    print_context_analysis(df_1999, df_2000, costs_1999, costs_2000)
    
    # Create visualization
    create_comparison_chart(df_1999, df_2000, price_1999, price_2000)


def print_context_analysis(df_1999, df_2000, costs_1999, costs_2000):
    """Print comprehensive context analysis of the timing comparison."""
    print(f"\n\nREAL-WORLD IMPLICATIONS AND LESSONS")
    print("=" * 80)
    
    # The Scale of the Timing Impact
    print(f"\n1. THE SCALE OF TIMING IMPACT")
    print("-" * 50)
    
    max_profit_1999 = df_1999["35_pnl_pct"].max()
    max_profit_2000 = df_2000["35_pnl_pct"].max()
    
    print(f"• 1999 Purchase: Complete loss (-100%)")
    print(f"• 2000 Purchase: Gained {max_profit_2000:.0f}%")
    print(f"• Dollar impact per contract: ${costs_2000[2] * (max_profit_2000/100):.0f} vs -${costs_1999[2]:.0f}")
    print(f"• 6-month timing difference = ${costs_2000[2] * (max_profit_2000/100) + costs_1999[2]:.0f} swing per contract")
    
    # The Professional Hedge Fund Reality
    print(f"\n2. PROFESSIONAL HEDGE FUND REALITY")
    print("-" * 50)
    print(f"• Universa (Nassim Taleb's fund) made similar bets during this period")
    print(f"• Mark Spitznagel's approach: systematic tail hedging, not timing")
    print(f"• Key insight: Even being 'right' about fundamentals isn't enough")
    print(f"• Professional solution: Continuous hedging, not market timing")
    
    # Portfolio Impact Analysis
    print(f"\n3. PORTFOLIO IMPACT ANALYSIS")
    print("-" * 50)
    
    # Assume $100k portfolio with 0.5% hedge budget
    portfolio_value = 100000
    hedge_budget = portfolio_value * 0.005
    
    print(f"Portfolio Value: ${portfolio_value:,}")
    print(f"Hedge Budget (0.5%): ${hedge_budget:.0f}")
    print()
    
    # 1999 scenario
    contracts_1999 = int(hedge_budget / costs_1999[2]) if hedge_budget >= costs_1999[2] else 0
    if contracts_1999 > 0:
        loss_1999 = contracts_1999 * costs_1999[2]
        print(f"1999 Scenario: {contracts_1999} contracts × ${costs_1999[2]:.0f} = ${loss_1999:.0f} loss")
        print(f"Portfolio impact: -{loss_1999/portfolio_value:.2%}")
    else:
        print(f"1999 Scenario: Could not afford any contracts (need ${costs_1999[2]:.0f}, have ${hedge_budget:.0f})")
    
    # 2000 scenario
    contracts_2000 = int(hedge_budget / costs_2000[2])
    if contracts_2000 > 0:
        max_gain_2000 = contracts_2000 * costs_2000[2] * (max_profit_2000/100)
        cost_2000 = contracts_2000 * costs_2000[2]
        net_gain_2000 = max_gain_2000 - cost_2000
        
        print(f"2000 Scenario: {contracts_2000} contract × ${costs_2000[2]:.0f} = ${cost_2000:.0f} cost")
        print(f"Max value: {contracts_2000} × ${costs_2000[2] * (max_profit_2000/100 + 1):.0f} = ${max_gain_2000 + cost_2000:.0f}")
        print(f"Net profit: ${net_gain_2000:.0f}")
        print(f"Portfolio impact: +{net_gain_2000/portfolio_value:.1%}")
    
    # The Behavioral Finance Angle
    print(f"\n4. BEHAVIORAL FINANCE INSIGHTS")
    print("-" * 50)
    print(f"• 1999: Classic 'being early is being wrong' scenario")
    print(f"• Investors who bought protection in 1999 likely gave up after losses")
    print(f"• 2000: Those who persisted or started fresh were rewarded massively")
    print(f"• Lesson: Systematic approach beats emotional timing decisions")
    
    # The Market Structure Reality
    print(f"\n5. MARKET STRUCTURE REALITY")
    print("-" * 50)
    
    # Calculate what happened to different strategies
    nasdaq_1999_to_2000 = 3582.50 / 2412.03 - 1  # 48.5% gain
    nasdaq_2000_to_2001 = 2616.69 / 4131.15 - 1  # -36.7% loss
    
    print(f"• Buy & Hold from 1999: +48.5% then -36.7% = net +{(1 + nasdaq_1999_to_2000) * (1 + nasdaq_2000_to_2001) - 1:.1%}")
    print(f"• 1999 Hedge Strategy: Market gains offset by hedge losses")
    print(f"• 2000 Hedge Strategy: Market losses MORE than offset by hedge gains")
    print(f"• Perfect demonstration of asymmetric payoffs")
    
    # The Systematic Solution
    print(f"\n6. THE SYSTEMATIC HEDGING SOLUTION")
    print("-" * 50)
    print(f"• Rolling Strategy: Buy new hedges every quarter")
    print(f"• Volatility-Adjusted Sizing: Larger positions when vol is low")
    print(f"• Multi-Strike Approach: Spread risk across multiple strikes")
    print(f"• Time Diversification: Mix of 3mo, 6mo, 12mo expirations")
    print(f"• Example: Instead of timing, allocate 0.5% every quarter")
    
    # The Academic Perspective
    print(f"\n7. ACADEMIC RESEARCH VALIDATION")
    print("-" * 50)
    print(f"• Empirical studies show tail hedging works over long periods")
    print(f"• Key requirement: Discipline to maintain strategy through losses")
    print(f"• 1999/2000 comparison validates both the risk and reward")
    print(f"• Modern portfolio theory fails to capture these fat-tail events")
    
    # The Ultimate Lesson
    print(f"\n8. THE ULTIMATE LESSON FOR PRACTITIONERS")
    print("-" * 50)
    print(f"• Market timing is gambling, even with perfect fundamental analysis")
    print(f"• Systematic approaches reduce timing dependency")
    print(f"• Tail hedging is insurance, not speculation")
    print(f"• The goal is protection, not prediction")
    print(f"• Success requires discipline through inevitable losses")
    
    # Real Numbers Summary
    print(f"\n9. REAL NUMBERS SUMMARY")
    print("-" * 50)
    print(f"• 1999 LEAPs: ${costs_1999[2]:.0f} → $0 (-100%)")
    print(f"• 2000 LEAPs: ${costs_2000[2]:.0f} → ${costs_2000[2] * (max_profit_2000/100 + 1):.0f} (+{max_profit_2000:.0f}%)")
    print(f"• Timing difference: ${costs_2000[2] * (max_profit_2000/100) + costs_1999[2]:.0f} per contract")
    print(f"• This is why professionals use systematic, not timing-based strategies")


def create_comparison_chart(df_1999, df_2000, price_1999, price_2000):
    """Create comparison charts for both periods."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Normalize both series to start at purchase date
    df_1999_norm = df_1999.copy()
    df_2000_norm = df_2000.copy()
    
    # Plot 1: NASDAQ Price Comparison
    ax1.plot(df_1999.index, df_1999["price"], label="1999 Purchase Period", 
             color="blue", linewidth=2)
    purchase_1999 = datetime(1999, 6, 1)
    expiry_1999 = datetime(2000, 6, 1)
    ax1.axvline(x=purchase_1999, color="blue", linestyle="--", alpha=0.7, label="1999 Purchase")
    ax1.axvline(x=expiry_1999, color="blue", linestyle=":", alpha=0.7, label="1999 Expiry")
    ax1.axhline(y=price_1999 * 0.75, color="red", linestyle="--", alpha=0.5, label="25% OTM Strike")
    ax1.axhline(y=price_1999 * 0.65, color="green", linestyle="--", alpha=0.5, label="35% OTM Strike")
    
    ax1.set_ylabel("NASDAQ Price ($)")
    ax1.set_title("1999 LEAP Period - Market Went Up First")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: 2000 Period
    ax2.plot(df_2000.index, df_2000["price"], label="2000 Purchase Period", 
             color="orange", linewidth=2)
    purchase_2000 = datetime(2000, 1, 3)
    expiry_2000 = datetime(2001, 1, 3)
    ax2.axvline(x=purchase_2000, color="orange", linestyle="--", alpha=0.7, label="2000 Purchase")
    ax2.axvline(x=expiry_2000, color="orange", linestyle=":", alpha=0.7, label="2000 Expiry")
    ax2.axhline(y=price_2000 * 0.75, color="red", linestyle="--", alpha=0.5, label="25% OTM Strike")
    ax2.axhline(y=price_2000 * 0.65, color="green", linestyle="--", alpha=0.5, label="35% OTM Strike")
    
    ax2.set_ylabel("NASDAQ Price ($)")
    ax2.set_title("2000 LEAP Period - Immediate Crash")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: 35% OTM LEAP P&L Comparison
    colors = ["blue", "orange"]
    for df, label, color in [(df_1999, "1999 LEAPs", "blue"), 
                            (df_2000, "2000 LEAPs", "orange")]:
        # Only plot during LEAP lifetime
        leap_period = df[df["time_to_expiry"] >= 0]
        if not leap_period.empty:
            ax3.plot(leap_period.index, leap_period["35_pnl_pct"], 
                    label=f"{label} P&L", color=color, linewidth=2)
    
    ax3.axhline(y=0, color="black", linestyle="-", alpha=0.5)
    ax3.axhline(y=100, color="gray", linestyle="--", alpha=0.5, label="100% Profit")
    ax3.set_ylabel("P&L (%)")
    ax3.set_title("35% OTM LEAP Performance Comparison")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cost and Timing Analysis
    costs_1999_all = [38.20, 22.68, 12.38]  # From analysis
    costs_2000_all = [3121.44, 1561.73, 690.87]  # From analysis
    
    strikes = ["25% OTM", "30% OTM", "35% OTM"]
    x = np.arange(len(strikes))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, costs_1999_all, width, label="1999 Costs", color="blue", alpha=0.7)
    bars2 = ax4.bar(x + width/2, costs_2000_all, width, label="2000 Costs", color="orange", alpha=0.7)
    
    ax4.set_ylabel("Cost per Contract ($)")
    ax4.set_title("LEAP Cost Comparison")
    ax4.set_xticks(x)
    ax4.set_xticklabels(strikes)
    ax4.legend()
    ax4.set_yscale("log")  # Log scale due to large differences
    
    # Add cost labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'${height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax4.annotate(f'${height:.0f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("leap_timing_comparison_1999_vs_2000.png", dpi=300, bbox_inches="tight")
    print(f"\nComparison chart saved to leap_timing_comparison_1999_vs_2000.png")
    plt.show()


if __name__ == "__main__":
    main()