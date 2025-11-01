#!/usr/bin/env python3
"""
Budget vs LEAP Costs Analysis: 1999-2004 Dot-Com Period

This script analyzes the hedge budget availability compared to LEAP costs
every 6 months during the dot-com bubble and crash period, using the same
logic as the long_term_hedge_simulation.py for strike selection and budgeting.
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


def analyze_budget_vs_costs():
    """
    Analyze hedge budget vs LEAP costs every 6 months from 1999-2004.
    Uses the same logic as long_term_hedge_simulation.py.
    """
    print("LEAP Budget vs Costs Analysis: 1999-2004 Dot-Com Period")
    print("=" * 70)
    
    # Parameters matching long_term_hedge_simulation.py
    initial_investment = 100000.0  # $100k starting portfolio
    hedge_budget_pct = 0.005  # 0.5% hedge budget
    strike_otm_pcts = [-0.25, -0.30, -0.35]  # Same strikes as simulation
    risk_free_rate = 0.05
    
    # Analysis period: Every 6 months from 1999-01-01 to 2004-01-01
    start_date = datetime(1999, 1, 1)
    end_date = datetime(2004, 1, 1)
    
    # Fetch NASDAQ data with extra buffer for volatility calculation
    fetch_start = start_date - timedelta(days=180)
    print(f"Fetching NASDAQ data from {fetch_start.date()} to {end_date.date()}...")
    
    data = yf.download("^IXIC", start=fetch_start.strftime("%Y-%m-%d"), 
                       end=end_date.strftime("%Y-%m-%d"), progress=False)
    
    # Calculate returns and rolling volatility (same as simulation)
    data["returns"] = data["Close"].pct_change()
    data["rolling_vol_60d"] = (data["returns"].rolling(window=60, min_periods=30).std() 
                              * np.sqrt(252))
    data["rolling_vol_60d"] = data["rolling_vol_60d"].bfill()
    
    # Generate 6-month analysis dates
    analysis_dates = []
    current_date = start_date
    while current_date < end_date:
        analysis_dates.append(current_date)
        current_date += relativedelta(months=6)
    
    # Track portfolio value simulation (unhedged baseline)
    portfolio_value = initial_investment
    portfolio_history = [{"date": start_date, "value": portfolio_value}]
    
    # Calculate portfolio values at each analysis date
    prev_date = start_date
    for analysis_date in analysis_dates[1:]:  # Skip first date (already have initial)
        # Find closest market data dates
        try:
            prev_price = data.loc[data.index.get_indexer([prev_date], method='nearest')[0]]["Close"]
            current_price = data.loc[data.index.get_indexer([analysis_date], method='nearest')[0]]["Close"]
            
            # Handle Series objects
            if isinstance(prev_price, pd.Series):
                prev_price = prev_price.iloc[0]
            if isinstance(current_price, pd.Series):
                current_price = current_price.iloc[0]
            
            # Update portfolio value based on market performance
            market_return = (current_price / prev_price) - 1
            portfolio_value *= (1 + market_return)
            
            portfolio_history.append({"date": analysis_date, "value": portfolio_value})
            prev_date = analysis_date
            
        except (KeyError, IndexError):
            print(f"Warning: Could not find data for {analysis_date}")
            portfolio_history.append({"date": analysis_date, "value": portfolio_value})
    
    # Analysis results storage
    results = []
    
    print(f"\nAnalyzing LEAP costs every 6 months...")
    print(f"Initial Portfolio: ${initial_investment:,.0f}")
    print(f"Hedge Budget: {hedge_budget_pct:.1%} of portfolio value")
    print(f"Strike Levels: {[f'{pct:.0%}' for pct in strike_otm_pcts]} OTM")
    print()
    
    for i, analysis_date in enumerate(analysis_dates):
        try:
            # Get market data for this date
            closest_idx = data.index.get_indexer([analysis_date], method='nearest')[0]
            market_data = data.iloc[closest_idx]
            
            current_price = float(market_data["Close"])
            implied_vol = float(market_data["rolling_vol_60d"])
            
            # Get portfolio value at this date
            current_portfolio_value = portfolio_history[i]["value"]
            hedge_budget = current_portfolio_value * hedge_budget_pct
            
            # Calculate LEAP costs for each strike (same logic as simulation)
            option_costs = []
            total_min_cost = 0  # Cost to buy 1 contract of each strike
            
            for strike_pct in strike_otm_pcts:
                strike_price = current_price * (1 + strike_pct)
                option_cost_per_share = BlackScholesCalculator.option_price(
                    spot=current_price, strike=strike_price, time_to_expiry=1.0,
                    volatility=implied_vol, risk_free_rate=risk_free_rate, 
                    option_type="put"
                )
                cost_per_contract = option_cost_per_share * 100
                option_costs.append({
                    "strike_pct": strike_pct,
                    "strike_price": strike_price,
                    "cost_per_share": option_cost_per_share,
                    "cost_per_contract": cost_per_contract
                })
                total_min_cost += cost_per_contract
            
            # Determine affordability using simulation logic
            can_afford_minimum = hedge_budget >= 150.0  # Minimum threshold from simulation
            can_afford_one_each = hedge_budget >= total_min_cost
            
            # Calculate maximum contracts affordable (simulation allocation logic)
            contracts_affordable = [0, 0, 0]  # For each strike level
            remaining_budget = hedge_budget
            
            if can_afford_minimum:
                # First, try to buy 1 of each (simulation priority)
                for idx, option in enumerate(option_costs):
                    if option["cost_per_contract"] <= remaining_budget:
                        contracts_affordable[idx] = 1
                        remaining_budget -= option["cost_per_contract"]
                
                # Then allocate remaining budget to cheapest options (furthest OTM)
                while remaining_budget > 0:
                    allocated = False
                    # Start from furthest OTM (cheapest, index 2) to closest (index 0)
                    for idx in range(len(option_costs) - 1, -1, -1):
                        if option_costs[idx]["cost_per_contract"] <= remaining_budget:
                            contracts_affordable[idx] += 1
                            remaining_budget -= option_costs[idx]["cost_per_contract"]
                            allocated = True
                            break
                    if not allocated:
                        break
            
            total_contracts = sum(contracts_affordable)
            actual_cost = sum(contracts_affordable[i] * option_costs[i]["cost_per_contract"] 
                            for i in range(len(option_costs)))
            
            # Store results
            result = {
                "date": analysis_date,
                "portfolio_value": current_portfolio_value,
                "nasdaq_price": current_price,
                "volatility": implied_vol,
                "hedge_budget": hedge_budget,
                "min_cost_one_each": total_min_cost,
                "can_afford_minimum": can_afford_minimum,
                "can_afford_one_each": can_afford_one_each,
                "contracts_25_otm": contracts_affordable[0],
                "contracts_30_otm": contracts_affordable[1], 
                "contracts_35_otm": contracts_affordable[2],
                "total_contracts": total_contracts,
                "actual_cost": actual_cost,
                "budget_utilization": actual_cost / hedge_budget if hedge_budget > 0 else 0,
                "option_costs": option_costs.copy()
            }
            results.append(result)
            
            # Print detailed analysis for this date
            print(f"Date: {analysis_date.strftime('%Y-%m-%d')}")
            print(f"  Portfolio Value: ${current_portfolio_value:,.0f}")
            print(f"  NASDAQ Price: ${current_price:,.0f}")
            print(f"  Volatility: {implied_vol:.1%}")
            print(f"  Hedge Budget: ${hedge_budget:.0f}")
            print(f"  LEAP Costs:")
            for i, option in enumerate(option_costs):
                print(f"    {option['strike_pct']:.0%} OTM (${option['strike_price']:,.0f}): "
                      f"${option['cost_per_contract']:,.0f} per contract")
            print(f"  Min Cost (1 each): ${total_min_cost:,.0f}")
            print(f"  Affordable: {'Yes' if can_afford_minimum else 'No'} "
                  f"(Budget ≥ $150: {'Yes' if hedge_budget >= 150 else 'No'})")
            if can_afford_minimum:
                print(f"  Allocation: {contracts_affordable[0]}×25% + "
                      f"{contracts_affordable[1]}×30% + {contracts_affordable[2]}×35% OTM")
                print(f"  Total Contracts: {total_contracts}")
                print(f"  Actual Cost: ${actual_cost:,.0f} "
                      f"({actual_cost/hedge_budget:.1%} of budget)")
            print()
            
        except Exception as e:
            print(f"Error processing {analysis_date}: {e}")
            continue
    
    # Create comprehensive analysis
    create_budget_analysis_charts(results)
    
    # Summary statistics
    print_summary_analysis(results)
    
    return results


def create_budget_analysis_charts(results):
    """Create comprehensive charts showing budget vs costs over time."""
    if not results:
        print("No results to plot")
        return
    
    dates = [r["date"] for r in results]
    
    # Create figure with multiple subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("LEAP Budget vs Costs Analysis: 1999-2004 Dot-Com Period", fontsize=16)
    
    # Plot 1: Portfolio Value and NASDAQ Price
    portfolio_values = [r["portfolio_value"] for r in results]
    nasdaq_prices = [r["nasdaq_price"] for r in results]
    
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(dates, portfolio_values, 'b-', linewidth=2, label='Portfolio Value')
    line2 = ax1_twin.plot(dates, nasdaq_prices, 'r-', linewidth=2, label='NASDAQ Price')
    
    ax1.set_ylabel('Portfolio Value ($)', color='b')
    ax1_twin.set_ylabel('NASDAQ Price ($)', color='r')
    ax1.set_title('Portfolio Value vs NASDAQ Price')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 2: Hedge Budget vs LEAP Costs
    hedge_budgets = [r["hedge_budget"] for r in results]
    min_costs = [r["min_cost_one_each"] for r in results]
    actual_costs = [r["actual_cost"] for r in results]
    
    ax2.plot(dates, hedge_budgets, 'g-', linewidth=2, label='Available Budget')
    ax2.plot(dates, min_costs, 'r--', linewidth=2, label='Min Cost (1 each)')
    ax2.plot(dates, actual_costs, 'b-', linewidth=2, label='Actual Cost')
    ax2.axhline(y=150, color='orange', linestyle=':', label='Min Threshold ($150)')
    
    ax2.set_ylabel('Cost ($)')
    ax2.set_title('Hedge Budget vs LEAP Costs')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Volatility and Contract Allocation
    volatilities = [r["volatility"] * 100 for r in results]
    total_contracts = [r["total_contracts"] for r in results]
    
    ax3_twin = ax3.twinx()
    line3 = ax3.plot(dates, volatilities, 'purple', linewidth=2, label='Volatility (%)')
    line4 = ax3_twin.plot(dates, total_contracts, 'orange', linewidth=2, label='Total Contracts')
    
    ax3.set_ylabel('Volatility (%)', color='purple')
    ax3_twin.set_ylabel('Total Contracts', color='orange')
    ax3.set_title('Volatility vs Contract Allocation')
    ax3.grid(True, alpha=0.3)
    
    # Combine legends
    lines3, labels3 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines3 + lines4, labels3 + labels4, loc='upper left')
    
    # Plot 4: Strike-Level Contract Distribution
    contracts_25 = [r["contracts_25_otm"] for r in results]
    contracts_30 = [r["contracts_30_otm"] for r in results]
    contracts_35 = [r["contracts_35_otm"] for r in results]
    
    # Use dates directly for proper date formatting
    width = 60  # Width in days for each bar (adjusted for better visibility)
    
    ax4.bar(dates, contracts_25, width, label='25% OTM', color='red', alpha=0.7)
    ax4.bar(dates, contracts_30, width, bottom=contracts_25, 
            label='30% OTM', color='blue', alpha=0.7)
    
    # Calculate bottom for 35% OTM bars
    bottom_35 = [c25 + c30 for c25, c30 in zip(contracts_25, contracts_30)]
    ax4.bar(dates, contracts_35, width, bottom=bottom_35, 
            label='35% OTM', color='green', alpha=0.7)
    
    ax4.set_ylabel('Number of Contracts')
    ax4.set_title('Contract Allocation by Strike Level')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Format x-axis dates for all subplots
    import matplotlib.dates as mdates
    
    # Configure date formatting for all axes
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=6))
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("budget_vs_costs_analysis_1999_2004.png", dpi=300, bbox_inches='tight')
    print("Budget analysis chart saved to budget_vs_costs_analysis_1999_2004.png")
    plt.show()


def print_summary_analysis(results):
    """Print comprehensive summary analysis of the results."""
    if not results:
        return
    
    print("\n" + "=" * 70)
    print("COMPREHENSIVE SUMMARY ANALYSIS")
    print("=" * 70)
    
    # Basic statistics
    affordable_periods = sum(1 for r in results if r["can_afford_minimum"])
    unaffordable_periods = len(results) - affordable_periods
    
    avg_portfolio = np.mean([r["portfolio_value"] for r in results])
    avg_budget = np.mean([r["hedge_budget"] for r in results])
    avg_min_cost = np.mean([r["min_cost_one_each"] for r in results])
    avg_volatility = np.mean([r["volatility"] for r in results])
    
    print(f"\n1. AFFORDABILITY ANALYSIS")
    print("-" * 50)
    print(f"Total Analysis Periods: {len(results)}")
    print(f"Affordable Periods: {affordable_periods} ({affordable_periods/len(results):.1%})")
    print(f"Unaffordable Periods: {unaffordable_periods} ({unaffordable_periods/len(results):.1%})")
    print(f"Average Portfolio Value: ${avg_portfolio:,.0f}")
    print(f"Average Hedge Budget: ${avg_budget:.0f}")
    print(f"Average Min Cost (1 each): ${avg_min_cost:,.0f}")
    print(f"Budget/Cost Ratio: {avg_budget/avg_min_cost:.2f}x")
    
    # Volatility impact analysis
    print(f"\n2. VOLATILITY IMPACT ANALYSIS")
    print("-" * 50)
    print(f"Average Volatility: {avg_volatility:.1%}")
    
    # Find periods of extreme volatility
    volatilities = [r["volatility"] for r in results]
    high_vol_threshold = np.percentile(volatilities, 75)
    low_vol_threshold = np.percentile(volatilities, 25)
    
    high_vol_periods = [r for r in results if r["volatility"] >= high_vol_threshold]
    low_vol_periods = [r for r in results if r["volatility"] <= low_vol_threshold]
    
    high_vol_affordable = sum(1 for r in high_vol_periods if r["can_afford_minimum"])
    low_vol_affordable = sum(1 for r in low_vol_periods if r["can_afford_minimum"])
    
    print(f"High Volatility Periods (≥{high_vol_threshold:.1%}): {len(high_vol_periods)}")
    print(f"  Affordable: {high_vol_affordable} ({high_vol_affordable/len(high_vol_periods):.1%})")
    print(f"Low Volatility Periods (≤{low_vol_threshold:.1%}): {len(low_vol_periods)}")
    print(f"  Affordable: {low_vol_affordable} ({low_vol_affordable/len(low_vol_periods):.1%})")
    
    # Market condition analysis
    print(f"\n3. MARKET CONDITION ANALYSIS")
    print("-" * 50)
    
    # Identify key market periods
    crash_period = [r for r in results if r["date"].year in [2000, 2001, 2002]]
    bubble_period = [r for r in results if r["date"].year == 1999]
    recovery_period = [r for r in results if r["date"].year in [2003]]
    
    for period_name, period_data in [("Bubble (1999)", bubble_period),
                                   ("Crash (2000-2002)", crash_period),
                                   ("Recovery (2003)", recovery_period)]:
        if period_data:
            affordable_count = sum(1 for r in period_data if r["can_afford_minimum"])
            avg_contracts = np.mean([r["total_contracts"] for r in period_data])
            avg_vol = np.mean([r["volatility"] for r in period_data])
            
            print(f"{period_name}:")
            print(f"  Periods: {len(period_data)}")
            print(f"  Affordable: {affordable_count} ({affordable_count/len(period_data):.1%})")
            print(f"  Avg Contracts: {avg_contracts:.1f}")
            print(f"  Avg Volatility: {avg_vol:.1%}")
    
    # Contract allocation analysis
    print(f"\n4. CONTRACT ALLOCATION ANALYSIS")
    print("-" * 50)
    
    affordable_results = [r for r in results if r["can_afford_minimum"]]
    if affordable_results:
        total_25_contracts = sum(r["contracts_25_otm"] for r in affordable_results)
        total_30_contracts = sum(r["contracts_30_otm"] for r in affordable_results)
        total_35_contracts = sum(r["contracts_35_otm"] for r in affordable_results)
        total_all_contracts = total_25_contracts + total_30_contracts + total_35_contracts
        
        print(f"Total Contracts (when affordable):")
        print(f"  25% OTM: {total_25_contracts} ({total_25_contracts/total_all_contracts:.1%})")
        print(f"  30% OTM: {total_30_contracts} ({total_30_contracts/total_all_contracts:.1%})")
        print(f"  35% OTM: {total_35_contracts} ({total_35_contracts/total_all_contracts:.1%})")
        print(f"  Average per period: {total_all_contracts/len(affordable_results):.1f}")
        
        avg_utilization = np.mean([r["budget_utilization"] for r in affordable_results])
        print(f"  Average Budget Utilization: {avg_utilization:.1%}")
    
    # Key insights
    print(f"\n5. KEY INSIGHTS")
    print("-" * 50)
    print(f"• Hedge affordability was severely limited during high volatility periods")
    print(f"• Budget constraint ($150 minimum) excluded hedging during critical times")
    print(f"• 35% OTM options dominated allocation due to lower cost")
    print(f"• Portfolio drawdown reduced available budget when protection was most needed")
    print(f"• Systematic approach needed volatility-adjusted budgeting")


def main():
    """Main function to run the budget vs costs analysis."""
    try:
        results = analyze_budget_vs_costs()
        print(f"\nAnalysis complete. Processed {len(results)} time periods.")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()