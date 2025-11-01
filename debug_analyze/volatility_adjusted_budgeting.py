#!/usr/bin/env python3
"""
Volatility-Adjusted Budgeting for Tail Hedging

This script demonstrates how to implement volatility-adjusted budgeting
in practice, solving the problem where hedges become unaffordable 
exactly when they're most needed.
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


class VolatilityAdjustedBudgeting:
    """
    Implements volatility-adjusted budgeting strategies for tail hedging.
    
    Key principles:
    1. Increase allocation when volatility is low (cheap protection)
    2. Maintain minimum protection when volatility is high (expensive)
    3. Use multiple time horizons to smooth costs
    4. Systematic rebalancing vs market timing
    """
    
    def __init__(self, base_allocation_pct=0.005, vol_adjustment_factor=2.0,
                 min_allocation_pct=0.002, max_allocation_pct=0.015):
        """
        Initialize volatility-adjusted budgeting parameters.
        
        Args:
            base_allocation_pct: Base hedge allocation (0.5% default)
            vol_adjustment_factor: How aggressively to adjust for volatility
            min_allocation_pct: Minimum allocation in high-vol periods
            max_allocation_pct: Maximum allocation in low-vol periods
        """
        self.base_allocation = base_allocation_pct
        self.vol_adjustment_factor = vol_adjustment_factor
        self.min_allocation = min_allocation_pct
        self.max_allocation = max_allocation_pct
        
    def calculate_vol_adjusted_budget(self, current_vol, vol_regime, 
                                    portfolio_value):
        """
        Calculate volatility-adjusted hedge budget.
        
        Three main approaches demonstrated:
        1. Inverse volatility scaling
        2. Volatility regime-based allocation
        3. Rolling average smoothing
        """
        # Method 1: Inverse Volatility Scaling
        # Allocate more when vol is low, less when vol is high
        vol_normalized = current_vol / 0.20  # Normalize to 20% baseline
        inverse_vol_multiplier = 1.0 / (vol_normalized ** 0.5)
        
        inverse_vol_budget = (self.base_allocation * inverse_vol_multiplier * 
                            portfolio_value)
        inverse_vol_budget = np.clip(inverse_vol_budget,
                                   self.min_allocation * portfolio_value,
                                   self.max_allocation * portfolio_value)
        
        # Method 2: Volatility Regime-Based Allocation
        if vol_regime == "low":  # vol < 15%
            regime_multiplier = 2.0
        elif vol_regime == "normal":  # 15% <= vol < 30%
            regime_multiplier = 1.0
        elif vol_regime == "high":  # 30% <= vol < 50%
            regime_multiplier = 0.5
        else:  # crisis: vol >= 50%
            regime_multiplier = 0.3
            
        regime_budget = (self.base_allocation * regime_multiplier * 
                       portfolio_value)
        
        # Method 3: Target Protection Level
        # Maintain consistent "protection units" regardless of cost
        target_protection_units = 1000  # Target: protect against 1000 point drop
        current_price = 4000  # Example NASDAQ level
        
        # Calculate cost for target protection
        strike_35_otm = current_price * 0.65
        cost_per_unit = BlackScholesCalculator.option_price(
            spot=current_price, strike=strike_35_otm, time_to_expiry=1.0,
            volatility=current_vol, risk_free_rate=0.05, option_type="put"
        )
        target_cost = target_protection_units * cost_per_unit
        target_budget = min(target_cost, self.max_allocation * portfolio_value)
        
        return {
            "inverse_vol": inverse_vol_budget,
            "regime_based": regime_budget, 
            "target_protection": target_budget,
            "vol_multiplier": inverse_vol_multiplier,
            "regime": vol_regime
        }
    
    def determine_vol_regime(self, current_vol, rolling_vol_avg):
        """Determine volatility regime for allocation decisions."""
        if current_vol < 0.15:
            return "low"
        elif current_vol < 0.30:
            return "normal" 
        elif current_vol < 0.50:
            return "high"
        else:
            return "crisis"


def analyze_vol_adjusted_strategies():
    """
    Analyze different volatility-adjusted budgeting strategies
    using historical data from 1999-2004.
    """
    print("Volatility-Adjusted Budgeting Analysis: 1999-2004")
    print("=" * 60)
    
    # Fetch historical data
    start_date = datetime(1999, 1, 1)
    end_date = datetime(2004, 1, 1)
    fetch_start = start_date - timedelta(days=180)
    
    print("Fetching NASDAQ data...")
    data = yf.download("^IXIC", start=fetch_start.strftime("%Y-%m-%d"),
                       end=end_date.strftime("%Y-%m-%d"), progress=False)
    
    # Calculate volatility metrics
    data["returns"] = data["Close"].pct_change()
    data["rolling_vol_60d"] = (data["returns"].rolling(window=60, min_periods=30)
                              .std() * np.sqrt(252))
    data["rolling_vol_20d"] = (data["returns"].rolling(window=20, min_periods=10)
                              .std() * np.sqrt(252))
    data["vol_ma_120d"] = data["rolling_vol_60d"].rolling(120).mean()
    
    # Initialize budgeting strategies
    budgeting = VolatilityAdjustedBudgeting()
    
    # Analysis dates (quarterly rebalancing)
    analysis_dates = []
    current_date = start_date
    while current_date < end_date:
        analysis_dates.append(current_date)
        current_date += relativedelta(months=3)
    
    # Track portfolio value (unhedged)
    initial_portfolio = 100000
    portfolio_value = initial_portfolio
    
    results = []
    
    print(f"\nAnalyzing volatility-adjusted strategies...")
    print(f"Base allocation: {budgeting.base_allocation:.1%}")
    print(f"Range: {budgeting.min_allocation:.1%} - {budgeting.max_allocation:.1%}")
    print()
    
    prev_date = start_date
    for i, analysis_date in enumerate(analysis_dates):
        try:
            # Get market data
            closest_idx = data.index.get_indexer([analysis_date], method='nearest')[0]
            market_data = data.iloc[closest_idx]
            
            current_price = float(market_data["Close"])
            current_vol = float(market_data["rolling_vol_60d"])
            vol_ma = float(market_data["vol_ma_120d"]) if not pd.isna(market_data["vol_ma_120d"]) else current_vol
            
            # Update portfolio value based on market performance
            if i > 0:
                prev_idx = data.index.get_indexer([prev_date], method='nearest')[0]
                prev_price = float(data.iloc[prev_idx]["Close"])
                market_return = (current_price / prev_price) - 1
                portfolio_value *= (1 + market_return)
            
            # Determine volatility regime
            vol_regime = budgeting.determine_vol_regime(current_vol, vol_ma)
            
            # Calculate different budget approaches
            budgets = budgeting.calculate_vol_adjusted_budget(
                current_vol, vol_regime, portfolio_value)
            
            # Calculate LEAP costs for comparison
            strikes = [current_price * 0.75, current_price * 0.70, current_price * 0.65]
            leap_costs = []
            
            for strike in strikes:
                cost_per_share = BlackScholesCalculator.option_price(
                    spot=current_price, strike=strike, time_to_expiry=1.0,
                    volatility=current_vol, risk_free_rate=0.05, option_type="put"
                )
                leap_costs.append(cost_per_share * 100)
            
            # Analyze affordability under each strategy
            fixed_budget = portfolio_value * budgeting.base_allocation
            cheapest_leap = min(leap_costs)
            
            affordability = {}
            for strategy_name, budget_amount in budgets.items():
                if strategy_name in ["inverse_vol", "regime_based", "target_protection"]:
                    contracts = int(budget_amount / cheapest_leap) if cheapest_leap > 0 else 0
                    affordability[strategy_name] = {
                        "budget": budget_amount,
                        "contracts": contracts,
                        "utilization": min(contracts * cheapest_leap / budget_amount, 1.0) if budget_amount > 0 else 0
                    }
            
            # Fixed budget for comparison
            fixed_contracts = int(fixed_budget / cheapest_leap) if cheapest_leap > 0 else 0
            affordability["fixed"] = {
                "budget": fixed_budget,
                "contracts": fixed_contracts,
                "utilization": min(fixed_contracts * cheapest_leap / fixed_budget, 1.0) if fixed_budget > 0 else 0
            }
            
            result = {
                "date": analysis_date,
                "portfolio_value": portfolio_value,
                "nasdaq_price": current_price,
                "volatility": current_vol,
                "vol_regime": vol_regime,
                "leap_costs": leap_costs,
                "cheapest_leap": cheapest_leap,
                "budgets": budgets,
                "affordability": affordability
            }
            results.append(result)
            
            # Print detailed analysis
            print(f"Date: {analysis_date.strftime('%Y-%m-%d')}")
            print(f"  Portfolio: ${portfolio_value:,.0f}, NASDAQ: ${current_price:,.0f}")
            print(f"  Volatility: {current_vol:.1%} ({vol_regime} regime)")
            print(f"  LEAP costs: ${leap_costs[0]:.0f} / ${leap_costs[1]:.0f} / ${leap_costs[2]:.0f}")
            print(f"  Budget strategies:")
            print(f"    Fixed (0.5%): ${fixed_budget:.0f} → {fixed_contracts} contracts")
            for strategy in ["inverse_vol", "regime_based", "target_protection"]:
                budget_amt = affordability[strategy]["budget"]
                contracts = affordability[strategy]["contracts"]
                print(f"    {strategy.replace('_', ' ').title()}: ${budget_amt:.0f} → {contracts} contracts")
            print()
            
            prev_date = analysis_date
            
        except Exception as e:
            print(f"Error processing {analysis_date}: {e}")
            continue
    
    # Create comprehensive analysis
    create_vol_adjusted_analysis_charts(results)
    print_vol_adjusted_summary(results)
    
    return results


def create_vol_adjusted_analysis_charts(results):
    """Create comprehensive charts for volatility-adjusted budgeting analysis."""
    if not results:
        return
        
    dates = [r["date"] for r in results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Volatility-Adjusted Budgeting Strategies: 1999-2004", fontsize=16)
    
    # Plot 1: Volatility and Regimes
    volatilities = [r["volatility"] * 100 for r in results]
    nasdaq_prices = [r["nasdaq_price"] for r in results]
    
    ax1_twin = ax1.twinx()
    line1 = ax1.plot(dates, volatilities, 'purple', linewidth=2, label='Volatility (%)')
    line2 = ax1_twin.plot(dates, nasdaq_prices, 'blue', linewidth=2, label='NASDAQ Price')
    
    # Add regime shading
    regime_colors = {"low": "green", "normal": "yellow", "high": "orange", "crisis": "red"}
    prev_date = dates[0]
    for i, result in enumerate(results):
        if i < len(dates) - 1:
            next_date = dates[i + 1]
            color = regime_colors.get(result["vol_regime"], "gray")
            ax1.axvspan(result["date"], next_date, alpha=0.2, color=color)
    
    ax1.set_ylabel('Volatility (%)', color='purple')
    ax1_twin.set_ylabel('NASDAQ Price ($)', color='blue')
    ax1.set_title('Volatility Regimes Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 2: Budget Allocation Comparison
    fixed_budgets = [r["affordability"]["fixed"]["budget"] for r in results]
    inverse_vol_budgets = [r["affordability"]["inverse_vol"]["budget"] for r in results]
    regime_budgets = [r["affordability"]["regime_based"]["budget"] for r in results]
    
    ax2.plot(dates, fixed_budgets, 'gray', linewidth=2, label='Fixed (0.5%)', linestyle='--')
    ax2.plot(dates, inverse_vol_budgets, 'blue', linewidth=2, label='Inverse Volatility')
    ax2.plot(dates, regime_budgets, 'red', linewidth=2, label='Regime-Based')
    
    ax2.set_ylabel('Budget Allocation ($)')
    ax2.set_title('Budget Allocation Strategies')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Contract Affordability
    fixed_contracts = [r["affordability"]["fixed"]["contracts"] for r in results]
    inverse_vol_contracts = [r["affordability"]["inverse_vol"]["contracts"] for r in results]
    regime_contracts = [r["affordability"]["regime_based"]["contracts"] for r in results]
    
    ax3.plot(dates, fixed_contracts, 'gray', linewidth=2, label='Fixed', linestyle='--')
    ax3.plot(dates, inverse_vol_contracts, 'blue', linewidth=2, label='Inverse Vol')
    ax3.plot(dates, regime_contracts, 'red', linewidth=2, label='Regime-Based')
    
    ax3.set_ylabel('Contracts Affordable')
    ax3.set_title('Hedge Contract Affordability')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Cost vs Budget Efficiency
    leap_costs = [r["cheapest_leap"] for r in results]
    inverse_vol_efficiency = [r["affordability"]["inverse_vol"]["utilization"] * 100 for r in results]
    regime_efficiency = [r["affordability"]["regime_based"]["utilization"] * 100 for r in results]
    
    ax4_twin = ax4.twinx()
    line3 = ax4.plot(dates, leap_costs, 'orange', linewidth=2, label='LEAP Cost ($)')
    line4 = ax4_twin.plot(dates, inverse_vol_efficiency, 'blue', linewidth=2, label='Inverse Vol Efficiency (%)')
    line5 = ax4_twin.plot(dates, regime_efficiency, 'red', linewidth=2, label='Regime Efficiency (%)')
    
    ax4.set_ylabel('LEAP Cost ($)', color='orange')
    ax4_twin.set_ylabel('Budget Utilization (%)')
    ax4.set_title('Cost vs Budget Efficiency')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    # Combine legends
    lines3, labels3 = ax4.get_legend_handles_labels()
    lines4, labels4 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines3 + lines4 + line5, labels3 + labels4 + ['Regime Efficiency (%)'], loc='upper left')
    
    # Format dates
    import matplotlib.dates as mdates
    for ax in [ax1, ax2, ax3, ax4]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("volatility_adjusted_budgeting_analysis.png", dpi=300, bbox_inches='tight')
    print("Chart saved to volatility_adjusted_budgeting_analysis.png")
    plt.show()


def print_vol_adjusted_summary(results):
    """Print comprehensive summary of volatility-adjusted strategies."""
    if not results:
        return
        
    print("\n" + "=" * 70)
    print("VOLATILITY-ADJUSTED BUDGETING SUMMARY")
    print("=" * 70)
    
    # Strategy performance comparison
    strategies = ["fixed", "inverse_vol", "regime_based", "target_protection"]
    strategy_names = {
        "fixed": "Fixed Allocation (0.5%)",
        "inverse_vol": "Inverse Volatility",
        "regime_based": "Regime-Based", 
        "target_protection": "Target Protection"
    }
    
    print("\n1. STRATEGY PERFORMANCE COMPARISON")
    print("-" * 50)
    
    for strategy in strategies:
        if strategy == "target_protection":
            continue  # Skip for summary (complex calculation)
            
        total_budget = sum(r["affordability"][strategy]["budget"] for r in results)
        total_contracts = sum(r["affordability"][strategy]["contracts"] for r in results)
        avg_utilization = np.mean([r["affordability"][strategy]["utilization"] for r in results])
        
        # Count periods with zero contracts (unaffordable)
        zero_periods = sum(1 for r in results if r["affordability"][strategy]["contracts"] == 0)
        affordable_rate = (len(results) - zero_periods) / len(results)
        
        print(f"{strategy_names[strategy]}:")
        print(f"  Total Budget Allocated: ${total_budget:,.0f}")
        print(f"  Total Contracts: {total_contracts:,}")
        print(f"  Average Utilization: {avg_utilization:.1%}")
        print(f"  Affordability Rate: {affordable_rate:.1%}")
        print()
    
    # Volatility regime analysis
    print("\n2. VOLATILITY REGIME ANALYSIS")
    print("-" * 50)
    
    regime_stats = {}
    for result in results:
        regime = result["vol_regime"]
        if regime not in regime_stats:
            regime_stats[regime] = {
                "count": 0,
                "avg_vol": 0,
                "avg_cost": 0,
                "inverse_vol_contracts": 0,
                "fixed_contracts": 0
            }
        
        regime_stats[regime]["count"] += 1
        regime_stats[regime]["avg_vol"] += result["volatility"]
        regime_stats[regime]["avg_cost"] += result["cheapest_leap"]
        regime_stats[regime]["inverse_vol_contracts"] += result["affordability"]["inverse_vol"]["contracts"]
        regime_stats[regime]["fixed_contracts"] += result["affordability"]["fixed"]["contracts"]
    
    for regime, stats in regime_stats.items():
        count = stats["count"]
        avg_vol = stats["avg_vol"] / count
        avg_cost = stats["avg_cost"] / count
        total_inverse_contracts = stats["inverse_vol_contracts"]
        total_fixed_contracts = stats["fixed_contracts"]
        
        print(f"{regime.upper()} Volatility Regime ({count} periods):")
        print(f"  Average Volatility: {avg_vol:.1%}")
        print(f"  Average LEAP Cost: ${avg_cost:.0f}")
        print(f"  Inverse Vol Strategy: {total_inverse_contracts} total contracts")
        print(f"  Fixed Strategy: {total_fixed_contracts} total contracts")
        print(f"  Advantage: {total_inverse_contracts - total_fixed_contracts:+d} contracts")
        print()
    
    # Key insights and implementation guidance
    print("\n3. PRACTICAL IMPLEMENTATION GUIDANCE")
    print("-" * 50)
    print("• INVERSE VOLATILITY SCALING:")
    print("  - Increase allocation when vol < 20% (cheap protection)")
    print("  - Reduce allocation when vol > 40% (expensive protection)")
    print("  - Formula: allocation = base × (20% / current_vol)^0.5")
    print()
    print("• REGIME-BASED ALLOCATION:")
    print("  - Low vol (<15%): 2x base allocation")
    print("  - Normal vol (15-30%): 1x base allocation")
    print("  - High vol (30-50%): 0.5x base allocation") 
    print("  - Crisis vol (>50%): 0.3x base allocation")
    print()
    print("• TARGET PROTECTION APPROACH:")
    print("  - Define target protection level (e.g., 1000 index points)")
    print("  - Calculate required budget for target protection")
    print("  - Cap at maximum allocation limit")
    print()
    print("• IMPLEMENTATION BEST PRACTICES:")
    print("  - Use quarterly rebalancing (avoid over-trading)")
    print("  - Set allocation floors and ceilings")
    print("  - Consider multiple time horizons (3mo, 6mo, 12mo)")
    print("  - Monitor regime changes systematically")
    print("  - Document allocation rules in advance")
    
    # ROI analysis
    print("\n4. RETURN ON INVESTMENT IMPLICATIONS")
    print("-" * 50)
    
    # Calculate periods where vol-adjusted strategies could afford hedges vs fixed
    vol_periods = len([r for r in results if r["volatility"] > 0.30])
    fixed_coverage = len([r for r in results if r["affordability"]["fixed"]["contracts"] > 0])
    inverse_vol_coverage = len([r for r in results if r["affordability"]["inverse_vol"]["contracts"] > 0])
    
    print(f"High Volatility Periods (>30%): {vol_periods} out of {len(results)}")
    print(f"Fixed Strategy Coverage: {fixed_coverage}/{len(results)} periods ({fixed_coverage/len(results):.1%})")
    print(f"Inverse Vol Coverage: {inverse_vol_coverage}/{len(results)} periods ({inverse_vol_coverage/len(results):.1%})")
    print()
    print("Key Benefits of Volatility-Adjusted Budgeting:")
    print("• Maintains hedge coverage during market stress")
    print("• Opportunistic allocation during low-volatility periods")
    print("• Reduces timing dependency")
    print("• Improves long-term hedge effectiveness")


def demonstrate_implementation_example():
    """
    Demonstrate a practical implementation example for portfolio managers.
    """
    print("\n" + "=" * 70)
    print("PRACTICAL IMPLEMENTATION EXAMPLE")
    print("=" * 70)
    
    print("\nScenario: $10M equity portfolio, quarterly rebalancing")
    print("-" * 60)
    
    # Example scenarios
    scenarios = [
        {"name": "Bull Market", "vol": 0.12, "market_level": 4500},
        {"name": "Normal Market", "vol": 0.20, "market_level": 4000}, 
        {"name": "Stressed Market", "vol": 0.35, "market_level": 3500},
        {"name": "Crisis Market", "vol": 0.60, "market_level": 2500}
    ]
    
    portfolio_value = 10_000_000
    budgeting = VolatilityAdjustedBudgeting()
    
    print(f"Portfolio Value: ${portfolio_value:,}")
    print(f"Base Allocation: {budgeting.base_allocation:.1%}")
    print(f"Allocation Range: {budgeting.min_allocation:.1%} - {budgeting.max_allocation:.1%}")
    print()
    
    for scenario in scenarios:
        vol = scenario["vol"]
        market_level = scenario["market_level"]
        
        # Determine regime
        regime = budgeting.determine_vol_regime(vol, vol)
        
        # Calculate budgets
        budgets = budgeting.calculate_vol_adjusted_budget(
            vol, regime, portfolio_value)
        
        # Calculate LEAP cost (35% OTM)
        strike = market_level * 0.65
        leap_cost = BlackScholesCalculator.option_price(
            spot=market_level, strike=strike, time_to_expiry=1.0,
            volatility=vol, risk_free_rate=0.05, option_type="put"
        ) * 100
        
        # Fixed vs adjusted allocation
        fixed_budget = portfolio_value * budgeting.base_allocation
        adjusted_budget = budgets["inverse_vol"]
        
        fixed_contracts = int(fixed_budget / leap_cost)
        adjusted_contracts = int(adjusted_budget / leap_cost)
        
        print(f"{scenario['name']} (vol={vol:.0%}, level={market_level}):")
        print(f"  Regime: {regime}")
        print(f"  LEAP Cost (35% OTM): ${leap_cost:,.0f}")
        print(f"  Fixed Budget: ${fixed_budget:,.0f} → {fixed_contracts} contracts")
        print(f"  Adjusted Budget: ${adjusted_budget:,.0f} → {adjusted_contracts} contracts")
        print(f"  Improvement: {adjusted_contracts - fixed_contracts:+d} contracts")
        print()


def main():
    """Main function to run volatility-adjusted budgeting analysis."""
    try:
        print("Volatility-Adjusted Budgeting for Tail Hedging")
        print("=" * 60)
        print("Solving the problem: 'Hedges become unaffordable when most needed'")
        print()
        
        # Run historical analysis
        results = analyze_vol_adjusted_strategies()
        
        # Demonstrate practical implementation
        demonstrate_implementation_example()
        
        print(f"\nAnalysis complete. Processed {len(results)} time periods.")
        print("\nKey Takeaway: Volatility-adjusted budgeting maintains hedge")
        print("coverage during market stress while optimizing allocation")
        print("during calm periods.")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()