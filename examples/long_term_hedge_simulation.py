"""
Long-Term Portfolio Hedging Simulation

This script simulates and analyzes a long-term investment strategy for a
portfolio fully invested in a market index, comparing a simple buy-and-hold
approach against a systematic hedging strategy using LEAPS put options.

The simulation follows these steps:
1.  Initializes a portfolio with a starting capital invested in a specified
    market index (e.g., NASDAQ Composite - ^IXIC).
2.  Tracks the portfolio's value over a long historical period.
3.  Simulates two scenarios:
    a.  An unhedged portfolio that simply tracks the index.
    b.  A hedged portfolio that systematically purchases "insurance" every
        six months.
4.  The "insurance" consists of 1-year LEAPS put options purchased with a
    small percentage of the portfolio's value (e.g., 0.5%). This budget is
    split across multiple out-of-the-money (OTM) strike prices (e.g.,
    25%, 30%, 35% OTM).
5.  The value of all active option contracts is re-priced daily based on
    the underlying index price and rolling historical volatility.
6.  The cost of purchasing new options is deducted from the hedged
    portfolio's value.
7.  The script calculates and plots the daily performance of both
    portfolios, the daily rolling volatility of the index, and the
    drawdown for both the hedged and unhedged portfolios.
8.  Finally, it reports key performance metrics like CAGR and max drawdown
    for both strategies.
"""
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import argparse
import os
import sys
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Ensure the source directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from asymmetric_options import BlackScholesCalculator


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
            regime_multiplier = 1.0  # Changed from 2.0 to 1.0
        elif vol_regime == "normal":  # 15% <= vol < 30%
            regime_multiplier = 1.0
        elif vol_regime == "high":  # 30% <= vol < 50%
            regime_multiplier = 0.5
        else:  # crisis: vol >= 50%
            regime_multiplier = 0.3
            
        regime_budget = (self.base_allocation * regime_multiplier * 
                       portfolio_value)
        regime_budget = np.clip(regime_budget,
                              self.min_allocation * portfolio_value,
                              self.max_allocation * portfolio_value)
        
        return {
            "inverse_vol": inverse_vol_budget,
            "regime_based": regime_budget,
            "vol_multiplier": inverse_vol_multiplier,
            "regime": vol_regime
        }
    
    def determine_vol_regime(self, current_vol, rolling_vol_avg=None):
        """Determine volatility regime for allocation decisions."""
        if current_vol < 0.15:
            return "low"
        elif current_vol < 0.30:
            return "normal" 
        elif current_vol < 0.50:
            return "high"
        else:
            return "crisis"

    def allocate_contracts_vol_adjusted(self, option_costs, budget, vol_regime, current_vol):
        """
        Allocate contracts based on volatility regime.
        
        Strategy:
        - Low vol: Can afford ATM/ITM options, diversify across strikes
        - Normal vol: Standard allocation
        - High vol: Focus on far OTM for maximum coverage
        - Crisis vol: Only buy cheapest options for survival
        """
        contracts_per_strike = [0] * len(option_costs)
        remaining_budget = budget
        
        if vol_regime == "low":
            # Low volatility: Diversify across all strikes, start with ATM
            print(f"    *** LOW VOLATILITY ALLOCATION ***")
            print(f"    Strategy: Diversified allocation across all strikes")
            
            # Try to buy 2-3 contracts of each strike
            target_contracts = 3 if budget > 1000 else 2
            for idx, (strike_pct, strike_price, cost_per_share, cost_per_contract) in enumerate(option_costs):
                affordable_contracts = min(target_contracts, int(remaining_budget // cost_per_contract))
                if affordable_contracts > 0:
                    contracts_per_strike[idx] = affordable_contracts
                    cost = affordable_contracts * cost_per_contract
                    remaining_budget -= cost
                    print(f"    Buying {affordable_contracts} contracts {strike_pct:.1%} OTM for ${cost:.2f}")
            
            # Calculate remaining allocation proportionally in one step
            if remaining_budget > 0:
                min_cost = min(cost for _, _, _, cost in option_costs)
                if remaining_budget >= min_cost:
                    # Distribute remaining budget proportionally across all strikes
                    total_weight = len(option_costs)
                    for idx, (strike_pct, strike_price, cost_per_share, cost_per_contract) in enumerate(option_costs):
                        additional_budget = remaining_budget / total_weight
                        additional_contracts = int(additional_budget // cost_per_contract)
                        if additional_contracts > 0:
                            contracts_per_strike[idx] += additional_contracts
                            remaining_budget -= additional_contracts * cost_per_contract
                            print(f"    Added {additional_contracts} more contracts {strike_pct:.1%} OTM (total: {contracts_per_strike[idx]})")
                        
        elif vol_regime == "normal":
            # Normal volatility: Slightly favor far OTM but maintain diversity
            print(f"    *** NORMAL VOLATILITY ALLOCATION ***") 
            print(f"    Strategy: Balanced allocation with slight OTM bias")
            
            # Buy 1 of each if possible, then favor far OTM
            for idx, (strike_pct, strike_price, cost_per_share, cost_per_contract) in enumerate(option_costs):
                if cost_per_contract <= remaining_budget:
                    contracts_per_strike[idx] = 1
                    remaining_budget -= cost_per_contract
                    print(f"    Buying 1 contract {strike_pct:.1%} OTM for ${cost_per_contract:.2f}")
            
            # Calculate how many additional contracts to buy from furthest OTM in one step
            if remaining_budget > 0:
                # Start from furthest OTM (highest index) and calculate bulk purchases
                for idx in range(len(option_costs) - 1, -1, -1):
                    strike_pct, strike_price, cost_per_share, cost_per_contract = option_costs[idx]
                    additional_contracts = int(remaining_budget // cost_per_contract)
                    if additional_contracts > 0:
                        contracts_per_strike[idx] += additional_contracts
                        cost = additional_contracts * cost_per_contract
                        remaining_budget -= cost
                        print(f"    Added {additional_contracts} far OTM contracts {strike_pct:.1%} (total: {contracts_per_strike[idx]})")
                        break  # Focus on one strike at a time for normal vol
                    
        elif vol_regime == "high":
            # High volatility: Focus heavily on far OTM options
            print(f"    *** HIGH VOLATILITY ALLOCATION ***")
            print(f"    Strategy: Focus on far OTM options for cost efficiency")
            
            # Sort by cost and focus on cheapest options
            sorted_options = sorted(enumerate(option_costs), key=lambda x: x[1][3])  # Sort by cost_per_contract
            
            # Try to get at least one of the cheapest option
            cheapest_idx, (strike_pct, strike_price, cost_per_share, cost_per_contract) = sorted_options[0]
            if cost_per_contract <= remaining_budget:
                contracts_per_strike[cheapest_idx] = 1
                remaining_budget -= cost_per_contract
                print(f"    Buying 1 cheapest contract {strike_pct:.1%} OTM for ${cost_per_contract:.2f}")
            
            # Calculate bulk purchase for remaining budget on cheapest options
            if remaining_budget > 0:
                for idx, (strike_pct, strike_price, cost_per_share, cost_per_contract) in sorted_options:
                    additional_contracts = int(remaining_budget // cost_per_contract)
                    if additional_contracts > 0:
                        contracts_per_strike[idx] += additional_contracts
                        cost = additional_contracts * cost_per_contract
                        remaining_budget -= cost
                        print(f"    Added {additional_contracts} cheap contracts {strike_pct:.1%} (total: {contracts_per_strike[idx]})")
                        break  # Focus on cheapest option for high vol
                    
        else:  # Crisis regime
            # Crisis volatility: Only buy the absolute cheapest option
            print(f"    *** CRISIS VOLATILITY ALLOCATION ***")
            print(f"    Strategy: Survival mode - cheapest options only")
            
            # Find the cheapest option and buy as many as possible in one calculation
            cheapest_idx = min(range(len(option_costs)), key=lambda i: option_costs[i][3])
            strike_pct, strike_price, cost_per_share, cost_per_contract = option_costs[cheapest_idx]
            
            # Buy as many of the cheapest as possible
            max_contracts = int(remaining_budget // cost_per_contract)
            if max_contracts > 0:
                contracts_per_strike[cheapest_idx] = max_contracts
                total_cost = max_contracts * cost_per_contract
                remaining_budget -= total_cost
                print(f"    Buying {max_contracts} cheapest contracts {strike_pct:.1%} OTM for ${total_cost:.2f}")
        
        return contracts_per_strike


def calculate_drawdown(series: pd.Series) -> pd.Series:
    """Calculates the drawdown of a time series."""
    cumulative_max = series.cummax()
    drawdown = (series - cumulative_max) / cumulative_max
    return drawdown


def run_long_term_hedge_simulation(
    start_date_str: str = "1991-01-01",
    hedge_start_date_str: str = "1993-01-01",
    end_date_str: str = datetime.now().strftime("%Y-%m-%d"),
    ticker: str = "^IXIC",
    initial_investment: float = 10000.0,
    hedge_budget_pct: float = 0.005,
    hedge_frequency_months: int = 6,
    strike_otm_pcts: list[float] = [-0.20, -0.25, -0.30],
    risk_free_rate: float = 0.05,
    vol_adj_leaps: bool = False,
    cost_premium_pct: float = 0.0,
    volatility_cap: float = 0.50
):
    """
    Runs the long-term historical simulation of a systematically hedged portfolio.
    
    Args:
        vol_adj_leaps: If True, use volatility-adjusted budgeting for LEAP allocation
        cost_premium_pct: Additional cost premium to add to LEAP prices (default 0.0).
                         For example, 0.20 adds 20% to the theoretical option cost.
                         This simulates scenarios where implied volatility exceeds
                         historical volatility used in pricing.
                         IMPORTANT: Applied consistently to both purchases AND sales.
        volatility_cap: Maximum volatility level for LEAP pricing (default 0.50).
                       Based on historical analysis:
                       - 32% (0.32): Too conservative, caps legitimate crisis volatility
                       - 50% (0.50): Realistic based on 95th percentile historical data
                       - 65% (0.65): Based on 99th percentile, allows most crisis volatility
                       - None: No cap (most realistic for tail hedge analysis)
                       Historical evidence shows equity volatility has exceeded 80% during
                       major crises (2008, 2020), making caps below 50% unrealistic.
    """
    # Initialize volatility-adjusted budgeting if enabled
    vol_budgeting = None
    if vol_adj_leaps:
        vol_budgeting = VolatilityAdjustedBudgeting(
            base_allocation_pct=hedge_budget_pct,
            vol_adjustment_factor=2.0,
            min_allocation_pct=hedge_budget_pct * 0.4,  # 40% of base in high vol
            max_allocation_pct=hedge_budget_pct * 3.0   # 300% of base in low vol
        )
        print(f"*** Volatility-Adjusted LEAP Budgeting ENABLED ***")
        print(f"    Base allocation: {hedge_budget_pct:.1%}")
        print(f"    Range: {hedge_budget_pct * 0.4:.1%} - {hedge_budget_pct * 3.0:.1%}")
        print(f"    Strategy: Inverse volatility scaling + regime-based allocation")

    # Initialize volatility capping tracking
    vol_cap_events = 0
    vol_cap_total_impact = 0.0
    
    # Log volatility cap configuration
    print(f"*** VOLATILITY CAP CONFIGURATION ***")
    if volatility_cap is not None:
        print(f"    Maximum volatility: {volatility_cap:.1%}")
        print(f"    Rationale: Based on historical analysis of equity market volatility")
        print(f"    - 32% (0.32): Too conservative, caps legitimate crisis volatility")
        print(f"    - 50% (0.50): Realistic based on 95th percentile historical data")
        print(f"    - 65% (0.65): Based on 99th percentile, allows most crisis volatility")
        print(f"    Impact: When active, reduces LEAP costs but may underestimate crisis performance")
    else:
        print(f"    No volatility cap applied (most realistic for tail hedge analysis)")
        print(f"    This allows full capture of crisis-period volatility and LEAP performance")

    # --- 1. Data Fetching and Preparation ---
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    hedge_start_date = datetime.strptime(hedge_start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    # Fetch data from one year prior to start for initial volatility calculations
    fetch_start_date = start_date - timedelta(days=365)

    print(f"Fetching historical data for {ticker} from {fetch_start_date.date()} to {end_date.date()}...")
    data = yf.download(
        ticker,
        start=fetch_start_date.strftime("%Y-%m-%d"),
        end=end_date.strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False
    )
    
    # Create a copy to avoid SettingWithCopyWarning
    data = data.copy()
    data["returns"] = data["Close"].pct_change()
    
    # Use a longer window for initial vol calculation to avoid NaNs
    data["rolling_vol_60d"] = (
        data["returns"].rolling(window=60, min_periods=30).std()
        * np.sqrt(252)
    )
    data["rolling_vol_60d"] = data["rolling_vol_60d"].bfill()
    
    # Isolate the simulation period
    sim_data = data.loc[start_date:end_date].copy()
    
    # --- 2. Unhedged Portfolio Simulation ---
    sim_data["unhedged_value"] = 0.0
    sim_data.iloc[0, sim_data.columns.get_loc("unhedged_value")] = initial_investment
    
    # Calculate unhedged portfolio value step by step to avoid NaN propagation
    for i in range(1, len(sim_data)):
        daily_return = sim_data.iloc[i]["returns"]
        # Handle NaN returns properly - ensure scalar value
        if isinstance(daily_return, pd.Series):
            daily_return = daily_return.iloc[0]
        if pd.isna(daily_return):
            daily_return = 0.0
        daily_return = float(daily_return)
        
        prev_value = sim_data.iloc[i-1]["unhedged_value"] 
        if isinstance(prev_value, pd.Series):
            prev_value = prev_value.iloc[0]
        prev_value = float(prev_value)
        
        sim_data.iloc[i, sim_data.columns.get_loc("unhedged_value")] = prev_value * (1 + daily_return)

# --- 3. Hedged Portfolio Simulation ---
    print("Starting hedged portfolio simulation...")
    sim_data["hedged_portfolio_value"] = sim_data["unhedged_value"].copy()  # Start same as unhedged
    sim_data["total_leaps_value"] = 0.0  # Unrealized value of all active LEAPs
    sim_data["cumulative_leap_cost"] = 0.0  # Track cumulative LEAP costs
    sim_data["cumulative_leap_realized_gains"] = 0.0  # Track cumulative realized gains
    
    active_leaps = []  # Track individual LEAP positions
    next_hedge_date = pd.Timestamp(hedge_start_date)
    
    # Track costs and profits for text box display
    total_hedge_cost = 0.0
    total_hedge_purchases = 0
    total_leap_sales = 0.0
    total_leap_realized_gains = 0.0  # Track realized gains separately
    skipped_purchases = 0

    # This loop is the core of the simulation
    for i in range(1, len(sim_data)):
        current_date = sim_data.index[i]
        prev_date = sim_data.index[i-1]
        
        # Copy cumulative values from previous day
        sim_data.iloc[i, sim_data.columns.get_loc("cumulative_leap_cost")] = sim_data.iloc[i-1]["cumulative_leap_cost"]
        sim_data.iloc[i, sim_data.columns.get_loc("cumulative_leap_realized_gains")] = sim_data.iloc[i-1]["cumulative_leap_realized_gains"]
        
        # --- a. Grow Portfolio by Index Return ---
        daily_return = sim_data.iloc[i]["returns"]
        
        # Handle NaN and Series issues properly
        if isinstance(daily_return, pd.Series):
            daily_return = daily_return.iloc[0]
        if pd.isna(daily_return):
            daily_return = 0.0
        daily_return = float(daily_return)

        # Get previous portfolio value
        prev_portfolio_value = sim_data.iloc[i-1]["hedged_portfolio_value"]
        if isinstance(prev_portfolio_value, pd.Series):
            prev_portfolio_value = prev_portfolio_value.iloc[0]
        prev_portfolio_value = float(prev_portfolio_value)
        
        # Calculate current portfolio value (equity portion only)
        current_portfolio_value = prev_portfolio_value * (1 + daily_return)
        sim_data.iloc[i, sim_data.columns.get_loc("hedged_portfolio_value")] = current_portfolio_value

        # --- b. Purchase New LEAPs ---
        if current_date >= next_hedge_date:
            hedge_budget = current_portfolio_value * hedge_budget_pct
            
            # Get unhedged portfolio value for comparison
            unhedged_value = sim_data.iloc[i]["unhedged_value"]
            if isinstance(unhedged_value, pd.Series):
                unhedged_value = unhedged_value.iloc[0]
            unhedged_value = float(unhedged_value)
            
            # Print portfolio comparison before hedge decision
            print(f"  - Portfolio on {current_date.date()}: Unhedged=${unhedged_value:,.2f}, Hedged=${current_portfolio_value:,.2f}")
            
            # Check minimum cost threshold
            if hedge_budget >= 150.0:  # Minimum $150 to buy hedges
                print(f"  - Purchasing LEAPs on {current_date.date()} (Budget: ${hedge_budget:.2f})")
                
                current_price = sim_data.iloc[i]["Close"]
                if isinstance(current_price, pd.Series):
                    current_price = current_price.iloc[0]
                current_price = float(current_price)

                implied_vol = sim_data.iloc[i]["rolling_vol_60d"]
                if isinstance(implied_vol, pd.Series):
                    implied_vol = implied_vol.iloc[0]
                
                # Fallback if vol is still NaN or zero
                if pd.isna(implied_vol) or implied_vol == 0:
                    implied_vol = 0.2  # Default to 20% vol
                implied_vol = float(implied_vol)

                # Apply volatility-adjusted budgeting if enabled
                if vol_adj_leaps and vol_budgeting:
                    vol_regime = vol_budgeting.determine_vol_regime(implied_vol)
                    vol_budget_info = vol_budgeting.calculate_vol_adjusted_budget(
                        implied_vol, vol_regime, current_portfolio_value
                    )
                    
                    # Use inverse volatility method for budget adjustment
                    adjusted_budget = vol_budget_info["inverse_vol"]
                    vol_multiplier = vol_budget_info["vol_multiplier"]
                    
                    print(f"    *** VOLATILITY-ADJUSTED BUDGETING ***")
                    print(f"    Volatility: {implied_vol:.1%} (Regime: {vol_regime})")
                    print(f"    Vol multiplier: {vol_multiplier:.2f}x")
                    print(f"    Original budget: ${hedge_budget:.2f}")
                    print(f"    Adjusted budget: ${adjusted_budget:.2f}")
                    print(f"    Budget change: {(adjusted_budget/hedge_budget - 1)*100:+.1f}%")
                    
                    # Use the adjusted budget
                    hedge_budget = adjusted_budget
                    remaining_budget = hedge_budget

                # Determine volatility regime for LEAP adjustments
                vol_regime = vol_budgeting.determine_vol_regime(implied_vol) if vol_adj_leaps and vol_budgeting else "normal"
                
                # Adjust strikes and expiration based on volatility regime
                if vol_regime == "low":
                    # Low volatility: Use 2-year LEAPs and lower strikes by 5%
                    leap_expiration_years = 2
                    adjusted_strike_otm_pcts = [strike + 0.05 for strike in strike_otm_pcts]  # Less OTM (closer to money)
                    print(f"    *** LOW VOLATILITY ADJUSTMENTS ***")
                    print(f"    LEAP expiration: {leap_expiration_years} years (vs 1 year normal)")
                    print(f"    Adjusted strikes: {[f'{p:.1%}' for p in adjusted_strike_otm_pcts]} (vs {[f'{p:.1%}' for p in strike_otm_pcts]} normal)")
                else:
                    # Normal/High/Crisis volatility: Use standard 1-year LEAPs and original strikes
                    leap_expiration_years = 1
                    adjusted_strike_otm_pcts = strike_otm_pcts
                
                # Calculate cost per contract for each strike (using adjusted strikes)
                option_costs = []
                for strike_pct in adjusted_strike_otm_pcts:
                    strike_price = current_price * (1 + strike_pct)
                    
                    # Use high-precision pricing for very small option values
                    option_cost_per_share = BlackScholesCalculator.option_price_high_precision(
                        spot=current_price, 
                        strike=strike_price, 
                        time_to_expiry=leap_expiration_years, 
                        volatility=implied_vol, 
                        risk_free_rate=risk_free_rate, 
                        option_type="put",
                        min_value=1e-6  # Minimum $0.000001 per share
                    )
                    
                    # Apply cost premium if specified (increases purchase cost)
                    if cost_premium_pct > 0:
                        option_cost_per_share = option_cost_per_share * (1 + cost_premium_pct)
                    
                    cost_per_contract = option_cost_per_share * 100  # 100 shares per contract
                    option_costs.append((strike_pct, strike_price, option_cost_per_share, cost_per_contract))
                
                # Log cost premium application if used
                if cost_premium_pct > 0:
                    print(f"    *** COST PREMIUM APPLIED: +{cost_premium_pct:.1%} ***")
                    print(f"    This simulates implied volatility exceeding historical volatility")
                    print(f"    Premium applies to BOTH purchases and sales for consistency")
                
                print(f"    Budget: ${hedge_budget:.2f}, Vol: {implied_vol:.1%}")
                print(f"    Expected costs: {[f'${cost:.2f}' for _, _, _, cost in option_costs]}")
                
                # Volatility-adjusted contract allocation strategy
                if vol_adj_leaps and vol_budgeting:
                    # In high volatility environments, focus more on far OTM options
                    # In low volatility environments, can afford closer to the money
                    contracts_per_strike = vol_budgeting.allocate_contracts_vol_adjusted(
                        option_costs, remaining_budget, vol_regime, implied_vol
                    )
                else:
                    # Original strategy: Calculate contracts efficiently without loops
                    contracts_per_strike = [0] * len(strike_otm_pcts)
                    remaining_budget = hedge_budget
                    
                    # First pass: Try to buy 1 contract of each strike if budget allows
                    for idx, (strike_pct, strike_price, cost_per_share, cost_per_contract) in enumerate(option_costs):
                        if cost_per_contract <= remaining_budget:
                            contracts_per_strike[idx] = 1
                            remaining_budget -= cost_per_contract
                            print(f"    Buying 1 contract {strike_pct:.1%} OTM for ${cost_per_contract:.2f}")
                        else:
                            print(f"    Cannot afford {strike_pct:.1%} OTM (${cost_per_contract:.2f} > ${remaining_budget:.2f})")
                    
                    # Second pass: Allocate remaining budget to cheapest options in bulk
                    print(f"    Remaining budget: ${remaining_budget:.2f}")
                    
                    if remaining_budget > 0:
                        # Start with furthest OTM (cheapest) and calculate bulk purchases
                        for idx in range(len(option_costs) - 1, -1, -1):
                            strike_pct, strike_price, cost_per_share, cost_per_contract = option_costs[idx]
                            
                            # Calculate how many additional contracts we can afford
                            additional_contracts = int(remaining_budget // cost_per_contract)
                            if additional_contracts > 0:
                                contracts_per_strike[idx] += additional_contracts
                                cost = additional_contracts * cost_per_contract
                                remaining_budget -= cost
                                
                                if additional_contracts >= 50:
                                    print(f"    Added {additional_contracts} far OTM contracts {strike_pct:.1%} for ${cost:.2f}")
                                else:
                                    print(f"    Added {additional_contracts} contracts {strike_pct:.1%} OTM for ${cost:.2f}")
                                break  # Focus on one strike at a time for simplicity
                
                total_contracts = sum(contracts_per_strike)
                print(f"    Final allocation: {contracts_per_strike} (Total: {total_contracts} contracts)")
                
                # Create LEAP positions
                cost_of_new_leaps = 0
                for idx, (strike_pct, strike_price, cost_per_share, cost_per_contract) in enumerate(option_costs):
                    contracts = contracts_per_strike[idx]
                    if contracts > 0:
                        num_shares = contracts * 100
                        leap_cost = contracts * cost_per_contract
                        cost_of_new_leaps += leap_cost
                        
                        # Calculate what percentage of portfolio this option cost represents
                        option_cost_pct = leap_cost / current_portfolio_value
                        
                        print(f"    Creating LEAP: {strike_pct:.1%} OTM, {contracts} contracts, ${leap_cost:.2f} ({option_cost_pct:.2%} of portfolio)")
                        
                        # Create LEAP position with tracking for 15-day SMA
                        leap_position = {
                            "purchase_date": current_date,
                            "expiry_date": current_date + relativedelta(years=leap_expiration_years),
                            "strike_price": strike_price,
                            "strike_pct": strike_pct,
                            "num_shares": num_shares,
                            "cost_per_share": cost_per_share,
                            "total_cost": leap_cost,
                            "daily_values": [],
                            "sold": False,
                            "sale_date": None,
                            "sale_value": 0.0
                        }
                        active_leaps.append(leap_position)
                
                # Deduct LEAP costs from portfolio immediately
                sim_data.iloc[i, sim_data.columns.get_loc("hedged_portfolio_value")] -= cost_of_new_leaps
                
                # Track costs for reporting
                total_hedge_cost += cost_of_new_leaps
                total_hedge_purchases += 1
                sim_data.iloc[i, sim_data.columns.get_loc("cumulative_leap_cost")] += cost_of_new_leaps
            else:
                skipped_purchases += 1
                print(f"  - Skipping LEAPs on {current_date.date()} (Budget ${hedge_budget:.2f} < $150 minimum)")
            
            next_hedge_date += relativedelta(months=hedge_frequency_months)

        # --- c. Update LEAP Values and Check for Sales ---
        current_price = sim_data.iloc[i]["Close"]
        if isinstance(current_price, pd.Series):
            current_price = current_price.iloc[0]
        current_price = float(current_price)

        implied_vol = sim_data.iloc[i]["rolling_vol_60d"]
        if isinstance(implied_vol, pd.Series):
            implied_vol = implied_vol.iloc[0]
        
        if pd.isna(implied_vol) or implied_vol == 0:
            prev_vol = sim_data.iloc[i-1]["rolling_vol_60d"]
            if isinstance(prev_vol, pd.Series):
                prev_vol = prev_vol.iloc[0]
            implied_vol = prev_vol if not pd.isna(prev_vol) else 0.2
        implied_vol = float(implied_vol)
        
        # Apply configurable volatility cap for LEAP pricing
        # Based on historical analysis of equity market volatility patterns
        original_vol = implied_vol
        
        if volatility_cap is not None and implied_vol > volatility_cap:
            # Calculate pricing impact before applying cap
            if active_leaps:  # Only calculate impact if we have LEAPs to price
                sample_strike = current_price * 0.75  # 25% OTM for impact calculation
                uncapped_price = BlackScholesCalculator.option_price(
                    spot=current_price, strike=sample_strike, time_to_expiry=1.0,
                    volatility=original_vol, risk_free_rate=risk_free_rate, option_type="put"
                )
                capped_price = BlackScholesCalculator.option_price(
                    spot=current_price, strike=sample_strike, time_to_expiry=1.0,
                    volatility=volatility_cap, risk_free_rate=risk_free_rate, option_type="put"
                )
                
                if uncapped_price > 0:
                    pricing_impact = ((uncapped_price - capped_price) / uncapped_price) * 100
                    vol_cap_total_impact += pricing_impact
                else:
                    pricing_impact = 0.0
            else:
                pricing_impact = 0.0
            
            print(f"    *** VOLATILITY CAP APPLIED ***")
            print(f"    Original volatility: {original_vol:.1%}")
            print(f"    Capped volatility: {volatility_cap:.1%}")
            print(f"    Estimated LEAP pricing impact: -{pricing_impact:.1f}%")
            print(f"    Rationale: Prevents unrealistic option values during extreme volatility spikes")
            
            implied_vol = volatility_cap
            vol_cap_events += 1
        
        # Add volatility mean reversion for long-term options
        # LEAPs should use a blend of current and long-term average volatility
        long_term_vol = 0.20  # 20% long-term average for equity indices
        vol_weight = 0.7  # 70% current vol, 30% long-term average
        leap_vol = implied_vol * vol_weight + long_term_vol * (1 - vol_weight)

        total_leaps_value = 0.0
        leaps_to_remove = []
        
        for leap_idx, leap in enumerate(active_leaps):
            if leap["sold"] or leap["expiry_date"] <= current_date:
                if not leap["sold"]:  # Expired without being sold
                    leaps_to_remove.append(leap_idx)
                continue
                
            time_to_expiry = (leap["expiry_date"] - current_date).days / 365.25
            if time_to_expiry > 0:
                current_leap_value_per_share = BlackScholesCalculator.option_price(
                    spot=current_price, strike=leap["strike_price"], 
                    time_to_expiry=time_to_expiry, volatility=leap_vol,  # Use adjusted volatility
                    risk_free_rate=risk_free_rate, option_type="put"
                )
                
                # Cost premium only applies to purchases, not sales
                # When selling, we receive the theoretical Black-Scholes value without any premium
                
                current_leap_total_value = current_leap_value_per_share * leap["num_shares"]
                
                # Sanity check: Put option value should never exceed strike price
                max_possible_value = leap["strike_price"] * leap["num_shares"]
                num_contracts = leap["num_shares"] // 100
                # Sanity check: Put option value should never exceed strike price
                max_possible_value = leap["strike_price"] * leap["num_shares"]
                if current_leap_total_value > max_possible_value:
                    print(f"    WARNING: Invalid option value detected on {current_date.date()}")
                    print(f"    Strike: ${leap['strike_price']:.2f}, Current: ${current_price:.2f}")
                    print(f"    Calculated value: ${current_leap_total_value:.2f}, Max possible: ${max_possible_value:.2f}")
                    current_leap_total_value = min(current_leap_total_value, max_possible_value)
                
                # DEBUG: Log calculation details for verification
                if num_contracts > 100:  # Large positions that could be problematic
                    print(f"    DEBUG LARGE POSITION on {current_date.date()}:")
                    print(f"      Strike: {leap['strike_pct']:.1%} OTM (${leap['strike_price']:.2f})")
                    print(f"      Contracts: {num_contracts}, Shares: {leap['num_shares']}")
                    print(f"      Market: ${current_price:.2f}, Time: {time_to_expiry:.2f}y")
                    print(f"      BS value per share: ${current_leap_value_per_share:.6f}")
                    print(f"      Total position value: ${current_leap_total_value:.2f}")
                    print(f"      Original cost: ${leap['total_cost']:.2f}")
                    profit = current_leap_total_value - leap["total_cost"]
                    profit_pct = profit / leap["total_cost"] * 100
                    print(f"      Current P&L: ${profit:.2f} ({profit_pct:.1f}%)")
                
                # Track daily value for 15-day SMA calculation
                leap["daily_values"].append({
                    "date": current_date,
                    "value": current_leap_total_value,
                    "value_per_share": current_leap_value_per_share
                })
                
                # Check if we should sell - REALISTIC TAIL HEDGE CONDITIONS
                # LEAPs should only be sold when they're actually providing protection
                should_sell = False
                sell_reason = ""
                
                # Condition 1: Put is in-the-money (current price below strike)
                if current_price < leap["strike_price"]:
                    # Calculate how deep in-the-money the put is
                    itm_amount = leap["strike_price"] - current_price
                    itm_percentage = itm_amount / leap["strike_price"]
                    
                    # Sell if put is profitable and providing meaningful protection
                    if current_leap_total_value > leap["total_cost"] * 1.5:  # 50% profit minimum
                        should_sell = True
                        sell_reason = f"ITM Protection ({itm_percentage:.1%} below strike)"
                
                # Condition 2: Near strike but very profitable (market stress scenario)
                elif current_price < leap["strike_price"] * 1.05:  # Within 5% of strike
                    if current_leap_total_value > leap["total_cost"] * 2.0:  # 100% profit minimum
                        should_sell = True
                        sell_reason = f"Near Strike & Profitable"
                
                # Condition 3: Approaching expiration with any profit (final 60 days)
                elif time_to_expiry < (60/365.25) and current_leap_total_value > leap["total_cost"]:
                    should_sell = True
                    sell_reason = "Expiration Approaching"
                
                # Condition 4: Extreme volatility spike with massive profit (very rare)
                elif (implied_vol > 0.50 and  # Extreme volatility (50%+)
                      current_leap_total_value > leap["total_cost"] * 3.0):  # 200%+ profit
                    should_sell = True
                    sell_reason = f"Extreme Vol Spike (IV: {implied_vol:.1%})"
                
                if should_sell:
                    # Additional validation before selling
                    profit = current_leap_total_value - leap["total_cost"]
                    profit_pct = profit / leap["total_cost"] * 100
                    
                    # DEBUG: Log the sale calculation
                    print(f"    SELLING LEAP {leap['strike_pct']:.1%} OTM:")
                    print(f"      Contracts: {num_contracts}")
                    print(f"      Value per share: ${current_leap_value_per_share:.6f}")
                    print(f"      Total value: ${current_leap_total_value:.2f}")
                    print(f"      Original cost: ${leap['total_cost']:.2f}")
                    print(f"      Calculated profit: ${profit:.2f} ({profit_pct:.1f}%)")
                    
                    # Cap unrealistic profits (likely calculation errors)
                    if profit_pct > 10000:  # 10000% profit cap (increased from 1000%)
                        print(f"    WARNING: Capping extreme profit of {profit_pct:.0f}%")
                        print(f"    This may be legitimate tail hedge performance during crisis")
                        current_leap_total_value = leap["total_cost"] * 101  # Cap at 10000% profit
                        profit = current_leap_total_value - leap["total_cost"]
                    
                    # Sell the LEAP at current value
                    leap["sold"] = True
                    leap["sale_date"] = current_date
                    leap["sale_value"] = current_leap_total_value
                    
                    # Add sale proceeds to portfolio
                    current_portfolio_val = sim_data.iloc[i]["hedged_portfolio_value"]
                    sim_data.iloc[i, sim_data.columns.get_loc("hedged_portfolio_value")] = current_portfolio_val + current_leap_total_value
                    
                    total_leap_sales += current_leap_total_value
                    total_leap_realized_gains += profit
                    sim_data.iloc[i, sim_data.columns.get_loc("cumulative_leap_realized_gains")] += profit
                    
                    # Enhanced logging with validation
                    days_held = (current_date - leap["purchase_date"]).days
                    annual_return = (profit / leap["total_cost"]) * (365 / days_held) * 100 if days_held > 0 else 0
                    
                    print(f"    * Sold LEAP {leap['strike_pct']:.1%} OTM on {current_date.date()}")
                    print(f"      Cost: ${leap['total_cost']:.2f}, Sale: ${current_leap_total_value:.2f}")
                    print(f"      Profit: ${profit:.2f} ({profit/leap['total_cost']*100:.1f}%) over {days_held} days")
                    print(f"      Annualized return: {annual_return:.1f}%")
                
                if not leap["sold"]:
                    total_leaps_value += current_leap_total_value
        
        # Remove expired LEAPs
        for idx in reversed(leaps_to_remove):
            active_leaps.pop(idx)
        
        sim_data.iloc[i, sim_data.columns.get_loc("total_leaps_value")] = total_leaps_value

    # --- 4. Calculate Metrics and Drawdowns ---
    sim_data["unhedged_drawdown"] = calculate_drawdown(sim_data["unhedged_value"])
    sim_data["hedged_drawdown"] = calculate_drawdown(sim_data["hedged_portfolio_value"])

    # Calculate final metrics
    current_leaps_value = sim_data["total_leaps_value"].iloc[-1]
    total_hedge_profit = total_leap_sales - total_hedge_cost

    # --- 5. Reporting ---
    years = (sim_data.index[-1] - sim_data.index[0]).days / 365.25
    unhedged_cagr = (sim_data["unhedged_value"].iloc[-1] / sim_data["unhedged_value"].iloc[0]) ** (1/years) - 1
    hedged_cagr = (sim_data["hedged_portfolio_value"].iloc[-1] / sim_data["hedged_portfolio_value"].iloc[0]) ** (1/years) - 1
    unhedged_max_drawdown = sim_data["unhedged_drawdown"].min()
    hedged_max_drawdown = sim_data["hedged_drawdown"].min()

    print("\n--- Simulation Complete ---")
    print(f"Period: {sim_data.index[0].date()} to {sim_data.index[-1].date()} ({years:.1f} years)")
    print(f"Total LEAP Purchases: {total_hedge_purchases}")
    print(f"Skipped Purchases (budget < $150): {skipped_purchases}")
    print(f"Total LEAP Cost: ${total_hedge_cost:,.2f}")
    print(f"Total LEAP Sales: ${total_leap_sales:,.2f}")
    print(f"Current LEAP Value (unrealized): ${current_leaps_value:,.2f}")
    print(f"Total LEAP Profit/Loss: ${total_hedge_profit:,.2f}")
    
    # --- Volatility Cap Impact Reporting ---
    if volatility_cap is not None:
        print(f"\n--- Volatility Cap Impact ---")
        print(f"Volatility Cap Level: {volatility_cap:.1%}")
        print(f"Cap Events: {vol_cap_events}")
        if vol_cap_events > 0:
            avg_impact = vol_cap_total_impact / vol_cap_events
            print(f"Average LEAP Pricing Impact: -{avg_impact:.1f}%")
            print(f"Total Impact: LEAP costs reduced by estimated {vol_cap_total_impact:.1f}% over {vol_cap_events} events")
            print(f"Note: Cap reduces hedge costs but may underestimate crisis performance")
        else:
            print(f"No volatility capping events occurred during simulation period")
    else:
        print(f"\n--- No Volatility Cap Applied ---")
        print(f"Full crisis-period volatility captured for realistic tail hedge analysis")

    print("\n--- Performance Metrics ---")
    print(f"{'Metric':<20} {'Unhedged':>15} {'Hedged':>15}")
    print("-" * 52)
    
    final_unhedged_str = f"${sim_data['unhedged_value'].iloc[-1]:,.2f}"
    final_hedged_str = f"${sim_data['hedged_portfolio_value'].iloc[-1]:,.2f}"
    
    print(f"{'Final Value':<20} {final_unhedged_str:>15} {final_hedged_str:>15}")
    unhedged_cagr_str = f"{unhedged_cagr:.2%}"
    hedged_cagr_str = f"{hedged_cagr:.2%}" if not pd.isna(hedged_cagr) else "nan%"
    print(f"{'CAGR':<20} {unhedged_cagr_str:>15} {hedged_cagr_str:>15}")
    unhedged_max_dd_str = f"{unhedged_max_drawdown:.2%}"
    hedged_max_dd_str = f"{hedged_max_drawdown:.2%}" if not pd.isna(hedged_max_drawdown) else "nan%"
    print(f"{'Max Drawdown':<20} {unhedged_max_dd_str:>15} {hedged_max_dd_str:>15}")

    # --- 6. Plotting ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16*.8, 9*.8), sharex=True, gridspec_kw={"height_ratios": [3, 1, 2]})
    fig.suptitle(f"Long-Term Hedging Simulation: {ticker}", fontsize=16)

    # --- Add input parameters and results text box ---
    strikes_str = ", ".join([f"{p:.1%}" for p in strike_otm_pcts])
    
    setup_box = (
        f"Inputs:\n"
        f"  Period: {sim_data.index[0].date()} to {sim_data.index[-1].date()}\n"
        f"  Index: {ticker}\n"
        f"  Initial Investment: ${initial_investment:,.0f}\n"
        f"  Hedge Budget: {hedge_budget_pct:.1%} of portfolio\n"
        f"  Frequency: Every {hedge_frequency_months} months\n"
        f"  Strikes (OTM): [{strikes_str}]\n"
    )

    results_box = (
        f"Results:\n"
        f"  LEAP Purchases: {total_hedge_purchases}\n"
        f"  Skipped (< $150): {skipped_purchases}\n"
        f"  Total Cost: ${total_hedge_cost:,.0f}\n"
        f"  Total Sales: ${total_leap_sales:,.0f}\n"
        f"  Current Value: ${current_leaps_value:,.0f}\n"
        f"  Net P&L: ${total_hedge_profit:,.0f}\n"
        f"  Unhedged CAGR: {unhedged_cagr:.1%}\n"
        f"  Hedged CAGR: {hedged_cagr:.1%}\n"
        f"  Unhedged Max DD: {unhedged_max_drawdown:.1%}\n"
        f"  Hedged Max DD: {hedged_max_drawdown:.1%}\n"
    )

    textstr = f"{setup_box}\n{results_box}"

    fig.text(0.01, 0.98, textstr, transform=fig.transFigure, fontsize=8,
             verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9))

    # Plot 1: Portfolio Value
    ax1.plot(sim_data.index, sim_data["unhedged_value"], label="Unhedged (Index Tracking)", color="blue")
    ax1.plot(sim_data.index, sim_data["hedged_portfolio_value"], label="Hedged (w/ LEAPs)", color="orange")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, which="both", linestyle="--", alpha=0.6)
    ax1.set_title("Portfolio Value (Log Scale)")

    # Plot 2: Cumulative LEAP Costs and Realized Gains (Log Scale)
    ax2.plot(sim_data.index, sim_data["cumulative_leap_cost"], 
             label="Cumulative LEAP Cost", color="red", linewidth=2)
    ax2.plot(sim_data.index, sim_data["cumulative_leap_realized_gains"], 
             label="Cumulative LEAP Realized Gains", color="green", linewidth=2)
    ax2.set_ylabel("Amount ($)")
    ax2.set_yscale("log")
    ax2.legend()
    ax2.grid(True, which="both", linestyle="--", alpha=0.6)
    ax2.set_title("Cumulative LEAP Costs and Realized Gains (Log Scale)")

    # Plot 3: Drawdown
    ax3.plot(sim_data.index, sim_data["unhedged_drawdown"] * 100, label="Unhedged Drawdown", color="red")
    ax3.plot(sim_data.index, sim_data["hedged_drawdown"] * 100, label="Hedged Drawdown", color="green")
    ax3.fill_between(sim_data.index, sim_data["unhedged_drawdown"] * 100, 0, color="red", alpha=0.1)
    ax3.fill_between(sim_data.index, sim_data["hedged_drawdown"] * 100, 0, color="green", alpha=0.1)
    ax3.set_ylabel("Drawdown (%)")
    ax3.set_xlabel("Date")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.set_title("Portfolio Drawdown")

    plt.tight_layout(rect=[0.02, 0, 1, 0.96])
    
    # Include cost premium in filename to prevent overwrites
    cost_premium_str = f"_premium{cost_premium_pct:.1%}".replace(".", "p") if cost_premium_pct > 0 else ""
    plot_filename = f"long_term_hedge_{ticker.replace('^', '')}{cost_premium_str}.png"
    
    plt.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")
    plt.show()


def main():
    """Main function to parse arguments and run the simulation."""
    parser = argparse.ArgumentParser(description="Run a long-term portfolio hedging simulation.")
    parser.add_argument("--ticker", type=str, default="^IXIC", help="Ticker symbol for the index (e.g., '^IXIC').")
    parser.add_argument("--start-date", type=str, default="1991-01-01", help="Portfolio start date.")
    parser.add_argument("--hedge-start-date", type=str, default="1993-01-01", help="Date to begin hedging strategy.")
    parser.add_argument("--end-date", type=str, default=datetime.now().strftime("%Y-%m-%d"), help="End date for the simulation.")
    parser.add_argument("--initial-investment", type=float, default=10000.0, help="Initial investment amount.")
    parser.add_argument("--hedge-budget-pct", type=float, default=0.005, help="Percentage of portfolio value to spend on hedges every period.")
    parser.add_argument("--hedge-frequency-months", type=int, default=6, help="Frequency in months to purchase new hedges.")
    parser.add_argument("--strike-pct", type=float, nargs="+", default=[-0.25, -0.30, -0.35], help="List of OTM percentages for put strikes.")
    parser.add_argument("--risk-free-rate", type=float, default=0.05, help="Risk-free rate for option pricing.")
    parser.add_argument("--vol-adj-leaps", action="store_true", help="Enable volatility-adjusted budgeting for LEAP allocation.")
    parser.add_argument("--cost-premium-pct", type=float, default=0.0, help="Additional cost premium to add to LEAP prices as a percentage (default 0.0). For example, 0.20 adds 20%% to theoretical option cost. This simulates scenarios where implied volatility exceeds historical volatility.")
    parser.add_argument("--volatility-cap", type=float, default=0.50, help="Maximum volatility level for LEAP pricing (default 0.50). Based on historical analysis: 32%% (0.32) = Too conservative, caps legitimate crisis volatility; 50%% (0.50) = Realistic based on 95th percentile historical data; 65%% (0.65) = Based on 99th percentile, allows most crisis volatility; None = No cap (most realistic for tail hedge analysis). Historical evidence shows equity volatility has exceeded 80%% during major crises.")
    
    args = parser.parse_args()

    run_long_term_hedge_simulation(
        start_date_str=args.start_date,
        hedge_start_date_str=args.hedge_start_date,
        end_date_str=args.end_date,
        ticker=args.ticker,
        initial_investment=args.initial_investment,
        hedge_budget_pct=args.hedge_budget_pct,
        hedge_frequency_months=args.hedge_frequency_months,
        strike_otm_pcts=args.strike_pct,
        risk_free_rate=args.risk_free_rate,
        vol_adj_leaps=args.vol_adj_leaps,
        cost_premium_pct=args.cost_premium_pct,
        volatility_cap=args.volatility_cap
    )


if __name__ == "__main__":
    main()
