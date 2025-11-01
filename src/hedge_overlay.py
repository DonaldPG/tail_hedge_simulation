"""
Hedge Overlay Module

Evaluates the effectiveness of tail-risk hedging strategies (e.g., put options)
by simulating their performance under various market conditions, including
asymmetric volatility and stress scenarios.

This module connects the asymmetric simulation with practical hedging implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict

# Assuming asymmetric_options is in the same src directory
from .asymmetric_options import (
    AsymmetricVolatilitySimulator,
    BlackScholesCalculator,
    OptionSpecification,
    AsymmetricVolParams
)


def evaluate_hedge_effectiveness(
    portfolio_returns: pd.Series,
    vol_params: AsymmetricVolParams,
    hedge_allocation: float = 0.03,
    strike_offset: float = -0.05,
    leverage: int = 5,
    option_expiry_days: int = 30
) -> pd.DataFrame:
    """
    Evaluates a systematic put option hedging strategy on a portfolio.

    This function simulates the cost and payoff of a protective put strategy
    that is rolled over a period of time (e.g., monthly). It uses the
    Black-Scholes model to price the options for cost calculation but
    evaluates their performance against the actual (asymmetric) returns.

    Args:
        portfolio_returns: A pandas Series of daily returns for the unhedged portfolio.
        vol_params: Parameters for the asymmetric volatility model.
        hedge_allocation: The percentage of the portfolio value spent on hedges.
        strike_offset: The moneyness of the put options (e.g., -0.05 for 5% OTM).
        leverage: Not directly used in this realistic model but kept for compatibility.
                  The natural leverage of options provides the payoff.
        option_expiry_days: The tenor of the options being purchased.

    Returns:
        A DataFrame containing the returns of the unhedged portfolio, the hedge
        itself, and the final combined (hedged) portfolio.
    """
    # Initialize components
    bs_calculator = BlackScholesCalculator()
    
    # Assume portfolio starts at a value of 100
    portfolio_value = 100.0
    
    # Store results
    hedge_returns = []
    unhedged_portfolio_values = []
    hedged_portfolio_values = []

    # Simulate day-by-day
    current_portfolio_value = portfolio_value
    days_since_last_hedge = 0
    
    for daily_return in portfolio_returns:
        # Update the value of the unhedged portfolio
        current_portfolio_value *= (1 + daily_return)
        unhedged_portfolio_values.append(current_portfolio_value)
        
        # At the start of each period, buy a new hedge
        if days_since_last_hedge % option_expiry_days == 0:
            # Define the put option to be purchased
            option_spec = OptionSpecification(
                option_type="put",
                strike=current_portfolio_value * (1 + strike_offset),
                expiry_days=option_expiry_days,
                underlying_price=current_portfolio_value,
                risk_free_rate=0.02
            )
            
            # Calculate the cost of the option using Black-Scholes with base volatility
            # This simulates buying the option in a "normal" environment
            time_to_expiry_years = option_spec.expiry_days / 365.25
            option_cost_per_unit = bs_calculator.option_price(
                spot=option_spec.underlying_price,
                strike=option_spec.strike,
                time_to_expiry=time_to_expiry_years,
                volatility=vol_params.base_volatility,
                risk_free_rate=option_spec.risk_free_rate,
                option_type='put'
            )
            
            # The total cost is a percentage of the portfolio
            total_hedge_cost = current_portfolio_value * hedge_allocation
            
            # Store the option spec and cost for the current period
            current_hedge = {
                "spec": option_spec,
                "cost": total_hedge_cost,
                "initial_portfolio_value": current_portfolio_value
            }

        # Calculate hedge return for the day
        daily_hedge_return = 0
        
        # At the end of the option's life, calculate its payoff
        if (days_since_last_hedge + 1) % option_expiry_days == 0:
            final_price = current_portfolio_value
            initial_price = current_hedge["initial_portfolio_value"]
            
            # Payoff is the intrinsic value at expiry
            payoff = max(0, current_hedge["spec"].strike - final_price)
            
            # The return is the payoff minus the cost, scaled by the initial value
            hedge_period_return = (payoff - current_hedge["cost"]) / initial_price
            
            # For simplicity, we book the entire period's hedge return on the last day
            daily_hedge_return = hedge_period_return
        
        hedge_returns.append(daily_hedge_return)
        days_since_last_hedge += 1

    # Combine the returns
    hedge_returns = pd.Series(hedge_returns, index=portfolio_returns.index)
    
    # The traditional portfolio returns are simply the market returns
    # The cost of the hedge is implicitly included in the hedge_returns series
    combined_returns = portfolio_returns + hedge_returns

    return pd.DataFrame({
        "Traditional": portfolio_returns,
        "Hedge": hedge_returns,
        "Combined": combined_returns
    })
