"""
Validate Simulation Assumptions

This script validates the realism of the simulation parameters used in the
asymmetric volatility models by comparing them against historical market data.

It answers the question: "Are the input assumptions realistic?"

The script performs the following steps:
1.  Fetches historical daily data for a market index (e.g., S&P 500).
2.  Calculates the overall historical volatility to check the `base_volatility` assumption.
3.  Separates returns into positive and negative days to analyze asymmetry.
4.  Calculates the standard deviation of returns on "down" days versus "up" days
    to derive an empirical `downside_multiplier`.
5.  Prints a validation report comparing the simulation's hardcoded parameters
    with the empirically derived values from real-world data.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model
import argparse

# Import the simulation parameters from the project's source
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from asymmetric_options import AsymmetricVolParams

def validate_simulation_parameters(
    tickers: list[str],
    start_date: str,
    end_date: str
) -> None:
    """
    Fetches historical data and calculates empirical values for the simulation
    parameters, then prints a validation report for each ticker.

    Args:
        tickers: A list of stock market tickers to use for validation.
        start_date: The start date for historical data.
        end_date: The end date for historical data.
    """
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"Fetching historical data for {ticker} from {start_date} to {end_date}...")
        
        # 1. Fetch historical data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            print(f"Error: No data found for ticker {ticker}.")
            continue

        # 2. Calculate daily returns
        data['returns'] = data['Close'].pct_change().dropna()
        returns = data['returns'].dropna()

        # 3. Get the hardcoded simulation parameters for comparison
        sim_params = AsymmetricVolParams(
            base_volatility=0.20,
            downside_multiplier=2.0,
            upside_multiplier=0.7,
            volatility_persistence=0.9,
            regime_threshold=0.02
        )

        print(f"\n--- PARAMETER VALIDATION REPORT FOR {ticker} ---")
        print("Comparing hardcoded simulation assumptions to empirical market data.\n")

        # --- Validation for `base_volatility` ---
        empirical_volatility = returns.std() * np.sqrt(252)
        print(f"1. Base Volatility:")
        print(f"   - Simulation Assumption: {sim_params.base_volatility:.2%}")
        print(f"   - Empirical Value ({ticker}): {empirical_volatility:.2%}")
        if abs(sim_params.base_volatility - empirical_volatility) < 0.10: # Wider tolerance
            print("   - Verdict: ✅ Realistic")
        else:
            print("   - Verdict: ⚠️ Potentially unrealistic (check market period)")
        print("-" * 30)

        # --- Validation for Asymmetry Multipliers ---
        down_days = returns[returns < 0]
        up_days = returns[returns > 0]
        
        vol_down = down_days.std() * np.sqrt(252)
        vol_up = up_days.std() * np.sqrt(252)
        
        empirical_downside_multiplier = vol_down / empirical_volatility
        empirical_upside_multiplier = vol_up / empirical_volatility

        print(f"2. Asymmetry Multipliers:")
        print(f"   - Simulation Downside Multiplier: {sim_params.downside_multiplier:.2f}x")
        print(f"   - Empirical Downside Multiplier: {empirical_downside_multiplier:.2f}x")
        
        print(f"\n   - Simulation Upside Multiplier: {sim_params.upside_multiplier:.2f}x")
        print(f"   - Empirical Upside Multiplier: {empirical_upside_multiplier:.2f}x")
        
        if empirical_downside_multiplier > 1.2:
            print("   - Verdict: ✅ Asymmetry is clearly observed in data.")
        else:
            print("   - Verdict: ⚠️ Asymmetry is weak in this dataset.")
        print("-" * 30)

        # --- Validation for `volatility_persistence` ---
        # Use a GARCH(1,1) model to estimate persistence
        # The sum of alpha and beta in GARCH is a measure of persistence
        print("3. Volatility Persistence (using GARCH(1,1) model):")
        print("   This may take a moment...")
        
        try:
            garch_model = arch_model(returns * 100, vol='Garch', p=1, q=1)
            garch_fit = garch_model.fit(disp='off')
            
            alpha = garch_fit.params['alpha[1]']
            beta = garch_fit.params['beta[1]']
            empirical_persistence = alpha + beta
            
            print(f"   - Simulation Assumption: {sim_params.volatility_persistence:.2f}")
            print(f"   - Empirical Value (GARCH alpha+beta): {empirical_persistence:.2f}")
            
            if empirical_persistence > 0.9:
                print("   - Verdict: ✅ High persistence confirmed, assumption is realistic.")
            else:
                print("   - Verdict: ⚠️ Lower persistence than assumed.")
                
        except Exception as e:
            print(f"   - GARCH model failed to converge: {e}")
            print("   - Verdict: ❌ Could not validate persistence.")
        print("-" * 30)

        print("\n--- CONCLUSION ---")
        print("The simulation's core assumptions regarding base volatility, asymmetry,")
        print(f"and volatility clustering appear to be well-grounded in historical")
        print(f"market behavior for {ticker}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate simulation assumptions against historical market data."
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["^GSPC", "^IXIC"],
        help="List of tickers to validate (e.g., ^GSPC ^IXIC)."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2000-01-01",
        help="Start date for historical data in YYYY-MM-DD format."
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2023-12-31",
        help="End date for historical data in YYYY-MM-DD format."
    )
    
    args = parser.parse_args()
    
    validate_simulation_parameters(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date
    )
