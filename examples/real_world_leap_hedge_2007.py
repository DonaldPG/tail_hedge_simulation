"""
Real-World LEAPS Hedge Simulation for 2007

This script provides a concrete, historical example of using a LEAPS put
option as a hedge during the turbulent year of 2007, demonstrating the
user's thesis about the failure of constant volatility assumptions.

It answers the question: "Can I model the cost and payoff for LEAPS
purchased at the beginning of 2007 and see its value over time?"

The script performs the following steps:
1.  Fetches historical S&P 500 data for 2006-2007.
2.  Defines a 1-year, 10% Out-of-the-Money (OTM) LEAPS put option.
3.  Calculates the initial cost of this option on the first trading day
    of 2007, using the low volatility environment of late 2006 as a proxy
    for the implied volatility at the time of purchase.
4.  Iterates through every trading day of 2007, re-pricing the option
    based on the actual market price and the evolving (rising) volatility.
5.  Tracks the daily Profit and Loss (P&L) of the option.
6.  Generates a plot showing the S&P 500 price, the rolling volatility,
    and the net value of the LEAPS hedge over the year.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import argparse
import os
import sys
from datetime import datetime, timedelta

# Ensure the source directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from asymmetric_options import BlackScholesCalculator, OptionSpecification


def run_2007_leap_hedge_simulation(
    leap_date_str: str = "2007-01-01",
    ticker: str = "^IXIC",
    strike_otm_pcts: list[float] = [0.15],
    risk_free_rate: float = 0.05
):
    """
    Runs the historical simulation for the 2007 LEAPS hedge.

    Args:
        leap_date_str: The start date of the LEAPS contract.
        ticker: The market index to use.
        strike_otm_pcts: How far OTM to set the put's strike price.
        risk_free_rate: The risk-free rate for pricing.
    """
    # Determine the date range for fetching data
    leap_date = datetime.strptime(leap_date_str, "%Y-%m-%d")
    sim_start_date = leap_date
    sim_end_date = leap_date + timedelta(days=365)
    fetch_start_date = leap_date - timedelta(days=365)  # Need prior year for initial vol

    print(f"Fetching historical data for {ticker} from {fetch_start_date.date()} to {sim_end_date.date()}...")
    data = yf.download(
        ticker,
        start=fetch_start_date.strftime("%Y-%m-%d"),
        end=sim_end_date.strftime("%Y-%m-%d"),
        progress=False
    )
    data["returns"] = data["Close"].pct_change()

    # Isolate the simulation period data
    sim_period = data.loc[sim_start_date.strftime("%Y-%m-%d"):sim_end_date.strftime("%Y-%m-%d")].copy()

    # --- Step 1: Define the LEAPS Option and Calculate Initial Cost for each strike ---
    start_price = sim_period["Close"].iloc[0]
    if isinstance(start_price, pd.Series):
        start_price = start_price.item()
    
    expiry_days = 365

    # Use volatility from H2 of the prior year as a proxy for implied vol
    initial_vol_start = leap_date - timedelta(days=182)
    initial_vol_end = leap_date - timedelta(days=1)
    initial_vol_period = data.loc[
        initial_vol_start.strftime("%Y-%m-%d"):initial_vol_end.strftime("%Y-%m-%d")
    ]["returns"]
    initial_implied_vol = initial_vol_period.std() * np.sqrt(252)

    results = {}
    initial_costs = []
    strike_prices = []
    contracts_to_buy = []
    actual_costs = []

    print("\n--- Simulation Setup ---")
    print(f"LEAPS Purchase Date: {sim_period.index[0].date()}")
    print(f"Underlying Price: ${start_price:.2f}")
    print(f"Initial Implied Vol (from prior 6 months): {initial_implied_vol:.2%}")

    for strike_pct in strike_otm_pcts:
        # Ensure strike_pct is negative for puts if user provides positive value
        if strike_pct > 0:
            strike_pct = -strike_pct
            
        strike_price = start_price * (1 + strike_pct)
        strike_prices.append(strike_price)

        initial_cost = BlackScholesCalculator.option_price(
            spot=start_price,
            strike=strike_price,
            time_to_expiry=expiry_days / 365.25,
            volatility=initial_implied_vol,
            risk_free_rate=risk_free_rate,
            option_type="put"
        )
        initial_costs.append(initial_cost)

        # Calculate contracts for ~$200 cost
        cost_per_contract = initial_cost * 100
        num_contracts = round(200 / cost_per_contract) if cost_per_contract > 0 else 0
        contracts_to_buy.append(num_contracts)
        actual_costs.append(num_contracts * cost_per_contract)

        results[strike_pct] = {
            "pnl": [],
            "value": [],
        }
        print(f"  Strike {strike_pct:.2%} OTM: ${strike_price:.2f} (Initial Cost: ${initial_cost:.4f})")


    # --- Step 2: Simulate Through the Year Day by Day ---
    vol_history = []

    for i in range(len(sim_period)):
        current_date = sim_period.index[i]
        current_price = sim_period["Close"].iloc[i]
        days_passed = (current_date - sim_period.index[0]).days
        time_to_expiry = (expiry_days - days_passed) / 365.25

        # Use a 60-day rolling volatility as a proxy for current implied vol
        rolling_vol = data["returns"].rolling(window=60).std().loc[current_date] * np.sqrt(252)
        if isinstance(rolling_vol, pd.Series):
            rolling_vol = rolling_vol.item()
        vol_history.append(rolling_vol)

        # Re-price the option for each strike
        for idx, strike_pct in enumerate(strike_otm_pcts):
            current_value = BlackScholesCalculator.option_price(
                spot=current_price,
                strike=strike_prices[idx],
                time_to_expiry=time_to_expiry,
                volatility=rolling_vol,
                risk_free_rate=risk_free_rate,
                option_type="put"
            )
            results[strike_pct]["value"].append(current_value)
            results[strike_pct]["pnl"].append(current_value - initial_costs[idx])

    sim_period["rolling_vol"] = vol_history
    for strike_pct in strike_otm_pcts:
        sim_period[f"pnl_{strike_pct}"] = results[strike_pct]["pnl"]
        sim_period[f"value_{strike_pct}"] = results[strike_pct]["value"]

    # --- Calculate additional metrics for the plot ---
    # Calculate the 15-day SMA of rolling volatility
    sim_period['rolling_vol_15d_sma'] = sim_period['rolling_vol'].rolling(window=15).mean()

    # Calculate 15-day SMA of P&L for each strike
    for strike_pct in strike_otm_pcts:
        sim_period[f"pnl_{strike_pct}_15d_sma"] = sim_period[f"pnl_{strike_pct}"].rolling(window=15).mean()

    # --- Step 3: Report and Visualize the Results ---
    # --- Find P&L at max 15d SMA for each strike ---
    pnl_at_max_sma = []
    date_at_max_sma = []
    for p in strike_otm_pcts:
        max_sma_date = sim_period[f"pnl_{p}_15d_sma"].idxmax()
        pnl = sim_period.loc[max_sma_date, f"pnl_{p}"]
        if isinstance(pnl, pd.Series):
            pnl = pnl.iloc[0]
        pnl_at_max_sma.append(pnl)
        date_at_max_sma.append(max_sma_date.strftime('%Y-%m-%d'))

    print("\n--- Simulation Results (at Max 15d P&L SMA) ---")
    
    # Table 1: P&L per share
    print(f"{'Strike (% OTM)':<15} {'P&L ($)':>10} {'Profit (%)':>12} {'Date':>12}")
    for i, strike_pct in enumerate(strike_otm_pcts):
        pnl = pnl_at_max_sma[i]
        profit_perc = pnl / initial_costs[i] if initial_costs[i] > 0 else 0
        print(f"{strike_pct:<15.2%} {f'${pnl:.2f}':>10} {f'{profit_perc:.2%}':>12} {date_at_max_sma[i]:>12}")

    print() # Blank line

    # Table 2: Hedge position
    print(f"{'Strike (% OTM)':<15} {'Contracts':>10} {'Cost ($)':>12} {'Net Profit ($)':>15}")
    for i, strike_pct in enumerate(strike_otm_pcts):
        pnl = pnl_at_max_sma[i]
        num_contracts = contracts_to_buy[i]
        actual_cost = actual_costs[i]
        net_profit = (pnl * 100 * num_contracts)
        print(f"{strike_pct:<15.2%} {num_contracts:>10} {f'${actual_cost:.2f}':>12} {f'${net_profit:.2f}':>15}")


    # Create the plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    fig.suptitle(f"LEAPS Put Hedge Performance: {ticker} in {leap_date.year}", fontsize=16)

    # --- Add input parameters text box ---
    strikes_str = ", ".join([f"{p:.1%}" for p in strike_otm_pcts])
    costs_str = ", ".join([f"${c:.2f}" for c in initial_costs])

    setup_box = (
        f"Inputs:\n"
        f"  LEAP Date: {leap_date.strftime('%Y-%m-%d')}\n"
        f"  Index: {ticker}\n"
        f"  Strikes (OTM): [{strikes_str}]\n"
        f"  Initial Costs: [{costs_str}]\n"
    )

    # Build the results box string with P&L at max 15d SMA
    strike_lines = []
    contract_lines = []
    for i, strike_pct in enumerate(strike_otm_pcts):
        pnl = pnl_at_max_sma[i]
        date = date_at_max_sma[i]
        profit_perc = pnl / initial_costs[i] if initial_costs[i] > 0 else 0
        if isinstance(pnl, pd.Series):
            pnl = pnl.iloc[0]
        if isinstance(profit_perc, pd.Series):
            profit_perc = profit_perc.iloc[0]
        
        # --- New: Add contract and net profit details to plot ---
        num_contracts = contracts_to_buy[i]
        actual_cost = actual_costs[i]
        net_profit = (pnl * 100 * num_contracts)

        strike_lines.append(f"  Strike {strike_pct:.1%}: P&L ${pnl:.2f} ({profit_perc:.1%}) on {date}")
        contract_lines.append(f"    {num_contracts} contracts | Cost: ${actual_cost:.2f} | Net Profit: ${net_profit:.2f}")

    results_lines = strike_lines + [""] + contract_lines

    results_box = (
        f"Results (at Max 15d P&L SMA):\n"
        + "\n".join(results_lines)
    )
    textstr = f"{setup_box}\n{results_box}"

    fig.text(0.01, 0.98, textstr, transform=fig.transFigure, fontsize=6,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    # Plot 1: Underlying Index Price
    ax1.plot(sim_period.index, sim_period["Close"], label=f"{ticker} Price", color="navy")
    ax1.set_ylabel("Index Price")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Plot 2: Rolling Volatility
    ax2.plot(sim_period.index, sim_period["rolling_vol"] * 100, label="60-Day Rolling Volatility", color="purple")
    ax2.axhline(initial_implied_vol * 100, color='red', linestyle='--', label=f'Initial Vol ({initial_implied_vol:.2%})')
    ax2.plot(sim_period.index, sim_period["rolling_vol_15d_sma"] * 100, label="15-Day SMA of Volatility", linestyle="--", color="green")
    ax2.set_ylabel("Volatility (%)", fontsize=8)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.legend(loc="upper left", fontsize=6)
    ax2.grid(True, linestyle="--", alpha=0.6)

    # Plot 3: P&L of the Hedge
    for strike_pct in strike_otm_pcts:
        ax3.plot(sim_period.index, sim_period[f"pnl_{strike_pct}"], label=f"Hedge P&L ({strike_pct:.1%} Strike)", alpha=0.7)
    for strike_pct in strike_otm_pcts:
        ax3.plot(sim_period.index, sim_period[f"pnl_{strike_pct}_15d_sma"], label=f"P&L 15d SMA ({strike_pct:.1%} Strike)", linestyle='--')

    ax3.axhline(0, color="black", linestyle="--")
    ax3.set_xlabel("Date", fontsize=8)
    ax3.set_ylabel("Profit/Loss ($)", fontsize=8)
    ax3.set_yscale("symlog")
    ax3.tick_params(axis='both', labelsize=8)
    ax3.legend(loc="upper left", fontsize=6)
    ax3.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0.02, 0, 1, 0.96])
    
    # Generate a unique filename
    index_name = ticker.strip('^')
    plot_filename = f"leap_hedge_{index_name}_{leap_date.year}.png"
    plt.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")
    plt.show()


def main():
    """Main function to parse arguments and run the simulation."""
    parser = argparse.ArgumentParser(description="Run a historical LEAPS hedge simulation.")
    parser.add_argument(
        "--leap-date",
        type=str,
        default="2007-01-03",
        help="Start date for the LEAPS contract (e.g., '2007-01-03')."
    )
    parser.add_argument(
        "--index",
        type=str,
        default="gspc",
        choices=["gspc", "naz100"],
        help="Index to use: 'gspc' for S&P 500 or 'naz100' for NASDAQ 100."
    )
    parser.add_argument(
        "--strike-pct",
        type=float,
        nargs='+',
        default=[-0.15],
        help="List of OTM percentages for put strikes (e.g., -0.10 for 10%% OTM)."
    )
    parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.05,
        help="Risk-free rate for option pricing."
    )
    args = parser.parse_args()

    ticker_map = {
        "gspc": "^GSPC",
        "naz100": "^IXIC"
    }
    ticker = ticker_map[args.index]

    run_2007_leap_hedge_simulation(
        leap_date_str=args.leap_date,
        ticker=ticker,
        strike_otm_pcts=args.strike_pct,
        risk_free_rate=args.risk_free_rate
    )

if __name__ == "__main__":
    main()
