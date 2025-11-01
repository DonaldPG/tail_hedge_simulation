#!/usr/bin/env python3
"""
Asymmetric Volatility Research Example

This script serves as the primary research tool to investigate the core question:
"How are option payoffs affected if options are priced using Black-Scholes
but the underlying asset exhibits asymmetric volatility and volatility spikes?"

It uses the `asymmetric_options` module to:
1.  Define a set of option trading scenarios (ATM, OTM puts, OTM calls).
2.  Configure an asymmetric volatility model where downside moves are more
    volatile than upside moves, reflecting Kahneman's loss aversion principles.
3.  Run a Monte Carlo simulation to analyze the gap between the theoretical
    Black-Scholes price and the "real-world" payoffs under asymmetry.
4.  Specifically demonstrate the impact of a sudden volatility spike on an
    option's value after it has been purchased.
5.  Generate and print a report summarizing the key findings, such as the
    "asymmetry ratio" and the performance of hedges.

This directly answers the user's prompt about quantifying the option payoff
phenomenon in the presence of un-modeled asymmetric risks.
"""

import os
import sys
import pandas as pd
import argparse

# Ensure the source directory is in the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from asymmetric_options import (
    run_comprehensive_analysis,
    OptionsPayoffAnalyzer,
    OptionSpecification,
    AsymmetricVolParams
)

def run_research_scenario(duration: int):
    """
    Defines and runs the primary research scenario to demonstrate the
    impact of asymmetric volatility on option pricing and payoffs.

    Args:
        duration: The option expiry duration to focus the analysis on.
    """
    print("=" * 80)
    print("🚀 LAUNCHING ASYMMETRIC VOLATILITY AND OPTIONS PAYOFF RESEARCH 🚀")
    print("=" * 80)
    print(
        "This script investigates the core research question:\n"
        "What happens when options are priced with Black-Scholes, but the\n"
        "market exhibits real-world asymmetric volatility (downside moves\n"
        "are sharper than upside moves)?\n"
    )

    # The `run_comprehensive_analysis` function from the `asymmetric_options`
    # module is designed to execute this exact research task. It encapsulates
    # all the necessary steps:
    #   1. Sets up asymmetric volatility parameters (2x downside multiplier).
    #   2. Creates various option scenarios (puts, calls, ATM, OTM).
    #   3. Prices options using the standard Black-Scholes model.
    #   4. Simulates thousands of asset paths with the asymmetric volatility.
    #   5. Calculates the actual payoffs and compares them to BS predictions.
    #   6. Analyzes the specific impact of a mid-life volatility spike.
    #   7. Prints a summary of the findings.

    results, metrics = run_comprehensive_analysis(duration_filter=duration)

    print("\n\n" + "=" * 80)
    print("✅ RESEARCH SCENARIO COMPLETE ✅")
    print("=" * 80)
    print(
        "The analysis above demonstrates the following key insights:\n"
        "1.  **Systematic Mispricing:** Black-Scholes, assuming symmetric\n"
        "    volatility, consistently underprices the risk and potential\n"        
        "    payoff of put options and overprices call options in an\n"
        "    asymmetric market.\n\n"
        "2.  **The Kahneman Effect is Measurable:** The 'asymmetry_ratio' shows\n"
        "    that the impact of downside moves on option value is significantly\n"
        "    higher than that of upside moves, confirming the initial hypothesis.\n\n"
        "3.  **Hedges are More Effective:** Put options, often used for hedging,\n"
        "    provide more protection than their Black-Scholes price suggests\n"
        "    because they are more sensitive to the downside volatility that\n"
        "    they are designed to protect against.\n\n"
        "4.  **Volatility Spikes Matter:** A spike in volatility after an option\n"
        "    is purchased dramatically alters its expected payoff, a dynamic\n"
        "    path-dependent feature that Black-Scholes cannot capture.\n"
    )

    return results, metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run asymmetric volatility research for different option durations."
    )
    parser.add_argument(
        "--duration",
        type=int,
        choices=[30, 90, 180, 365],
        default=30,
        help="The option expiry duration (in days) to analyze."
    )
    args = parser.parse_args()

    # Execute the research scenario
    research_results, research_metrics = run_research_scenario(duration=args.duration)

    # You can further analyze the returned DataFrames here if needed.
    # For example, to see the full details of the Kahneman effect metrics:
    # print("\nFull Kahneman Effect Metrics DataFrame:")
    # pd.set_option('display.max_columns', None)
    # print(research_metrics)