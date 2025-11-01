# Project Summary: Tail Hedge Simulation Analysis

## 1. Core Research Question

This project is designed to investigate and demonstrate a key thesis: **Standard options pricing models, like Black-Scholes, systematically undervalue protective put options (hedges) because they fail to account for the asymmetric nature of market volatility.**

Specifically, it addresses:
- What is the real-world performance of a long-term option hedge (LEAPS) purchased in a low-volatility environment that subsequently experiences a major market crisis?
- How does the actual profit and loss (P&L) of the hedge evolve as market volatility spikes, and how does this compare to the initial, "static" price calculated by Black-Scholes?
- Can we model a practical scenario to show that the value of a hedge is not just in its final expiration payoff, but in its changing value during periods of market stress?

## 2. Key Files and Their Purpose

### `examples/real_world_leap_hedge_2007.py`

This is the main driver script that runs a historical case study.

**What it does:**
It simulates the purchase of a long-term, out-of-the-money (OTM) LEAPS put option on a major index (e.g., NASDAQ 100) at the beginning of 2007—a period of relative calm before the Global Financial Crisis.

**How it answers the core questions:**
1.  **Fetches Historical Data:** It pulls real market data for the chosen index from 2006 to 2008.
2.  **Calculates Initial Cost:** It uses the `BlackScholesCalculator` to determine the "fair" price of the LEAPS put on the purchase date. Crucially, it uses the low, backward-looking volatility from late 2006 as the input for this calculation, mimicking how a real-world buyer would price the option at that time.
3.  **Simulates Day-by-Day:** It iterates through every trading day of 2007, re-calculating the option's value based on:
    - The actual closing price of the index for that day.
    - A rolling 60-day historical volatility, which serves as a proxy for the evolving implied volatility in the market. This is the key step that introduces dynamic, real-world volatility into the model.
4.  **Tracks and Visualizes P&L:** It calculates the daily P&L of the hedge (current value minus initial cost) and generates a comprehensive plot showing:
    - The index price over time.
    - The rolling volatility, clearly illustrating the spike during the crisis.
    - The P&L of the hedge, demonstrating how its value increases dramatically as the market falls and volatility rises.
5.  **Identifies Optimal Exit:** Instead of just showing the final P&L at expiration, the script identifies the date when the hedge's value was at its peak (using a 15-day moving average of the P&L), providing insight into optimal profit-taking.

### `src/asymmetric_options.py`

This file provides the foundational financial modeling tools used by the simulation.

**What it contains:**
-   **`OptionSpecification`**: A simple data class that defines the parameters of an option contract (strike, expiry, type, etc.), keeping the code clean and organized.
-   **`BlackScholesCalculator`**: A standard implementation of the Black-Scholes formula for pricing European options. In the context of this project, it serves as the "naïve" model. The simulation script uses it to price the option under the assumption of a static volatility environment, thereby setting up the core conflict: a static model versus a dynamic reality.

---

## 3. High-Level Summary

In essence, the `real_world_leap_hedge_2007.py` script uses the `BlackScholesCalculator` to "buy" a theoretical hedge in a historical context. It then tracks the hedge's performance through the 2007-2008 market turmoil, demonstrating that the hedge's value grew far beyond what its initial, low-volatility cost would have suggested. This provides a concrete, data-driven example that validates the project's central thesis about the mispricing of options in the face of asymmetric, real-world volatility.
