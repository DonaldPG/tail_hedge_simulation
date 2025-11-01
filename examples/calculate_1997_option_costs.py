#!/usr/bin/env python3
"""
Calculate expected 1997 option costs to validate simulation logic
"""
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from asymmetric_options import BlackScholesCalculator

def main():
    """Calculate expected LEAP costs for 1997-01-02 purchase"""
    print("=== 1997 LEAP Cost Validation ===")
    
    # Values from verification
    purchase_price = 1280.70
    volatility = 0.20  # 20% default
    risk_free_rate = 0.05
    time_to_expiry = 1.0  # 1 year LEAPS
    
    print(f"IXIC Price on 1997-01-02: ${purchase_price:.2f}")
    print(f"Using volatility: {volatility:.1%}")
    print(f"Risk-free rate: {risk_free_rate:.1%}")
    print()
    
    # Calculate costs for each strike
    strike_otm_pcts = [-0.25, -0.30, -0.35]
    total_budget = 309.93  # From simulation output
    
    print("Expected LEAP costs:")
    print("-" * 40)
    
    total_cost = 0
    for strike_pct in strike_otm_pcts:
        strike_price = purchase_price * (1 + strike_pct)
        
        cost_per_share = BlackScholesCalculator.option_price(
            spot=purchase_price,
            strike=strike_price,
            time_to_expiry=time_to_expiry,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            option_type="put"
        )
        
        cost_per_contract = cost_per_share * 100
        total_cost += cost_per_contract
        
        print(f"{strike_pct:.1%} OTM (${strike_price:.2f}): "
              f"${cost_per_share:.4f}/share = ${cost_per_contract:.2f}/contract")
    
    print("-" * 40)
    print(f"Total cost for 1 contract each: ${total_cost:.2f}")
    print(f"Available budget: ${total_budget:.2f}")
    print(f"Budget utilization: {total_cost/total_budget:.1%}")
    
    # Check what prices would be needed for huge profits
    print("\n=== Profit Analysis ===")
    print("For -35% OTM to be worth $2,363.63:")
    
    strike_35 = purchase_price * 0.65  # 832.45
    target_value = 2363.63 / 100  # Per share value needed
    
    print(f"Strike price: ${strike_35:.2f}")
    print(f"Would need put value of: ${target_value:.2f}/share")
    print(f"For ITM payout alone: IXIC would need to be below ${strike_35:.2f}")
    print(f"Actual low in 1997: IXIC never went below ~1200")
    print(f"This confirms the calculation is WRONG!")

if __name__ == "__main__":
    main()
