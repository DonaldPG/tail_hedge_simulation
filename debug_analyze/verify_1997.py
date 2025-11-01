import yfinance as yf
import pandas as pd
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from asymmetric_options import BlackScholesCalculator

# Fetch IXIC data around the 1997 period
print("Fetching IXIC data for 1997 verification...")
data = yf.download("^IXIC", start="1996-12-01", end="1997-05-01", progress=False)

# Key dates from the example
purchase_date = "1997-01-02"
sale_date_35 = "1997-03-20"
sale_date_25_30 = "1997-04-23"

print("IXIC Price Analysis for 1997 LEAP Example")
print("=" * 50)

purchase_price = float(data.loc[purchase_date, "Close"])
sale_price_35 = float(data.loc[sale_date_35, "Close"])
sale_price_25_30 = float(data.loc[sale_date_25_30, "Close"])

print(f"Purchase date ({purchase_date}): ${purchase_price:.2f}")
print(f"Sale date 35% OTM ({sale_date_35}): ${sale_price_35:.2f}")
print(f"Sale date 25%/30% OTM ({sale_date_25_30}): ${sale_price_25_30:.2f}")
print()

# Calculate strike prices
strike_25 = purchase_price * 0.75  # 25% OTM
strike_30 = purchase_price * 0.70  # 30% OTM  
strike_35 = purchase_price * 0.65  # 35% OTM

print("Strike prices at purchase:")
print(f"  25% OTM: ${strike_25:.2f}")
print(f"  30% OTM: ${strike_30:.2f}")
print(f"  35% OTM: ${strike_35:.2f}")
print()

# Check price changes
change_35 = (sale_price_35 / purchase_price - 1) * 100
change_25_30 = (sale_price_25_30 / purchase_price - 1) * 100
print("Price changes:")
print(f"  {purchase_date} to {sale_date_35}: {change_35:.1f}%")
print(f"  {purchase_date} to {sale_date_25_30}: {change_25_30:.1f}%")
print()

# Check if strikes were in-the-money
in_money_35 = sale_price_35 < strike_35
in_money_30 = sale_price_25_30 < strike_30
in_money_25 = sale_price_25_30 < strike_25

print("Strike vs Market on sale dates:")
print(f"  35% OTM strike ${strike_35:.2f} vs market ${sale_price_35:.2f} - In money: {in_money_35}")
print(f"  30% OTM strike ${strike_30:.2f} vs market ${sale_price_25_30:.2f} - In money: {in_money_30}")
print(f"  25% OTM strike ${strike_25:.2f} vs market ${sale_price_25_30:.2f} - In money: {in_money_25}")
print()

print("CONCLUSION:")
if not (in_money_35 or in_money_30 or in_money_25):
    print("❌ PROBLEM: None of the puts were in-the-money!")
    print("   The reported profits appear to be calculation errors.")
else:
    print("✅ Some puts were in-the-money, profits could be legitimate.")
