"""
Stock Dividend Scraping Module

This module reads stock data from an Excel spreadsheet, scrapes current stock
prices and dividend information from Yahoo Finance, and updates the spreadsheet
with verification results.

Dependencies:
    - yfinance: For stock data scraping
    - openpyxl: For Excel file manipulation
    - pandas: For data handling

Author: GitHub Copilot
Date: October 17, 2025
"""

import logging
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Optional
from openpyxl import load_workbook
from pathlib import Path


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockDividendScraper:
    """
    A comprehensive class for scraping stock prices and dividend data
    and updating Excel spreadsheets with verification results.
    """
    
    def __init__(self, excel_file_path: str):
        """
        Initialize the scraper with the Excel file path.
        
        Args:
            excel_file_path (str): Path to the Excel file containing stock data
        """
        self.excel_file_path = Path(excel_file_path)
        self.stock_symbols = []
        self.expected_prices = []
        self.start_date = "2024-12-24"
        self.end_date = "2025-01-01"
        self.verification_date = "2024-12-24"
        
        if not self.excel_file_path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_file_path}")
    
    def read_excel_data(self) -> None:
        """
        Read stock symbols and expected closing prices from the Excel file.
        Extracts data from Column A (symbols) and Column C (prices).
        """
        try:
            logger.info(f"Reading Excel file: {self.excel_file_path}")
            
            # Read the Excel file
            df = pd.read_excel(self.excel_file_path)
            
            # Extract stock symbols from Column A (assuming first column)
            self.stock_symbols = df.iloc[:, 0].dropna().tolist()
            
            # Extract expected closing prices from Column C (assuming third column)
            raw_prices = df.iloc[:, 2].dropna().tolist()
            
            # Clean and convert prices to float, handling any string issues
            self.expected_prices = []
            for price in raw_prices:
                try:
                    # Handle various string formats that might appear
                    if isinstance(price, str):
                        # Remove any non-numeric characters except decimal point
                        cleaned_price = ''.join(c for c in price if c.isdigit() or c == '.')
                        if cleaned_price:
                            self.expected_prices.append(float(cleaned_price))
                        else:
                            logger.warning(f"Could not parse price: {price}, skipping")
                            continue
                    else:
                        self.expected_prices.append(float(price))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert price {price} to float: {e}, skipping")
                    continue
            
            # Ensure both lists have the same length
            min_length = min(len(self.stock_symbols), len(self.expected_prices))
            self.stock_symbols = self.stock_symbols[:min_length]
            self.expected_prices = self.expected_prices[:min_length]
            
            logger.info(f"Successfully read {len(self.stock_symbols)} stock symbols")
            print(f"Stock symbols loaded: {self.stock_symbols}")
            print(f"Expected prices loaded: {self.expected_prices}")
            
        except Exception as e:
            logger.error(f"Error reading Excel file: {str(e)}")
            raise
    
    def get_stock_data(self, symbol: str) -> Tuple[Optional[float], Optional[float], Optional[float], List[Tuple[str, float]]]:
        """
        Scrape stock price and dividend data for a given symbol from Yahoo Finance.
        
        Args:
            symbol (str): Stock ticker symbol
            
        Returns:
            Tuple[Optional[float], Optional[float], Optional[float], List[Tuple[str, float]]]: 
                - Closing price on verification date (or None if not found)
                - Lowest price in range 12/24-12/26 (or None if not found)
                - Highest price in range 12/24-12/26 (or None if not found)
                - List of (date, amount) tuples for dividends in the date range
        """
        try:
            logger.info(f"Fetching data for {symbol}")
            
            # Clean symbol (remove trailing spaces)
            symbol = symbol.strip()
            
            # Create yfinance Ticker object
            ticker = yf.Ticker(symbol)
            
            # Get historical price data for verification date and surrounding days
            price_range_start = "2024-12-24"
            price_range_end = "2024-12-27"  # End date is exclusive, so this gets through 12/26
            
            hist_data = ticker.history(
                start=price_range_start, 
                end=price_range_end,
                interval="1d"
            )
            
            # Get closing price on verification date and price range
            verification_price = None
            min_price = None
            max_price = None
            
            if not hist_data.empty:
                # Convert verification date to timezone-aware datetime
                verification_dt = pd.to_datetime(self.verification_date)
                
                # Make verification_dt timezone-aware to match hist_data index
                if hist_data.index.tz is not None:
                    verification_dt = verification_dt.tz_localize(hist_data.index.tz)
                
                # Get verification date price
                for idx in hist_data.index:
                    if pd.to_datetime(idx).date() == verification_dt.date():
                        verification_price = hist_data.loc[idx, "Close"]
                        break
                
                # Calculate min and max prices for the range (12/24-12/26)
                if len(hist_data) > 0:
                    min_price = hist_data["Low"].min()
                    max_price = hist_data["High"].max()
            
            # Get dividend data for the original date range
            dividends = ticker.dividends
            
            # Filter dividends for our date range and capture dates
            dividend_data = []
            if not dividends.empty:
                # Convert dates to timezone-aware format for comparison
                start_dt = pd.to_datetime(self.start_date)
                end_dt = pd.to_datetime(self.end_date)
                
                # Make timezone-aware if dividends have timezone info
                if dividends.index.tz is not None:
                    start_dt = start_dt.tz_localize(dividends.index.tz)
                    end_dt = end_dt.tz_localize(dividends.index.tz)
                
                # Filter dividends in date range and store date/amount pairs
                for div_date, div_amount in dividends.items():
                    div_dt = pd.to_datetime(div_date)
                    if start_dt <= div_dt < end_dt:
                        # Format date as string for easier handling
                        date_str = div_dt.strftime("%Y-%m-%d")
                        dividend_data.append((date_str, div_amount))
            
            logger.info(f"{symbol}: Price={verification_price}, Range=[{min_price}-{max_price}], Dividends={dividend_data}")
            return verification_price, min_price, max_price, dividend_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None, None, None, []
    
    def verify_prices_and_dividends(self) -> Dict[str, Dict]:
        """
        Verify closing prices and collect dividend information for all stocks.
        
        Returns:
            Dict[str, Dict]: Dictionary with verification results for each symbol
        """
        results = {}
        
        print("Starting price verification and dividend collection...")
        print("Price matching logic: Expected price must fall within 12/24-12/26 range")
        
        for i, symbol in enumerate(self.stock_symbols):
            expected_price = self.expected_prices[i]
            
            print(f"Processing {symbol} (Expected price: ${expected_price:.2f})")
            
            # Get actual data from Yahoo Finance
            actual_price, min_price, max_price, dividend_data = self.get_stock_data(symbol)
            
            # Determine price match status using range-based logic
            match_status = "No data available"
            if min_price is not None and max_price is not None:
                # Check if expected price falls within the range of 12/24-12/26
                if min_price <= expected_price <= max_price:
                    match_status = "Match"
                else:
                    match_status = "Mismatch"
            elif actual_price is not None:
                # Fallback to exact price comparison if range data unavailable
                if abs(actual_price - expected_price) <= 0.01:
                    match_status = "Match"
                else:
                    match_status = "Mismatch"
            
            # Process dividend information
            dividend_summary = "No dividends"
            dividend_dates_list = ""
            dividend_amounts_list = ""
            total_dividends = 0.0
            
            if dividend_data:
                num_payments = len(dividend_data)
                dividend_summary = f"Dividends in {num_payments} payments"
                
                # Extract dates and amounts
                dates = [date for date, amount in dividend_data]
                amounts = [amount for date, amount in dividend_data]
                
                dividend_dates_list = "; ".join(dates)
                dividend_amounts_list = "; ".join([f"${amount:.4f}" for amount in amounts])
                total_dividends = sum(amounts)
            
            results[symbol] = {
                "expected_price": expected_price,
                "actual_price": actual_price,
                "min_price": min_price,
                "max_price": max_price,
                "match_status": match_status,
                "dividend_data": dividend_data,
                "dividend_summary": dividend_summary,
                "dividend_dates_list": dividend_dates_list,
                "dividend_amounts_list": dividend_amounts_list,
                "total_dividends": total_dividends
            }
            
            print(f"  Result: {match_status}")
            if min_price is not None and max_price is not None:
                print(f"  Price range: ${min_price:.2f} - ${max_price:.2f}")
            print(f"  Dividends: {dividend_summary}")
            if dividend_data:
                print(f"  Total dividend: ${total_dividends:.4f}")
        
        return results
    
    def update_excel_file(self, results: Dict[str, Dict]) -> None:
        """
        Update the Excel file with verification results in separate columns.
        
        Args:
            results (Dict[str, Dict]): Verification results from verify_prices_and_dividends
        """
        try:
            logger.info("Updating Excel file with results")
            
            # Load the workbook
            workbook = load_workbook(self.excel_file_path)
            worksheet = workbook.active
            
            # Define column mappings (starting from column G)
            column_headers = {
                "G": "Price Match Status",
                "H": "Low Price (12/24-12/26)",
                "I": "High Price (12/24-12/26)", 
                "J": "Dividend Summary",
                "K": "Dividend Dates",
                "L": "Dividend Amounts",
                "M": "Total Dividends"
            }
            
            # Add headers in row 1
            for col, header in column_headers.items():
                worksheet[f"{col}1"] = header
            
            # Update data for each stock
            for i, symbol in enumerate(self.stock_symbols):
                row_num = i + 2  # Data starts in row 2
                
                if symbol in results:
                    result = results[symbol]
                    
                    # Column G: Match/Mismatch status
                    worksheet[f"G{row_num}"] = result["match_status"]
                    
                    # Column H: Low price in range
                    if result["min_price"] is not None:
                        worksheet[f"H{row_num}"] = f"${result['min_price']:.2f}"
                    else:
                        worksheet[f"H{row_num}"] = "No data"
                    
                    # Column I: High price in range
                    if result["max_price"] is not None:
                        worksheet[f"I{row_num}"] = f"${result['max_price']:.2f}"
                    else:
                        worksheet[f"I{row_num}"] = "No data"
                    
                    # Column J: Dividend summary
                    worksheet[f"J{row_num}"] = result["dividend_summary"]
                    
                    # Column K: Dividend dates
                    worksheet[f"K{row_num}"] = result["dividend_dates_list"] if result["dividend_dates_list"] else ""
                    
                    # Column L: Dividend amounts
                    worksheet[f"L{row_num}"] = result["dividend_amounts_list"] if result["dividend_amounts_list"] else ""
                    
                    # Column M: Total dividends
                    if result["total_dividends"] > 0:
                        worksheet[f"M{row_num}"] = f"${result['total_dividends']:.4f}"
                    else:
                        worksheet[f"M{row_num}"] = "$0.0000"
                else:
                    # Handle empty rows - insert empty values
                    for col in ["G", "H", "I", "J", "K", "L", "M"]:
                        worksheet[f"{col}{row_num}"] = ""
            
            # Save the workbook
            workbook.save(self.excel_file_path)
            logger.info("Excel file updated successfully")
            print(f"Excel file updated with 7 new columns: {self.excel_file_path}")
            print("Columns added:")
            for col, header in column_headers.items():
                print(f"  Column {col}: {header}")
            
        except Exception as e:
            logger.error(f"Error updating Excel file: {str(e)}")
            raise
    
    def print_dividend_summary(self, results: Dict[str, Dict]) -> None:
        """
        Print a detailed, human-readable summary of dividend findings.
        
        Args:
            results (Dict[str, Dict]): Verification results to summarize
        """
        print("\n" + "=" * 60)
        print("DIVIDEND PAYMENT SUMMARY")
        print("=" * 60)
        
        # Collect dividend information
        dividend_stocks = {}
        for symbol, result in results.items():
            if result['dividends']:
                dividend_stocks[symbol] = result
        
        if not dividend_stocks:
            print("❌ No dividend payments found in the period 12/24/2024 to 1/1/2025")
            return
        
        print(f"✅ Found {len(dividend_stocks)} stocks with dividend payments:")
        print()
        
        total_dividend_value = 0
        
        for symbol, result in dividend_stocks.items():
            print(f"📈 {symbol.upper()}")
            print(f"   Expected Price: ${result['expected_price']:.2f}")
            print(f"   Price Status: {result['price_match']}")
            print(f"   Dividend Summary: {result['dividend_info']}")
            print(f"   Payment Details: {result['dividend_dates']}")
            
            # Calculate total dividend value for this stock
            stock_total = sum(amount for date, amount in result['dividends'])
            total_dividend_value += stock_total
            print()
        
        print("-" * 40)
        print(f"💰 TOTAL DIVIDEND VALUE: ${total_dividend_value:.4f}")
        print(f"📅 Period Analyzed: December 24, 2024 - January 1, 2025")
        print(f"📊 Average dividend per paying stock: ${total_dividend_value/len(dividend_stocks):.4f}")
        
        # Show breakdown by payment date
        all_payments = []
        for result in dividend_stocks.values():
            for date, amount in result['dividends']:
                all_payments.append((date, amount))
        
        if all_payments:
            print("\n📋 CHRONOLOGICAL DIVIDEND SCHEDULE:")
            # Sort by date
            all_payments.sort(key=lambda x: x[0])
            for date, amount in all_payments:
                print(f"   {date}: ${amount:.4f}")

    def run_complete_analysis(self) -> Dict[str, Dict]:
        """
        Run the complete analysis workflow:
        1. Read Excel data
        2. Verify prices and collect dividends
        3. Update Excel file with results
        
        Returns:
            Dict[str, Dict]: Complete verification results
        """
        print("=" * 60)
        print("STOCK DIVIDEND SCRAPING ANALYSIS")
        print("=" * 60)
        
        # Step 1: Read Excel data
        self.read_excel_data()
        
        # Step 2: Verify prices and collect dividends
        results = self.verify_prices_and_dividends()
        
        # Step 3: Update Excel file
        self.update_excel_file(results)
        
        # Step 4: Print summaries
        self.print_summary(results)
        self.print_dividend_summary(results)
        
        return results
    
    def print_summary(self, results: Dict[str, Dict]) -> None:
        """
        Print a summary of the analysis results.
        
        Args:
            results (Dict[str, Dict]): Verification results to summarize
        """
        print("\n" + "=" * 60)
        print("ANALYSIS SUMMARY")
        print("=" * 60)
        
        total_stocks = len(results)
        
        # Count different types of matches using updated variable names
        matches = sum(1 for r in results.values() 
                     if r["match_status"] == "Match")
        mismatches = sum(1 for r in results.values() 
                        if r["match_status"] == "Mismatch")
        no_data_count = sum(1 for r in results.values() 
                           if r["match_status"] == "No data available")
        
        stocks_with_dividends = sum(1 for r in results.values() 
                                   if r["dividend_data"])
        
        print(f"Total stocks analyzed: {total_stocks}")
        print(f"Price matches: {matches}/{total_stocks}")
        print(f"Price mismatches: {mismatches}/{total_stocks}")
        print(f"No data available: {no_data_count}/{total_stocks}")
        print(f"Stocks with dividends: {stocks_with_dividends}/{total_stocks}")
        
        print(f"\nPrice verification logic: Expected price within 12/24-12/26 range")
        print(f"Dividend search period: {self.start_date} to {self.end_date}")
        
        # Show stocks with dividends
        if stocks_with_dividends > 0:
            print(f"\nStocks with dividends in period:")
            for symbol, result in results.items():
                if result["dividend_data"]:
                    print(f"  {symbol}: {result['dividend_summary']}")
        
        # Show some example verification results
        print(f"\nExample verification results:")
        count = 0
        for symbol, result in results.items():
            if count >= 3:  # Show first 3 examples
                break
            status = result["match_status"]
            if result["min_price"] and result["max_price"]:
                print(f"  {symbol}: {status} (Range: ${result['min_price']:.2f}-${result['max_price']:.2f})")
            else:
                print(f"  {symbol}: {status}")
            count += 1


def main():
    """
    Main function to run the stock dividend scraping analysis.
    """
    # File path (modify as needed)
    excel_file_path = "/Users/donaldpg/demo_input_dividend_search.xlsx"
    
    try:
        # Create scraper instance
        scraper = StockDividendScraper(excel_file_path)
        
        # Run complete analysis
        results = scraper.run_complete_analysis()
        
        print("\n✅ Analysis completed successfully!")
        
        return results
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        logger.error(f"File not found: {e}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        logger.error(f"Analysis error: {e}")
        raise


if __name__ == "__main__":
    main()