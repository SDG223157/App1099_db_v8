import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import yfinance as yf
import json
import re
import logging
from app.utils.analysis.analysis_service import AnalysisService
from app.utils.data.data_service import DataService

# Configure logging to suppress debug messages
logging.getLogger('yfinance').setLevel(logging.WARNING)
yf.pdr_override()

def clean_ticker_for_table_name(ticker):
    """Clean ticker symbol for use in table name."""
    cleaned = ''.join(c if c.isalnum() else '_' for c in ticker)
    cleaned = cleaned.strip('_').lower()
    return cleaned if cleaned else 'unknown'

def get_stock_data(ticker):
    """Get historical stock data from yfinance."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=11*365)  # Get 11 years to ensure enough data
        
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=start_date, end=end_date)
        
        if df.empty:
            return None
            
        df.index = df.index.tz_localize(None)
        return df
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def calculate_period_score(data, period_years):
    """Calculate score for a specific time period using AnalysisService."""
    try:
        if len(data) < 50:  # Minimum data points required
            return None
            
        end_date = data.index[-1]
        start_date = end_date - pd.DateOffset(years=period_years)
        period_data = data[data.index >= start_date].copy()
        
        if len(period_data) < 50:
            return None
            
        # Use AnalysisService to perform regression and calculate score
        regression_results = AnalysisService.perform_polynomial_regression(period_data)
        
        if regression_results['total_score']['score'] == 0:
            return None
            
        return {
            'score': float(regression_results['total_score']['score']),
            'trend_score': float(regression_results['total_score']['components']['trend']['score']),
            'return_score': float(regression_results['total_score']['components']['return']['score']),
            'volatility_score': float(regression_results['total_score']['components']['volatility']['score']),
            'annual_return': float(regression_results['total_score']['components']['return']['value']),
            'annual_volatility': float(regression_results['total_score']['components']['volatility']['value']),
            'r2': float(regression_results['r2']),
            'quad_coef': float(regression_results['coefficients'][2]),
            'linear_coef': float(regression_results['coefficients'][1]),
            'raw_score': float(regression_results['total_score']['raw_score']),
            'scaling_factor': float(regression_results['total_score']['scaling']['factor']),
            'sp500_base': float(regression_results['total_score']['scaling']['sp500_base'])
        }
            
    except Exception as e:
        print(f"Error calculating {period_years}-year score: {e}")
        return None

def parse_tickers_file(content, limit=100):
    """
    Parse the TypeScript tickers file content.
    
    Args:
        content: Content of the tickers.ts file
        limit: Maximum number of tickers to return (default: 100)
    """
    try:
        matches = re.search(r'export const tickers = \[(.*?)\];', content, re.DOTALL)
        if not matches:
            raise ValueError("Could not find tickers array in file")
            
        array_content = matches.group(1)
        ticker_objects = []
        current_object = ""
        brace_count = 0
        
        for line in array_content.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            brace_count += line.count('{') - line.count('}')
            current_object += line
            
            if brace_count == 0 and current_object:
                obj_str = current_object.strip(' ,')
                if obj_str:
                    symbol_match = re.search(r'symbol:\s*["\']([^"\']+)["\']', obj_str)
                    name_match = re.search(r'name:\s*["\']([^"\']+)["\']', obj_str)
                    
                    if symbol_match and name_match:
                        ticker_objects.append({
                            'symbol': symbol_match.group(1),
                            'name': name_match.group(1)
                        })
                        
                        # Check if we've reached the limit
                        if len(ticker_objects) >= limit:
                            print(f"\nLimited to first {limit} tickers")
                            return ticker_objects
                            
                current_object = ""
                
        return ticker_objects
    except Exception as e:
        print(f"Error parsing tickers file: {e}")
        return []

def analyze_tickers(tickers_file, limit=100):
    """
    Analyze tickers and store results in DataFrame.
    
    Args:
        tickers_file: Path to tickers.ts file
        limit: Maximum number of tickers to analyze (default: 100)
    """
    with open(tickers_file, 'r') as f:
        content = f.read()
        tickers_data = parse_tickers_file(content, limit)
    
    if not tickers_data:
        raise ValueError("No tickers found in file")
    
    results = []
    total_tickers = len(tickers_data)
    time_periods = [1, 2, 3, 5, 10]  # Years to analyze
    
    for idx, ticker_info in enumerate(tickers_data, 1):
        ticker = ticker_info['symbol']
        name = ticker_info['name']
        
        print(f"Analyzing {ticker} ({name})... ({idx}/{total_tickers})")
        
        data = get_stock_data(ticker)
        if data is None or data.empty:
            print(f"No data available for {ticker}")
            continue
        
        stock_result = {
            'ticker': ticker,
            'name': name
        }
        
        # Calculate scores for each time period
        for years in time_periods:
            period_scores = calculate_period_score(data, years)
            if period_scores:
                prefix = f"{years}y_"
                for key, value in period_scores.items():
                    stock_result[prefix + key] = value
        
        results.append(stock_result)
    
    df = pd.DataFrame(results)
    
    # Add ratings for each time period
    for years in time_periods:
        score_col = f"{years}y_score"
        if score_col in df.columns:
            df[f"{years}y_rating"] = df[score_col].apply(lambda x: 
                'Excellent' if float(x) >= 90 else
                'Very Good' if float(x) >= 75 else
                'Good' if float(x) >= 65 else
                'Fair' if float(x) >= 40 else
                'Poor'
            )
    
    return df, time_periods

def save_to_excel(results_df, time_periods):
    """Save results to Excel with formatting."""
    try:
        print("\nSaving results to Excel...")
        
        try:
            import xlsxwriter
            excel_file = 'stock_regression_results.xlsx'
            writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
            
            results_df.to_excel(writer, sheet_name='Stock Analysis', index=False)
            
            workbook = writer.book
            worksheet = writer.sheets['Stock Analysis']
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'bg_color': '#D9E1F2',
                'border': 1
            })
            
            number_format = workbook.add_format({
                'num_format': '0.0000',
                'align': 'right'
            })
            
            percent_format = workbook.add_format({
                'num_format': '0.0000%',
                'align': 'right'
            })
            
            score_format = workbook.add_format({
                'num_format': '0.0000',
                'align': 'right'
            })
            
            text_format = workbook.add_format({
                'align': 'left'
            })
            
            # Write headers
            for col_num, value in enumerate(results_df.columns.values):
                worksheet.write(0, col_num, value, header_format)
            
            # Set column widths and formats
            base_formats = {
                'ticker': {'width': 12, 'format': text_format},
                'name': {'width': 30, 'format': text_format}
            }
            
            # Add formats for each time period
            for years in time_periods:
                prefix = f"{years}y_"
                base_formats.update({
                    prefix + 'score': {'width': 12, 'format': score_format},
                    prefix + 'trend_score': {'width': 12, 'format': score_format},
                    prefix + 'return_score': {'width': 12, 'format': score_format},
                    prefix + 'volatility_score': {'width': 12, 'format': score_format},
                    prefix + 'annual_return': {'width': 15, 'format': percent_format},
                    prefix + 'annual_volatility': {'width': 15, 'format': percent_format},
                    prefix + 'r2': {'width': 12, 'format': number_format},
                    prefix + 'quad_coef': {'width': 15, 'format': number_format},
                    prefix + 'linear_coef': {'width': 15, 'format': number_format},
                    prefix + 'rating': {'width': 12, 'format': text_format},
                    prefix + 'raw_score': {'width': 12, 'format': number_format},
                    prefix + 'scaling_factor': {'width': 12, 'format': number_format},
                    prefix + 'sp500_base': {'width': 12, 'format': number_format}
                })
            
            # Apply formats to columns
            for col_num, column in enumerate(results_df.columns):
                if column in base_formats:
                    worksheet.set_column(col_num, col_num, 
                                      base_formats[column]['width'],
                                      base_formats[column]['format'])
            
            # Add conditional formatting for scores
            score_columns = []
            for years in time_periods:
                score_columns.extend([
                    f"{years}y_score",
                    f"{years}y_trend_score",
                    f"{years}y_return_score",
                    f"{years}y_volatility_score"
                ])
            
            for col_name in score_columns:
                if col_name in results_df.columns:
                    col_idx = results_df.columns.get_loc(col_name)
                    worksheet.conditional_format(1, col_idx, len(results_df), col_idx, {
                        'type': '3_color_scale',
                        'min_color': "#FF9999",
                        'mid_color': "#FFFF99",
                        'max_color': "#99FF99"
                    })
            
            writer.close()
            print(f"Results saved with full formatting to {excel_file}")
            
        except ImportError:
            try:
                print("xlsxwriter not found, using openpyxl with basic formatting...")
                excel_file = 'stock_regression_results.xlsx'
                writer = pd.ExcelWriter(excel_file, engine='openpyxl')
                results_df.to_excel(writer, sheet_name='Stock Analysis', index=False)
                
                workbook = writer.book
                worksheet = writer.sheets['Stock Analysis']
                
                # Set column widths
                for column in worksheet.columns:
                    max_length = 0
                    column = [cell for cell in column]
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = (max_length + 2)
                    worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
                
                writer.close()
                print(f"Results saved with basic formatting to {excel_file}")
                
            except ImportError:
                print("Excel writers not available. Please install xlsxwriter or openpyxl.")
                return
    
    except Exception as e:
        print(f"Error saving to Excel: {e}")
        return

# Main execution
if __name__ == "__main__":
    tickers_file = "tickers.ts"
    results_df, time_periods = analyze_tickers(tickers_file, limit=1000)
    
    # Save results
    save_to_excel(results_df, time_periods)
    
    # Print summary
    print("\nAnalysis Summary:")
    print(f"Total stocks analyzed: {len(results_df)}")
    
    # Print ratings distribution for each time period
    for years in time_periods:
        rating_col = f"{years}y_rating"
        if rating_col in results_df.columns:
            print(f"\n{years}-Year Rating Distribution:")
            print(results_df[rating_col].value_counts())
    
    pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))
    for years in time_periods:
        score_col = f"{years}y_score"
        if score_col in results_df.columns:
            print(f"\nTop 10 stocks by {years}-Year Score:")
            print("="*120)
            
            # Create display DataFrame with selected columns
            display_cols = [
                'ticker', 'name', 
                f"{years}y_score", f"{years}y_rating",
                f"{years}y_annual_return", f"{years}y_annual_volatility",
                f"{years}y_r2", f"{years}y_raw_score", 
                f"{years}y_scaling_factor", f"{years}y_sp500_base"
            ]
            
            # Convert to numeric for sorting
            results_df[score_col] = pd.to_numeric(results_df[score_col], errors='coerce')
            
            # Get and format top 10
            top_10 = results_df.nlargest(10, score_col)[display_cols]
            
            # Rename columns for display
            display_names = {
                'ticker': 'Ticker',
                'name': 'Name',
                f"{years}y_score": 'Score',
                f"{years}y_rating": 'Rating',
                f"{years}y_annual_return": 'Annual Return',
                f"{years}y_annual_volatility": 'Volatility',
                f"{years}y_r2": 'R²',
                f"{years}y_raw_score": 'Raw Score',
                f"{years}y_scaling_factor": 'Scale Factor',
                f"{years}y_sp500_base": 'S&P500 Score'
            }
            
            top_10 = top_10.rename(columns=display_names)
            print(top_10.to_string(index=False))
            print("="*120)