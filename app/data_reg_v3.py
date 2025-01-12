import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import yfinance as yf
import mysql.connector
import json
import re
import logging
from dateutil.relativedelta import relativedelta
from urllib.parse import quote_plus
from sqlalchemy import create_engine, inspect, text
from app.utils.data.data_service import DataService
from app.utils.analysis.analysis_service import AnalysisService
# Configure logging
logging.basicConfig(level=logging.WARNING)


    
def calculate_period_score(data, period_years):
    """Calculate score for a specific time period."""
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
    """Parse the TypeScript tickers file content."""
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
                        
                        if len(ticker_objects) >= limit:
                            print(f"\nLimited to first {limit} tickers")
                            return ticker_objects
                            
                current_object = ""
                
        return ticker_objects
    except Exception as e:
        print(f"Error parsing tickers file: {e}")
        return []


def analyze_tickers(tickers_file, data_service, end_date=None, limit=1000):
    """
    Analyze tickers and store results in DataFrame.
    
    Parameters:
    tickers_file (str): Path to the file containing ticker symbols
    data_service (DataService): Service for fetching historical data
    end_date (str or datetime, optional): End date for analysis. Defaults to current date.
    limit (int, optional): Maximum number of tickers to analyze. Defaults to 1000.
    """
    # Handle end_date parameter
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    
    # Format dates
    end_date_str = end_date.strftime('%Y-%m-%d')
    start_date = (end_date - timedelta(days=11*365)).strftime('%Y-%m-%d')  # 11 years before end_date
    
    with open(tickers_file, 'r') as f:
        content = f.read()
        tickers_data = parse_tickers_file(content, limit)
    
    if not tickers_data:
        raise ValueError("No tickers found in file")
    
    results = []
    total_tickers = len(tickers_data)
    time_periods = [1, 2, 3, 5, 10]  # Years to analyze
    
    # Define base weights for general score calculation
    base_weights = {
        1: 0.10,   # 10% weight for 1-year score
        2: 0.15,   # 15% weight for 2-year score
        3: 0.20,   # 20% weight for 3-year score
        5: 0.25,   # 25% weight for 5-year score
        10: 0.30   # 30% weight for 10-year score
    }
    
    for idx, ticker_info in enumerate(tickers_data, 1):
        ticker = ticker_info['symbol']
        name = ticker_info['name']
        data = data_service.get_historical_data(ticker, start_date, end_date_str)
        
        if data is None or data.empty:
            print(f"No data found for {ticker}")
            continue
        
        # Get latest price date
        latest_date = data.index[-1]
        
        # Calculate one year forward from latest date
        forward_date = latest_date + pd.DateOffset(years=1)
        
        # Get forward data for a full year after latest_date
        forward_date = latest_date + pd.DateOffset(years=1)
        # Get fresh data for the forward period
        forward_data = data_service.get_historical_data(ticker, latest_date.strftime('%Y-%m-%d'), forward_date.strftime('%Y-%m-%d'))
        
        # Calculate forward return
        if forward_data is not None and not forward_data.empty:
            start_price = data['Close'].iloc[-1]  # Price at latest_date
            end_price = forward_data['Close'].iloc[-1]  # Last available price in forward period
            end_date_actual = forward_data.index[-1]  # Actual last date in forward data
            forward_return = (end_price / start_price - 1) * 100
            days_forward = (end_date_actual - latest_date).days
        else:
            forward_return = None
            days_forward = 0
        
        stock_result = {
            'ticker': ticker,
            'name': name,
            'latest_price': round(data['Close'].iloc[-1], 2),
            'latest_date': latest_date.strftime('%Y-%m-%d'),
            'forward_return': round(forward_return, 2) if forward_return is not None else None,
            'days_forward': days_forward  # Add this to show how many days of forward data we have
        }
        
        # Calculate scores for each time period
        period_scores = {}
        for years in time_periods:
            scores = calculate_period_score(data, years)
            if scores:
                prefix = f"{years}y_"
                for key, value in scores.items():
                    stock_result[prefix + key] = value
                period_scores[years] = scores['score']
        
        # Calculate general score with available periods
        if period_scores:
            available_weights = {year: base_weights[year] 
                              for year in period_scores.keys()}
            
            weight_sum = sum(available_weights.values())
            normalized_weights = {year: weight/weight_sum 
                               for year, weight in available_weights.items()}
            
            general_score = sum(period_scores[year] * normalized_weights[year]
                              for year in period_scores.keys())
            
            stock_result['general_score'] = general_score
            stock_result['analyzed_periods'] = '+'.join(str(y) for y in sorted(period_scores.keys()))
            stock_result['available_periods'] = len(period_scores)
            
            base_rating = (
                'Excellent' if general_score >= 90 else
                'Very Good' if general_score >= 75 else
                'Good' if general_score >= 65 else
                'Fair' if general_score >= 40 else
                'Poor'
            )
            
            if len(period_scores) == len(time_periods):
                confidence = "High"
            elif len(period_scores) >= 3:
                confidence = "Medium"
            else:
                confidence = "Low"
            
            stock_result['general_rating'] = f"{base_rating} ({confidence} Confidence)"
        
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
    
    # Sort by general score (if exists) and available periods
    if 'general_score' in df.columns:
        df = df.sort_values(['available_periods', 'general_score'], 
                          ascending=[False, False])
    
    return df, time_periods

def save_to_excel(results_df, time_periods,tickers_file,output_file='stock_regression_results.xlsx',end_date="2023-01-11"):
    """Save results to Excel with formatting."""
    try:
        try:
            import xlsxwriter
            output_file = f"{end_date}_{tickers_file}_stock_regression_results.xlsx"
            writer = pd.ExcelWriter(output_file, engine='xlsxwriter')
            
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
            
            # Add price format
            price_format = workbook.add_format({
                'num_format': '$#,##0.00',
                'align': 'right'
            })
            
            # Update base_formats to include latest_price
            base_formats = {
                'ticker': {'width': 12, 'format': text_format},
                'name': {'width': 30, 'format': text_format},
                'latest_price': {'width': 12, 'format': price_format},  # Add this line
                'general_score': {'width': 12, 'format': score_format},
                'general_rating': {'width': 12, 'format': text_format}
            }
                
            # Set column widths and formats
          
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
            score_columns = ['general_score']
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
            
        except ImportError:
            try:
                print("xlsxwriter not found, using openpyxl with basic formatting...")
                writer = pd.ExcelWriter(output_file, engine='openpyxl')
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
                print(f"Results saved with basic formatting to {output_file}")
                
            except ImportError:
                print("Excel writers not found. Please install xlsxwriter or openpyxl.")
                return
    
    except Exception as e:
        print(f"Error saving to Excel: {e}")
        return

import pandas as pd

def insert_last_n_columns_at_position_reindex(df, n, m):
    """
    Inserts the last n columns of the DataFrame into the m-th position using reindex.
    
    Parameters:
    df (pd.DataFrame): The original DataFrame.
    n (int): Number of columns to move from the end.
    m (int): The position index where the columns should be inserted (0-based).
    
    Returns:
    pd.DataFrame: A new DataFrame with reordered columns.
    """
    # Validate inputs
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if not isinstance(n, int) or not isinstance(m, int):
        raise TypeError("n and m must be integers.")
    if n <= 0:
        raise ValueError("n must be a positive integer.")
    if m < 0:
        raise ValueError("m must be a non-negative integer.")
    
    total_columns = len(df.columns)
    
    # Handle cases where n exceeds the number of columns
    if n > total_columns:
        raise ValueError(f"n ({n}) cannot be greater than the number of columns ({total_columns}).")
    
    # Handle cases where m exceeds the number of columns after removal
    if m > total_columns - n:
        raise ValueError(f"m ({m}) is too large after removing the last n ({n}) columns.")
    
    # Get list of columns
    cols = df.columns.tolist()
    
    # Extract last n columns
    last_n = cols[-n:]
    
    # Remove last n columns
    remaining = cols[:-n]
    
    # Insert last n columns at position m
    new_order = remaining[:m] + last_n + remaining[m:]
    
    # Reindex the DataFrame with the new column order
    return df.reindex(columns=new_order)

    # Configure logging
    # test_fetch.py

# Main execution
if __name__ == "__main__":
    try:
       
        # Initialize DataService
        # Initialize service and get some data
        # data_service = DataService()

        # Try to get S&P 500 data for a recent period
        # data_service = DataService()

# Get Apple data for the same period as S&P 500
        # start_date = "2015-01-01"
        # end_date = "2024-01-11"
        # df = data_service.get_historical_data("AAPL", start_date, end_date)
        # exit()
# Show table again to see if it was created
        end_date = "2023-01-11"
        data_service = DataService()
        
        tickers_file = "US_tickers.ts"
        results_df, time_periods = analyze_tickers(tickers_file, data_service, limit=1000,end_date=end_date)
        results_df = insert_last_n_columns_at_position_reindex(results_df, 9, 6)
        
        
        # Save results to Excel
        save_to_excel(results_df, time_periods,tickers_file,end_date=end_date)
        
        # Print summary
        print("\nAnalysis Summary:")
        print(f"Total stocks analyzed: {len(results_df)}")
        
        # Print ratings distribution for each time period
        for years in time_periods:
            rating_col = f"{years}y_rating"
            if rating_col in results_df.columns:
                print(f"\n{years}-Year Rating Distribution:")
                print(results_df[rating_col].value_counts())
        
        # Set float format for display
        pd.set_option('display.float_format', lambda x: '{:.4f}'.format(x))
        
        # Print top performers for each time period
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
        
    except Exception as e:
        print(f"Error in main execution: {e}")
    
    finally:
        # Dispose of the engine if it exists
        if 'data_service' in locals() and hasattr(data_service, 'engine'):
            data_service.engine.dispose()