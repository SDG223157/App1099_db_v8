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

# Configure logging
logging.basicConfig(level=logging.WARNING)

class DataService:
    def __init__(self):
        """Initialize DataService with database configuration"""
        self.connection = mysql.connector.connect(
            host='localhost',
            user='username',
            password='Gern@8280',
            database='my_database'
        )

    def clean_ticker_for_table_name(self, ticker: str) -> str:
        """Clean ticker symbol for use in table name."""
        cleaned = ''.join(c if c.isalnum() else '_' for c in ticker)
        cleaned = cleaned.strip('_').lower()
        return cleaned if cleaned else 'unknown'

    def table_exists(self, table_name: str) -> bool:
        """Check if table exists in database"""
        try:
            cursor = self.connection.cursor()
            cursor.execute("""
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_schema = %s
                AND table_name = %s
            """, (self.connection.database, table_name))
            exists = cursor.fetchone()[0] == 1
            cursor.close()
            return exists
        except Exception as e:
            print(f"Error checking table existence: {e}")
            return False

    def store_dataframe(self, df: pd.DataFrame, table_name: str) -> bool:
        """Store DataFrame in MySQL database"""
        try:
            cursor = self.connection.cursor()
            
            # Create table if it doesn't exist
            columns = []
            for col in df.columns:
                if df[col].dtype in ['float64', 'float32']:
                    columns.append(f"{col} DOUBLE")
                elif df[col].dtype == 'int64':
                    columns.append(f"{col} BIGINT")
                else:
                    columns.append(f"{col} VARCHAR(255)")
            
            create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    Date DATETIME,
                    {', '.join(columns)},
                    PRIMARY KEY (Date)
                )
            """
            cursor.execute(create_table_query)
            
            # Insert data
            placeholders = ', '.join(['%s'] * (len(df.columns) + 1))
            insert_query = f"""
                INSERT INTO {table_name} 
                (Date, {', '.join(df.columns)})
                VALUES ({placeholders})
                ON DUPLICATE KEY UPDATE
                {', '.join(f'{col} = VALUES({col})' for col in df.columns)}
            """
            
            values = []
            for idx, row in df.iterrows():
                values.append([idx] + row.tolist())
            
            cursor.executemany(insert_query, values)
            self.connection.commit()
            cursor.close()
            return True
            
        except Exception as e:
            print(f"Error storing DataFrame in table {table_name}: {e}")
            self.connection.rollback()
            return False

    def _fetch_yfinance_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data directly from yfinance without storing"""
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=start_date, end=end_date)
        if df.empty:
            return None
        df.index = df.index.tz_localize(None)
        return df

    def _fetch_and_store_yfinance_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch data from yfinance and store in database"""
        df = self._fetch_yfinance_data(ticker, start_date, end_date)
        if df is not None:
            table_name = f"his_{self.clean_ticker_for_table_name(ticker)}"
            self.store_dataframe(df, table_name)
        return df

    def _fetch_and_merge_yfinance_data(self, ticker: str, start_date: str, db_start: str, table_name: str) -> pd.DataFrame:
        """Fetch additional historical data and merge with existing database data"""
        # Fetch new data from yfinance
        new_data = self._fetch_yfinance_data(ticker, start_date, db_start)
        if new_data is None:
            return None

        # Get existing data from database
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(f"SELECT * FROM {table_name} ORDER BY Date")
        rows = cursor.fetchall()
        cursor.close()
        
        existing_data = pd.DataFrame(rows)
        existing_data.set_index('Date', inplace=True)
        
        # Combine datasets
        combined_data = pd.concat([new_data, existing_data])
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        combined_data.sort_index(inplace=True)
        
        # Update database
        self.store_dataframe(combined_data, table_name)
        return combined_data

    def _update_historical_data(self, ticker: str, db_end: str, current_date: str, table_name: str) -> pd.DataFrame:
        """Update historical data with new data from yfinance"""
        # Delete the last 10 days of data to ensure clean update
        cutoff_date = pd.to_datetime(db_end) - pd.Timedelta(days=10)
        
        # Get existing data before cutoff
        cursor = self.connection.cursor(dictionary=True)
        cursor.execute(f"SELECT * FROM {table_name} WHERE Date < %s", (cutoff_date,))
        rows = cursor.fetchall()
        cursor.close()
        
        existing_data = pd.DataFrame(rows)
        existing_data.set_index('Date', inplace=True)
        
        # Fetch new data
        new_data = self._fetch_yfinance_data(ticker, cutoff_date.strftime('%Y-%m-%d'), current_date)
        if new_data is None:
            return existing_data
        
        # Combine datasets
        combined_data = pd.concat([existing_data, new_data])
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        combined_data.sort_index(inplace=True)
        
        # Update database
        self.store_dataframe(combined_data, table_name)
        return combined_data

    def get_historical_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical data from MySQL database or yfinance.
        If data doesn't exist in database, fetch from yfinance and store it.
        """
        try:
            cleaned_ticker = self.clean_ticker_for_table_name(ticker)
            table_name = f"his_{cleaned_ticker}"
            
            # Get the latest trading day (last Friday if weekend)
            latest_trading_day = pd.Timestamp.now()
            while latest_trading_day.weekday() > 4:  # 5 = Saturday, 6 = Sunday
                latest_trading_day -= pd.Timedelta(days=1)
            latest_trading_day = latest_trading_day.strftime('%Y-%m-%d')
            
            # Adjust end_date if it's beyond latest trading day
            end_date = min(pd.to_datetime(end_date), pd.to_datetime(latest_trading_day)).strftime('%Y-%m-%d')
            
            # First check if table exists
            if not self.table_exists(table_name):
                # Create table and get data from yfinance
                df = self._fetch_yfinance_data(ticker, start_date, end_date)
                if df is not None:
                    # Create the table structure
                    cursor = self.connection.cursor()
                    create_table_query = f"""
                        CREATE TABLE IF NOT EXISTS {table_name} (
                            Date DATETIME PRIMARY KEY,
                            Open DOUBLE,
                            High DOUBLE,
                            Low DOUBLE,
                            Close DOUBLE
                            
                        )
                    """
                    cursor.execute(create_table_query)
                    self.connection.commit()
                    cursor.close()
                    
                    # Store the data
                    self.store_dataframe(df, table_name)
                    return df
                return None
            
            # Table exists, check if data needs updating
            cursor = self.connection.cursor(dictionary=True)
            date_range_query = f"""
                SELECT MIN(Date) as min_date, MAX(Date) as max_date 
                FROM {table_name}
            """
            cursor.execute(date_range_query)
            date_range = cursor.fetchone()
            cursor.close()
            
            min_date = date_range['min_date']
            max_date = date_range['max_date']
            current_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            
            # Check if we need to update the data
            if max_date is None or min_date is None:
                # Table exists but is empty, populate it
                df = self._fetch_yfinance_data(ticker, start_date, end_date)
                if df is not None:
                    self.store_dataframe(df, table_name)
                    return df
                return None
            
            # Convert dates for comparison
            db_end = pd.to_datetime(max_date).strftime('%Y-%m-%d')
            days_difference = (pd.to_datetime(current_date) - pd.to_datetime(db_end)).days
            
            # Update if data is more than 10 days old
            if days_difference > 10:
                df = self._update_historical_data(ticker, db_end, current_date, table_name)
            else:
                # Get data from database
                cursor = self.connection.cursor(dictionary=True)
                query = f"""
                    SELECT * FROM {table_name}
                    WHERE Date BETWEEN %s AND %s
                    ORDER BY Date
                """
                cursor.execute(query, (start_date, end_date))
                rows = cursor.fetchall()
                cursor.close()
                
                if not rows:
                    df = self._fetch_yfinance_data(ticker, start_date, end_date)
                    if df is not None:
                        self.store_dataframe(df, table_name)
                    return df
                
                df = pd.DataFrame(rows)
                df.set_index('Date', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error in get_historical_data for {ticker}: {str(e)}")
            return None

class AnalysisService:
    @staticmethod
    def perform_polynomial_regression(data, future_days=180):
        """Perform polynomial regression analysis with three-component scoring"""
        try:
            # Input validation
            if data is None or data.empty:
                print("Error: Input data is None or empty")
                return {
                    'predictions': [],
                    'upper_band': [],
                    'lower_band': [],
                    'r2': 0,
                    'coefficients': [0, 0, 0],
                    'intercept': 0,
                    'std_dev': 0,
                    'equation': "No data available",
                    'max_x': 0,
                    'total_score': {
                        'score': 0,
                        'rating': 'Error',
                        'components': {
                            'trend': {'score': 0, 'type': 'Unknown'},
                            'return': {'score': 0},
                            'volatility': {'score': 0}
                        }
                    }
                }

            # Get S&P 500 data for benchmarking
            try:
                data_service = DataService()
                end_date = data.index[-1].strftime('%Y-%m-%d')
                start_date = data.index[0].strftime('%Y-%m-%d')
                
                sp500_data = data_service.get_historical_data('^GSPC', start_date, end_date)
                
                if sp500_data is not None and not sp500_data.empty:
                    sp500_data['Log_Close'] = np.log(sp500_data['Close'])
                    X_sp = (sp500_data.index - sp500_data.index[0]).days.values.reshape(-1, 1)
                    y_sp = sp500_data['Log_Close'].values
                    X_sp_scaled = X_sp / (np.max(X_sp) * 1)
                    
                    poly_features = PolynomialFeatures(degree=2)
                    X_sp_poly = poly_features.fit_transform(X_sp_scaled)
                    sp500_model = LinearRegression()
                    sp500_model.fit(X_sp_poly, y_sp)
                    
                    sp500_r2 = r2_score(y_sp, sp500_model.predict(X_sp_poly))
                    sp500_returns = sp500_data['Close'].pct_change().dropna()
                    sp500_annual_return = sp500_returns.mean() * 252
                    sp500_annual_volatility = sp500_returns.std() * np.sqrt(252)
                    
                    sp500_params = {
                        'quad_coef': sp500_model.coef_[2],
                        'linear_coef': sp500_model.coef_[1],
                        'r_squared': sp500_r2,
                        'annual_return': sp500_annual_return,
                        'annual_volatility': sp500_annual_volatility
                    }
                else:
                    sp500_params = {
                        'quad_coef': -0.1134,
                        'linear_coef': 0.4700,
                        'r_squared': 0.9505,
                        'annual_return': 0.2384,
                        'annual_volatility': 0.125
                    }
            except Exception as sp_error:
                print(f"Error calculating S&P 500 parameters: {str(sp_error)}")
                sp500_params = {
                    'quad_coef': -0.1134,
                    'linear_coef': 0.4700,
                    'r_squared': 0.9505,
                    'annual_return': 0.2384,
                    'annual_volatility': 0.125
                }

            # Perform regression analysis
            try:
                data['Log_Close'] = np.log(data['Close'])
                X = (data.index - data.index[0]).days.values.reshape(-1, 1)
                y = data['Log_Close'].values
                X_scaled = X / (np.max(X) * 1)
                
                poly_features = PolynomialFeatures(degree=2)
                X_poly = poly_features.fit_transform(X_scaled)
                model = LinearRegression()
                model.fit(X_poly, y)
                
                coef = model.coef_
                intercept = model.intercept_
                max_x = np.max(X)
                
                # Calculate predictions
                X_future = np.arange(len(data) + future_days).reshape(-1, 1)
                X_future_scaled = X_future / np.max(X) * 1
                X_future_poly = poly_features.transform(X_future_scaled)
                y_pred_log = model.predict(X_future_poly)
                y_pred = np.exp(y_pred_log)
                
                # Calculate confidence bands
                residuals = y - model.predict(X_poly)
                std_dev = np.std(residuals)
                y_pred_upper = np.exp(y_pred_log + 2 * std_dev)
                y_pred_lower = np.exp(y_pred_log - 2 * std_dev)
                
                # Calculate R²
                r2 = r2_score(y, model.predict(X_poly))
                
                # Format equation
                equation = AnalysisService.format_regression_equation(coef, intercept, max_x)
                
            except Exception as e:
                print(f"Error in regression calculation: {str(e)}")
                return {
                    'predictions': data['Close'].values.tolist(),
                    'upper_band': data['Close'].values.tolist(),
                    'lower_band': data['Close'].values.tolist(),
                    'r2': 0,
                    'coefficients': [0, 0, 0],
                    'intercept': 0,
                    'std_dev': 0,
                    'equation': "Regression failed",
                    'max_x': len(data),
                    'total_score': {
                        'score': 0,
                        'rating': 'Error',
                        'components': {
                            'trend': {'score': 0, 'type': 'Unknown'},
                            'return': {'score': 0},
                            'volatility': {'score': 0}
                        }
                    }
                }

            # Calculate scoring
            def evaluate_trend_score(quad_coef, linear_coef, r_squared):
                """Calculate trend score from 0-100 based on coefficients"""
                try:
                    # Calculate asset's own volatility for benchmarks
                    returns = data['Close'].pct_change().dropna()
                    annual_vol = returns.std() * np.sqrt(252)
                    period_days = len(data)
                    period_years = period_days / 252
                    
                    # Calculate benchmarks using asset's own volatility
                    vol_linear = annual_vol * np.sqrt(period_years)  
                    vol_quad = annual_vol / np.sqrt(period_days)
                    
                    # Calculate base trend score (50 is neutral)
                    trend_score = 50
                    
                    # Linear component contribution (±25 points)
                    linear_impact = linear_coef / vol_linear
                    trend_score += 25 * min(1, max(-1, linear_impact))
                    
                    # Quadratic component contribution (±15 points)
                    quad_impact = quad_coef / vol_quad
                    if (quad_coef > 0 and linear_coef > 0) or (quad_coef < 0 and linear_coef < 0):
                        # Reinforcing trend
                        trend_score += 15 * min(1, max(-1, quad_impact))
                    else:
                        # Counteracting trend
                        trend_score -= 15 * min(1, max(-1, abs(quad_impact)))
                        
                    # Apply strength multiplier based on R-squared
                    strength_multiplier = 0.5 + (0.5 * r_squared)  # Range: 0.5-1.0
                    
                    # Calculate final score
                    final_score = trend_score * strength_multiplier
                    
                    # Normalize to 0-100 range
                    final_score = min(100, max(0, final_score))
                    
                    # Calculate additional metrics for compatibility
                    ratio = abs(quad_coef / linear_coef) if linear_coef != 0 else float('inf')
                    credibility_level = int(r_squared * 5)  # 1-5 scale
                    
                    return final_score, ratio, credibility_level
                    
                except Exception as e:
                    print(f"Error in trend score calculation: {str(e)}")
                    return 50, 0, 1

            def score_metric(value, benchmark, metric_type='return'):
                """Score metrics based on comparison to benchmark"""
                if metric_type == 'return':
                    diff = (value - benchmark) * 100
                    if diff >= 40: return 100
                    if diff >= 35: return 95
                    if diff >= 30: return 90
                    if diff >= 25: return 85
                    if diff >= 20: return 80
                    if diff >= 15: return 75
                    if diff >= 10: return 70
                    if diff >= 5:  return 65
                    if diff >= 0:  return 60
                    if diff >= -5: return 45
                    if diff >= -10: return 40
                    if diff >= -15: return 30
                    if diff >= -20: return 20
                    if diff >= -25: return 10
                    return 5
                else:  # volatility
                    ratio = value / benchmark
                    if ratio <= 0.6: return 100
                    if ratio <= 0.7: return 90
                    if ratio <= 0.8: return 85
                    if ratio <= 0.9: return 80
                    if ratio <= 1.0: return 75
                    if ratio <= 1.1: return 70
                    if ratio <= 1.2: return 65
                    if ratio <= 1.3: return 60
                    if ratio <= 1.4: return 55
                    if ratio <= 1.5: return 50
                    return 40

            try:
                # Calculate returns and volatility
                returns = data['Close'].pct_change().dropna()
                annual_return = returns.mean() * 252
                annual_volatility = returns.std() * np.sqrt(252)
                
                # Calculate component scores
                trend_score, ratio, credibility_level = evaluate_trend_score(coef[2], coef[1], r2)
                return_score = score_metric(annual_return, sp500_params['annual_return'], 'return')
                vol_score = score_metric(annual_volatility, sp500_params['annual_volatility'], 'volatility')
                
                # Calculate weighted score
                weights = {'trend': 0.4, 'return': 0.4, 'volatility': 0.20}
                raw_score = (
                    trend_score * weights['trend'] +
                    return_score * weights['return'] +
                    vol_score * weights['volatility']
                )

                # Calculate SP500's score for scaling
                sp500_trend_score, _, _ = evaluate_trend_score(
                    sp500_params['quad_coef'],
                    sp500_params['linear_coef'],
                    sp500_params['r_squared']
                )
                sp500_return_score = score_metric(
                    sp500_params['annual_return'],
                    sp500_params['annual_return'],
                    'return'
                )
                sp500_vol_score = score_metric(
                    sp500_params['annual_volatility'],
                    sp500_params['annual_volatility'],
                    'volatility'
                )
                
                sp500_raw_score = (
                    sp500_trend_score * weights['trend'] +
                    sp500_return_score * weights['return'] +
                    sp500_vol_score * weights['volatility']
                )
                
                scaling_factor = 75 / sp500_raw_score
                final_score = min(98, raw_score * scaling_factor)

                # Determine rating
                if final_score >= 90: rating = 'Excellent'
                elif final_score >= 75: rating = 'Very Good'
                elif final_score >= 65: rating = 'Good'
                elif final_score >= 40: rating = 'Fair'
                else: rating = 'Poor'

            except Exception as e:
                print(f"Error in scoring calculation: {str(e)}")
                return_score = vol_score = trend_score = final_score = 0
                rating = 'Error'
                ratio = 0
                credibility_level = 0

            return {
                'predictions': y_pred.tolist(),
                'upper_band': y_pred_upper.tolist(),
                'lower_band': y_pred_lower.tolist(),
                'r2': float(r2),
                'coefficients': coef.tolist(),
                'intercept': float(intercept),
                'std_dev': float(std_dev),
                'equation': equation,
                'max_x': int(max_x),
                'total_score': {
                    'score': float(final_score),
                    'raw_score': float(raw_score),
                    'rating': rating,
                    'components': {
                        'trend': {
                            'score': float(trend_score),
                            'details': {
                                'ratio': float(ratio),
                                'credibility_level': credibility_level,
                                'quad_coef': float(coef[2]),
                                'linear_coef': float(coef[1])
                            }
                        },
                        'return': {
                            'score': float(return_score),
                            'value': float(annual_return)
                        },
                        'volatility': {
                            'score': float(vol_score),
                            'value': float(annual_volatility)
                        }
                    },
                    'scaling': {
                        'factor': float(scaling_factor),
                        'sp500_base': float(sp500_raw_score)
                    },
                    'weights': weights
                }
            }

        except Exception as e:
            print(f"Error in polynomial regression: {str(e)}")
            return {
                'predictions': data['Close'].values.tolist() if data is not None else [],
                'upper_band': data['Close'].values.tolist() if data is not None else [],
                'lower_band': data['Close'].values.tolist() if data is not None else [],
                'r2': 0,
                'coefficients': [0, 0, 0],
                'intercept': 0,
                'std_dev': 0,
                'equation': "Error occurred",
                'max_x': len(data) if data is not None else 0,
                'total_score': {
                    'score': 0,
                    'raw_score': 0,
                    'rating': 'Error',
                    'components': {
                        'trend': {'score': 0, 'details': {}},
                        'return': {'score': 0, 'value': 0},
                        'volatility': {'score': 0, 'value': 0}
                    },
                    'scaling': {
                        'factor': 0,
                        'sp500_base': 0
                    }
                }
            }

    @staticmethod
    def format_regression_equation(coefficients, intercept, max_x):
        """Format regression equation string"""
        terms = []
        if coefficients[2] != 0:
            terms.append(f"{coefficients[2]:.4f}(x/{max_x})²")
        if coefficients[1] != 0:
            sign = "+" if coefficients[1] > 0 else ""
            terms.append(f"{sign}{coefficients[1]:.4f}(x/{max_x})")
        if intercept != 0:
            sign = "+" if intercept > 0 else ""
            terms.append(f"{sign}{intercept:.4f}")
        equation = "ln(y) = " + " ".join(terms)
        return equation

def get_stock_data(ticker, data_service):
    """Get historical stock data using DataService."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=11*365)  # Get 11 years of data
        
        return data_service.get_historical_data(
            ticker, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

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

def analyze_tickers(tickers_file, data_service, limit=100):
    """Analyze tickers and store results in DataFrame."""
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
        data = get_stock_data(ticker, data_service)
        
        if data is None or data.empty:
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

def save_to_excel(results_df, time_periods, output_file='stock_regression_results.xlsx'):
    """Save results to Excel with formatting."""
    try:
        print("\nSaving results to Excel...")
        
        try:
            import xlsxwriter
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
            print(f"Results saved with full formatting to {output_file}")
            
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

# Main execution
if __name__ == "__main__":
    try:
        # Initialize DataService
        data_service = DataService()
        
        tickers_file = "tickers.ts"
        results_df, time_periods = analyze_tickers(tickers_file, data_service, limit=100)
        
        # Save results to Excel
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
        if 'data_service' in locals():
            data_service.connection.close()