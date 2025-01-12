import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, inspect, text
from urllib.parse import quote_plus
import logging
from datetime import datetime, timedelta
import re

class DatabaseService:
    def __init__(self):
        """Initialize database connection"""
        try:
            password = quote_plus("Gern@8280")
            base_url = f"mysql+pymysql://username:{password}@localhost"
            
            # Create database
            temp_engine = create_engine(base_url)
            with temp_engine.connect() as conn:
                conn.execute(text("DROP DATABASE IF EXISTS my_database"))
                conn.execute(text("CREATE DATABASE if not exists my_database"))
            temp_engine.dispose()
            
            # Connect to database
            self.engine = create_engine(f"{base_url}/my_database")
            print("Database initialized successfully")
            
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise

    def store_stock_data(self, ticker, start_date, end_date):
        """
        Fetch and store stock data, handling existing data intelligently.
        If table exists, only fetch missing date ranges. Otherwise, fetch all data.
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns:
        --------
        pd.DataFrame or None
            DataFrame containing the stock data if successful, None otherwise
        """
        try:
            print(f"\nProcessing data for {ticker}...")
            ticker_cleaned = self.clean_ticker_for_table_name(ticker)
            table_name = f"his_{ticker_cleaned}"
            
            # Convert dates to datetime for comparison
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            existing_data = None
            existing_start = None
            existing_end = None
            
            # Check if table exists and get existing date range
            inspector = inspect(self.engine)
            if table_name in inspector.get_table_names():
                with self.engine.connect() as conn:
                    existing_dates = conn.execute(
                        text(f"SELECT MIN(Date), MAX(Date) FROM {table_name}")
                    ).fetchone()
                    
                    if existing_dates[0] is not None:
                        existing_start = pd.to_datetime(existing_dates[0])
                        existing_end = pd.to_datetime(existing_dates[1])
                        
                        # If requested range is within existing data, return
                        if start_dt >= existing_start and end_dt <= existing_end:
                            print(f"Requested date range already exists in table {table_name}")
                            return None
                        
                        # Load existing data
                        existing_data = pd.read_sql(f"SELECT * FROM {table_name}", self.engine, index_col='Date')
                        existing_data.index = pd.to_datetime(existing_data.index)
            
            # Initialize stock data fetcher
            stock = yf.Ticker(ticker)
            new_data = None
            
            # Determine what data ranges we need to fetch
            if existing_data is not None:
                # Fetch data before existing range if needed
                if start_dt < existing_start:
                    before_data = stock.history(start=start_date, end=existing_start)
                    if not before_data.empty:
                        before_data.index = before_data.index.tz_localize(None)
                        new_data = before_data
                
                # Fetch data after existing range if needed
                if end_dt > existing_end:
                    after_data = stock.history(start=existing_end, end=end_date)
                    if not after_data.empty:
                        after_data.index = after_data.index.tz_localize(None)
                        new_data = after_data if new_data is None else pd.concat([new_data, after_data])
            else:
                # No existing data, fetch entire range
                new_data = stock.history(start=start_date, end=end_date)
                if not new_data.empty:
                    new_data.index = new_data.index.tz_localize(None)
            
            if new_data is None or new_data.empty:
                print(f"No new data to add for {ticker}")
                return None
                
            # Combine existing and new data if we have both
            final_data = new_data if existing_data is None else pd.concat([existing_data, new_data])
            final_data = final_data.loc[~final_data.index.duplicated(keep='last')]
            final_data = final_data.sort_index()
            
            # Store in database
            final_data.to_sql(
                name=table_name,
                con=self.engine,
                if_exists='replace',
                index=True,
                index_label='Date'
            )
            
            # Verify storage
            with self.engine.connect() as conn:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()
                print(f"Stored total of {count} rows in {table_name}")
                if existing_data is not None:
                    new_rows = len(final_data) - len(existing_data)
                    print(f"Added {new_rows} new rows to existing data")
            
            return final_data
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            return None
    def clean_ticker_for_table_name(self, ticker: str) -> str:
        """
        Clean ticker symbol for use in table name.
        Removes special characters and converts to valid table name format.
        
        Parameters:
        -----------
        ticker : str
            Original ticker symbol
        
        Returns:
        --------
        str
            Cleaned ticker symbol safe for use in table names
        """
        # Replace any non-alphanumeric characters with underscore
        cleaned = ''.join(c if c.isalnum() else '_' for c in ticker)
        # Remove leading/trailing underscores
        cleaned = cleaned.strip('_')
        # Convert to lowercase
        cleaned = cleaned.lower()
        # If the cleaned string is empty, use a default
        if not cleaned:
            cleaned = 'unknown'
        return cleaned
    
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

try:
    # Initialize database
    days_of_data = 12*365
    db_service = DatabaseService()
    tickers_file = 'tickers.ts'
    
    # Set date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days_of_data)).strftime('%Y-%m-%d')  # 11 years of data
    
    print(f"\nProcessing data from {start_date} to {end_date}")
    print("=" * 80)
    
    # First store S&P 500 for benchmarking
    print("\nStoring S&P 500 data...")
    sp500_df = db_service.store_stock_data("^GSPC", start_date, end_date)
    with open(tickers_file, 'r') as f:
        content = f.read()
        tickers_data = parse_tickers_file(content, limit=1000)
    
    # Store data for each stock
    for idx, ticker_info in enumerate(tickers_data, 1):
        ticker = ticker_info['symbol']
        df = db_service.store_stock_data(ticker, start_date, end_date)
        if df is not None:
            print(f"Successfully stored {ticker} data")
    
    # Show database state
    inspector = inspect(db_service.engine)
    tables = inspector.get_table_names()
    print("\nTables in database:", tables)
    
    # Show row counts for each table
    print("\nRow counts:")
    for table in tables:
        with db_service.engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"{table}: {count} rows")

except Exception as e:
    print(f"Error in main execution: {e}")
