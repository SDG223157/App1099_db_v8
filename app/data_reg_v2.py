def get_stock_data(ticker, data_service):
    """Get historical stock data."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=11*365)  # Get 11 years of data
        return data_service.get_historical_data(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None 