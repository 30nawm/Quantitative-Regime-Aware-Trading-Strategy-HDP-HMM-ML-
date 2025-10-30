import yfinance as yf
import pandas as pd

def fetch_data(tickers_dict, start_date, end_date):
    print(f"Fetching data for {list(tickers_dict.values())}...")
    
    data_frames = []
    for ticker, name in tickers_dict.items():
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print(f"Warning: No data found for {ticker} ({name})")
            continue
        
        data.columns = pd.MultiIndex.from_tuples(
            [(col, name) for col in data.columns],
            names=['Field', 'Ticker']
        )
        data_frames.append(data)
    
    if not data_frames:
        raise ValueError("No data could be downloaded for any ticker.")
        
    multi_data = pd.concat(data_frames, axis=1).sort_index()
    multi_data.ffill(inplace=True)
    multi_data.bfill(inplace=True)
    
    print("Data fetching complete.")
    return multi_data