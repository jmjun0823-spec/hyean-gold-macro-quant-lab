import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from datetime import datetime

def fetch_macro_data(start_date="2011-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    print("[*] Collecting macro economic data (FRED & Yahoo Finance)...")
    
    yf_tickers = ['GC=F', 'GLD', 'DX-Y.NYB']
    try:
        df_yf = yf.download(yf_tickers, start=start_date, end=end_date, progress=False)
        if isinstance(df_yf.columns, pd.MultiIndex):
            df_yf = df_yf['Adj Close']
    except Exception as e:
        print(f"[!] Yahoo Finance download warning: {e}")
        df_yf = pd.DataFrame()
    
    try:
        df_tips = web.DataReader('DFII10', 'fred', start=start_date, end=end_date)
    except Exception as e:
        print(f"[!] FRED data fetch warning: {e}.")
        df_tips = pd.DataFrame()

    combined_df = pd.concat([df_yf, df_tips], axis=1).dropna(how='all')
    expected_cols = ['Gold_Futures', 'GLD', 'DXY', 'Real_Yield_10Y']
    if combined_df.shape[1] == len(expected_cols):
        combined_df.columns = expected_cols
    combined_df = combined_df.ffill().dropna()
    
    print(f"[+] Data collection completed. Shape: {combined_df.shape}")
    return combined_df

if __name__ == "__main__":
    df = fetch_macro_data()
    print(df.tail())
