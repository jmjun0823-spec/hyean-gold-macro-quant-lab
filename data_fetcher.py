import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
from datetime import datetime

def fetch_macro_data(start_date="2011-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    
    print("[*] Collecting macro economic data (FRED & Yahoo Finance)...")
    
    yf_tickers = ['GC=F', 'GLD', 'DX-Y.NYB']
    df_yf = pd.DataFrame()
    try:
        raw_yf = yf.download(yf_tickers, start=start_date, end=end_date, progress=False)
        if not raw_yf.empty:
            if isinstance(raw_yf.columns, pd.MultiIndex):
                if 'Adj Close' in raw_yf.columns.levels[0]:
                    df_yf = raw_yf['Adj Close']
                else:
                    df_yf = raw_yf.iloc[:, :len(yf_tickers)]
            else:
                df_yf = raw_yf
            # 타임존 제거 (Naive 변환)
            if df_yf.index.tz is not None:
                df_yf.index = df_yf.index.tz_localize(None)
    except Exception as e:
        print(f"[!] Yahoo Finance download warning: {e}")

    df_tips = pd.DataFrame()
    try:
        df_tips = web.DataReader('DFII10', 'fred', start=start_date, end=end_date)
        if not df_tips.empty and df_tips.index.tz is not None:
            df_tips.index = df_tips.index.tz_localize(None)
    except Exception as e:
        print(f"[!] FRED data fetch warning: {e}.")

    # 데이터 병합
    combined_df = pd.concat([df_yf, df_tips], axis=1)
    
    # 컬럼명 매핑 및 정돈
    cols = []
    for col in combined_df.columns:
        c_str = str(col)
        if 'GC=F' in c_str or 'Gold' in c_str:
            cols.append('Gold_Futures')
        elif 'GLD' in c_str:
            cols.append('GLD')
        elif 'DX' in c_str or 'DXY' in c_str:
            cols.append('DXY')
        elif 'DFII10' in c_str or 'Real' in c_str:
            cols.append('Real_Yield_10Y')
        else:
            cols.append(c_str)
    
    combined_df.columns = cols[:len(combined_df.columns)]
    
    # 필수 컬럼 보정
    for required in ['Gold_Futures', 'GLD', 'DXY', 'Real_Yield_10Y']:
        if required not in combined_df.columns:
            combined_df[required] = 0.0

    combined_df = combined_df[['Gold_Futures', 'GLD', 'DXY', 'Real_Yield_10Y']]
    combined_df = combined_df.ffill().dropna()
    
    print(f"[+] Data collection completed. Shape: {combined_df.shape}")
    return combined_df

if __name__ == "__main__":
    df = fetch_macro_data()
    print(df.tail())
