import pandas as pd
import numpy as np

class FlowAndRiskAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def diagnose_retail_overheating(self):
        print("\n--- [Analysis 3] Retail Investor Overheating & Dip-Buying Diagnosis ---")
        gold_col = 'Gold_Futures' if 'Gold_Futures' in self.df.columns else self.df.columns[0]
        self.df['Returns'] = self.df[gold_col].pct_change()
        self.df['Rolling_Volatility'] = self.df['Returns'].rolling(window=20).std() * np.sqrt(252)
        
        delta = self.df[gold_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        
        recent_rsi = self.df['RSI'].iloc[-1] if not self.df['RSI'].empty else 50.0
        print(f"Current Gold RSI (14): {recent_rsi:.2f}")
        if recent_rsi < 30:
            print("[!] Alert: Technical oversold condition (RSI < 30) reached.")
        return self.df

    def evaluate_tracking_error_and_tail_risk(self):
        print("\n--- [Analysis 4] Tracking Error & Tail Risk (End-of-Day Rebalancing) ---")
        gld_col = 'GLD' if 'GLD' in self.df.columns else self.df.columns[1]
        daily_ret = self.df[gld_col].pct_change().dropna()
        var_95 = np.percentile(daily_ret, 5)
        tail_events = daily_ret[daily_ret < var_95]
        
        print(f"95% Daily Value at Risk (VaR): {var_95:.4f}")
        print(f"Total Tail Risk / Liquidity Gap Shock Events Count: {len(tail_events)}")
        return tail_events
