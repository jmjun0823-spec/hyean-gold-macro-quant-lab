import pandas as pd
import numpy as np
import statsmodels.api as sm

class MacroAnalyzer:
    def __init__(self, df: pd.DataFrame):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ['_'.join(str(c) for c in col).strip() for col in df.columns]
        else:
            df.columns = [str(c) for c in df.columns]
        self.df = df.copy()

    def evaluate_real_yield_and_dxy(self):
        print("\n--- [Analysis 1] Macro Impact & Correlation Evaluation ---")
        col_map = {}
        for c in self.df.columns:
            c_upper = c.upper()
            if 'GOLD' in c_upper or 'GC=F' in c_upper:
                col_map[c] = 'Gold'
            elif 'REAL' in c_upper or 'DFII10' in c_upper:
                col_map[c] = 'Real_Yield_10Y'
            elif 'DXY' in c_upper or 'DX' in c_upper:
                col_map[c] = 'DXY'
        
        sub_df = self.df.rename(columns=col_map)
        required = ['Gold', 'Real_Yield_10Y', 'DXY']
        
        if not all(r in sub_df.columns for r in required):
            numeric_cols = sub_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 3:
                sub_df = sub_df[numeric_cols].copy()
                sub_df.columns = required
            else:
                for r in required:
                    if r not in sub_df.columns:
                        sub_df[r] = 0.0
                sub_df = sub_df[required]

        sub_df = sub_df[required].dropna()
        if sub_df.empty:
            print("[!] Warning: sub_df is empty after dropna.")
            return pd.DataFrame(), None

        corr_matrix = sub_df.corr()
        print("Correlation Matrix:")
        print(corr_matrix)
        
        X = sub_df[['Real_Yield_10Y', 'DXY']]
        X = sm.add_constant(X)
        y = np.log(sub_df['Gold'].replace(0, np.nan)).dropna()
        
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        if len(common_idx) < 10:
            print("[!] Warning: Not enough data points for regression.")
            return corr_matrix, None

        model = sm.OLS(y, X).fit()
        print("\nOLS Regression Results (Log Gold Price vs Real Yield & DXY):")
        print(model.summary())
        return corr_matrix, model

    def test_central_bank_momentum_shift(self, window=252):
        print("\n--- [Analysis 2] Central Bank Buying Momentum Test ---")
        gold_col = None
        for c in self.df.columns:
            if 'GOLD' in c.upper() or 'GC=F' in c.upper():
                gold_col = c
                break
        if gold_col is None:
            gold_col = self.df.select_dtypes(include=[np.number]).columns[0]

        self.df['Gold_Mom'] = self.df[gold_col].diff(window)
        
        recent_momentum = self.df['Gold_Mom'].iloc[-1]
        print(f"Current 1-Year Rolling Gold Price Change Momentum: {recent_momentum:.2f}")
        
        momentum_breakdown = self.df[self.df['Gold_Mom'] < 0].tail(10)
        print("Recent periods indicating momentum deceleration/breakdown:")
        print(momentum_breakdown)
        return self.df
