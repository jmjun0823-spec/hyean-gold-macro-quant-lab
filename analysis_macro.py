import pandas as pd
import numpy as np
import statsmodels.api as sm

class MacroAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def evaluate_real_yield_and_dxy(self):
        print("\n--- [Analysis 1] Macro Impact & Correlation Evaluation ---")
        sub_cols = [c for c in ['Gold_Futures', 'Real_Yield_10Y', 'DXY'] if c in self.df.columns]
        if len(sub_cols) < 3:
            print("[!] Warning: Missing columns for macro analysis. Using available data.")
            sub_df = self.df.select_dtypes(include=[np.number]).iloc[:, :3]
            sub_df.columns = ['Gold', 'Real_Yield_10Y', 'DXY']
        else:
            sub_df = self.df[sub_cols].rename(columns={'Gold_Futures': 'Gold'})
        
        sub_df = sub_df.dropna()
        corr_matrix = sub_df.corr()
        print("Correlation Matrix:")
        print(corr_matrix)
        
        X = sub_df[['Real_Yield_10Y', 'DXY']]
        X = sm.add_constant(X)
        y = np.log(sub_df['Gold'])
        
        model = sm.OLS(y, X).fit()
        print("\nOLS Regression Results (Log Gold Price vs Real Yield & DXY):")
        print(model.summary())
        return corr_matrix, model

    def test_central_bank_momentum_shift(self, window=252):
        print("\n--- [Analysis 2] Central Bank Buying Momentum Test ---")
        gold_col = 'Gold_Futures' if 'Gold_Futures' in self.df.columns else self.df.columns[0]
        self.df['Gold_Mom'] = self.df[gold_col].diff(window)
        
        recent_momentum = self.df['Gold_Mom'].iloc[-1]
        print(f"Current 1-Year Rolling Gold Price Change Momentum: {recent_momentum:.2f}")
        
        momentum_breakdown = self.df[self.df['Gold_Mom'] < 0].tail(10)
        print("Recent periods indicating momentum deceleration/breakdown:")
        print(momentum_breakdown)
        return self.df
