import pandas as pd
import numpy as np

class QuantAllocationModel:
    def __init__(self, df: pd.DataFrame):
        self.df = df.dropna()

    def run_dynamic_hedge_strategy(self):
        print("\n--- [Analysis 5] Quant-Based Asset Allocation & Hedge Strategy ---")
        df_strat = self.df.copy()
        df_strat['Signal'] = 0
        
        if 'RSI' in df_strat.columns:
            df_strat.loc[df_strat['RSI'] < 30, 'Signal'] = 1
        
        gold_col = 'Gold_Futures' if 'Gold_Futures' in df_strat.columns else df_strat.columns[0]
        df_strat['Strategy_Returns'] = df_strat['Signal'].shift(1) * df_strat[gold_col].pct_change()
        df_strat['Benchmark_Returns'] = df_strat[gold_col].pct_change()
        
        df_strat['Cum_Strategy'] = (1 + df_strat['Strategy_Returns'].fillna(0)).cumprod()
        df_strat['Cum_Benchmark'] = (1 + df_strat['Benchmark_Returns'].fillna(0)).cumprod()
        
        print(f"Final Benchmark Cumulative Return: {df_strat['Cum_Benchmark'].iloc[-1]:.4f}")
        print(f"Final Quant Strategy Cumulative Return: {df_strat['Cum_Strategy'].iloc[-1]:.4f}")
        
        return df_strat
