from data_fetcher import fetch_macro_data
from analysis_macro import MacroAnalyzer
from analysis_flows import FlowAndRiskAnalyzer
from quant_allocation import QuantAllocationModel

def main():
    print("==================================================")
    print(" Macro-Quant Analyzer: Gold & Macro Diagnostic Tool")
    print("==================================================")
    
    df = fetch_macro_data(start_date="2011-01-01")
    
    macro_analyzer = MacroAnalyzer(df)
    corr, ols_model = macro_analyzer.evaluate_real_yield_and_dxy()
    df_analyzed = macro_analyzer.central_bank_momentum_shift()
    
    flow_analyzer = FlowAndRiskAnalyzer(df_analyzed)
    df_flow = flow_analyzer.diagnose_retail_overheating()
    tail_events = flow_analyzer.evaluate_tracking_error_and_tail_risk()
    
    quant_model = QuantAllocationModel(df_flow)
    strat_results = quant_model.run_dynamic_hedge_strategy()
    
    print("\n==================================================")
    print("[+] All quantitative validation pipelines executed successfully.")
    print("==================================================")

if __name__ == "__main__":
    main()
