from data_fetcher import fetch_macro_data
from analysis_macro import MacroAnalyzer
from analysis_flows import FlowAndRiskAnalyzer
from quant_allocation import QuantAllocationModel

def main():
    print("==================================================")
    print(" Macro-Quant Analyzer: Gold & Macro Diagnostic Tool")
    print("==================================================")
    
    # 1. 데이터 수집 (최근 15년 장기 시계열)
    df = fetch_macro_data(start_date="2011-01-01")
    
    # 2. 매크로 및 중앙은행 수급 분석 (메서드명 일치)
    macro_analyzer = MacroAnalyzer(df)
    corr, ols_model = macro_analyzer.evaluate_real_yield_and_dxy()
    df_analyzed = macro_analyzer.test_central_bank_momentum_shift()
    
    # 3. 개인 수급 및 테일 리스크 분석
    flow_analyzer = FlowAndRiskAnalyzer(df_analyzed)
    df_flow = flow_analyzer.diagnose_retail_overheating()
    tail_events = flow_analyzer.evaluate_tracking_error_and_tail_risk()
    
    # 4. 퀀트 기반 자산 배분 및 혜지 모델 백테스트
    quant_model = QuantAllocationModel(df_flow)
    strat_results = quant_model.run_dynamic_hedge_strategy()
    
    print("\n==================================================")
    print("[+] All quantitative validation pipelines executed successfully.")
    print("==================================================")

if __name__ == "__main__":
    main()
