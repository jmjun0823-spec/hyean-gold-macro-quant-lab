import unittest
import numpy as np
import pandas as pd

from analysis_flows import FlowAndRiskAnalyzer
from analysis_macro import MacroAnalyzer
from quant_allocation import QuantAllocationModel


class TestQuantPipelineSemantics(unittest.TestCase):
    def setUp(self):
        # 휴장일(공백/결측)이 섞인 고정 소규모 시계열 데이터
        idx = pd.to_datetime([
            "2026-01-01",  # 휴장일 가정 (결측)
            "2026-01-02",
            "2026-01-05",
            "2026-01-06",
            "2026-01-07",
            "2026-01-08",
        ])
        self.sample_df = pd.DataFrame(
            {
                "Gold_Futures": [np.nan, 2000.0, 1960.0, 1920.0, 1940.0, 1980.0],
                "GLD": [np.nan, 185.0, 181.0, 177.0, 179.0, 183.0],
                "DXY": [102.0, 102.5, 103.2, 103.5, 103.1, 102.8],
                "Real_Yield_10Y": [1.8, 1.85, 1.95, 2.00, 1.92, 1.88],
            },
            index=idx,
        )

    def test_holiday_ffill_and_empty_response_handling(self):
        # 1) 휴장일/초기 결측이 포함된 데이터에서 전방 채움 및 분석 정상 동작 확인
        aligned = self.sample_df.ffill().dropna()
        self.assertEqual(aligned.index[0], pd.Timestamp("2026-01-02"))
        self.assertFalse(aligned.isna().any().any())

        # 2) 완전 빈 DataFrame 입력 시에도 파이프라인이 중단되지 않는지 확인
        empty_df = pd.DataFrame(columns=["Gold_Futures", "GLD", "DXY", "Real_Yield_10Y"])
        flow_empty = FlowAndRiskAnalyzer(empty_df)
        tail_empty = flow_empty.evaluate_tracking_error_and_tail_risk()
        self.assertGreater(len(tail_empty), 0)

        alloc_empty = QuantAllocationModel(empty_df)
        res_empty = alloc_empty.run_dynamic_hedge_strategy()
        self.assertIn("Cum_Strategy", res_empty.columns)
        self.assertIn("Cum_Benchmark", res_empty.columns)

    def test_signal_lag_prevents_lookahead(self):
        # 시그널 발생 당일이 아닌 다음 시점(shift(1)) 수익률에 곱해지는지 검증
        df = self.sample_df.ffill().dropna().copy()
        df["RSI"] = [50.0, 25.0, 28.0, 45.0, 55.0]  # 2026-01-05, 01-06에 RSI < 30 시그널

        alloc = QuantAllocationModel(df)
        out = alloc.run_dynamic_hedge_strategy()

        expected_returns = df["Gold_Futures"].pct_change().fillna(0.0)
        expected_lagged_signal = out["Signal"].shift(1).fillna(0.0)
        expected_strat_ret = expected_lagged_signal * expected_returns

        pd.testing.assert_series_equal(
            out["Strategy_Returns"],
            expected_strat_ret,
            check_names=False,
        )
        # 첫 행의 전략 수익률은 이전 시그널이 없으므로 0.0이어야 함
        self.assertEqual(out["Strategy_Returns"].iloc[0], 0.0)

    def test_macro_analyzer_runs_on_small_fixed_data(self):
        macro = MacroAnalyzer(self.sample_df.ffill().dropna())
        corr, model = macro.evaluate_real_yield_and_dxy()
        self.assertFalse(corr.empty)
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
