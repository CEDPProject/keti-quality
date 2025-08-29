import pandas as pd
import numpy as np

class OverallQualityMetrics:
    """
    전체 데이터프레임의 품질을 평가하는 클래스.
    - 설명: 이 클래스는 전체 시계열 데이터에 대한 품질 메트릭스를 계산하여,
            데이터의 결측치, 중복 타임스탬프, 인덱스 규칙성, 데이터 밀도를 평가함.
    """

    def __init__(self, df):
        """
        클래스 초기화 메서드
        - 매개변수: 
            - df: 시계열 데이터가 포함된 pandas DataFrame
        """
        self.df = df

    def calculate_missing_ratio(self):
        """
        전체 결측치 비율을 계산하는 함수.
        - 범위: 0 ~ 1 (비율), 값이 낮을수록 결측치가 적어 품질이 높음.
        - 수식: 결측치 비율 = (전체 결측치 수) / (전체 데이터 수)
        - 설명: 전체 데이터에서 결측치가 차지하는 비율을 계산하여 데이터의 신뢰성을 평가함.
                결측치 비율이 낮을수록 신뢰성이 높고, 비율이 높을수록 데이터 품질에 문제가 있음을 의미함.
        """
        return float(round(self.df.isna().mean().mean(), 2))

    def calculate_duplicate_timestamps_ratio(self):
        """
        중복된 타임스탬프 비율을 계산하는 함수.
        - 범위: 0 ~ 1 (비율), 값이 낮을수록 좋음.
        - 수식: 중복된 타임스탬프 비율 = (중복된 타임스탬프 수) / (전체 타임스탬프 수)
        - 설명: 타임스탬프가 중복되면 데이터의 시간적 순서에 혼란이 생길 수 있으며, 
                분석과 예측의 신뢰성을 떨어뜨림. 중복된 타임스탬프 비율이 낮을수록 시간적 순서가 명확함.
        """
        return float(round(self.df.index.duplicated().mean(), 2))

    def calculate_index_irregularity_score(self):
        """
        인덱스 규칙성 점수를 계산하는 함수.
        - 범위: 0 ~ ∞, 값이 낮을수록 시간 간격이 일정하여 규칙적인 데이터임을 의미
        - 수식: 불규칙성 점수 = 변동 계수 (CV) + 불규칙성 비율
        - 설명: 시간 인덱스 간격이 일정할수록 값이 낮아짐.
                불규칙성이 클수록 시간 인덱스가 일정하지 않아 분석에 문제가 생길 가능성이 높음.
        """
        time_diffs = self.df.index.to_series().diff().dropna().dt.total_seconds() / (24 * 60 * 60)
        mean_diff = np.mean(time_diffs)
        std_diff = np.std(time_diffs)
        cv_diff = std_diff / mean_diff if mean_diff != 0 else np.inf
        irregularity_ratio = np.sum(np.abs(time_diffs - mean_diff) > (2 * std_diff)) / len(time_diffs)
        return float(round(cv_diff + irregularity_ratio, 2))

    def calculate_density_ratio(self):
        """
        데이터 밀도 비율을 계산하는 함수.
        - 범위: 0 ~ 1, 값이 1에 가까울수록 좋음.
        - 수식: 데이터 밀도 비율 = 고유 타임스탬프 수 / 전체 데이터 수
        - 설명: 데이터가 시간에 따라 균등하게 분포되어 있는지 확인하며, 밀도가 높을수록 신뢰성이 높음.
        """
        return float(round(self.df.index.nunique() / len(self.df), 2))

    def get_metrics(self):
        """
        전체 품질 메트릭스를 반환하는 함수.
        - 결과: 전체 결측치 비율, 중복 타임스탬프 비율, 인덱스 불규칙성 점수, 밀도 비율을 포함한 딕셔너리 반환
        """
        return {
            "overall_missing_ratio": self.calculate_missing_ratio(),
            "duplicate_timestamps_ratio": self.calculate_duplicate_timestamps_ratio(),
            "index_irregularity_score": self.calculate_index_irregularity_score(),
            "density_ratio": self.calculate_density_ratio()
        }
