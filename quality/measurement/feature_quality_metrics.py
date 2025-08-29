import pandas as pd
import numpy as np

class FeatureQualityMetrics:
    """
    각 피처별 품질 메트릭스를 계산하는 클래스.
    개별 피처에 대한 특성을 평가하며, 범위 및 데이터 타입 검사도 포함함.
    """
    
    def __init__(self, df, range_limits=None, expected_types=None):
        """
        클래스 초기화 메서드.
        
        Args:
            df (pd.DataFrame): 품질을 평가할 데이터프레임.
            range_limits (dict, optional): 피처별 최소/최대 허용 값. 예: {'max_num': {'feature1': 100, 'feature2': 50}, 'min_num': {'feature1': 0, 'feature2': 10}}
            expected_types (dict, optional): 피처별 기대 데이터 타입. 예: {'feature1': int, 'feature2': float}
        """
        self.df = df
        self.range_limits = range_limits
        self.expected_types = expected_types

    def _convert_invalid_types_to_nan(self, series, column):
        """
        지정된 데이터 타입과 다른 값은 NaN으로 변환하는 함수.
        
        Args:
            series (pd.Series): 평가할 피처 데이터.
            column (str): 평가할 피처 이름.
        
        Returns:
            pd.Series: 잘못된 데이터 타입을 NaN으로 변환한 시리즈.
        
        설명: 메트릭 계산 전에 잘못된 타입을 NaN으로 변환하여 오류를 방지함.
        """
        if self.expected_types and column in self.expected_types:
            expected_type = self.expected_types[column]
            return series.apply(lambda x: x if isinstance(x, expected_type) or pd.isna(x) else np.nan)
        return series

    def calculate_out_of_range_ratio(self, series, column):
        """
        범위를 벗어나는 데이터 비율을 계산하는 함수.
        
        범위: 0 ~ 1 (비율), 값이 낮을수록 품질이 좋음.
        수식: 범위를 벗어난 비율 = (범위를 벗어난 값 개수) / (전체 데이터 수)
        
        Args:
            series (pd.Series): 평가할 피처 데이터.
            column (str): 평가할 피처 이름.
        
        Returns:
            dict: 범위 벗어난 값 개수 및 비율을 포함한 딕셔너리, 예: {"out_of_range_count": 5, "out_of_range_ratio": 0.05}.
        """
        # 숫자형 데이터만 남기고, 문자열이나 기타 타입은 NaN으로 변환
        series = pd.to_numeric(series, errors='coerce')
        
        max_limit = self.range_limits.get('max_num', {}).get(column, None)
        min_limit = self.range_limits.get('min_num', {}).get(column, None)
        
        if max_limit is not None and min_limit is not None:
            out_of_range = (series > max_limit) | (series < min_limit)
            out_of_range_count = out_of_range.sum()
            out_of_range_ratio = out_of_range.mean()
            return {"out_of_range_count": int(out_of_range_count), "out_of_range_ratio": float(round(out_of_range_ratio, 2))}
        
        return None


    def calculate_invalid_data_type_ratio(self, series, column):
        """
        피처별 지정된 데이터 타입과 다른 데이터 비율을 계산하는 함수.
        
        범위: 0 ~ 1 (비율), 값이 낮을수록 데이터 품질이 높음.
        수식: 올바르지 않은 데이터 타입 비율 = (잘못된 타입 값 개수) / (전체 데이터 수)
        
        Args:
            series (pd.Series): 평가할 피처 데이터.
            column (str): 평가할 피처 이름.
        
        Returns:
            dict: 잘못된 데이터 타입 값 개수 및 비율을 포함한 딕셔너리, 예: {"invalid_type_count": 2, "invalid_type_ratio": 0.02}.
        
        설명: 특정 피처의 값이 지정된 데이터 타입과 다른 경우 비율을 계산하여 일관성을 평가함.
               값이 높을수록 비정상적인 데이터 타입이 섞여 있어 데이터 일관성이 떨어짐을 의미함.
        """
        if self.expected_types is None:
            return None
        expected_type = self.expected_types.get(column, None)
        if expected_type:
            invalid_types = ~series.apply(lambda x: isinstance(x, expected_type))
            invalid_type_count = invalid_types.sum()
            invalid_type_ratio = invalid_types.mean()
            return {"invalid_type_count": int(invalid_type_count), "invalid_type_ratio": float(round(invalid_type_ratio, 2))}
        return None

    def calculate_outlier_ratio(self, series, column):
        """
        이상치 비율을 계산하는 함수.
        Z-점수와 IQR 기법을 결합하여 이상치를 탐지함.
        Args:
            series (pd.Series): 평가할 피처 데이터.
            column (str): 평가할 피처 이름.
        Returns:
            dict: 이상치 개수 및 비율을 포함한 딕셔너리.
        설명: Z-점수 기반 탐지와 IQR 기법을 함께 사용하여 이상치를 탐지합니다.
             Z-점수 임계값과 IQR 범위를 벗어나는 값이 겹칠 경우 이상치로 판단합니다.
        """
        # 숫자형 데이터만 남기고 NaN 값 제거
        series = pd.to_numeric(series, errors='coerce').dropna()
        self.outlier_threshold = 3
        # Z-점수 기반 이상치 탐지
        mean = series.mean()
        std = series.std()
        if std == 0:
            return {"outlier_count": 0, "outlier_ratio": 0.0}  # 표준편차가 0일 경우 이상치가 없음
        z_scores = np.abs((series - mean) / std)
        z_outliers = z_scores > self.outlier_threshold
        # IQR 기반 이상치 탐지
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        iqr_outliers = (series < lower_bound) | (series > upper_bound)
        # Z-점수와 IQR 모두에서 이상치로 판정된 값
        combined_outliers = z_outliers & iqr_outliers
        outlier_count = combined_outliers.sum()
        outlier_ratio = combined_outliers.mean()
        return outlier_ratio

    def get_metrics(self):
        """
        각 피처별 품질 메트릭스를 계산하여 딕셔너리로 반환하는 함수.
        
        Returns:
            dict: 피처별 품질 메트릭스를 포함한 딕셔너리.
        
        설명: 각 피처에 대해 범위를 벗어나는 비율, 잘못된 데이터 타입 비율 등을 포함한 품질 메트릭스를 계산함.
        """
        feature_metrics = {}
        for column in self.df.columns:
            series = self.df[column]
            feature_metrics[column] = {
                "out_of_range_ratio": self.calculate_out_of_range_ratio(series, column),
                "invalid_data_type_ratio": self.calculate_invalid_data_type_ratio(series, column),
                # 추가 메트릭 계산 함수 호출 가능
                "outlier_ratio": self.calculate_outlier_ratio(series, column),  # 이상치 비율 계산 추가
            }
        return feature_metrics
