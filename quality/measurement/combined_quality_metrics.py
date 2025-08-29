import pandas as pd
import numpy as np
from .feature_quality_metrics import FeatureQualityMetrics
from .overall_quality_metrics import OverallQualityMetrics
from .data_quality_metrices import DataQualityMetrics

class CombinedQualityMetrics:
    """
    전체 품질 메트릭스와 피처별 품질 메트릭스를 결합하는 클래스.
    범위 검사 및 데이터 타입 검사 결과를 포함하여 품질을 종합적으로 평가
    """
    
    def __init__(self, df, range_limits=None, expected_types=None, error_values = None , z_threshold=None, percentile_range=None):
        self.df = df
        self.overall_metrics = OverallQualityMetrics(df)
        self.feature_metrics = FeatureQualityMetrics(df, range_limits, expected_types)
        # self.data_metrics = DataQualityMetrics(df, range_limits, expected_types, error_values, z_threshold, percentile_range)
        # 필요한 인자만 kwargs에 추가
        data_metrics_kwargs = {}
        if z_threshold is not None:
            data_metrics_kwargs["z_threshold"] = z_threshold
        if percentile_range is not None:
            data_metrics_kwargs["percentile_range"] = percentile_range

        self.data_metrics = DataQualityMetrics(df, range_limits, expected_types, error_values, **data_metrics_kwargs)
    def get_combined_metrics(self):
        """
        전체 품질 메트릭스와 피처별 품질 메트릭스를 결합하여 반환하는 함수.
        - 결과: 전체 품질 메트릭스와 각 피처별 품질 메트릭스가 포함된 딕셔너리
        """
        combined_metrics = {
            "overall_quality_metrics": self.overall_metrics.get_metrics(),
            "feature_quality_metrics": self.feature_metrics.get_metrics(),
            "data_quality_metrics": self.data_metrics.get_metrics()
        }
        return combined_metrics