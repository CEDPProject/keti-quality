import numpy as np
import pandas as pd
from scipy.special import expit  # 시그모이드 함수 (expit)

class DataQualityMetrics:
    def __init__(self, df, range_limits=None, expected_types=None, error_values = None , z_threshold=2.5, percentile_range=(0.05, 0.95)):
        self.df = df
        self.range_limits = range_limits
        self.expected_types = expected_types
        self.error_values = error_values
        self.z_threshold = z_threshold  # Z-점수 임계값 저장
        self.percentile_range = percentile_range  # IQR 범위 저장

    def syntactic_accuracy(self):
        """
        전체 데이터프레임(df)의 필드별 형식 정확도를 평가하는 함수.
        (한 필드에서 하나라도 틀린 값이 있으면 그 필드는 0점 처리)

        Args:
            None (클래스 내부에서 `self.df`, `self.expected_types`를 사용)

        Returns:
            float: 구문 데이터 정확성 점수 (0~1 사이)
                - 1.0: 모든 필드의 형식이 올바름
                - 0.0: 모든 필드의 형식이 틀림
                - 올바른 필드 수 / 전체 검사 필드 수로 계산

        설명:
            - `expected_types`에 정의된 컬럼만 검사
            - 한 컬럼에서 하나라도 형식이 틀리면 0점 처리 (완전 일치해야 1점)
            - `timestamp`(인덱스)가 `"%Y-%m-%d %H:%M:%S"` 형식을 따르는지도 검사
            - `expected_types`가 아예 없으면 100% (1.0) 반환
        """
        if not self.expected_types:
            return 1.0  # 기대 타입이 정의되지 않았다면 전체 필드 수에서 제외

        # 기대 타입이 정의된 컬럼만 검사
        target_columns = [col for col in self.df.columns if col in self.expected_types]

        # `expected_types`에 정의된 필드가 없으면 평가 불가능 → 100% 반환
        if not target_columns:
            return 1.0  

        checked_fields = len(target_columns)  # 검사할 필드 개수 (expected_types에 정의된 필드만)
        correct_fields = 0  # 올바른 필드 개수

        # 컬럼별 데이터 타입 검사
        for column in target_columns:
            expected_type = self.expected_types[column]
            if self.df[column].apply(lambda x: isinstance(x, expected_type)).all():
                correct_fields += 1  # 모든 값이 올바른 형식이면 1점 추가

        # 인덱스(`timestamp`) 형식 검사
        try:
            pd.to_datetime(self.df.index, format="%Y-%m-%d %H:%M:%S", errors="raise")
            correct_fields += 1  # 형식이 맞으면 1점 추가
            checked_fields += 1  # timestamp를 검사 대상으로 추가
        except (ValueError, TypeError):  
            pass  # 형식이 틀리면 점수 추가 안 함

        # 정확도 계산 (총 검사 필드 대비 올바른 필드 비율)
        accuracy = correct_fields / checked_fields if checked_fields > 0 else 1.0
        return round(accuracy, 2)
    

    def semantic_accuracy(self):
        """
        데이터프레임 전체의 의미적 데이터 정확성을 평가하는 함수.

        의미적 정확성은 데이터의 중복, 결측치(NaN), 그리고 사용자가 정의한 오류 값의 비율을 고려하여 계산됨.

        수식:
            의미적 정확성 = 1 - (중복 비율 + 결측치 비율 + 오류 값 비율)

        Args:
            없음 (클래스 내부의 self.df를 사용)

        Returns:
            float: 의미적 정확성 점수 (0~1 사이).  
                1.0이면 데이터가 완전히 깨끗하며, 0.0이면 모든 데이터가 중복되었거나 오류가 포함됨.

        예제:
            error_values = {
                "temperature": [9999, -9999],  
                "humidity": [-1, 9999]  
            }

            data_quality = DataQualityMetrics(df, error_values=error_values)
            semantic_score = data_quality.semantic_accuracy()
            print("Semantic Accuracy:", semantic_score)
        """

        # 1. 중복 개수 및 비율 계산 (인덱스 기준)
        duplicate_count = self.df.index.duplicated().sum()
        duplicate_ratio = duplicate_count / len(self.df) if len(self.df) > 0 else 0

        # 중복 제거 후 데이터프레임 생성
        unique_df = self.df[~self.df.index.duplicated(keep='first')]
        unique_total_values = unique_df.size  # 중복 제거 후 전체 값 개수

        # 2. 결측치 개수 및 비율 계산
        missing_count = self.df.isna().sum().sum()
        missing_ratio = missing_count / unique_total_values

        # 3. 사용자 정의 오류 값 개수 및 비율 계산
        error_count = 0
        if self.error_values:  # 오류 값이 정의된 경우만 검사
            for column, error_vals in (self.error_values or {}).items(): 
                if column in self.df.columns:
                    if not isinstance(error_vals, list):
                        error_vals = [error_vals]  # 단일 값도 리스트로 변환
                    error_count += self.df[column].isin(error_vals).sum()

        error_ratio = error_count / unique_total_values if unique_total_values > 0 else 0

        # 4. 의미적 정확성 점수 계산 (1 - (중복 비율 + 결측치 비율 + 오류 값 비율))
        semantic_score = 1 - ((duplicate_ratio + missing_ratio + error_ratio) / 2)

        return float(round(semantic_score, 2))


    def inaccuracy_risk(self, z_threshold=2.5, percentile_range=(0.05, 0.95)):
        """
        이상치 비율을 계산하는 함수.

        - Z-점수와 백분위수 기법을 결합하여 이상치를 탐지.
        - timestamp의 main duration(주요 시간 간격)을 찾고, 이에 맞지 않는 값 개수 계산.

        Args:
            z_threshold (float, optional): Z-점수 기반 이상치 탐지 임계값 (기본값: 2.5).
            percentile_range (tuple, optional): 백분위수 기반 이상치 탐지 범위 (기본값: (0.05, 0.95)).

        Returns:
            float: 최종 부정확성 점수
            float or None: 주요 시간 간격
        """
        numeric_df = self.df.apply(pd.to_numeric, errors="coerce")  # 숫자로 변환
        total_data_count = numeric_df.size

        # duration_mismatch_ratio와 Timestamp Main Duration 계산
        main_duration = None
        duration_mismatch_ratio = 0.0
        if isinstance(self.df.index, pd.DatetimeIndex):
            time_diffs = self.df.index.to_series().diff().dropna().dt.total_seconds()  # 시간 간격(초 단위) 계산
            if not time_diffs.empty:
                main_duration = time_diffs.mode()[0]  # 가장 많이 나타나는 간격 (최빈값)
                mismatch_count = (time_diffs != main_duration).sum()  # 다른 간격의 개수
                duration_mismatch_ratio = mismatch_count / len(time_diffs)  # 비율 계산

        total_outlier_count = 0

        for column in numeric_df.columns:
            col_data = numeric_df[column].dropna()
            if col_data.empty:
                continue
            
            # Z-점수 기반 이상치 탐지
            mean, std = col_data.mean(), col_data.std(ddof=0)
            if std == 0:
                z_outliers = pd.Series(False, index=col_data.index)
            else:
                z_scores = np.abs((col_data - mean) / std)
                z_outliers = z_scores > z_threshold

            # 백분위수 기반 이상치 탐지
            Q_low, Q_high = col_data.quantile(percentile_range[0]), col_data.quantile(percentile_range[1])
            lower_bound, upper_bound = Q_low, Q_high  # 기존 IQR 방식 제거, 백분위수만 활용
            percentile_outliers = (col_data < lower_bound) | (col_data > upper_bound)

            # 두 기준에서 모두 이상치로 판정된 값
            combined_outliers = z_outliers & percentile_outliers
            total_outlier_count += combined_outliers.sum()
        
        total_outlier_ratio = total_outlier_count / total_data_count if total_data_count > 0 else 0.0

        # 최종 점수 계산 (이상치 비율 + 시간 불규칙 비율)
        inaccuracy_score = float(round((total_outlier_ratio + duration_mismatch_ratio) / 2, 2))

        return inaccuracy_score, float((round(main_duration, 2) if main_duration else None))
        

    def range_accuracy(self):
        """
        데이터 범위 정확성 평가 함수.

        - `range_limits`에 최소/최대 범위가 정의된 필드만 평가.
        - 평가 대상 필드의 모든 값이 범위 내에 존재해야 1점, 하나라도 벗어나면 0점.

        Args:
            None (클래스 내부 속성 사용)
            - `self.df`: 평가할 데이터프레임
            - `self.range_limits`: 필드별 최소/최대 허용 범위를 담은 딕셔너리
              - `self.range_limits['min_num']`: 최소 허용값 (컬럼별)
              - `self.range_limits['max_num']`: 최대 허용값 (컬럼별)

        Returns:
            float: 
                - 평가 대상 필드 중 모든 값이 허용 범위 내에 있는 필드의 비율 (0~1 사이)
                - `valid_field_count / total_fields` 계산
                - 소수점 2자리까지 반올림 (`float(round(accuracy_ratio, 2))`)
                - 평가 대상 필드가 없을 경우 `1.0` 반환 (100%)
        """
        # 평가 대상 필드 목록 (범위가 정의된 필드만)
        target_columns = [
            col for col in self.df.columns
            if col in self.range_limits.get('min_num', {}) and col in self.range_limits.get('max_num', {})
        ]

        if not target_columns:  # 평가할 필드가 없으면 100% 처리
            return 1.0

        valid_field_count = 0  # 모든 값이 범위 내에 있는 필드 개수
        total_fields = len(target_columns)  # 평가 대상 필드 수

        for column in target_columns:
            # 숫자형 변환 (NaN 허용)
            series = pd.to_numeric(self.df[column], errors='coerce')

            # 최소/최대 범위 가져오기
            min_limit = self.range_limits['min_num'][column]
            max_limit = self.range_limits['max_num'][column]

            # 모든 값이 범위 내 존재하면 1점 추가
            if ((series >= min_limit) & (series <= max_limit)).all():
                valid_field_count += 1

        # 정확도 계산
        accuracy_ratio = valid_field_count / total_fields
        return float(round(accuracy_ratio, 2))
    
    def overall_data_quality(self, A, B, C, D):
        """
        최종 데이터 품질 정확성 계산
        """

        return float((A + B + (1 - C) + D) / 4 * 100)
    

    def get_metrics(self):
        """
        전체 데이터셋에 대한 품질 메트릭스를 계산하여 딕셔너리로 반환하는 함수.
        
        Returns:
            dict: 품질 메트릭스를 포함한 딕셔너리.
        
        설명: 전체 데이터프레임(df)에 대해 구문 정확성(A), 의미 정확성(B), 
            부정확성 위험(C), 범위 정확성(D)을 계산하고, 최종 품질 점수를 포함함.
        """
        # 1. 평가 지표 계산
        A = self.syntactic_accuracy()
        B = self.semantic_accuracy()
        
        # inaccuracy_risk 결과에서 C 값과 main_duration 추출
        inaccuracy_result = self.inaccuracy_risk()

        C, main_duration = inaccuracy_result
        D = self.range_accuracy()

        # 2. 최종 품질 점수 계산
        overall_quality = self.overall_data_quality(A, B, C, D)

        # 3. 결과 딕셔너리 반환
        return {
            "syntactic_accuracy_A": A,
            "semantic_accuracy_B": B,
            "risk_of_inaccuracy_C": C,
            "main_duration(s)": main_duration,
            "range_accuracy_D": D,
            "overall_data_quality": overall_quality,   
        }