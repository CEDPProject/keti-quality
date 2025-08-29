import matplotlib.pyplot as plt
import numpy as np

def visualize_metrics(metrics):
    """
    데이터 품질 지표(A, B, C, D, Overall Quality)를 막대 그래프로 시각화하는 함수.
    
    Args:
        metrics (dict): 데이터 품질 지표 딕셔너리.
    """
    # main_duration 제거 & overall_data_quality 별도 저장
    overall_quality = metrics.pop("overall_data_quality", None)  
    metrics = {k: v for k, v in metrics.items() if k != "main_duration(s)"}

    labels = list(metrics.keys())
    values = list(metrics.values())

    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red', 'purple']  # 막대 색상 (A, B, C, D)

    bars = plt.bar(labels, values, color=colors, alpha=0.7)

    # 값 표시
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), 
                 ha='center', va='bottom', fontsize=12)

    # 그래프 스타일 설정
    plt.ylim(0, 1.1)  # Y축 0~1.1 사이로 제한 (모든 지표가 0~1이므로)
    plt.ylabel("Score")
    plt.title("Data Quality Metrics Visualization")
    plt.xticks(rotation=30, ha='right')  # X축 레이블 회전

    # Overall Quality 별도 텍스트로 표시 (그래프 아래)
    if overall_quality is not None:
        plt.figtext(0.5, -0.05, f"Overall Data Quality: {overall_quality:.2f}%", 
                    fontsize=14, fontweight='bold', color='darkred', ha='center')

    # 그래프 출력
    plt.show()


def compare_metrics(original_metrics, processed_metrics):
    """
    원본 데이터와 전처리된 데이터의 품질 지표(A, B, C, D, Overall Quality)를 비교하는 막대 그래프.
    
    Args:
        original_metrics (dict): 원본 데이터의 품질 지표.
        processed_metrics (dict): 전처리된 데이터의 품질 지표.
    """
    # main_duration 제외 & overall_data_quality 별도 저장
    overall_quality_original = original_metrics.pop("overall_data_quality", None)
    overall_quality_processed = processed_metrics.pop("overall_data_quality", None)
    
    # main_duration 제거
    original_metrics = {k: v for k, v in original_metrics.items() if k != "main_duration(s)"}
    processed_metrics = {k: v for k, v in processed_metrics.items() if k != "main_duration(s)"}
    
    labels = list(original_metrics.keys())  # X축 라벨 (A, B, C, D)
    values_original = list(original_metrics.values())
    values_processed = list(processed_metrics.values())

    x = np.arange(len(labels))  # X축 인덱스
    width = 0.35  # 막대 너비

    plt.figure(figsize=(10, 6))

    # 막대 그래프 생성 (원본 vs 전처리)
    bars1 = plt.bar(x - width/2, values_original, width, label='Corrupted', color='gray', alpha=0.7)
    bars2 = plt.bar(x + width/2, values_processed, width, label='Preprocessed', color='blue', alpha=0.7)

    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), 
                     ha='center', va='bottom', fontsize=12)

    # 그래프 스타일 설정
    plt.ylim(0, 1.1)  # Y축 0~1.1 사이로 제한 (모든 지표가 0~1이므로)
    plt.ylabel("Score")
    plt.title("Comparison of Data Quality Metrics (Corrupted vs Preprocessed)")
    plt.xticks(x, labels, rotation=30, ha='right')  # X축 레이블 회전
    plt.legend()

    # Overall Data Quality 별도 텍스트로 표시
    if overall_quality_original is not None and overall_quality_processed is not None:
        plt.figtext(0.5, -0.1, 
                    f"Overall Data Quality - Original: {overall_quality_original:.2f}%   |   Processed: {overall_quality_processed:.2f}%", 
                    fontsize=14, fontweight='bold', color='darkred', ha='center')

    # 그래프 출력
    plt.show()