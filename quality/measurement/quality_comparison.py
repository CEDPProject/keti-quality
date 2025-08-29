import matplotlib.pyplot as plt

def plot_quality_metrics_comparison(good_result, bad_result):
    """
    good_result와 bad_result의 품질 메트릭스를 비교하는 플롯을 생성하는 함수.
    
    Args:
        good_result (dict): 정상 품질 데이터의 메트릭스 결과
        bad_result (dict): 열화된 품질 데이터의 메트릭스 결과
    """
    # 전체 품질 메트릭스 비교 데이터 준비
    overall_keys = good_result['overall_quality_metrics'].keys()
    overall_good = [good_result['overall_quality_metrics'][key] for key in overall_keys]
    overall_bad = [bad_result['overall_quality_metrics'][key] for key in overall_keys]

    # 피처별 품질 메트릭스 비교 데이터 준비
    feature_names = good_result['feature_quality_metrics'].keys()
    out_of_range_good = [
        good_result['feature_quality_metrics'][feat]['out_of_range_ratio']['out_of_range_ratio'] 
        if good_result['feature_quality_metrics'][feat]['out_of_range_ratio'] else 0
        for feat in feature_names
    ]
    out_of_range_bad = [
        bad_result['feature_quality_metrics'][feat]['out_of_range_ratio']['out_of_range_ratio'] 
        if bad_result['feature_quality_metrics'][feat]['out_of_range_ratio'] else 0
        for feat in feature_names
    ]
    invalid_type_good = [
        good_result['feature_quality_metrics'][feat]['invalid_data_type_ratio']['invalid_type_ratio'] 
        if good_result['feature_quality_metrics'][feat]['invalid_data_type_ratio'] else 0
        for feat in feature_names
    ]
    invalid_type_bad = [
        bad_result['feature_quality_metrics'][feat]['invalid_data_type_ratio']['invalid_type_ratio'] 
        if bad_result['feature_quality_metrics'][feat]['invalid_data_type_ratio'] else 0
        for feat in feature_names
    ]

    # 플롯 생성
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle("Quality Metrics Comparison: Good vs Bad Data")

    # 전체 품질 메트릭스 플롯
    ax1.bar(overall_keys, overall_good, alpha=0.7, label='Good Data', color='blue')
    ax1.bar(overall_keys, overall_bad, alpha=0.7, label='Bad Data', color='red')
    ax1.set_title("Overall Quality Metrics")
    ax1.set_ylabel("Metric Value")
    ax1.legend()

    # 피처별 품질 메트릭스 플롯
    bar_width = 0.35
    index = range(len(feature_names))

    ax2.bar([i - bar_width / 2 for i in index], out_of_range_good, width=bar_width, label='Out of Range (Good)', color='blue', alpha=0.6)
    ax2.bar([i + bar_width / 2 for i in index], out_of_range_bad, width=bar_width, label='Out of Range (Bad)', color='red', alpha=0.6)
    ax2.bar([i - bar_width / 2 for i in index], invalid_type_good, width=bar_width, bottom=out_of_range_good, label='Invalid Type (Good)', color='lightblue', alpha=0.6)
    ax2.bar([i + bar_width / 2 for i in index], invalid_type_bad, width=bar_width, bottom=out_of_range_bad, label='Invalid Type (Bad)', color='salmon', alpha=0.6)
    
    ax2.set_xticks(list(index))
    ax2.set_xticklabels(feature_names)
    ax2.set_title("Feature-Specific Quality Metrics")
    ax2.set_ylabel("Metric Ratio")
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
