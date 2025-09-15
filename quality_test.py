from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    
    data_dir = Path("./data/paprika_F00012_part1.csv")
    data = pd.read_csv(data_dir, parse_dates=["time"], index_col="time")
    
    range_limits = {
        'max_num': {
            'EXT_TMP': 37.76, 'EXT_WIND_DIRECT': 270, 'EXT_WIND_SPEED': 6.33, 
            'INT_CO2': 1148.83, 'INT_REL_HD': 100, 'INT_TMP': 37.00, 
            'MAX_EXT_TMP': 38.10, 'MIN_EXT_TMP': 37.70
        },
        'min_num': {
            'EXT_TMP': -20.82, 'EXT_WIND_DIRECT': 90, 'EXT_WIND_SPEED': 0.00, 
            'INT_CO2': -300.00, 'INT_REL_HD': 18.23, 'INT_TMP': -2.48, 
            'MAX_EXT_TMP': -20.00, 'MIN_EXT_TMP': -21.40
        }
    }

    expected_types = {'EXT_TMP': float, 'EXT_WIND_DIRECT': float, 'EXT_WIND_SPEED': float, 'INT_CO2' : float, 
                    'INT_REL_HD': float, 'INT_TMP':float, 'MAX_EXT_TMP': float, 'MIN_EXT_TMP':float}

    error_values = {
        "INT_CO2": [9999, -9999],  
        "INT_TMP": [9999, -9999],  
    }
    
    # measure data quality
    from quality.measurement.data_quality_metrices import DataQualityMetrics
    import pprint

    print("original_data_quality")
    metrics = DataQualityMetrics(data, range_limits=range_limits, expected_types=expected_types, error_values = error_values, 
                                    z_threshold = 2.5, percentile_range = (0.1, 0.9))
    original_result= metrics.get_metrics()
    pprint.pprint(original_result, sort_dicts=False, width=80)
    
    from quality.measurement.visualisation_data_quality import visualize_metrics
    visualize_metrics(original_result)