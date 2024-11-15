# utils/statistics.py

import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, median_absolute_error
import scipy.stats as stats

def get_stats(x, y, thresholds):
    """
    Calculate various statistical metrics between two arrays.

    Parameters:
    - x (array-like): The first array.
    - y (array-like): The second array.
    - thresholds (float or list of floats): The threshold(s) used for calculating the Error Below Threshold (EBO).

    Returns:
    - stats_dict (dict): A dictionary containing the calculated statistical metrics:
        - 'RMSE' (float): Root Mean Squared Error.
        - 'MAE' (float): Mean Absolute Error.
        - 'KendallTau' (float): Kendall's Tau correlation coefficient.
        - 'MedianAE' (float): Median Absolute Error.
        - 'Error_below_{threshold}' (float or list of floats): Error Below Threshold (EBO) for each specified threshold.

    Note:
    - If thresholds is a single float, a single EBO value will be calculated.
    - If thresholds is a list of floats, multiple EBO values will be calculated, each corresponding to a threshold in the list.
    """
    RMSE = root_mean_squared_error(np.array(x), np.array(y))
    MAE = mean_absolute_error(np.array(x), np.array(y))
    #if tresholds is a list, iterate. otherwise use the one value
    if isinstance(thresholds, list):
        EBO = [np.mean(np.abs(np.array(x) - np.array(y)) < threshold) for threshold in thresholds]
    else:
        EBO = np.mean(np.abs(np.array(x) - np.array(y)) < thresholds)
    KT = stats.kendalltau(x, y)[0]
    if np.isnan(KT):
        #raise error, and print out the x and y values
        raise ValueError(f'Kendall Tau is NaN. x: {x}, y: {y}')
    median_AE = median_absolute_error(np.array(x), np.array(y))

    #return dict
    stats_dict = {
        'RMSE': RMSE,
        'MAE': MAE,
        'KendallTau': KT,
        'MedianAE': median_AE
    }
    if isinstance(thresholds, list):
        for i, threshold in enumerate(thresholds):
            stats_dict[f'Error_below_{threshold}'] = EBO[i]
    else:
        stats_dict[f'Error_below_{thresholds}'] = EBO

    return stats_dict
