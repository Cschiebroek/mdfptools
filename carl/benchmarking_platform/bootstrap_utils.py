
from sklearn.utils import resample
from scipy.stats import kendalltau, pearsonr, spearmanr
import numpy as np

def bootstrap_error_estimate(pred, truth, method, method_name="", alpha=0.95, sample_frac=0.9, iterations=1000):
    """
    Generate a bootstrapped estimate of confidence intervals
    :param pred: list of predicted values
    :param truth: list of experimental values
    :param method: method to evaluate performance, e.g. matthews_corrcoef
    :param alpha: confidence limit (e.g. 0.95 for 95% confidence interval)
    :param sample_frac: fraction to resample for bootstrap confidence interval
    :param iterations: number of iterations for resampling
    :return: lower and upper bounds for confidence intervals
    """
    index_list = range(0, len(pred))
    num_samples = int(len(index_list) * sample_frac)
    stats = []
    for _ in range(0, iterations):
        sample_idx = resample(index_list, n_samples=num_samples)
        pred_sample = [pred[x] for x in sample_idx]
        truth_sample = [truth[x] for x in sample_idx]
        stats.append(method(pred_sample, truth_sample))
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    if method_name in ["root_mean_squared_error", "mean_absolute_error"]:
        upper = np.percentile(stats, p)
    else:
        upper = min(1.0, np.percentile(stats, p))
    median = method(pred, truth)
    return lower, upper, median

def get_kendall_tau(pred, truth):
    """
    Calculate Kendall tau correlation
    :param pred: list of predicted values
    :param truth: list of experimental values
    :return: Kendall tau correlation
    """
    return kendalltau(pred, truth).correlation

def get_pearson_r(pred, truth):
    """
    Calculate Pearson correlation
    :param pred: list of predicted values
    :param truth: list of experimental values
    :return: Pearson correlation
    """
    return pearsonr(pred, truth).correlation

def get_spearman_rho(pred, truth):
    """
    Calculate Spearman rho correlation
    :param pred: list of predicted values
    :param truth: list of experimental values
    :return: Spearman rho correlation
    """
    return spearmanr(pred, truth).correlation

def get_fraction_withing_1_log(pred, truth):
    """
    Calculate the fraction of predicted values that are within 1 log of the experimental values
    :param pred: list of predicted values
    :param truth: list of experimental values
    :return: fraction of predicted values within 1 log of the experimental values
    """
    return np.mean(np.abs(np.array(pred) - np.array(truth)) < 1)