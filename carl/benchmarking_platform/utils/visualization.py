import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
from utils.stats import get_stats

def density_plots(reals_list, predictions_list, molregnos_list, print_stats=True, bounds=None, title=None, name=None, dims=(1, 3), thresholds=1):
    """
    Generate density plots comparing real vs predicted values for multiple datasets.

    Parameters:
    reals_list (list of lists): List of real values for multiple datasets
    predictions_list (list of lists): List of predicted values for multiple datasets
    molregnos_list (list of lists): List of molecular IDs for multiple datasets
    print_stats (bool): Whether to print statistical metrics on the plots
    bounds (tuple): Bounds for the axes
    title (list or str): Titles for the plots
    name (str): File name to save the plot; if None, the plot is shown
    dims (tuple): Dimensions for the subplot grid
    thresholds (list or float): Threshold(s) for calculating the fraction of errors below this value
    """
    assert len(reals_list) <= dims[0] * dims[1]
    fig, axes = plt.subplots(dims[0], dims[1], figsize=(10 * dims[1], 8 * dims[0]))
    axes = axes.flatten()

    for i, (reals, predictions, molregnos) in enumerate(zip(reals_list, predictions_list, molregnos_list)):
        ax = axes[i] if len(reals_list) > 1 else axes

        # Debug: Check for NaNs in reals and predictions
        if np.any(pd.isnull(reals)) or np.any(pd.isnull(predictions)):
            print(f"NaN detected in reals or predictions for split {i+1}")
            print(f"Reals: {reals}")
            print(f"Predictions: {predictions}")

        # Collect stats for each set of predictions
        stats_list = [get_stats(r, p, thresholds) for r, p in zip(reals, predictions)]
        
        # Debug: Check for NaNs in stats_list
        if np.any(pd.isnull([stat['RMSE'] for stat in stats_list])):
            print(f"NaN detected in RMSE for split {i+1}")
            print(f"Stats List: {stats_list}")

        # Calculate means and confidence intervals for each metric
        rmse_mean = np.mean([stat['RMSE'] for stat in stats_list])
        mae_mean = np.mean([stat['MAE'] for stat in stats_list])
        kt_mean = np.mean([stat['KendallTau'] for stat in stats_list])
        median_AE_mean = np.mean([stat['MedianAE'] for stat in stats_list])

        rmse_90_low, rmse_90_high = stats.norm.interval(0.90, loc=rmse_mean, scale=stats.sem([stat['RMSE'] for stat in stats_list]))
        mae_90_low, mae_90_high = stats.norm.interval(0.90, loc=mae_mean, scale=stats.sem([stat['MAE'] for stat in stats_list]))
        kt_90_low, kt_90_high = stats.norm.interval(0.90, loc=kt_mean, scale=stats.sem([stat['KendallTau'] for stat in stats_list]))
        median_AE_90_low, median_AE_90_high = stats.norm.interval(0.90, loc=median_AE_mean, scale=stats.sem([stat['MedianAE'] for stat in stats_list]))

        ebo_means = {}
        ebo_cis = {}
        if isinstance(thresholds, list):
            for threshold in thresholds:
                ebo_values = [stat[f'Error_below_{threshold}'] for stat in stats_list]
                ebo_means[threshold] = np.mean(ebo_values)
                ebo_cis[threshold] = stats.norm.interval(0.90, loc=np.mean(ebo_values), scale=stats.sem(ebo_values))
        else:
            ebo_values = [stat[f'Error_below_{thresholds}'] for stat in stats_list]
            ebo_means[thresholds] = np.mean(ebo_values)
            ebo_cis[thresholds] = stats.norm.interval(0.90, loc=np.mean(ebo_values), scale=stats.sem(ebo_values))

        # Group data by molregno to calculate mean predictions and true values
        mrn = [item for sublist in molregnos for item in sublist]
        real = [item for sublist in reals for item in sublist]
        prediction = [item for sublist in predictions for item in sublist]

        df = pd.DataFrame({'molregno': mrn, 'real': real, 'prediction': prediction})
        df = df.groupby('molregno').mean()
        real = df['real'].tolist()
        prediction = df['prediction'].tolist()

        # Plotting
        ax.plot([min(prediction + real), max(prediction + real)], [min(prediction + real), max(prediction + real)], 'k-')
        ax.plot([min(prediction + real), max(prediction + real)], [min(prediction + real) - 1, max(prediction + real) - 1], 'k--')
        ax.plot([min(prediction + real), max(prediction + real)], [min(prediction + real) + 1, max(prediction + real) + 1], 'k--')

        dens_u = sm.nonparametric.KDEMultivariate(data=[real, prediction], var_type='cc', bw='normal_reference')
        z = dens_u.pdf([real, prediction])

        sc = ax.scatter(real, prediction, lw=0, c=z, s=10, alpha=0.9)

        ax.set_xlabel(r'Exp. VP (log10 Pa)', fontsize=14)
        ax.set_ylabel(r'Predicted VP (log10 Pa)', fontsize=14)
        ax.grid(True, which="both")

        if bounds is None:
            lower = min(prediction + real) - 2
            upper = max(prediction + real) + 2
        else:
            lower = bounds[0]
            upper = bounds[1]

        ax.axis([lower, upper, lower, upper])


        if print_stats:
            stats_text = (
                f'RMSE: {rmse_mean:.2f} ({rmse_90_low:.2f}-{rmse_90_high:.2f})\n'
                f'Median AE: {median_AE_mean:.2f} ({median_AE_90_low:.2f}-{median_AE_90_high:.2f})\n'
                f'Mean AE: {mae_mean:.2f} ({mae_90_low:.2f}-{mae_90_high:.2f})\n'
                f'Kendalls Tau: {kt_mean:.2f} ({kt_90_low:.2f}-{kt_90_high:.2f})'
            )

            for threshold, ebo_mean in ebo_means.items():
                ebo_low, ebo_high = ebo_cis[threshold]
                stats_text += f'\nFrac AE < {threshold}: {ebo_mean:.2f} ({ebo_low:.2f}-{ebo_high:.2f})'

            ax.text(0.45, 0.25, stats_text,
                    transform=ax.transAxes, fontsize=14, verticalalignment='top',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

        if title is not None:
            ax.set_title(title[i] if len(reals_list) > 1 else title, fontsize=14)

        ax.set_aspect('equal', 'box')
    # #tight layout
    # plt.tight_layout()
    if name:
        plt.savefig(f'{name}.png', dpi=800, bbox_inches='tight')
    else:
        plt.show()
