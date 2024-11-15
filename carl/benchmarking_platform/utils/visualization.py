import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats as stats
from utils.stats import get_stats
import seaborn as sns
from scipy import stats

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


def compile_metrics_dataframe(data_path, thresholds=[1], 
                              statistic_names=['RMSE', 'MAE', 'KendallTau', 'MedianAE', 'Error_below_1']):
    # Load the saved data
    data = pd.read_pickle(data_path)

    # Extract the relevant data
    reals_list = data['reals_list']
    predictions_list = data['predictions_list']
    combined_titles = data['combined_titles']

    # Calculate the specified statistic for each combination
    all_stats = []  # to hold all statistic values for each combination
    for real, pred, title in zip(reals_list, predictions_list, combined_titles):
        stat_values = []
        for real_vals, pred_vals in zip(real, pred):
            stat_dict = get_stats(real_vals, pred_vals, thresholds)
            stat_values.append(stat_dict[statistic_name])
        all_stats.append(stat_values)

    # Convert to DataFrame
    df_stat = pd.DataFrame(all_stats).T

    # Debugging: Check lengths
    print(f"Length of combined_titles: {len(combined_titles)}")
    print(f"Shape of df_stat: {df_stat.shape}")

    if df_stat.shape[1] != len(combined_titles):
        raise ValueError(f"Expected {len(combined_titles)} columns, but got {df_stat.shape[1]}.")

    df_stat.columns = combined_titles

    # Initialize lists to hold the data for the new DataFrame
    models = []
    descriptors = []
    stat_values = []

    # Loop through each column and split into model, descriptor, and append the data
    for col in df_stat.columns:
        model, descriptor = col.split(' (')
        descriptor = descriptor.rstrip(')')
        models.extend([model] * df_stat.shape[0])
        descriptors.extend([descriptor] * df_stat.shape[0])
        stat_values.extend(df_stat[col].values)

    # Create a new DataFrame with the three columns
    new_df = pd.DataFrame({
        'Model': models,
        'Descriptor': descriptors,
        statistic_name: stat_values
    })

    # Group by Model and Descriptor, and calculate mean and 90% confidence intervals
    grouped_df = new_df.groupby(['Model', 'Descriptor']).agg(
        stat_mean=(statistic_name, 'mean'),
        stat_std=(statistic_name, 'std'),
        stat_count=(statistic_name, 'count')
    ).reset_index()

    # Calculate 90% confidence interval
    grouped_df['CI'] = grouped_df.apply(
        lambda row: stats.norm.interval(0.90, loc=row['stat_mean'], scale=row['stat_std'] / np.sqrt(row['stat_count'])),
        axis=1
    )

    # Convert the 'Model' and 'Descriptor' columns to categorical types with the defined order
    grouped_df['Model'] = pd.Categorical(grouped_df['Model'], categories=model_order, ordered=True)
    grouped_df['Descriptor'] = pd.Categorical(grouped_df['Descriptor'], categories=descriptor_order, ordered=True)

    # Re-sort the DataFrame based on the new categorical order
    grouped_df = grouped_df.sort_values(['Descriptor', 'Model'])

    # Pivot the DataFrame for heatmap
    heatmap_data = grouped_df.pivot(index="Descriptor", columns="Model", values="stat_mean")

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=False, cmap="coolwarm", fmt=".2f", linewidths=.5, cbar=True)

    plt.title(f"Heatmap of Mean {statistic_name} Values by Model and Descriptor")
    plt.xlabel("Model")
    plt.ylabel("Descriptor")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    # Add text boxes with the 90% CI
    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            mean_val = heatmap_data.iloc[i, j]
            ci = grouped_df[(grouped_df['Descriptor'] == heatmap_data.index[i]) & 
                            (grouped_df['Model'] == heatmap_data.columns[j])]['CI'].values[0]
            ci_text = f"{ci[0]:.2f} - {ci[1]:.2f}"
            plt.text(j + 0.5, i + 0.5, f"{mean_val:.2f}\n{ci_text}",
                     ha='center', va='center', color='black', fontsize=8)

    plt.show()


def density_plot_single(reals_list, predictions_list, molregnos_list, print_stats=True, bounds=None, title=None, name=None, thresholds=1):
    """
    Generate a density plot comparing real vs predicted values for a single model/descriptor combination over 10 splits.

    Parameters:
    reals_list (list of lists): List of real values for multiple datasets (10 splits)
    predictions_list (list of lists): List of predicted values for multiple datasets (10 splits)
    molregnos_list (list of lists): List of molecular IDs for multiple datasets (10 splits)
    print_stats (bool): Whether to print statistical metrics on the plots
    bounds (tuple): Bounds for the axes
    title (str): Title for the plot
    name (str): File name to save the plot; if None, the plot is shown
    thresholds (list or float): Threshold(s) for calculating the fraction of errors below this value
    """

    fig, ax = plt.subplots(figsize=(10, 8))

    # Collect stats for each set of predictions
    stats_list = [get_stats(reals, predictions, thresholds) for reals, predictions in zip(reals_list, predictions_list)]

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
    mrn = [item for sublist in molregnos_list for item in sublist]
    real = [item for sublist in reals_list for item in sublist]
    prediction = [item for sublist in predictions_list for item in sublist]

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
        ax.set_title(title, fontsize=14)

    ax.set_aspect('equal', 'box')

    if name:
        plt.savefig(f'{name}.png', dpi=800, bbox_inches='tight')
    else:
        plt.show()

import pandas as pd
import numpy as np
from scipy import stats

import pandas as pd
import numpy as np
from scipy import stats

def compile_metrics_dataframe(data_path, thresholds=[1], 
                              statistic_names=['RMSE', 'MAE', 'KendallTau', 'MedianAE', 'Error_below_1'],train_set='full_range'):
    """
    Compile a DataFrame with the mean and 90% confidence intervals for specified metrics.

    Parameters:
    data_path (str): Path to the pickled data file.
    thresholds (list): List of thresholds for calculating the fraction of errors below certain values.
    statistic_names (list): List of metric names to include in the DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with metrics, mean values, and confidence intervals.
    """
    # Load the saved data
    data = pd.read_pickle(data_path)

    # Extract relevant data
    reals_list = data['reals_list']
    if train_set=='full_range':
        predictions_list = data['predictions_train_full_range']
    elif train_set=='mid_range':
        predictions_list = data['predictions_train_mid_range']
    combined_titles = data['combined_titles']

    # Initialize a dictionary to hold metric data for each combination
    all_stats = {stat: [] for stat in statistic_names}

    # Calculate statistics for each combination
    for real, pred, title in zip(reals_list, predictions_list, combined_titles):
        metric_data = {stat: [] for stat in statistic_names}
        for real_vals, pred_vals in zip(real, pred):
            stat_dict = get_stats(real_vals, pred_vals, thresholds)
            for stat in statistic_names:
                metric_data[stat].append(stat_dict[stat])
        
        # Append the means and confidence intervals for each metric
        for stat in statistic_names:
            mean_val = np.mean(metric_data[stat])
            if stat == 'KendallTau':
                # Check for constant or insufficient data
                if len(real_vals) < 2 or len(set(real_vals)) == 1 or len(set(pred_vals)) == 1:
                    print(f'Warning: Kendall Tau is NaN due to constant or insufficient data.')
                    print(f'Real values: {real_vals}')
                    print(f'Predicted values: {pred_vals}')
                    mean_val = np.nan  # Explicitly set to NaN for this case
                elif np.isnan(mean_val):
                    raise ValueError(f'Kendall Tau is NaN. x: {real_vals}, y: {pred_vals}')

            ci_low, ci_high = stats.norm.interval(0.90, loc=mean_val, scale=stats.sem(metric_data[stat]))

            all_stats[stat].append({
                'Model': title.split(' (')[0],
                'Descriptor': title.split(' (')[1].rstrip(')'),
                'Mean': mean_val,
                'CI_Low': ci_low,
                'CI_High': ci_high
            })

    # Convert the collected data to a DataFrame
    df_list = []
    for stat, stat_data in all_stats.items():
        df = pd.DataFrame(stat_data)
        df['Metric'] = stat
        df_list.append(df)

    # Combine all metrics into a single DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)
    # Reorder the columns for readability
    combined_df = combined_df[['Model', 'Descriptor', 'Metric', 'Mean', 'CI_Low', 'CI_High']]

    # Sort the DataFrame by descriptor and model for better readability
    combined_df = combined_df.sort_values(['Descriptor', 'Model', 'Metric']).reset_index(drop=True)

    return combined_df
