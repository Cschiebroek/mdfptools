from sklearn.model_selection import RepeatedKFold
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pickle
from sklearn.preprocessing import StandardScaler

def train_grouped_kfold(df,params,descriptors,name='test',split_on = 'molregno',y='vp_log10pa',n_splits=5,n_repeats=3,random_state=2652124,normalize_x=False):
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    X = df[split_on].to_numpy()
    output = ([],[],[],[],[],[],[],[],[],[],[])
    for i, (train_index, test_index) in enumerate(rkf.split(X)):
        train_all = df[df[split_on].isin(X[train_index])]
        test_all = df[df[split_on].isin(X[test_index])]

        train_X = train_all[descriptors]
        train_y = train_all[y]
        test_X = test_all[descriptors]
        test_y = test_all[y]
        molregnos_test = test_all[split_on]

        if normalize_x:
            scaler = StandardScaler()
            train_X = scaler.fit_transform(train_X)
            test_X = scaler.transform(test_X)

        dtrain = xgb.DMatrix(train_X, label=train_y)
        dtest = xgb.DMatrix(test_X, label=test_y)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=999,
            verbose_eval=False
        )

        preds = model.predict(dtest)
        output[0].append(test_y.to_list())
        output[1].append(list(preds))
        output[2].append(molregnos_test.to_list())


        #print the RMSEs
        rmse = mean_squared_error(test_y, preds, squared=False)
        print(f"Fold {i + 1}: RMSE = {rmse}")

    with open(f'{name}.pkl', 'wb') as f:
        pickle.dump(output, f)

    return output
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats as stats

def getStatValues_v2(x,y,treshold=1):
    RMSE = mean_squared_error(np.array(x), np.array(y), squared=False)
    MAE = mean_absolute_error(np.array(x), np.array(y))
    EBO = np.mean(np.abs(np.array(x) - np.array(y)) < treshold)
    KT = stats.kendalltau(x, y)[0]

    return RMSE, MAE, EBO, KT
    

def density_plot_multiple_custom(reals, predictions,molregnos, print_stats=True, bounds=None, titles=None,global_title=None,print_spearman=False,name=None):
    num_plots = len(reals)
    print(f'making {num_plots} plots')
    num_cols = min(num_plots, 3)
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate the number of rows needed for the grid
    #if you cant devide by 3, but can divide by 2, do two rows
    if num_plots % 3 != 0 and num_plots % 2 == 0:
        num_rows = 2
        num_cols = 2   
    if num_plots == 4:
        num_rows = 2
        num_cols = 2
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    axes = axes.flatten()  # Flatten the 2D array of axes to 1D
    
    for i in range(num_plots):
        print(f'plot {i+1} of {num_plots}')
        real = reals[i]
        prediction = predictions[i]
        rmses,maes,ebos,kts = [],[],[],[]
        for r,p in zip(real,prediction):
            RMSE, MAE, EBO, KT = getStatValues_v2(r, p)
            rmses.append(RMSE)
            maes.append(MAE)
            ebos.append(EBO)
            kts.append(KT)

        rmse_90_low,rmse_90_high = stats.norm.interval(confidence=0.90, 
                 loc=np.mean(rmses), 
                 scale=stats.sem(rmses)) 
        rmse_mean = np.mean(rmses)
        mae_90_low,mae_90_high = stats.norm.interval(confidence=0.90,
                    loc=np.mean(maes), 
                    scale=stats.sem(maes))
        mae_mean = np.mean(maes)
        ebo_90_low,ebo_90_high = stats.norm.interval(confidence=0.90,
                    loc=np.mean(ebos), 
                    scale=stats.sem(ebos))
        ebo_mean = np.mean(ebos)
        kt_90_low,kt_90_high = stats.norm.interval(confidence=0.90,
                    loc=np.mean(kts), 
                    scale=stats.sem(kts))
        kt_mean = np.mean(kts)
        
                                             
        molregno = molregnos[i]

        mrn = [item for sublist in molregno for item in sublist]
        real = [item for sublist in real for item in sublist]
        prediction = [item for sublist in prediction for item in sublist]
        # #make df of these three, and average over molregno
        df = pd.DataFrame({'molregno': mrn, 'real': real, 'prediction': prediction})
        df = df.groupby('molregno').mean()
        real = df['real'].tolist()
        prediction = df['prediction'].tolist()


        print(f'{len(real)} points')
        
        ax = axes[i]
        
        if bounds is None:
            lower = min(prediction + real) - 2
            upper = max(prediction + real) + 2
        else:
            lower = bounds[0]
            upper = bounds[1]
        
        ax.plot([min(prediction + real), max(prediction + real)], [min(prediction + real), max(prediction + real)], 'k-')
        ax.plot([min(prediction + real), max(prediction + real)], [min(prediction + real) - 1, max(prediction + real) - 1], 'k--')
        ax.plot([min(prediction + real), max(prediction + real)], [min(prediction + real) + 1, max(prediction + real) + 1], 'k--')
        dens_u = sm.nonparametric.KDEMultivariate(data=[real, prediction], var_type='cc', bw='normal_reference')
        z = dens_u.pdf([real, prediction])

        sc = ax.scatter(real, prediction, lw=0, c=z, s=10, alpha=0.9)

        ax.set_xlabel(r'Exp. VP (log10 Pa)', fontsize=14)
        ax.set_ylabel(r'Predicted VP (log10 Pa)', fontsize=14)
        ax.grid(True, which="both")
        ax.axis([lower, upper, lower, upper])
        # ax.text(0.05, 0.95, f'RMSE: {rmse_mean:.2f} ({rmse_90_low:.2f}-{rmse_90_high:.2f}) \nMAE: {mae_mean:.2f} ({mae_90_low:.2f}-{mae_90_high:.2f})\nFraction errors < 1: {ebo_mean:.2f} ({ebo_90_low:.2f}-{ebo_90_high:.2f})\nKendalls Tau: {kt_mean:.2f} ({kt_90_low:.2f}-{kt_90_high:.2f})', transform=ax.transAxes, fontsize=10, verticalalignment='top')
        text_box = ax.text(0.05, 0.25, f'RMSE: {rmse_mean:.2f} ({rmse_90_low:.2f}-{rmse_90_high:.2f}) \nMAE: {mae_mean:.2f} ({mae_90_low:.2f}-{mae_90_high:.2f})\nFraction errors < 1: {ebo_mean:.2f} ({ebo_90_low:.2f}-{ebo_90_high:.2f})\nKendalls Tau: {kt_mean:.2f} ({kt_90_low:.2f}-{kt_90_high:.2f})', 
                    transform=ax.transAxes, fontsize=10, verticalalignment='top',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
        
        if titles is not None and len(titles) > i:
            ax.set_title(titles[i], fontsize=14)
        ax.set_aspect('equal', 'box')
    
    # Remove any unused subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])
    if global_title is not None:
        fig.suptitle(global_title, fontsize=16)
    
    # plt.tight_layout()
    if name:
        plt.savefig(name+'.png', dpi=600)
    plt.show()

 

def density_plot_single_outliers(reals, predictions, molregnos, outliers_reals, outliers_predictions, outliers_molregnos, print_stats=True, bounds=None, title=None, print_spearman=False, name=None):
    num_plots = 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    real = reals[0]
    prediction = predictions[0]
    rmses, maes, ebos, kts = [], [], [], []
    for r, p in zip(real, prediction):
        RMSE, MAE, EBO, KT = getStatValues_v2(r, p)
        rmses.append(RMSE)
        maes.append(MAE)
        ebos.append(EBO)
        kts.append(KT)

    rmse_90_low, rmse_90_high = stats.norm.interval(confidence=0.90, 
                                                   loc=np.mean(rmses), 
                                                   scale=stats.sem(rmses)) 
    rmse_mean = np.mean(rmses)
    mae_90_low, mae_90_high = stats.norm.interval(confidence=0.90,
                                                 loc=np.mean(maes), 
                                                 scale=stats.sem(maes))
    mae_mean = np.mean(maes)
    ebo_90_low, ebo_90_high = stats.norm.interval(confidence=0.90,
                                                 loc=np.mean(ebos), 
                                                 scale=stats.sem(ebos))
    ebo_mean = np.mean(ebos)
    kt_90_low, kt_90_high = stats.norm.interval(confidence=0.90,
                                               loc=np.mean(kts), 
                                               scale=stats.sem(kts))
    kt_mean = np.mean(kts)

    molregno = molregnos[0]

    mrn = [item for sublist in molregno for item in sublist]
    real = [item for sublist in real for item in sublist]
    prediction = [item for sublist in prediction for item in sublist]
    # #make df of these three, and average over molregno
    df = pd.DataFrame({'molregno': mrn, 'real': real, 'prediction': prediction})
    df = df.groupby('molregno').mean()
    real = df['real'].tolist()
    prediction = df['prediction'].tolist()

    ax.plot([min(prediction + real), max(prediction + real)], [min(prediction + real), max(prediction + real)], 'k-')
    ax.plot([min(prediction + real), max(prediction + real)], [min(prediction + real) - 1, max(prediction + real) - 1], 'k--')
    ax.plot([min(prediction + real), max(prediction + real)], [min(prediction + real) + 1, max(prediction + real) + 1], 'k--')
    dens_u = sm.nonparametric.KDEMultivariate(data=[real, prediction], var_type='cc', bw='normal_reference')
    z = dens_u.pdf([real, prediction])

    sc = ax.scatter(real, prediction, lw=0, c=z, s=10, alpha=0.9)

    # Add outliers
    # outliers_real = outliers_reals[0]
    # outliers_prediction = outliers_predictions[0]
    # outliers_molregno = outliers_molregnos[0]
    # outliers_real = [item for sublist in outliers_real for item in sublist]
    # outliers_prediction = [item for sublist in outliers_prediction for item in sublist]
    ax.scatter(outliers_reals, outliers_predictions, color='red', s=50, alpha=0.9)

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
    
    if title is not None:
        ax.set_title(title, fontsize=14)
    
    text_box = ax.text(0.05, 0.25, f'RMSE: {rmse_mean:.2f} ({rmse_90_low:.2f}-{rmse_90_high:.2f}) \nMAE: {mae_mean:.2f} ({mae_90_low:.2f}-{mae_90_high:.2f})\nFraction errors < 1: {ebo_mean:.2f} ({ebo_90_low:.2f}-{ebo_90_high:.2f})\nKendalls Tau: {kt_mean:.2f} ({kt_90_low:.2f}-{kt_90_high:.2f})', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
    #add custom legend: red = outliers
    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='red', label='Difference between values datasets > 0.3')
    #show legend
    plt.legend(handles=[red_patch])
    
    if name:
        plt.savefig(name+'.png', dpi=600)
    plt.show()