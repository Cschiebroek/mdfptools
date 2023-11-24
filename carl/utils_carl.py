import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import linregress
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
import json

import psycopg2
import pandas as pd

import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

from functools import reduce

hostname = 'scotland'
dbname = 'cs_mdfps'
username = 'cschiebroek'



def getStatValues(x,y,get_spearman=False):
    slope, intercept, r_value, p_value, std_err = linregress(x,y)
    this_rmse = mean_squared_error(np.array(x), np.array(y), squared=False)
    this_mue = mean_absolute_error(np.array(x), np.array(y))
    if not get_spearman:
        return slope, intercept, r_value**2, this_rmse, this_mue
    else:
        import scipy.stats as stats
        spearman = stats.spearmanr(x,y)
        return slope, intercept, r_value**2, this_rmse, this_mue, spearman[0]

def density_plot(real,prediction,print_stats=True,bounds=None,title=None):
    slope_mdfp_d, intercept_mdfp_d, r2, this_rmse, this_mae = getStatValues(real, prediction)
    if print_stats:
        print('RMSE: ', this_rmse)
        print('MAE: ', this_mae)
        print('R2: ', r2)
    fsize = 20
    fig = plt.figure(1, figsize=(10, 6.15))
    ax = plt.subplot(111)
    if bounds is None:
        lower = min(prediction + real) - 2
        upper = max(prediction + real) + 2
    else:
        lower = bounds[0]
        upper = bounds[1]
    x = np.linspace(lower, upper,100)
    y = slope_mdfp_d*x+intercept_mdfp_d
    plt.plot(x, y, '-r')
    plt.plot([min(prediction + real), max(prediction + real)], [min(prediction + real), max(prediction + real)], 'k-')
    plt.plot([min(prediction + real), max(prediction + real)], [min(prediction + real)-1, max(prediction + real) - 1], 'k--')
    plt.plot([min(prediction + real), max(prediction + real)], [min(prediction + real)+1, max(prediction + real)+1], 'k--')

    import statsmodels.api as sm
    dens_u = sm.nonparametric.KDEMultivariate(data=[real, prediction],var_type='cc', bw='normal_reference')
    z = dens_u.pdf([real, prediction])

    sc = plt.scatter(real, prediction, lw=0, c=z, s=10, alpha = 0.9)

    plt.xlabel(r'Exp. VP (log10 Pa)', fontsize=fsize)
    plt.ylabel(r'Predicted VP (log10 Pa)', fontsize=fsize)
    plt.setp(ax.get_xticklabels(), fontsize=fsize)
    plt.setp(ax.get_yticklabels(), fontsize=fsize)
    plt.grid(1,"both")
    plt.axis([lower, upper, lower, upper])
    plt.text(0.05, 0.95, f'RMSE: {this_rmse:.2f}\nMAE: {this_mae:.2f}\nR2: {r2:.2f}', transform=ax.transAxes, fontsize=fsize, verticalalignment='top')
    #make square
    if title is not None:
        plt.title(title, fontsize=fsize)
    ax.set_aspect('equal', 'box')
    plt.show()


def train_pred_xgboost(df,params,splits=5,return_confids=False,print_fold_rmses=False,average = False):
    gkf = GroupKFold(n_splits=splits)

    # Create an empty list to store the indices of each fold
    fold_indices = []

    # Group the data by 'molregno'
    groups = df['molregno']

    # Iterate over each fold
    for train_idx, test_idx in gkf.split(df, groups=groups):
        fold_indices.append((train_idx, test_idx))

    y = df['vp_log10pa']  
    params = params
    output = ([],[],[],[])

    # Iterate over each fold
    for fold, (train_idx, test_idx) in enumerate(fold_indices):
        # Split the data into train and test sets for this fold
        X_train = np.array(df['mdfp'].iloc[train_idx].tolist())  # Convert lists to NumPy arrays
        X_test = np.array(df['mdfp'].iloc[test_idx].tolist())
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


        # Create DMatrix for training and testing
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=999,
            evals=[(dtest, "Test")],
            early_stopping_rounds=10,
            verbose_eval=False

        )
        pp = model.predict(dtest)

        output[0].append(y_test)
        output[1].append(pp)
        molregnos_test = df['molregno'].iloc[test_idx]
        confids_test = df['confid'].iloc[test_idx]
        
        output[2].append(molregnos_test)
        output[3].append(confids_test)
        rmse = np.sqrt(mean_squared_error(y_test, pp, squared=False))
        if print_fold_rmses:
            print(f"Fold {fold + 1}: RMSE = {rmse}")
    vps = reduce(lambda a,b : list(a)+list(b) , output[0])
    preds = reduce(lambda a,b : list(a)+list(b), output[1])
    molregnos = reduce(lambda a,b : list(a)+list(b), output[2])
    confids = reduce(lambda a,b : list(a)+list(b), output[3])
    if average:
        df_preds = pd.DataFrame({'vp': vps, 'pred': preds, 'confid': confids, 'molregno': molregnos})
        df_preds = df_preds.groupby('molregno').mean()
        preds = df_preds['pred'].tolist()
        vps = df_preds['vp'].tolist()
        molregnos = df_preds.index.tolist()

    return vps,preds,molregnos,confids
import py3Dmol
from rdkit import Chem
from rdkit.Chem.Draw import IPythonConsole
def drawit(ms, p=None, confId=-1, removeHs=True,colors=('cyanCarbon','redCarbon','blueCarbon')):
        if p is None:
            p = py3Dmol.view(width=400, height=400)
        p.removeAllModels()
        for i,m in enumerate(ms):
            if removeHs:
                m = Chem.RemoveHs(m)
            IPythonConsole.addMolToView(m,p,confId=confId)
        for i,m in enumerate(ms):
            p.setStyle({'model':i,},
                            {'stick':{'colorscheme':colors[i%len(colors)]}})
        p.zoomTo()
        return p.show()



def density_plot_multiple(reals, predictions, print_stats=True, bounds=None, titles=None,global_title=None,print_spearman=False):
    num_plots = len(reals)
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
        real = reals[i]
        prediction = predictions[i]
        if not print_spearman:
            slope_mdfp_d, intercept_mdfp_d, r2, this_rmse, this_mae = getStatValues(real, prediction)
        else:
            slope_mdfp_d, intercept_mdfp_d, r2, this_rmse, this_mae, spearman = getStatValues(real, prediction,get_spearman=True)
            print(f'Spearman: {spearman:.2f}')
        if print_stats:
            print(f'Plot {i + 1} Stats:')
            print('RMSE: ', this_rmse)
            print('MAE: ', this_mae)
            print('R2: ', r2)
        
        ax = axes[i]
        
        if bounds is None:
            lower = min(prediction + real) - 2
            upper = max(prediction + real) + 2
        else:
            lower = bounds[0]
            upper = bounds[1]
        
        x = np.linspace(lower, upper, 100)
        y = slope_mdfp_d * x + intercept_mdfp_d
        ax.plot(x, y, '-r')
        ax.plot([min(prediction + real), max(prediction + real)], [min(prediction + real), max(prediction + real)], 'k-')
        ax.plot([min(prediction + real), max(prediction + real)], [min(prediction + real) - 1, max(prediction + real) - 1], 'k--')
        ax.plot([min(prediction + real), max(prediction + real)], [min(prediction + real) + 1, max(prediction + real) + 1], 'k--')
        import statsmodels.api as sm
        dens_u = sm.nonparametric.KDEMultivariate(data=[real, prediction], var_type='cc', bw='normal_reference')
        z = dens_u.pdf([real, prediction])

        sc = ax.scatter(real, prediction, lw=0, c=z, s=10, alpha=0.9)

        ax.set_xlabel(r'Exp. VP (log10 Pa)', fontsize=14)
        ax.set_ylabel(r'Predicted VP (log10 Pa)', fontsize=14)
        ax.grid(True, which="both")
        ax.axis([lower, upper, lower, upper])
        if not print_spearman:
            ax.text(0.05, 0.95, f'RMSE: {this_rmse:.2f}\nMAE: {this_mae:.2f}\nR2: {r2:.2f}', transform=ax.transAxes, fontsize=14, verticalalignment='top')
        else:
            ax.text(0.05, 0.95, f'RMSE: {this_rmse:.2f}\nMAE: {this_mae:.2f}\nR2: {r2:.2f}\nSpearman: {spearman:.2f}', transform=ax.transAxes, fontsize=14, verticalalignment='top')
        if titles is not None and len(titles) > i:
            ax.set_title(titles[i], fontsize=14)
        ax.set_aspect('equal', 'box')
    
    # Remove any unused subplots
    for i in range(num_plots, len(axes)):
        fig.delaxes(axes[i])
    if global_title is not None:
        fig.suptitle(global_title, fontsize=16)
    
    plt.tight_layout()
    plt.show()


def get_mdfps(which='all'):
    """
    Returns dataframe with mdfp and VP for specified conformers.
    Options: 'all', 'one_5ns', 'five_5ns','one_25ns'
    """
    cn = psycopg2.connect(host=hostname, dbname=dbname, user=username)
    cur = cn.cursor()

    # Define the SQL query to perform the joins with schema qualification
    sql_query = '''
        SELECT cs_mdfps_schema.mdfp_experiment_data.conf_id AS confid,
            public.conformers.molregno,
            cs_mdfps_schema.mdfp_experiment_data.mdfp,
            cs_mdfps_schema.experimental_data.vp_log10Pa
        FROM cs_mdfps_schema.mdfp_experiment_data
        INNER JOIN public.conformers
        ON cs_mdfps_schema.mdfp_experiment_data.conf_id = public.conformers.conf_id
        INNER JOIN cs_mdfps_schema.confid_data
        ON cs_mdfps_schema.mdfp_experiment_data.conf_id = cs_mdfps_schema.confid_data.conf_id
        INNER JOIN cs_mdfps_schema.experimental_data
        ON public.conformers.molregno = cs_mdfps_schema.experimental_data.molregno
    '''
    if which == 'all':
        pass
    elif which == 'one_5ns':
        sql_query = sql_query + " WHERE cs_mdfps_schema.mdfp_experiment_data.md_experiment_uuid = 'fc57851e-b654-4338-bcdd-faa28ec66253'"
    elif which == 'five_5ns':
        sql_query = sql_query + "WHERE cs_mdfps_schema.mdfp_experiment_data.md_experiment_uuid = 'e0f120fb-efa9-4c88-a964-e7b99253027c'"
    elif which == 'one_25ns':
        sql_query = sql_query +  "WHERE cs_mdfps_schema.mdfp_experiment_data.md_experiment_uuid = '80b643c8-5bdc-4b63-a12d-6f1ba3f7dd2a'"
    else:
        raise ValueError('Invalid value for which')

    # Execute the SQL query
    cur.execute(sql_query)

    # Fetch the results if needed
    results = cur.fetchall()
    print(f'{len(results)} results fetched')

    # Print the column names
    column_names = [desc[0] for desc in cur.description]
    print(column_names)
    confids = [r[0] for r in results]
    molregnos = [r[1] for r in results]
    mdfps = [json.loads(r[2]['mdfp']) for r in results]
    vps = [r[3] for r in results]
    df_mdfps = pd.DataFrame({'confid': confids, 'molregno': molregnos, 'mdfp': mdfps, 'vp_log10pa': vps})
    cur.close()
    cn.close()
    return df_mdfps

def print_package_versions():
    import openff.toolkit
    import openmm
    import rdkit
    print("ff_name: openff_unconstrained-2.1.0.offxml")
    print("ff_version: ", openff.toolkit.__version__)
    print("simulation_type: tMD water solution")
    print("md_engine: openMM")
    print("version: ", openmm.__version__)
    print("steps_time: 5.0")
    print("rdkit version: ", rdkit.__version__)

def train_pred_xgboost_2d(df,params,X_features,y_label,splits=5,scale = True):

    X = df[X_features]
    X = X.to_numpy()
    if scale:
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
    y = df[y_label]
    kf = KFold(n_splits=splits)
    output = ([], [])

    for train, test in kf.split(X):
            
            train_x = np.array(X)[train]
            train_y = np.array(y)[train]
    
            test_x = np.array(X)[test]
            test_y = np.array(y)[test]
            
            dtrain = xgb.DMatrix(train_x, label=train_y)
            dtest = xgb.DMatrix(test_x, label=test_y)

            model = xgb.train(
            params,
            dtrain,
            num_boost_round=999,
            evals=[(dtest, "Test")],
            early_stopping_rounds=10,
            verbose_eval=False
            )

            predictions = model.predict(dtest)
    
            output[0].append(test_y)
            output[1].append(predictions)

    return output