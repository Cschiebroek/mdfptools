from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
import os
import json
import uuid

import rdkit
print(rdkit.__version__)
import lwreg
from lwreg import standardization_lib
from lwreg import utils
import psycopg2
import pandas as pd
import psycopg2
import numpy as np
hostname = 'scotland'
dbname = 'cs_mdfps'
username = 'cschiebroek'

cn = psycopg2.connect(host=hostname, dbname=dbname, user=username)
cur = cn.cursor()

# Define the SQL query to perform the joins with schema qualification
sql_query = '''
    SELECT cs_mdfps_schema.mdfp_experiment_data.conf_id AS confid,
           public.conformers.molregno,
           cs_mdfps_schema.mdfp_experiment_data.mdfp,
           cs_mdfps_schema.experimental_data.vp
    FROM cs_mdfps_schema.mdfp_experiment_data
    INNER JOIN public.conformers
    ON cs_mdfps_schema.mdfp_experiment_data.conf_id = public.conformers.conf_id
    INNER JOIN cs_mdfps_schema.confid_data
    ON cs_mdfps_schema.mdfp_experiment_data.conf_id = cs_mdfps_schema.confid_data.conf_id
    INNER JOIN cs_mdfps_schema.experimental_data
    ON public.conformers.molregno = cs_mdfps_schema.experimental_data.molregno
'''
# Execute the SQL query
cur.execute(sql_query)

# Fetch the results if needed
results = cur.fetchall()
print(f'{len(results)} results fetched')
df = pd.DataFrame(results, columns=['confid', 'molregno', 'mdfp', 'vp'])
#mdfp; load the dict, get only the value for key 'mdfp' and json.loads it
df['mdfp'] = df['mdfp'].apply(lambda x: json.loads(x['mdfp']))


import xgboost as xgb

X = df['mdfp'].tolist() 
y = df['vp'].tolist()
# y = [np.log10(i) for i in y]
X = np.array(X)
y = np.array(y)
dtrain = xgb.DMatrix(X, label=y)

params = {
    # Parameters that we are going to tune.
    'max_depth':6,
    'min_child_weight': 1,
    'eta':.3,
    'subsample': 1,
    'colsample_bytree': 1,
    # Other parameters
    "objective" : 'reg:squarederror',
    "nthread" : 8,
}

gridsearch_params = [
    (max_depth, min_child_weight, eta, subsample, colsample_bytree)
    for max_depth in range(3,10)
    for min_child_weight in range(4,8)
    for eta in [0.3, 0.2, 0.1, 0.05, 0.01]
    for subsample in [0.5, 0.75, 1.0]
    for colsample_bytree in [0.5, 0.75, 1.0]
]

min_mae = float("Inf")
best_params = None
for max_depth, min_child_weight, eta, subsample, colsample_bytree in gridsearch_params:
    print(f"CV with max_depth={max_depth}, min_child_weight={min_child_weight} eta={eta}, subsample={subsample}, colsample_bytree={colsample_bytree}")
    # Update our parameters
    params['max_depth'] = max_depth
    params['min_child_weight'] = min_child_weight
    params['eta'] = eta
    params['subsample'] = subsample
    params['colsample_bytree'] = colsample_bytree

    # Run CV
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=999,
        seed=42,
        nfold=5,
        metrics={'mae', "rmse"},
        early_stopping_rounds=10,
        verbose_eval=False
    )
    # Update best MAE
    mean_mae = cv_results['test-rmse-mean'].min()
    boost_rounds = cv_results['test-rmse-mean'].argmin()
    print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
    if mean_mae < min_mae:
        min_mae = mean_mae
        best_params = (max_depth,min_child_weight, eta, subsample, colsample_bytree)
print("Best params: {}, {}, {}, {}, MAE: {}".format(best_params[0], best_params[1], best_params[2], best_params[3], min_mae))


