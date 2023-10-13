

import xgboost as xgb
import pandas as pd


df_predictions = pd.read_csv('002_delta_learning.csv')
mdfp_keys_full = ['NumHeavyAtoms', 'NumRotatableBonds', 'NumN', 'NumO', 'NumF', 'NumP', 'NumS', 'NumCl', 'NumBr', 'NumI', 'water_intra_crf_mean', 'water_intra_crf_std', 'water_intra_crf_median', 'water_intra_lj_mean', 'water_intra_lj_std', 'water_intra_lj_median', 'water_total_crf_mean', 'water_total_crf_std', 'water_total_crf_median', 'water_total_lj_mean', 'water_total_lj_std', 'water_total_lj_median', 'water_intra_ene_mean', 'water_intra_ene_std', 'water_intra_ene_median', 'water_total_ene_mean', 'water_total_ene_std', 'water_total_ene_median', 'water_rgyr_mean', 'water_rgyr_std', 'water_rgyr_median', 'water_sasa_mean', 'water_sasa_std', 'water_sasa_median'] 
X = df_predictions[mdfp_keys_full]
y = df_predictions['abs_error']
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


