from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, RepeatedKFold
import numpy as np
import joblib
import psycopg2
import pandas as pd
import logging
import warnings
from utils.data_preprocessing import prepare_data, preprocess_data_kfold, get_features
from models.xgboost import XGBoostModel
from models.RF import RandomForestModel
from descriptors.rdkit_physchem_decriptors import calculate_RDKit_PhysChem_descriptors
from models.neural_network import NeuralNetworkModelGregstyle
from models.ridge_regression import RidgeModel

# Constants for easier maintenance
N_SPLITS = 5  # Number of splits for cross-validation
N_REPEATS = 2  # Number of repetitions for cross-validation
RANDOM_STATE = 2841  # Random state for reproducibility
MAX_DEPTH_RF = 5  # Max depth for RF_SHORT model
MAX_RETRIES = 3  # Maximum retries for model training

# Model and descriptor definitions
model_classes = {
    'MLP': NeuralNetworkModelGregstyle,
    # 'XGB': XGBoostModel,
    # 'RF': RandomForestModel,
    # 'RF_SHORT': RandomForestModel,
    'RIDGE': RidgeModel
}

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

descriptor_functions = {
    'RDKit_PhysChem': calculate_RDKit_PhysChem_descriptors
}

def instantiate_model(model_name, input_shape=None, max_depth=MAX_DEPTH_RF):
    if model_name == 'MLP':
        return NeuralNetworkModelGregstyle(input_shape=input_shape)
    elif model_name == 'RF_SHORT':
        return RandomForestModel(max_depth=max_depth)
    else:
        return model_classes[model_name]()

def train_with_retries(model, X_train, y_train, X_val, max_retries=MAX_RETRIES):
    retry_count = 0
    while retry_count < max_retries:
        if retry_count > 0 and model.__class__.__name__ == 'NeuralNetworkModelGregstyle':
            model.set_model_seed(retry_count + 1)
        model.train(X_train, y_train)
        y_pred = model.predict(X_val)
        if len(y_pred) != 0 and len(np.unique(y_pred)) > 1:
            return y_pred
        retry_count += 1
        logging.info(f"Retrying model training (attempt {retry_count}/{max_retries}).")
    logging.error("Exceeded maximum retries. Skipping this combination.")
    return None

def main(descriptors_to_use, models_to_evaluate, wipe_database=False):
    # Database setup
    with psycopg2.connect("dbname=cs_mdfps user=cschiebroek host=lebanon") as conn:
        logging.info("Preparing data...")
        df = prepare_data(conn, list(descriptors_to_use))
        logging.info("Data prepared with selected descriptors.")

        # Cross-validation setup
        kf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
        
        # Initialize result storage
        y_list, molregno_list = [], []
        predictions_test = {descriptor: {model: [] for model in models_to_evaluate} for descriptor in descriptors_to_use}
        split_n = 0

        # Iterate over cross-validation splits
        for train_index, test_index in kf.split(df):
            logging.info(f"Starting split {split_n + 1}...")
            train_molregnos, val_molregnos, train_y, val_y, df_train, df_val = preprocess_data_kfold(df, train_index, test_index)

            for descriptor in descriptors_to_use:
                scale = True if descriptor == 'RDKit_PhysChem' else False
                train_X, val_X = get_features(df_train, df_val, descriptor, scale)

                for model_name in models_to_evaluate:
                    input_shape = (train_X.shape[1],) if model_name == 'MLP' else None
                    model = instantiate_model(model_name, input_shape=input_shape)

                    # Train model and make predictions
                    y_pred = train_with_retries(model, train_X, train_y, val_X)
                    if y_pred is None:
                        continue

                    # Collect predictions
                    predictions_test[descriptor][model_name].append(y_pred)

            y_list.append(val_y)
            molregno_list.append(val_molregnos)
            split_n += 1

        # Save results to pickle for later analysis
        results = {
            'reals_list': y_list,
            'molregnos_list': molregno_list,
            'predictions_test': predictions_test
        }
        pd.to_pickle(results, 'results/test_set_results_physchem_split_full_range.pkl')

if __name__ == "__main__":
    models_to_evaluate = model_classes.keys()
    descriptors_to_use = descriptor_functions.keys()
    main(descriptors_to_use, models_to_evaluate, wipe_database=True)
