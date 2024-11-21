from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib
import psycopg2
import pandas as pd
import logging
import warnings
from sklearn.model_selection import RepeatedKFold
from utils.data_preprocessing import prepare_data, preprocess_data_kfold, get_features
from models.xgboost import XGBoostModel
from models.RF import RandomForestModel
from descriptors.rdkit_physchem_decriptors import calculate_RDKit_PhysChem_descriptors
from descriptors.mdfp import extract_mdfp_features
from models.neural_network import NeuralNetworkModelGregstyle, NeuralNetworkModel
from models.ridge_regression import RidgeModel

# Define named constants for easier maintenance
N_SPLITS = 5  # Number of splits for cross-validation
N_REPEATS = 2  # Number of repetitions for cross-validation
RANDOM_STATE = 2841  # Random state for reproducibility
MAX_DEPTH_RF = 5  # Max depth for RF_SHORT model
MAX_RETRIES = 3  # Maximum retries for model training

# Add the new model to the dictionary of available models
model_classes = {
    'MLP': NeuralNetworkModelGregstyle,
    'XGB': XGBoostModel,
    'RF': RandomForestModel,
    'RF_SHORT': RandomForestModel,
    'RIDGE': RidgeModel
}

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

descriptor_functions = {
    'MDFP_2D_counts': extract_mdfp_features,
    'RDKit_PhysChem': calculate_RDKit_PhysChem_descriptors,
    'MDFP': extract_mdfp_features
}

def wipe_database_func(conn,descriptors,models):  # Utility function to wipe the database
    cur = conn.cursor()
    for descriptor in descriptors:
        for model in models:
            cur.execute("DELETE FROM cs_mdfps_schema.model_descriptor_results WHERE descriptor=%s AND model=%s", (descriptor, model))
    conn.commit()




def instantiate_model(model_name, input_shape=None, max_depth=MAX_DEPTH_RF):  # Utility function to instantiate models
    if model_name == 'MLP':
        return NeuralNetworkModelGregstyle(input_shape=input_shape)
    elif model_name == 'RF_SHORT':
        return RandomForestModel(max_depth=max_depth)
    else:
        return model_classes[model_name]()

def train_with_retries(model, X_train, y_train, X_val, max_retries=MAX_RETRIES):  # Utility function for retry mechanism
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

def main(descriptors_to_use, models_to_evaluate,wipe_database=False):  # Main function
    if wipe_database:
        with psycopg2.connect("dbname=cs_mdfps user=cschiebroek host=lebanon") as conn:
            logging.info("Wiping database...")
            wipe_database_func(conn, descriptors_to_use, models_to_evaluate)
            logging.info("Database wiped.")

    with psycopg2.connect("dbname=cs_mdfps user=cschiebroek host=lebanon") as conn:  # Use context manager for database connection
        logging.info("Preparing data...")
        df = prepare_data(conn, list(set(list(descriptors_to_use) + ['MDFP', 'RDKit_PhysChem'])))
        logging.info("Data prepared with selected descriptors.")
        df_mid_range = df[(df['vp_log10_pa'] > 1) & (df['vp_log10_pa'] < 8)]

        kf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)  # Use constants for splits and random state
        
        # Initialize lists to store results
        predictions_train_full_range = {(descriptor, model_name): [] for descriptor in descriptors_to_use for model_name in models_to_evaluate}
        predictions_train_mid_range = {(descriptor, model_name): [] for descriptor in descriptors_to_use for model_name in models_to_evaluate}
        predictions_on_train_full_range = {(descriptor, model_name): [] for descriptor in descriptors_to_use for model_name in models_to_evaluate}
        predictions_on_train_mid_range = {(descriptor, model_name): [] for descriptor in descriptors_to_use for model_name in models_to_evaluate}
        y_list, molregno_list = [], []
        y_list_train_full, molregno_list_train_full = [], []
        y_list_train_mid, molregno_list_train_mid = [], []
        split_n = 0
    
        for train_index, test_index in kf.split(df_mid_range):
            logging.info(f"Starting split {split_n + 1}...")
            train_molregnos_mid_range, val_molregnos, train_y_mid_range, val_y, df_train_mid_range, df_val = preprocess_data_kfold(df_mid_range, train_index, test_index)

            df_train_full_range = df[~df['molregno'].isin(val_molregnos)]
            train_y_full_range = df_train_full_range['vp_log10_pa']

            for descriptor in descriptors_to_use:
                scale = True if descriptor == 'RDKit_PhysChem' else False
                train_X_full_range, val_X_full_range = get_features(df_train_full_range, df_val, descriptor, scale)
                train_X_mid_range, val_X_mid_range = get_features(df_train_mid_range, df_val, descriptor, scale)

                for model_name in models_to_evaluate:
                    input_shape = (train_X_full_range.shape[1],) if model_name == 'MLP' else None

                    # Instantiate models using utility function
                    model_full_range = instantiate_model(model_name, input_shape=input_shape)
                    model_mid_range = instantiate_model(model_name, input_shape=input_shape)

                    #logging.info(f"Training and evaluating model: {model_name} with descriptor: {descriptor}")
                    # Train and retry if needed using utility function
                    y_pred_full_range = train_with_retries(model_full_range, train_X_full_range, train_y_full_range, val_X_full_range)
                    if y_pred_full_range is None:
                        continue

                    y_pred_mid_range = train_with_retries(model_mid_range, train_X_mid_range, train_y_mid_range, val_X_mid_range)
                    if y_pred_mid_range is None:
                        continue

                    # Predictions on training set
                    y_pred_train_full_range = model_full_range.predict(train_X_full_range)
                    y_pred_train_mid_range = model_mid_range.predict(train_X_mid_range)

                    # Collect results
                    predictions_train_full_range[(descriptor, model_name)].append(y_pred_full_range)
                    predictions_train_mid_range[(descriptor, model_name)].append(y_pred_mid_range)
                    predictions_on_train_full_range[(descriptor, model_name)].append(y_pred_train_full_range)
                    predictions_on_train_mid_range[(descriptor, model_name)].append(y_pred_train_mid_range)

            y_list.append(val_y)
            molregno_list.append(val_molregnos)
            y_list_train_full.append(train_y_full_range)
            molregno_list_train_full.append(train_molregnos_mid_range)
            y_list_train_mid.append(train_y_mid_range)
            molregno_list_train_mid.append(train_molregnos_mid_range)

                    # Store results in the database

            split_n += 1

        combined_titles = []
        for descriptor in descriptors_to_use:
            for model_name in models_to_evaluate:
                combined_titles.append(f"{model_name} ({descriptor})")

        # Save all data to a pickle to load again later
        data_test = {'reals_list': y_list, 'molregnos_list': molregno_list, 'predictions_train_full_range': predictions_train_full_range, 'predictions_train_mid_range': predictions_train_mid_range}
        pd.to_pickle(data_test, 'results/test_set_results_physchem_MDFP_2D_counts.pkl')
        data_train_full = {'reals_list': y_list_train_full, 'molregnos_list': molregno_list_train_full, 'predictions_on_train_full_range': predictions_on_train_full_range}
        pd.to_pickle(data_train_full, 'results/train_set_full_results_physchem_MDFP_2D_counts.pkl')
        data_train_mid = {'reals_list': y_list_train_mid, 'molregnos_list': molregno_list_train_mid, 'predictions_on_train_mid_range': predictions_on_train_mid_range}
        pd.to_pickle(data_train_mid, 'results/train_set_mid_results_physchem_MDFP_2D_counts.pkl')

if __name__ == "__main__":
    models_to_evaluate = model_classes.keys()
    descriptors_to_use = descriptor_functions.keys()
    main(descriptors_to_use, models_to_evaluate, wipe_database=True)