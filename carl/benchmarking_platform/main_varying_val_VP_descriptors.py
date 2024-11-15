import psycopg2
import pandas as pd
import logging
import warnings
from utils.data_preprocessing import prepare_data, preprocess_data_kfold, get_features
from models.xgboost import XGBoostModel
from models.RF import RandomForestModel
from descriptors.rdkit_physchem_decriptors import calculate_RDKit_PhysChem_descriptors
from descriptors.mdfp import extract_mdfp_features
from models.neural_network import NeuralNetworkModelGregstyle, NeuralNetworkModel
from models.ridge_regression import RidgeModel
from sklearn.model_selection import RepeatedKFold
import numpy as np

# Add the new model to the dictionary of available models
model_classes = {
    'MLP': NeuralNetworkModel,
    'XGB': XGBoostModel,
    'RF': RandomForestModel,
    'RF_SHORT': RandomForestModel,
    'RIDGE': RidgeModel
}

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

descriptor_functions = {
    'Counts': extract_mdfp_features,
    'RDKit_PhysChem': calculate_RDKit_PhysChem_descriptors,
}


def main(descriptors_to_use, models_to_evaluate):
    conn = psycopg2.connect("dbname=cs_mdfps user=cschiebroek host=lebanon")
    logging.info("Preparing data...")
    #log descriptors to use
    logging.info(f"Descriptors to use: {descriptors_to_use}")
    logging.info(f"Models to evaluate: {models_to_evaluate}")
    df = prepare_data(conn,list(descriptors_to_use) + ['MDFP'])
    logging.info("Data prepared with selected descriptors.")
    df_mid_range = df[(df['vp_log10_pa'] > 1) & (df['vp_log10_pa'] < 8)]

    predictions_train_full_range = {(descriptor, model_name): [] for descriptor in descriptors_to_use for model_name in models_to_evaluate}
    predictions_train_mid_range = {(descriptor, model_name): [] for descriptor in descriptors_to_use for model_name in models_to_evaluate}
    y_list, molregno_list = [], []
    predictions_on_train_set_train_full_range = {(descriptor, model_name): [] for descriptor in descriptors_to_use for model_name in models_to_evaluate}
    predictions_on_train_set_train_mid_range = {(descriptor, model_name): [] for descriptor in descriptors_to_use for model_name in models_to_evaluate}
    y_list_train_full, molregno_list_train_full = [], []
    y_list_train_mid, molregno_list_train_mid = [], []

    kf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2841)
    for train_index, test_index in kf.split(df_mid_range):
        logging.info(f"Training and evaluating split {len(y_list)+1}... \n \n \n")
        train_molregnos_mid_range, val_molregnos, train_y_mid_range, val_y, df_train_mid_range, df_val = preprocess_data_kfold(df_mid_range, train_index, test_index)
        logging.info(f"Number of train molregnos: {len(train_molregnos_mid_range)}")
        logging.info(f"Number of val molregnos: {len(val_molregnos)}")

        #now also get train from full df, removing the val_molregnos
        df_train_full_range = df[~df['molregno'].isin(val_molregnos)]
        train_y_full_range = df_train_full_range['vp_log10_pa']

        for descriptor in descriptors_to_use:
            logging.info(f"Extracting features for descriptor: {descriptor}")
            scale = True if descriptor == 'RDKit_PhysChem' else False
            train_X_full_range, val_X_full_range = get_features(df_train_full_range, df_val, descriptor, scale)
            train_X_mid_range, val_X_mid_range = get_features(df_train_mid_range, df_val, descriptor, scale)


            for model_name in models_to_evaluate:
                logging.info(f"Training and evaluating model: {model_name} with descriptor: {descriptor}")

                model_class = model_classes[model_name]
                
                if model_name == 'MLP':
                    # try:
                    model_full_range = NeuralNetworkModel(input_shape=(train_X_full_range.shape[1],))
                    model_mid_range = NeuralNetworkModel(input_shape=(train_X_mid_range.shape[1],))
                    # except AttributeError:
                    #     model_full_range = NeuralNetworkModelGregstyle(input_shape=(len(train_X_full_range[0]),))
                    #     model_mid_range = NeuralNetworkModelGregstyle(input_shape=(len(train_X_mid_range[0]),))
                elif model_name == 'RF_SHORT':
                    model_full_range = model_class(max_depth=5)
                    model_mid_range = model_class(max_depth=5)
                else:
                    model_full_range = model_class()  # Instantiate the class here
                    model_mid_range = model_class()   # Instantiate the class here

                # Train the models
                model_mid_range.train(train_X_mid_range, train_y_mid_range)
                model_full_range.train(train_X_full_range, train_y_full_range)


                y_pred_full_range = model_full_range.predict(val_X_full_range)
                y_pred_mid_range = model_mid_range.predict(val_X_mid_range)

                y_pred_train_set_full_range = model_full_range.predict(train_X_full_range)
                y_pred_train_set_mid_range = model_mid_range.predict(train_X_mid_range)

                #assert ys are not all the same
                try:
                    assert not all(y_pred_full_range == y_pred_full_range[0])
                    assert not all(y_pred_mid_range == y_pred_mid_range[0])
                except AssertionError:
                    logging.ERROR(f"Predictions are all the same for split {len(y_list)+1}, model {model_name}, descriptor {descriptor}")
                #same for ys
                try:
                    #val_y is a list, convert to numpy array
                    vall_y_array = np.array(val_y)
                    assert not all(vall_y_array == vall_y_array[0])
                except AssertionError:
                    logging.ERROR(f"Y values are all the same for split {len(y_list)+1}, model {model_name}, descriptor {descriptor}")

                y_true = val_y
                molregno = val_molregnos
                predictions_train_full_range[(descriptor, model_name)].append(y_pred_full_range)
                predictions_train_mid_range[(descriptor, model_name)].append(y_pred_mid_range)
                predictions_on_train_set_train_full_range[(descriptor, model_name)].append(y_pred_train_set_full_range)
                predictions_on_train_set_train_mid_range[(descriptor, model_name)].append(y_pred_train_set_mid_range)



        y_list.append(y_true)
        molregno_list.append(molregno)
        y_list_train_full.append(train_y_full_range)
        molregno_list_train_full.append(train_molregnos_mid_range)
        y_list_train_mid.append(train_y_mid_range)
        molregno_list_train_mid.append(train_molregnos_mid_range)

    combined_titles = []
    combined_preds_full_range = []
    combined_preds_mid_range = []
    combined_reals = []
    combined_molregnos = []
    combined_preds_on_train_set_full_range = []
    combined_preds_on_train_set_mid_range = []
    combined_molregnos_train_full = []
    combined_molregnos_train_mid = []
    combined_reals_train_full = []
    combined_reals_train_mid = []



    for descriptor in descriptors_to_use:
        for model_name in models_to_evaluate:
            combined_titles.append(f"{model_name} ({descriptor})")
            combined_preds_full_range.append(predictions_train_full_range[(descriptor, model_name)])
            combined_preds_mid_range.append(predictions_train_mid_range[(descriptor, model_name)])
            combined_reals.append(y_list)
            combined_molregnos.append(molregno_list)
            combined_preds_on_train_set_full_range.append(predictions_on_train_set_train_full_range[(descriptor, model_name)])
            combined_preds_on_train_set_mid_range.append(predictions_on_train_set_train_mid_range[(descriptor, model_name)])
            combined_molregnos_train_full.append(molregno_list_train_full)
            combined_molregnos_train_mid.append(molregno_list_train_mid)
            combined_reals_train_full.append(y_list_train_full)
            combined_reals_train_mid.append(y_list_train_mid)


    # Close the database connection
    conn.close()
    #assert that the lengths of the lists are the same
    assert len(combined_titles) == len(combined_reals) == len(combined_molregnos) == len(combined_preds_full_range) == len(combined_preds_mid_range)
    # Save all data to a pickle to load again later
    data_test = {'reals_list': combined_reals, 'molregnos_list': combined_molregnos, 'combined_titles': combined_titles, 'predictions_train_full_range': combined_preds_full_range, 'predictions_train_mid_range': combined_preds_mid_range}
    pd.to_pickle(data_test, 'results/test_set_overfitting_check_different_train_ranges_Counts.pkl')
    data_train_full = {'reals_list': combined_reals_train_full, 'molregnos_list': combined_molregnos_train_full, 'combined_titles': combined_titles, 'predictions_train_full_range': combined_preds_on_train_set_full_range}
    pd.to_pickle(data_train_full, 'results/train_set_full_overfitting_check_different_train_ranges_Counts.pkl')
    data_train_mid = {'reals_list': combined_reals_train_mid, 'molregnos_list': combined_molregnos_train_mid, 'combined_titles': combined_titles, 'predictions_train_mid_range': combined_preds_on_train_set_mid_range}
    pd.to_pickle(data_train_mid, 'results/train_set_mid_overfitting_check_different_train_ranges_Counts.pkl')

if __name__ == "__main__":
    models_to_evaluate = model_classes.keys()
    descriptors_to_use =descriptor_functions.keys()
    main(descriptors_to_use, models_to_evaluate)
