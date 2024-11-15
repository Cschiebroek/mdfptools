import psycopg2
import pandas as pd
import logging
import warnings
from utils.data_preprocessing import prepare_data, preprocess_data_kfold, get_features
from models.xgboost import XGBoostModel
from models.RF import RandomForestModel
from descriptors.rdkit_physchem_decriptors import calculate_RDKit_PhysChem_descriptors
from descriptors.fingerprints import calculate_bit_fingerprints
from models.neural_network import NeuralNetworkModelGregstyle, NeuralNetworkModel
from models.ridge_regression import RidgeModel
from sklearn.model_selection import RepeatedKFold

# Add the new model to the dictionary of available models
model_classes = {
    'XGB': XGBoostModel,
    'RF': RandomForestModel,
    'RF_SHORT': RandomForestModel,
    'MLP': NeuralNetworkModelGregstyle,
    'RIDGE': RidgeModel
}

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

descriptor_functions = {
    'mfp0': lambda df: calculate_bit_fingerprints(df, fingerprint_type='ecfp', n_bits=1024,radius=0),
    'RDKit_PhysChem': calculate_RDKit_PhysChem_descriptors,
    'mfp3': lambda df: calculate_bit_fingerprints(df, fingerprint_type='ecfp', n_bits=1024,radius=3)
}


def main(descriptors_to_use, models_to_evaluate):
    conn = psycopg2.connect("dbname=cs_mdfps user=cschiebroek host=lebanon")
    logging.info("Preparing data...")
    #log descriptors to use
    logging.info(f"Descriptors to use: {descriptors_to_use}")
    logging.info(f"Models to evaluate: {models_to_evaluate}")
    df = prepare_data(conn,descriptors_to_use)
    logging.info("Data prepared with selected descriptors.")

    predictions_train_full_range = {(descriptor, model_name): [] for descriptor in descriptors_to_use for model_name in models_to_evaluate}
    predictions_train_mid_range = {(descriptor, model_name): [] for descriptor in descriptors_to_use for model_name in models_to_evaluate}
    y_list, molregno_list = [], []


    kf = RepeatedKFold(n_splits=5, n_repeats=2, random_state=2841)
    for train_index, test_index in kf.split(df):
        logging.info(f"Training and evaluating split {len(y_list)+1}... \n \n \n")
        train_molregnos, val_molregnos, train_y, val_y, df_train, df_val = preprocess_data_kfold(df, train_index, test_index)
        #get val_molregnos which have an val_y between 1 and 8, this will be the val set
        val_molregnos = df_val[(df_val['vp_log10_pa'] > 1) & (df_val['vp_log10_pa'] < 8)]['molregno'].tolist()
        #make sure df_val and val_y is only for these
        df_val = df_val[df_val['molregno'].isin(val_molregnos)]
        val_y = df_val['vp_log10_pa']
        #now make two train examples, one with the full range and one with the mid range
        df_train_full_range = df_train
        df_train_mid_range = df_train[(df_train['vp_log10_pa'] > 1) & (df_train['vp_log10_pa'] < 8)]
        #get the y values for the train sets
        train_y_full_range = df_train_full_range['vp_log10_pa']
        train_y_mid_range = df_train_mid_range['vp_log10_pa']
        #get the molregnos for the train sets
        train_molregnos_full_range = df_train_full_range['molregno'].tolist()
        train_molregnos_mid_range = df_train_mid_range['molregno'].tolist()


        for descriptor in descriptors_to_use:
            logging.info(f"Extracting features for descriptor: {descriptor}")
            scale = True if descriptor == 'RDKit_PhysChem' else False
            train_X_full_range, val_X_full_range = get_features(df_train_full_range, df_val, descriptor, scale)
            train_X_mid_range, val_X_mid_range = get_features(df_train_mid_range, df_val, descriptor, scale)


            for model_name in models_to_evaluate:
                logging.info(f"Training and evaluating model: {model_name} with descriptor: {descriptor}")

                model_class = model_classes[model_name]
                
                if model_name == 'MLP':
                    try:
                        model_full_range = NeuralNetworkModelGregstyle(input_shape=(train_X_full_range.shape[1],))
                        model_mid_range = NeuralNetworkModelGregstyle(input_shape=(train_X_mid_range.shape[1],))
                    except AttributeError:
                        model_full_range = NeuralNetworkModelGregstyle(input_shape=(len(train_X_full_range[0]),))
                        model_mid_range = NeuralNetworkModelGregstyle(input_shape=(len(train_X_mid_range[0]),))
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
                y_true = val_y.tolist()
                molregno = val_molregnos
                predictions_train_full_range[(descriptor, model_name)].append(y_pred_full_range)
                predictions_train_mid_range[(descriptor, model_name)].append(y_pred_mid_range)

        y_list.append(y_true)
        molregno_list.append(molregno)

    combined_titles = []
    combined_preds_full_range = []
    combined_preds_mid_range = []
    combined_reals = []
    combined_molregnos = []

    for descriptor in descriptors_to_use:
        for model_name in models_to_evaluate:
            combined_titles.append(f"{model_name} ({descriptor})")
            combined_preds_full_range.append(predictions_train_full_range[(descriptor, model_name)])
            combined_preds_mid_range.append(predictions_train_mid_range[(descriptor, model_name)])
            combined_reals.append(y_list)
            combined_molregnos.append(molregno_list)

    # Close the database connection
    conn.close()
    #assert that the lengths of the lists are the same
    assert len(combined_titles) == len(combined_reals) == len(combined_molregnos) == len(combined_preds_full_range) == len(combined_preds_mid_range)
    # Save all data to a pickle to load again later
    data = {'reals_list': combined_reals, 'molregnos_list': combined_molregnos, 'combined_titles': combined_titles, 'predictions_train_full_range': combined_preds_full_range, 'predictions_train_mid_range': combined_preds_mid_range}
    pd.to_pickle(data, 'results/overfitting_check_different_train_ranges_gregorian_MLP.pkl')


if __name__ == "__main__":
    models_to_evaluate = model_classes.keys()
    descriptors_to_use =descriptor_functions.keys()
    main(descriptors_to_use, models_to_evaluate)
