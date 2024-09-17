import psycopg2
import pandas as pd
import logging
import warnings
from utils.data_preprocessing import prepare_data, preprocess_data, get_features
from models.xgboost import XGBoostModel
from models.PLS import PLSModel
from models.lasso import LassoModel
from models.RF import RandomForestModel
from models.kNN import KNNModel
from descriptors.rdkit_physchem_decriptors import calculate_RDKit_PhysChem_descriptors
from descriptors.mdfp import extract_mdfp_features
from descriptors.fingerprints import calculate_bit_fingerprints, calculate_count_fingerprints
from descriptors.codessa_descriptors import calculate_codessa_descriptor_df
from models.neural_network import NeuralNetworkModel
from models.linear_regression import MultilinearRegressionModel
from models.svm import SVMModel
from descriptors.padel import calculate_Padel_descriptors
from models.custom_lr import custom_LR
from descriptors.polarizability import calculate_liang_descriptors_df
from models.elasticnet import ElasticNetModel
from models.ridge_regression import RidgeModel
from descriptors.atom_contrib import add_crippen_atom_counts_to_df

# Add the new model to the dictionary of available models
model_classes = {
    'XGBoost': XGBoostModel,
    'PLS': PLSModel,
    'Lasso': LassoModel,
    'RandomForest': RandomForestModel,
    'kNN': KNNModel,
    'NeuralNetwork': NeuralNetworkModel,
    'MultilinearRegression': MultilinearRegressionModel,
    'SVM': SVMModel,
    'ElasticNet': ElasticNetModel,
    'RidgeRegression': RidgeModel
}

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

descriptor_functions = {
    'RDKit_PhysChem': calculate_RDKit_PhysChem_descriptors,
    'MDFP': extract_mdfp_features,
    'MACCS': lambda df: calculate_bit_fingerprints(df, fingerprint_type='maccs'),
    'ECFP4_bit': lambda df: calculate_bit_fingerprints(df, fingerprint_type='ecfp4'), 
    'ECFP4_count': lambda df: calculate_count_fingerprints(df, fingerprint_type='ecfp4'), 
    'codessa': calculate_codessa_descriptor_df,
    'padel': calculate_Padel_descriptors,
    'liang_descriptors': calculate_liang_descriptors_df ,
    'crippen_atoms': add_crippen_atom_counts_to_df
}

def check_results_in_db(conn, descriptor, model_name, seed):
    query = """
    SELECT y_true, y_pred, molregno FROM cs_mdfps_schema.model_descriptor_results 
    WHERE descriptor=%s AND model=%s AND seed=%s
    """
    cur = conn.cursor()
    cur.execute(query, (descriptor, model_name, seed))
    results = cur.fetchone()
    cur.close()
    return results

def store_results_in_db(conn, descriptor, model_name, seed, molregno, y_true, y_pred):
    query = """
    INSERT INTO cs_mdfps_schema.model_descriptor_results (descriptor, model, seed, molregno, y_true, y_pred) 
    VALUES (%s, %s, %s, %s, %s, %s)
    """
    cur = conn.cursor()
    cur.execute(query, (descriptor, model_name, seed, molregno, y_true, y_pred))
    conn.commit()
    cur.close()

def main(descriptors_to_use, models_to_evaluate):
    conn = psycopg2.connect("dbname=cs_mdfps user=cschiebroek host=lebanon")
    logging.info("Preparing data...")
    df = prepare_data(conn,descriptors_to_use)
    logging.info("Data prepared with selected descriptors.")

    predictions = {(descriptor, model_name): [] for descriptor in descriptors_to_use for model_name in models_to_evaluate}
    y_list, molregno_list = [], []

    for i in range(10):
        logging.info(f"Training and evaluating split {i+1}... \n \n \n")
        _, val_molregnos, train_y, val_y, df_train, df_val = preprocess_data(df, i)

        for descriptor in descriptors_to_use:
            logging.info(f"Extracting features for descriptor: {descriptor}")
            scale = descriptor not in ['ECFP4_bit', 'ECFP4_count', 'MACCS', 'crippen_atoms', 'fragments', 'Counts']
            train_X, val_X = get_features(df_train, df_val, descriptor, scale)

            for model_name in models_to_evaluate:
                logging.info(f"Training and evaluating model: {model_name} with descriptor: {descriptor}")

                # Check if results are already in the database
                result = check_results_in_db(conn, descriptor, model_name, i)
                if result:
                    y_true, y_pred, molregno = result
                    logging.info(f"Loaded results from database for {model_name} with {descriptor} at seed {i}")
                else:
                    if model_name == 'NeuralNetwork':
                        try:
                            model = NeuralNetworkModel(input_shape=(train_X.shape[1],))
                        except AttributeError:
                            model = NeuralNetworkModel(input_shape=(len(train_X[0]),)) 
                        model.train(train_X, train_y, validation_data=(val_X, val_y))
                    elif model_name == 'customLR':
                        model = custom_LR()
                    else:
                        model_class = model_classes[model_name]
                        model = model_class()
                        model.train(train_X, train_y)
                    y_pred = model.predict(val_X)

                    # Store results in the database
                    store_results_in_db(conn, descriptor, model_name, i, val_molregnos, val_y.tolist(), y_pred.tolist())

                predictions[(descriptor, model_name)].append(y_pred)

        y_list.append(val_y)
        molregno_list.append(val_molregnos)

    combined_titles = []
    combined_preds = []
    combined_reals = []
    combined_molregnos = []

    for descriptor in descriptors_to_use:
        for model_name in models_to_evaluate:
            combined_titles.append(f"{model_name} ({descriptor})")
            combined_preds.append(predictions[(descriptor, model_name)])
            combined_reals.append(y_list)
            combined_molregnos.append(molregno_list)

    # Close the database connection
    conn.close()

    # Save all data to a pickle to load again later
    data = {'reals_list': combined_reals, 'predictions_list': combined_preds, 'molregnos_list': combined_molregnos, 'combined_titles': combined_titles}
    pd.to_pickle(data, 'results/crippen_atoms.pkl')



if __name__ == "__main__":
    # models_to_evaluate = ['XGBoost', 'PLS', 'Lasso', 'RandomForest', 'kNN', 'NeuralNetwork', 'MultilinearRegression', 'SVM', 'ElasticNet', 'RidgeRegression']
    # descriptors_to_use = ['padel', 'liang_descriptors', 'RDKit_PhysChem', 'MDFP', 'MACCS', 'ECFP4_bit', 'codessa', 'ECFP4_count','carl_custom_features','fragments','Counts','crippen_atoms']
    models_to_evaluate = ['RidgeRegression','MultilinearRegression']
    descriptors_to_use = ['crippen_atoms']
    main(descriptors_to_use, models_to_evaluate)
