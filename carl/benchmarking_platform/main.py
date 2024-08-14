import psycopg2
from utils.data_preprocessing import prepare_data, preprocess_data, get_features
from models.xgboost import XGBoostModel
from models.PLS import PLSModel
from models.lasso import LassoModel
from models.RF import RandomForestModel
from models.kNN import KNNModel
from utils.visualization import density_plots
from descriptors.rdkit import calculate_RDKit_PhysChem_descriptors
from descriptors.mdfp import extract_mdfp_features
from descriptors.fingerprints import calculate_fingerprints
from descriptors.codessa_descriptors import calculate_codessa_descriptor_df
import logging
import warnings
import pandas as pd
from models.neural_network import NeuralNetworkModel
from models.linear_regression import MultilinearRegressionModel
from models.svm import SVMModel
from descriptors.padel import calculate_Padel_descriptors
from models.custom_lr import custom_LR
from descriptors.polarizability import calculate_liang_descriptors_df
from models.elasticnet import ElasticNetModel
from models.ridge_regression import RidgeModel

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
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define available descriptors
descriptor_functions = {
    'RDKit_PhysChem': calculate_RDKit_PhysChem_descriptors,
    'MDFP': extract_mdfp_features,
    'MACCS': lambda df: calculate_fingerprints(df, fingerprint_type='maccs'),
    'ECFP4': lambda df: calculate_fingerprints(df, fingerprint_type='ecfp4'), 
    'codessa': calculate_codessa_descriptor_df,
    'padel': calculate_Padel_descriptors,
    'liang_descriptors': calculate_liang_descriptors_df 
}

def main(descriptors_to_use, models_to_evaluate):
    # Database connection
    conn = psycopg2.connect("dbname=cs_mdfps user=cschiebroek host=lebanon")
    
    # Prepare data once
    logging.info("Preparing data...")
    df = prepare_data(conn)
    
    logging.info("Data prepared with selected descriptors.")

    # Dictionary to store results
    predictions = {(descriptor, model_name): [] for descriptor in descriptors_to_use for model_name in models_to_evaluate}
    y_list, molregno_list = [], []

    for i in range(10):  # Adjust the number of iterations as needed
        logging.info(f"Training and evaluating split {i+1}... \n \n \n")
        _, val_molregnos, train_y, val_y, df_train, df_val = preprocess_data(df, i)

        for descriptor in descriptors_to_use:
            logging.info(f"Extracting features for descriptor: {descriptor}")
            scale = descriptor not in ['ECFP4', 'MACCS']  # Scale only relevant descriptors
            train_X, val_X = get_features(df_train, df_val, descriptor, scale)

            for model_name in models_to_evaluate:
                logging.info(f"Training and evaluating model: {model_name} with descriptor: {descriptor}")

                if model_name == 'NeuralNetwork':
                    try:
                        model = NeuralNetworkModel(input_shape=(train_X.shape[1],))
                        logging.info(f"Shape of input: {train_X.shape[1]}")

                    except AttributeError:
                        model = NeuralNetworkModel(input_shape=(len(train_X[0]),)) 
                        logging.info(f"Shape of input: {len(train_X[0])}")

                    model.train(train_X, train_y, validation_data=(val_X, val_y))
                elif model_name == 'customLR':
                    model = custom_LR()
                else:
                    model_class = model_classes[model_name]
                    model = model_class()
                    model.train(train_X, train_y)
               
                y_pred = model.predict(val_X)
                predictions[(descriptor, model_name)].append(y_pred)
        
        y_list.append(val_y)
        molregno_list.append(val_molregnos)

    # Create a combined plot for all model-descriptor combinations
    logging.info("Creating a combined plot for all model-descriptor combinations...")
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

    # # Plot all combinations in a single plot
    # density_plots(reals_list=combined_reals, predictions_list=combined_preds, molregnos_list=combined_molregnos,
    #               print_stats=True, bounds=None, title=combined_titles,
    #               name="more_linear_models", dims=(len(descriptors_to_use), len(models_to_evaluate)), thresholds=1)

    # Close the database connection
    conn.close()

    # Save all data to a pickle to load again later
    data = {'reals_list': combined_reals, 'predictions_list': combined_preds, 'molregnos_list': combined_molregnos, 'combined_titles': combined_titles}
    pd.to_pickle(data, 'results/more_linear_models.pkl')


if __name__ == "__main__":
    models_to_evaluate = ['XGBoost', 'PLS', 'Lasso', 'RandomForest', 'kNN', 'NeuralNetwork', 'MultilinearRegression', 'SVM', 'ElasticNet', 'RidgeRegression']
    descriptors_to_use = ['padel','liang_descriptors', 'RDKit_PhysChem', 'MDFP', 'MACCS', 'ECFP4', 'codessa']


    main(descriptors_to_use, models_to_evaluate)
