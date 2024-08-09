import psycopg2
from utils.data_preprocessing import prepare_data, preprocess_data, get_features
from models.xgboost import XGBoostModel
from models.PLS import PLSModel
from models.lasso import LassoModel
from models.RF import RandomForestModel
from models.kNN import KNNModel
from utils.visualization import density_plots
from descriptors.rdkit import calculate_rdkit_descriptors
from descriptors.mdfp import extract_mdfp_features
from descriptors.fingerprints import calculate_maccs_keys, calculate_ecfp4
import logging
import warnings
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define available descriptors
descriptor_functions = {
    'RDKit': calculate_rdkit_descriptors,
    'MDFP': extract_mdfp_features,
    'MACCS': calculate_maccs_keys,
    'ECFP4': calculate_ecfp4,
}

# Define available models
model_classes = {
    'XGBoost': XGBoostModel,
    'PLS': PLSModel,
    'Lasso': LassoModel,
    'RandomForest': RandomForestModel,
    'kNN': KNNModel,
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
    pls_components = 10

    for i in range(10):  # Adjust the number of iterations as needed
        logging.info(f"Training and evaluating split {i+1}...")
        _, val_molregnos, train_y, val_y, df_train, df_val = preprocess_data(df, i)

        for descriptor in descriptors_to_use:
            logging.info(f"Extracting features for descriptor: {descriptor}")
            scale = descriptor in ['RDKit', 'MDFP']  # Scale only relevant descriptors
            train_X, val_X = get_features(df_train, df_val, descriptor, scale)
            for model_name in models_to_evaluate:
                logging.info(f"Training and evaluating model: {model_name} with descriptor: {descriptor}")
                model_class = model_classes[model_name]
                model = model_class() if model_name != 'PLS' else model_class(n_components=pls_components)
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

    # Plot all combinations in a single plot
    density_plots(reals_list=combined_reals, predictions_list=combined_preds, molregnos_list=combined_molregnos,
                  print_stats=True, bounds=None, title=combined_titles,
                  name="all_models_descriptors", dims=(len(descriptors_to_use), len(models_to_evaluate)), thresholds=1)

    # Close the database connection
    conn.close()

    #save all data to a pickle to load again later
    data = {'reals_list': combined_reals, 'predictions_list': combined_preds, 'molregnos_list': combined_molregnos, 'combined_titles': combined_titles}
    pd.to_pickle(data, 'all_models_descriptors.pkl')
    

if __name__ == "__main__":
    # Example usage
    descriptors_to_use = ['RDKit', 'MDFP', 'MACCS', 'ECFP4']
    models_to_evaluate = ['XGBoost', 'PLS', 'Lasso', 'RandomForest', 'kNN']

    main(descriptors_to_use, models_to_evaluate)
