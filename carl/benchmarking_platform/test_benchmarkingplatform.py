import psycopg2
import pandas as pd
from rdkit import Chem
from descriptors.padel import calculate_Padel_descriptors
from utils.data_preprocessing import preprocess_data, get_features
from models.xgboost import XGBoostModel

def test_padell_descriptors():
    # Connect to the database (adjust connection string if needed)
    conn = psycopg2.connect("dbname=cs_mdfps user=cschiebroek host=lebanon")

    # Create a small test DataFrame with SMILES strings
    test_data = {
        'molregno': [1, 2, 3],
        'smiles_column': ['CCO', 'CCN', 'CCC']  # Example SMILES strings for ethanol, ethylamine, propane
    }
    df = pd.DataFrame(test_data)

    # Calculate and store PaDEL descriptors
    df_with_descriptors = calculate_Padel_descriptors(df, conn)
    print("PaDEL descriptors calculated and stored.")
    print(df_with_descriptors.head())

    # Preprocess data (simulating train/test split)
    _, val_molregnos, train_y, val_y, df_train, df_val = preprocess_data(df_with_descriptors, seed=42)

    # Extract features for the 'PaDEL' descriptor
    train_X, val_X = get_features(df_train, df_val, 'PaDEL', scale=False)

    # Initialize a simple model (XGBoost)
    model = XGBoostModel()
    model.train(train_X, train_y)

    # Make predictions
    predictions = model.predict(val_X)
    print(f"Predictions: {predictions}")

    # Assert that predictions are not empty
    assert len(predictions) > 0, "Predictions should not be empty."

    # Close the database connection
    conn.close()
    print("Test completed successfully.")

if __name__ == "__main__":
    test_padell_descriptors()
