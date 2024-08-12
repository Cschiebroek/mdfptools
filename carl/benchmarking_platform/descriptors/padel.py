import pandas as pd
from padelpy import from_smiles
import logging
from rdkit import Chem
from tqdm import tqdm
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calc_padel_descriptors(smiles):
    """Calculate PaDEL descriptors for a single molecule."""
    try:
        descriptors_dict = from_smiles(smiles, fingerprints=True)  # You can adjust this to calculate only descriptors or fingerprints if needed
    except RuntimeError:
        logging.warning(f"Failed to calculate PaDEL descriptors for SMILES: {smiles}")
        descriptors_dict = {}
    return descriptors_dict

def get_padel_descriptors_from_db(conn):
    """Fetch PaDEL descriptors from the database."""
    query = """
    SELECT molregno, padeldescriptors
    FROM cs_mdfps_schema.padeldescriptors
    """
    df_sql = pd.read_sql(query, conn)

    # Normalize the padeldescriptors JSON column
    df_padel_json = pd.json_normalize(df_sql['padeldescriptors'])
    
    # If there is a 'molregno' column in the normalized DataFrame, drop it
    if 'molregno' in df_padel_json.columns:
        df_padel_json = df_padel_json.drop(columns=['molregno'])

    # Merge the original DataFrame with the normalized JSON DataFrame
    df_padel = df_sql.merge(df_padel_json, left_index=True, right_index=True)

    return df_padel

def calculate_Padel_descriptors(df, conn):
    """Main function to fetch or calculate PaDEL descriptors."""
    logging.info("Fetching PaDEL descriptors from the database...")

    df_padel_descriptors = get_padel_descriptors_from_db(conn)

    logging.info("Merging PaDEL descriptors with the main dataframe...")
    # Combine the two dataframes on molregno
    df = pd.merge(df, df_padel_descriptors, on='molregno', how='left')

    # Identify missing descriptors and calculate them
    try:
        missing_descriptors = df[df['padeldescriptors'].isnull()]  
    except KeyError:
        missing_descriptors = df
    
    if not missing_descriptors.empty:
        logging.info(f"Calculating PaDEL descriptors for {len(missing_descriptors)} molecules...")

        # Extract SMILES strings for missing molecules
        missing_molblocks = missing_descriptors['molblock'].tolist()
        missing_smiles = []
        for molblock in missing_molblocks:
            mol = Chem.MolFromMolBlock(molblock)
            if mol:
                missing_smiles.append(Chem.MolToSmiles(mol))
            else:
                missing_smiles.append(None)
                logging.warning(f"Failed to parse molblock: {molblock}")
        
        padel_results = []
        molregnos = missing_descriptors['molregno'].tolist()
        cur = conn.cursor()

        for smiles, molregno in tqdm(zip(missing_smiles, molregnos), total=len(molregnos)):
            if smiles:
                descriptors_dict = calc_padel_descriptors(smiles)
                padel_results.append(descriptors_dict)

                # Directly store to database
                cur.execute(
                    'INSERT INTO cs_mdfps_schema.padeldescriptors(molregno, padeldescriptors) VALUES(%s, %s)',
                    (molregno, json.dumps(descriptors_dict))
                )

        conn.commit()

        # Merge the newly calculated descriptors back into the main dataframe
        updated_descriptors_df = get_padel_descriptors_from_db(conn)
        df.update(pd.merge(df, updated_descriptors_df, on='molregno', how='left'))

    return df
