import pandas as pd
from padelpy import from_smiles
import logging
from rdkit import Chem
from tqdm import tqdm
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import json

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

    # Rename 'molregno_x' back to 'molregno' if needed
    if 'molregno_x' in df_padel.columns:
        df_padel = df_padel.rename(columns={'molregno_x': 'molregno'})

    # Drop 'molregno_y' if it exists
    df_padel = df_padel.drop(columns=['molregno_y'], errors='ignore')

    return df_padel

def calculate_Padel_descriptors(df, conn):
    """Main function to fetch or calculate PaDEL descriptors."""
    logging.info("Fetching PaDEL descriptors from the database...")

    df_padel_descriptors = get_padel_descriptors_from_db(conn)

    logging.info("Merging PaDEL descriptors with the main dataframe...")
    #combine the two dataframes on molregno
    df = pd.merge(df, df_padel_descriptors, on='molregno', how='left')

    # Identify missing descriptors and calculate them
    try:
        missing_descriptors = df[df['padeldescriptors'].isnull()]  
    except KeyError:
        missing_descriptors = df
    if not missing_descriptors.empty:
        logging.info(f"Calculating PaDEL descriptors for {len(missing_descriptors)} molecules...")
        
        # Extract SMILES strings for missing molecules
        missing_molblocks = missing_descriptors['molblock'].tolist()  # Replace 'smiles_column' with actual column name
        missing_smiles = [Chem.MolToSmiles(Chem.MolFromMolBlock(molblock)) for molblock in missing_molblocks]
        padel_results = []
        molregnos = missing_descriptors['molregno'].tolist()
        for smiles, molregno in tqdm(zip(missing_smiles, molregnos)):
            descriptors_dict = calc_padel_descriptors(smiles)
            #directly store to database
            cur = conn.cursor()
            cur.execute(
                'INSERT INTO cs_mdfps_schema.padeldescriptors(molregno, padeldescriptors) VALUES(%s, %s)',
                (molregno, json.dumps(descriptors_dict))  # Store descriptors as JSON
            )
            conn.commit()

            padel_results.append(descriptors_dict)

        padel_df = pd.DataFrame(padel_results)
        padel_df['molregno'] = missing_descriptors['molregno'].values
        
        # Insert the newly calculated descriptors into the database
        cur = conn.cursor()
        for i, row in padel_df.iterrows():
            cur.execute(
                'INSERT INTO cs_mdfps_schema.padeldescriptors(molregno, padeldescriptors) VALUES(%s, %s)',
                (row['molregno'], row.to_json())  # Store descriptors as JSON
            )
        conn.commit()

        # Merge the newly calculated descriptors back into the main dataframe
        updated_descriptors_df = get_padel_descriptors_from_db(conn)
        df.update(pd.merge(df, updated_descriptors_df, on='molregno', how='left'))

    return df