import pandas as pd
from rdkit.Chem import Descriptors, PandasTools
import logging
import json

def get_rdkit_descriptors_from_db(conn):
    """Fetch RDKit descriptors from the database."""
    query = """
    SELECT molregno, PhysChemDescriptors
    FROM cs_mdfps_schema.PhysChemDescriptors
    """
    df = pd.read_sql(query, conn)
    
    # If PhysChemDescriptors is a JSON string, convert it to individual columns
    df_descriptors = pd.json_normalize(df['PhysChemDescriptors'].apply(json.loads))
    df = pd.concat([df.drop(['PhysChemDescriptors'], axis=1), df_descriptors], axis=1)
    
    return df

def calc_rdkit_descriptors(mol):
    """Calculate RDKit descriptors for a single molecule."""
    try:
        descriptor_values = Descriptors.CalcMolDescriptors(mol)
        descriptors_dict = dict(zip([desc[0] for desc in Descriptors._descList], descriptor_values))
        return descriptors_dict
    except Exception as e:
        logging.error(f"Failed to calculate RDKit descriptors: {e}")
        return {}

def calculate_RDKit_PhysChem_descriptors(df, conn):
    """Main function to fetch or calculate RDKit descriptors."""
    logging.info("Fetching RDKit descriptors from the database...")

    df_rdkit_descriptors = get_rdkit_descriptors_from_db(conn)
    
    logging.info("Merging RDKit descriptors with the main dataframe...")
    df = pd.merge(df, df_rdkit_descriptors, on='molregno', how='left')

    # Identify missing descriptors and calculate them
    missing_descriptors = df[df['MaxAbsEStateIndex'].isnull()]
    if not missing_descriptors.empty:
        logging.info(f"Calculating RDKit descriptors for {len(missing_descriptors)} molecules...")
        PandasTools.AddMoleculeColumnToFrame(missing_descriptors, smilesCol='molblock')

        missing_descriptors['PhysChemDescriptors'] = missing_descriptors['ROMol'].apply(calc_rdkit_descriptors)
        
        # Insert the newly calculated descriptors into the database
        cur = conn.cursor()
        for i, row in missing_descriptors.iterrows():
            cur.execute(
                'INSERT INTO cs_mdfps_schema.PhysChemDescriptors (molregno, PhysChemDescriptors) VALUES (%s, %s)',
                (row['molregno'], json.dumps(row['PhysChemDescriptors']))
            )
        conn.commit()

        # Merge the newly calculated descriptors back into the main dataframe
        updated_descriptors_df = get_rdkit_descriptors_from_db(conn)
        df.update(pd.merge(df, updated_descriptors_df, on='molregno', how='left'))

    return df
