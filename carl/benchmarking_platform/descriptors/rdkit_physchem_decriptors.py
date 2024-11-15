import pandas as pd
from rdkit.Chem import Descriptors, PandasTools
import logging
import json

def get_rdkit_descriptors_from_db(conn):
    """Fetch RDKit descriptors from the database."""
    # Ensure molregnos is a flat list of unique integers
    query = f"""
    SELECT molregno, PhysChemDescriptors
    FROM cs_mdfps_schema.PhysChemDescriptors
    """
    df = pd.read_sql(query, conn)
    df = pd.concat([df.drop(['physchemdescriptors'], axis=1), df['physchemdescriptors'].apply(pd.Series)], axis=1)
    #drop fr_ and IPC descriptors
    df = df.drop([col for col in df.columns if 'Ipc' in col or 'fr_' in col], axis=1)
    return df

def calc_rdkit_descriptors(mol):
    """Calculate RDKit descriptors for a single molecule."""
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    descriptor_values = Descriptors.CalcMolDescriptors(mol)
    descriptors_dict = dict(zip(descriptor_names, descriptor_values))
    return json.dumps(descriptors_dict)  # Convert dict to JSON string

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
        cur = conn.cursor()
        
        missing_descriptors['PhysChemDescriptors'] = missing_descriptors['ROMol'].apply(calc_rdkit_descriptors)
        
        # Insert the missing descriptors into the database
        for i, row in missing_descriptors.iterrows():
            cur.execute(
                'INSERT INTO cs_mdfps_schema.PhysChemDescriptors(molregno, PhysChemDescriptors) VALUES(%s, %s)',
                (row['molregno'], row['PhysChemDescriptors'])
            )
        conn.commit()

        # Merge the newly calculated descriptors back into the main dataframe
        updated_descriptors_df = get_rdkit_descriptors_from_db(conn)
        df.update(pd.merge(df, updated_descriptors_df, on='molregno', how='left'))

    return df
