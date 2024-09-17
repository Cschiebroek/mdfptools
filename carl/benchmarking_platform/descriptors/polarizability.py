import pandas as pd
import logging
from serenityff.charge.tree.dash_tree import DASHTree, TreeType
from rdkit import Chem
from rdkit.Chem import Fragments

def wipe_existing_descriptors(conn):
    """Wipe existing entries in the Liang descriptors table."""
    logging.info("Wiping existing Liang descriptors from the database...")
    with conn.cursor() as cur:
        cur.execute("DELETE FROM cs_mdfps_schema.liang_descriptors_dash")
        conn.commit()

def wipe_existing_results(conn):
    """Wipe existing entries in the results database related to these descriptors."""
    logging.info("Wiping existing results related to Liang descriptors from the results database...")
    with conn.cursor() as cur:
        cur.execute("DELETE FROM results_database WHERE descriptor IN ('polarizability', 'amine', 'carbonyl', 'carboxylic_acid', 'hydroxyl', 'nitrile', 'nitro', 'alcohol')")
        conn.commit()

def get_liang_descriptors_from_db(conn):
    """Fetch Liang descriptors from the database."""
    query = """
    SELECT molregno, polarizability, amine, carbonyl, carboxylic_acid, hydroxyl, nitrile, nitro, alcohol
    FROM cs_mdfps_schema.liang_descriptors_dash
    """
    df = pd.read_sql(query, conn)
    return df

def calculate_dashtree_polarizability(molblock, tree):
    """Calculate polarizability using DASH-tree for a given molecule."""
    try:
        mol = Chem.MolFromMolBlock(molblock)
        mol = Chem.AddHs(mol, addCoords=True)
        polarizability = tree.get_molecular_polarizability(mol=mol)
    except Exception as e:
        logging.error(f"Error calculating polarizability: {e}")
        polarizability = None
    return polarizability

def count_functional_groups(molblock):
    """Count functional groups using RDKit fragment functions."""
    mol = Chem.MolFromMolBlock(molblock)
    descriptors = {
        'amine': Fragments.fr_NH1(mol) + Fragments.fr_NH2(mol),
        'carbonyl': Fragments.fr_C_O(mol),
        'carboxylic_acid': Fragments.fr_COO(mol),
        'hydroxyl': Fragments.fr_Al_OH(mol) + Fragments.fr_Ar_OH(mol),
        'nitrile': Fragments.fr_nitrile(mol),
        'nitro': Fragments.fr_nitro(mol),
        'alcohol': Fragments.fr_Al_OH(mol)
    }
    return descriptors

def calculate_liang_descriptors(molblock, tree):
    """Calculate Liang descriptors for a single molecule."""
    descriptors = {
        'polarizability': calculate_dashtree_polarizability(molblock, tree)
    }
    descriptors.update(count_functional_groups(molblock))
    return descriptors

def calculate_liang_descriptors_df(df, conn):

    logging.info("Fetching Liang descriptors from the database...")
    df_liang_descriptors = get_liang_descriptors_from_db(conn)
    
    logging.info("Merging Liang descriptors with the main dataframe...")
    df = pd.merge(df, df_liang_descriptors, on='molregno', how='left')

    # Identify missing descriptors and calculate them
    missing_descriptors = df[df['carboxylic_acid'].isnull()]
    if not missing_descriptors.empty:
        logging.info(f"Calculating Liang descriptors for {len(missing_descriptors)} molecules...")
        cur = conn.cursor()

        # Initialize the DASH-tree
        tree = DASHTree(tree_type=TreeType.FULL)
        
        # Apply calculate_liang_descriptors to each row and store results in a DataFrame
        calculated_descriptors = missing_descriptors['molblock'].apply(lambda molblock: pd.Series(calculate_liang_descriptors(molblock, tree)))
        
        # Add the molregno column to the DataFrame of calculated descriptors
        calculated_descriptors['molregno'] = missing_descriptors['molregno'].values
        
        # Insert the newly calculated descriptors into the database
        insert_values = [
            (
                row['molregno'], row['polarizability'], row['amine'], row['carbonyl'], 
                row['carboxylic_acid'], row['hydroxyl'], row['nitrile'], 
                row['nitro'], row['alcohol']
            )
            for _, row in calculated_descriptors.iterrows()
        ]
        cur.executemany(
            '''
            INSERT INTO cs_mdfps_schema.liang_descriptors_dash 
            (molregno, polarizability, amine, carbonyl, carboxylic_acid, hydroxyl, nitrile, nitro, alcohol) 
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ''', 
            insert_values
        )
        conn.commit()

        # Merge the newly calculated descriptors back into the main dataframe
        updated_descriptors_df = get_liang_descriptors_from_db(conn)
        df = pd.merge(df, updated_descriptors_df, on='molregno', how='left')

    return df
