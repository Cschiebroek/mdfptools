import pandas as pd
import logging
from serenityff.charge.tree.dash_tree import DASHTree, TreeType
from rdkit import Chem

# Define SMARTS patterns for functional groups
functional_group_smarts = {
    'amine': '[NX3]',  # Amine group
    'carbonyl': '[CX3]=[O]',  # Carbonyl group
    'carboxylic_acid': '[CX3](=O)[OX2H1]',  # Carboxylic acid group
    'hydroxyl': '[OX2H]',  # Hydroxyl group
    'nitrile': '[CX2]#[NX1]',  # Nitrile group
    'nitro': '[NX3](=O)(=O)',  # Nitro group
    'alcohol': '[CX4][OX2H]'  # Alcohol group
}

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

def count_functional_groups(molblock, func_group):
    """Count the number of functional groups in a molecule."""
    mol = Chem.MolFromMolBlock(molblock)
    smarts = functional_group_smarts[func_group]
    patt = Chem.MolFromSmarts(smarts)
    return len(mol.GetSubstructMatches(patt))

def calculate_liang_descriptors(molblock, tree):
    """Calculate Liang descriptors for a single molecule."""
    descriptors = {
        'polarizability': calculate_dashtree_polarizability(molblock, tree)
    }
    for func_group in functional_group_smarts.keys():
        descriptors[func_group] = count_functional_groups(molblock, func_group)
    return descriptors

def calculate_liang_descriptors_df(df, conn):
    """Main function to fetch or calculate Liang descriptors."""
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
