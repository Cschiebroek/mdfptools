from serenityff.charge.tree.dash_tree import DASHTree, TreeType
from rdkit import Chem

# Initialize the DASH-tree
TREE = DASHTree(tree_type=TreeType.FULL)

def calculate_dashtree_polarizability(molblock):
    """
    Calculate polarizability using DASH-tree for a given molecule.
    
    Parameters:
    smiles (str): SMILES string of the molecule
    
    Returns:
    float: Calculated polarizability
    """
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromMolBlock(molblock)
    mol = Chem.AddHs(mol, addCoords=True)
        
    try:
        polarizability = TREE.get_molecular_polarizability(mol=mol)
    except Exception as e:
        print(f"Error calculating polarizability: {e}")
        polarizability = None
        
    return polarizability

import pandas as pd

def calculate_custom_descriptors(df):
    # Calculate polarizability using DASH-tree and functional groups for each molecule
    df['polarizability'] = df['molblock'].apply(calculate_dashtree_polarizability)
    for func_group in functional_group_smarts.keys():
        df[func_group] = df['molblock'].apply(count_functional_groups, func_group=func_group)
    return df

from rdkit import Chem

# Define SMARTS patterns for functional groups
functional_group_smarts = {
    # 'amine': '[NX3;H2,H1;!$(NC=O)]',  # Primary amine --> make recursive for any amine, primary, secondary, tertiary
    'amine': '[NX3]',  # Primary amine
    'carbonyl': '[CX3]=[O]',                # Carbonyl group
    'carboxylic_acid': '[CX3](=O)[OX2H1]',  # Carboxylic acid
    'hydroxyl': '[OX2H]' ,             # Hydroxyl group
    'nitrile': '[CX2]#[NX1]',               # Nitrile group
    'nitro': '[NX3](=O)(=O)' ,           # Nitro group
    'alcohol': '[CX4][OX2H]'             # Alcohol group
}

def count_functional_groups(molblock,func_group):
    """
    Count the number of functional groups in a molecule.
    
    Parameters:
    smiles (str): SMILES string of the molecule
    
    Returns:
    dict: Counts of each functional group
    """
    mol = Chem.MolFromMolBlock(molblock)
    counts = {}
    
    smarts = functional_group_smarts[func_group]
    patt = Chem.MolFromSmarts(smarts)
    count = len(mol.GetSubstructMatches(patt))
    
    return count
