from padelpy import padeldescriptor
import pandas as pd
import os
from rdkit import Chem

def calculate_padel_descriptors(df):
    """
    Calculate PaDEL descriptors for each molecule in the DataFrame.
    
    Parameters:
    - df: pandas DataFrame containing the molecule information.
    - smiles_col: Column name containing the SMILES strings or molecular data.
    
    Returns:
    - df: Updated DataFrame with PaDEL descriptors.
    """
    # Specify the path to the PaDEL-Descriptor jar file
    padel_jar_path = '/path/to/PaDEL-Descriptor/PaDEL-Descriptor.jar'
    
    # Create a temporary file to store SMILES strings
    input_smiles_file = 'input_smiles.smi'
    output_csv_file = 'padel_descriptors.csv'
    
    # Write the SMILES to the input file
    df['smiles'] = df['molblock'].apply(lambda x: Chem.MolToSmiles(Chem.MolFromMolBlock(x)))
    df['smiles'].to_csv(input_smiles_file, index=False, header=False)

    # Run PaDEL-Descriptor
    padeldescriptor(
        mol_dir=input_smiles_file,
        d_file=output_csv_file,
        descriptors=True,
        fingerprints=False,
        convert3d=False,
        retain3d=False,
        threads=4,  # Adjust the number of threads if necessary
        log=False,
        waiting=False,
        padel_path=padel_jar_path
    )

    # Read the output descriptors
    padel_descriptors = pd.read_csv(output_csv_file)

    # Remove the temporary files
    os.remove(input_smiles_file)
    os.remove(output_csv_file)
    
    # Merge PaDEL descriptors with the original DataFrame
    df = pd.concat([df, padel_descriptors], axis=1)
    
    return df
