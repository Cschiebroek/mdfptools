from rdkit.Chem import AllChem, MACCSkeys
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

def calculate_bit_fingerprints(df, fingerprint_type='maccs', fpSize=2048, radius=2):
    """
    Calculate molecular fingerprints and add them to the dataframe.

    Parameters:
    - df: Pandas DataFrame containing a column 'molblock'.
    - fingerprint_type: Type of fingerprint to calculate. Options: 'maccs', 'ecfp4', 'rdkitfp', 'atompair', 'torsion'.
    - fpSize: Size of the fingerprint (default 2048).
    - radius: Radius for Morgan/ECFP fingerprints (default 2).

    Returns:
    - df: DataFrame with a new column containing the fingerprints.
    """

    # Generate molecules from molblocks
    ms = [Chem.MolFromMolBlock(x) for x in df['molblock']]

    # Select the appropriate fingerprint generator
    if fingerprint_type == 'maccs':
        fps = [MACCSkeys.GenMACCSKeys(mol) for mol in ms]
    elif fingerprint_type == 'ecfp4':
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
        fps = [mfpgen.GetFingerprint(mol) for mol in ms]
    elif fingerprint_type == 'rdkitfp':
        rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fpSize)
        fps = [rdkgen.GetFingerprint(mol) for mol in ms]
    elif fingerprint_type == 'atompair':
        apgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fpSize)
        fps = [apgen.GetFingerprint(mol) for mol in ms]
    elif fingerprint_type == 'torsion':
        ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=fpSize)
        fps = [ttgen.GetFingerprint(mol) for mol in ms]
    else:
        raise ValueError(f"Unknown fingerprint type: {fingerprint_type}")
    fingerprint_type = fingerprint_type + '_bit' if fingerprint_type != 'maccs' else fingerprint_type
    df[fingerprint_type] = fps
    return df

def calculate_count_fingerprints(df, fingerprint_type='maccs', fpSize=2048, radius=2):
    """
    Calculate molecular fingerprints and add them to the dataframe.

    Parameters:
    - df: Pandas DataFrame containing a column 'molblock'.
    - fingerprint_type: Type of fingerprint to calculate. Options: 'maccs', 'ecfp4', 'rdkitfp', 'atompair', 'torsion'.
    - fpSize: Size of the fingerprint (default 2048).
    - radius: Radius for Morgan/ECFP fingerprints (default 2).

    Returns:
    - df: DataFrame with a new column containing the fingerprints.
    """

    # Generate molecules from molblocks
    ms = [Chem.MolFromMolBlock(x) for x in df['molblock']]

    # Select the appropriate fingerprint generator
    if fingerprint_type == 'maccs':
        fps = [MACCSkeys.GenMACCSKeys(mol) for mol in ms]
    elif fingerprint_type == 'ecfp4':
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fpSize)
        fps = [mfpgen.GetCountFingerprint(mol) for mol in ms]
    elif fingerprint_type == 'rdkitfp':
        rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fpSize)
        fps = [rdkgen.GetCountFingerprint(mol) for mol in ms]
    elif fingerprint_type == 'atompair':
        apgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=fpSize)
        fps = [apgen.GetCountFingerprint(mol) for mol in ms]
    elif fingerprint_type == 'torsion':
        ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=fpSize)
        fps = [ttgen.GetCountFingerprint(mol) for mol in ms]
    else:
        raise ValueError(f"Unknown fingerprint type: {fingerprint_type}")
    fingerprint_type = fingerprint_type + '_count' if fingerprint_type != 'maccs' else fingerprint_type
    # Add the fingerprints to the dataframe
    df[fingerprint_type] = fps
    return df
