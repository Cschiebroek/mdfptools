from rdkit.Chem import AllChem, MACCSkeys
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit import Chem

def calculate_maccs_keys(df,conn):
    ms = [Chem.MolFromMolBlock(x) for x in df['molblock']]
    fps = [MACCSkeys.GenMACCSKeys(x) for x in ms]
    df['maccs'] = fps
    return df

def calculate_ecfp4(df,conn):
    ms = [Chem.MolFromMolBlock(x) for x in df['molblock']]
    fps = [AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in ms]
    df['ecfp4'] = fps
    return df
