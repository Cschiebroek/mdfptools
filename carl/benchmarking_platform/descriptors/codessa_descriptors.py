import psycopg2
import pandas as pd
import logging
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem, rdFreeSASA
import numpy as np

# Define SMARTS pattern for Hydrogen Bond Donors
HBD_SMARTS = "[N&!H0&v3,N&!H0&+1&v4,O&H1&+0,S&H1&+0,n&H1&+0]"
HBD_PATTERN = Chem.MolFromSmarts(HBD_SMARTS)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Calculate CODESSA Descriptors
def calculate_codessa_descriptors(mol):
    """
    Calculate the CODESSA descriptors for a given molecule.
    """
    try:
        descriptors = {}
        descriptors['GravitationalIndex'] = calculate_gi(mol)
        descriptors['HDCA'] = calculate_hdca(mol)
        descriptors['SA2_F'] = calculate_surface_area_of_fluorine(mol)
        descriptors['MNAC_Cl'] = calculate_max_net_atomic_charge_chlorine(mol)
        descriptors['SA_N'] = calculate_surface_area_of_nitrogen(mol)
        return descriptors
    except Exception as e:
        logging.error(f"Error calculating CODESSA descriptors for molecule: {e}")
        return None

# Helper functions to calculate individual descriptors
def calculate_gi(mol):
    try:
        mol = Chem.AddHs(mol, addCoords=True)
        conf = mol.GetConformer()
        gi_sum = 0.0
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            m_i, m_j = mol.GetAtomWithIdx(i).GetMass(), mol.GetAtomWithIdx(j).GetMass()
            r_ij = np.linalg.norm(np.array(conf.GetAtomPosition(i)) - np.array(conf.GetAtomPosition(j)))
            gi_sum += (m_i * m_j) / (r_ij ** 2)
        return gi_sum
    except Exception as e:
        logging.error(f"Error calculating Gravitational Index: {e}")
        return np.nan

def calculate_hdca(mol):
    try:
        if rdMolDescriptors.CalcNumHBD(mol) == 0:
            return 0.0
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.ComputeGasteigerCharges(mol)
        radii = rdFreeSASA.classifyAtoms(mol)
        stot = rdFreeSASA.CalcSASA(mol, radii)
        hdca_sum = 0.0
        hbd_matches = mol.GetSubstructMatches(HBD_PATTERN)
        for match in hbd_matches:
            donor_atom_idx = match[0]
            hydrogen_atoms = [n.GetIdx() for n in mol.GetAtomWithIdx(donor_atom_idx).GetNeighbors() if n.GetAtomicNum() == 1]
            for hydrogen_idx in hydrogen_atoms:
                qd = float(mol.GetAtomWithIdx(hydrogen_idx).GetProp('_GasteigerCharge'))
                sd = rdMolDescriptors.CalcLabuteASA(mol, hydrogen_idx)
                hdca_sum += (qd * np.sqrt(sd) / np.sqrt(stot))
        return hdca_sum
    except Exception as e:
        logging.error(f"Error calculating HDCA: {e}")
        return np.nan

def calculate_surface_area_of_fluorine(mol):
    try:
        fluorine_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'F']
        return sum(rdMolDescriptors.CalcLabuteASA(mol, atom.GetIdx()) for atom in fluorine_atoms) if fluorine_atoms else 0
    except Exception as e:
        logging.error(f"Error calculating Surface Area of Fluorine: {e}")
        return np.nan

def calculate_max_net_atomic_charge_chlorine(mol):
    try:
        chlorine_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'Cl']
        if chlorine_atoms:
            AllChem.ComputeGasteigerCharges(mol)
            return max(float(atom.GetProp('_GasteigerCharge')) for atom in chlorine_atoms)
        else:
            return 0
    except Exception as e:
        logging.error(f"Error calculating Maximum Net Atomic Charge for Chlorine: {e}")
        return np.nan

def calculate_surface_area_of_nitrogen(mol):
    try:
        nitrogen_atoms = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'N']
        return sum(rdMolDescriptors.CalcLabuteASA(mol, atom.GetIdx()) for atom in nitrogen_atoms) if nitrogen_atoms else 0
    except Exception as e:
        logging.error(f"Error calculating Surface Area of Nitrogen: {e}")
        return np.nan

# Functions to interact with the database
def get_codessa_descriptors_from_db(conn, molregnos):
    molregnos_str = ','.join(map(str, molregnos))
    query = f"""
    SELECT molregno, GravitationalIndex, HDCA, SA2_F, MNAC_Cl, SA_N
    FROM cs_mdfps_schema.codessa_descriptors
    WHERE molregno IN ({molregnos_str});
    """
    return pd.read_sql(query, conn)

def store_codessa_descriptors_to_db(conn, df):
    try:
        cur = conn.cursor()
        for _, row in df.iterrows():
            cur.execute(
                'INSERT INTO cs_mdfps_schema.codessa_descriptors(molregno, GravitationalIndex, HDCA, SA2_F, MNAC_Cl, SA_N) VALUES(%s, %s, %s, %s, %s, %s)',
                (row['molregno'], row['GravitationalIndex'], row['HDCA'], row['SA2_F'], row['MNAC_Cl'], row['SA_N'])
            )
        conn.commit()
        logging.info("CODESSA descriptors stored in the database.")
    except Exception as e:
        logging.error(f"Error storing CODESSA descriptors to database: {e}")

def calculate_codessa_descriptor_df(df, conn):
    try:
        codessa_descriptors_df = get_codessa_descriptors_from_db(conn, df['molregno'].tolist())
        missing_molregnos = df[~df['molregno'].isin(codessa_descriptors_df['molregno'])]['molregno'].tolist()
        if missing_molregnos:
            logging.info(f"Calculating CODESSA descriptors for {len(missing_molregnos)} molecules...")
            missing_df = df[df['molregno'].isin(missing_molregnos)]
            missing_df = calculate_descriptor_dfs(missing_df)
            store_codessa_descriptors_to_db(conn, missing_df[['molregno', 'GravitationalIndex', 'HDCA', 'SA2_F', 'MNAC_Cl', 'SA_N']])
            df = pd.merge(df, missing_df, on='molregno', how='left')
        else:
            logging.info("All CODESSA descriptors are already in the database.")
        df = pd.merge(df, codessa_descriptors_df, on='molregno', how='left')
        return df
    except Exception as e:
        logging.error(f"Error during CODESSA descriptor DataFrame calculation: {e}")
        return df

def calculate_descriptor_dfs(df):
    descriptors = []
    for molblock in df['molblock']:
        mol = Chem.MolFromMolBlock(molblock)
        descriptor_values = calculate_codessa_descriptors(mol)
        if descriptor_values:
            descriptors.append(descriptor_values)
    descriptors_df = pd.DataFrame(descriptors)
    df = pd.concat([df, descriptors_df], axis=1)
    return df
