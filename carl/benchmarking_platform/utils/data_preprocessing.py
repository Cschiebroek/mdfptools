import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
from rdkit.Chem import PandasTools
from rdkit import Chem
import json
from rdkit.Chem import Descriptors
import warnings

# Disable pandas userwarning
warnings.simplefilter(action='ignore', category=UserWarning)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import warnings
from descriptors.mdfp import extract_mdfp_features
from descriptors.rdkit import calculate_RDKit_PhysChem_descriptors
from descriptors.fingerprints import calculate_fingerprints
from descriptors.codessa_descriptors import calculate_codessa_descriptor_df
from descriptors.padel import calculate_Padel_descriptors

from padelpy import from_smiles
import os
import pickle

# Disable pandas userwarning
warnings.simplefilter(action='ignore', category=UserWarning)

def get_data_from_db(conn):
    query = """
    SELECT 
        e.molregno,
        c.conf_id,
        e.vp_log10_pa,
        m.mdfp,
        c.molblock,
        m.md_experiment_uuid,
        cd.confgen_uuid
    FROM 
        cs_mdfps_schema.experimental_data e
    JOIN 
        conformers c ON e.molregno = c.molregno
    LEFT JOIN 
        cs_mdfps_schema.mdfp_experiment_data m ON c.conf_id = m.conf_id
    LEFT JOIN 
        cs_mdfps_schema.confid_data cd ON c.conf_id = cd.conf_id;
    """
    df = pd.read_sql(query, conn)
    md_experiment_uuids_to_remove = ['80b643c8-5bdc-4b63-a12d-6f1ba3f7dd2a',
                                    '24e3946b-fb2c-47bf-9965-1682bb0d63c9',
                                    '5166be97-ef21-4cc5-bee1-719c7b9e3397',
                                    '13d08336-fb79-4042-83ce-af906193ff20']





    df = df[~df['md_experiment_uuid'].isin(md_experiment_uuids_to_remove)]
    df = df[df['confgen_uuid'] != '11093a30-b6d0-4e3f-a22b-8dcad60d6a11']
    df = df.dropna(subset=['mdfp'])
    return df

def prepare_data(conn):
    logging.info("Loading data from the database...")
    df = get_data_from_db(conn)

    # Load the test data (to be excluded from training)
    df_test = pd.read_csv('/localhome/cschiebroek/MDFP_VP/mdfptools/carl/data_curation/OPERA_Naef_Stratified_Test.csv')
    df = df[~df['molregno'].isin(df_test['molregno'])]

    logging.info("Calculating descriptors...")
    
    # Add descriptors
    df = calculate_RDKit_PhysChem_descriptors(df,conn)
    df = extract_mdfp_features(df,conn)
    df = calculate_fingerprints(df,'maccs')
    df = calculate_fingerprints(df,'ecfp4')
    df = calculate_codessa_descriptor_df(df,conn)
    df = calculate_Padel_descriptors(df,conn)

    return df

def get_descriptors_from_db(conn, molregnos):
    molregnos_str = ','.join([str(molregno) for molregno in molregnos])
    query = f"""
    SELECT molregno, PhysChemDescriptors
    FROM cs_mdfps_schema.PhysChemDescriptors
    WHERE molregno IN ({molregnos_str});
    """
    df = pd.read_sql(query, conn)
    df = pd.concat([df.drop(['physchemdescriptors'], axis=1), df['physchemdescriptors'].apply(pd.Series)], axis=1)
    return df

def calc_descriptors(mol):
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    descriptor_values = Descriptors.CalcMolDescriptors(mol)
    descriptors_dict = dict(zip(descriptor_names, descriptor_values))
    return json.dumps(descriptors_dict)  # Convert dict to JSON string


def preprocess_data(df, seed):
    logging.info("Splitting data into training and testing sets...")   
    # Split remaining data into training and validation sets
    train_molregnos, val_molregnos = train_test_split(df['molregno'].unique(), test_size=0.2, random_state=42 + seed)
    df_train = df[df['molregno'].isin(train_molregnos)]
    df_val = df[df['molregno'].isin(val_molregnos)]
    
    train_molregnos = df_train['molregno'].tolist()
    val_molregnos = df_val['molregno'].tolist()
    
    train_y = df_train['vp_log10_pa']
    val_y = df_val['vp_log10_pa']

    return train_molregnos, val_molregnos, train_y, val_y, df_train, df_val

def get_features(df_train, df_val, descriptor_name, scale=False):
    if descriptor_name == 'RDKit_PhysChem':
        features = [d[0] for d in Descriptors._descList if d[0] in df_train.columns]
    elif descriptor_name == 'MACCS':
        features = 'maccs'
    elif descriptor_name == 'ECFP4':
        features = 'ecfp4'
    elif descriptor_name == 'MDFP':
        features = ['NumHeavyAtoms', 'NumRotatableBonds', 'NumN', 'NumO', 'NumF', 'NumP', 'NumS', 'NumCl', 'NumBr', 'NumI',
                     'water_intra_crf_mean', 'water_intra_crf_std', 'water_intra_crf_median', 'water_intra_lj_mean',
                     'water_intra_lj_std', 'water_intra_lj_median', 'water_total_crf_mean', 'water_total_crf_std',
                     'water_total_crf_median', 'water_total_lj_mean', 'water_total_lj_std', 'water_total_lj_median',
                     'water_intra_ene_mean', 'water_intra_ene_std', 'water_intra_ene_median', 'water_total_ene_mean',
                     'water_total_ene_std', 'water_total_ene_median', 'water_rgyr_mean', 'water_rgyr_std',
                     'water_rgyr_median', 'water_sasa_mean', 'water_sasa_std', 'water_sasa_median']
    elif descriptor_name == 'codessa':
        features = ['gravitationalindex', 'hdca', 'sa2_f', 'mnac_cl', 'sa_n']

    elif descriptor_name == 'padel':
        if os.path.exists('padel_names.pkl'):
            with open('padel_names.pkl', 'rb') as f:
                features = pickle.load(f)
        else:
            features = from_smiles('CCO').keys()

        


    else:
        raise ValueError(f"Invalid descriptor name: {descriptor_name}")

    # Ensure that the features are numeric
    train_X = df_train[features].apply(pd.to_numeric, errors='coerce')
    val_X = df_val[features].apply(pd.to_numeric, errors='coerce')

    if descriptor_name == 'ECFP4' or descriptor_name == 'MACCS':
        train_X = [list(x) for x in train_X]
        val_X = [list(x) for x in val_X]

    else:
        df_train = df_train.drop(columns=['NumRotatableBonds'])
        df_train['NumRotatableBonds'] = df_train['molblock'].apply(lambda x: Descriptors.NumRotatableBonds(Chem.MolFromMolBlock(x)))
        df_val = df_val.drop(columns=['NumRotatableBonds'])
        df_val['NumRotatableBonds'] = df_val['molblock'].apply(lambda x: Descriptors.NumRotatableBonds(Chem.MolFromMolBlock(x)))
    if scale:
        scaler = StandardScaler()
        train_X = pd.DataFrame(scaler.fit_transform(train_X), columns=features, index=df_train.index)
        val_X = pd.DataFrame(scaler.transform(val_X), columns=features, index=df_val.index)
    return train_X, val_X