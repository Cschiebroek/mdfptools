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
from descriptors.rdkit_physchem_decriptors import calculate_RDKit_PhysChem_descriptors
from descriptors.fingerprints import calculate_bit_fingerprints, calculate_count_fingerprints
from descriptors.codessa_descriptors import calculate_codessa_descriptor_df
from descriptors.polarizability import calculate_liang_descriptors_df
from descriptors.atom_contrib import EMPTY_DICT, add_crippen_atom_counts_to_df

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

def prepare_data(conn,descriptor_to_use):
    logging.info("Loading data from the database...")
    df = get_data_from_db(conn)

    # Load the test data (to be excluded from training)
    df_test = pd.read_csv('/localhome/cschiebroek/MDFP_VP/mdfptools/carl/data_curation/OPERA_Naef_Stratified_Test.csv')
    df = df[~df['molregno'].isin(df_test['molregno'])]

    logging.info("Calculating descriptors...")


    # Add descriptors
    if 'RDKit_PhysChem' in descriptor_to_use:
        df = calculate_RDKit_PhysChem_descriptors(df, conn)
    if 'MDFP' in descriptor_to_use:
        df = extract_mdfp_features(df, conn)
    if 'MACCS' in descriptor_to_use:
        df = calculate_bit_fingerprints(df, 'maccs')
    if 'ECFP4_bit' in descriptor_to_use:
        df = calculate_bit_fingerprints(df, 'morgan')
    if 'ECFP4_count' in descriptor_to_use:
        df = calculate_count_fingerprints(df, 'morgan')
    if 'codessa' in descriptor_to_use:
        df = calculate_codessa_descriptor_df(df, conn)
    if 'padel' in descriptor_to_use:
        from descriptors.padel import calculate_Padel_descriptors
        df = calculate_Padel_descriptors(df, conn)
    if 'liang_descriptors' in descriptor_to_use:
        df = calculate_liang_descriptors_df(df, conn)
    if 'crippen_atoms' in descriptor_to_use:
        df = add_crippen_atom_counts_to_df(df)
    if 'mfp0' in descriptor_to_use:
        logging.info("Calculating mfp0 fingerprints")
        df = calculate_bit_fingerprints(df, fingerprint_type='morgan', fpSize=2048, radius=0)
    if 'mfp3' in descriptor_to_use:
        logging.info("Calculating mfp3 fingerprints")
        df = calculate_bit_fingerprints(df, fingerprint_type='morgan', fpSize=2048, radius=3)
    logging.info(df.columns)
    logging.info("Data loaded and descriptors calculated, dropping NaNs...")
    # Drop rows with any NaNs
    len_df_before = len(df)
    df = df.dropna()
    len_df_after = len_df_before - len(df)
    logging.info(f"Dropped {len_df_after} rows due to missing values.")

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

def preprocess_data_kfold(df, train_index, test_index):
    logging.info("Splitting data into training and testing sets...")   
    train_molregnos = df['molregno'].iloc[train_index].tolist()
    val_molregnos = df['molregno'].iloc[test_index].tolist()
    df_train = df[df['molregno'].isin(train_molregnos)]
    df_val = df[df['molregno'].isin(val_molregnos)]
    
    train_y = df_train['vp_log10_pa'].tolist()
    val_y = df_val['vp_log10_pa'].tolist()

    return train_molregnos, val_molregnos, train_y, val_y, df_train, df_val

def get_features_from_config(descriptor_name):
    with open('configs/descriptors.json', 'r') as file:
        config = json.load(file)
    if descriptor_name in config['descriptors']:
        return config['descriptors'][descriptor_name]['features']
    else:
        raise ValueError(f"Descriptor '{descriptor_name}' not found in configuration.")


def get_features(df_train, df_val, descriptor_name, scale=False):
    if descriptor_name == 'RDKit_PhysChem':
        features = [d[0] for d in Descriptors._descList if d[0] in df_train.columns]
        #drop IPC and fr_ descriptors
        features = [feature for feature in features if 'IPC' not in feature and 'fr_' not in feature]

    elif descriptor_name == 'padel':
        if os.path.exists('padel_names.pkl'):
            with open('padel_names.pkl', 'rb') as f:
                features = pickle.load(f)
        else:
            from padelpy import from_smiles
            features = from_smiles('CCO').keys()
            features = [feature for feature in features if feature in df_train.columns]
            #save the features to a file
            with open('padel_names.pkl', 'wb') as f:
                pickle.dump(features, f)

    elif descriptor_name == 'liang_descriptors':
        features = ['polarizability', 'hydroxyl', 'carbonyl', 'amine', 'carboxylic_acid', 'nitro', 'nitrile']
    elif descriptor_name == 'crippen_atoms':
        features = list(EMPTY_DICT.keys())

    else:
        try:
            features = get_features_from_config(descriptor_name)
        except ValueError:
            raise ValueError(f"Descriptor '{descriptor_name}' not found in configuration.")



    # Ensure that the features are numeric
    train_X = df_train[features].apply(pd.to_numeric, errors='coerce')
    val_X = df_val[features].apply(pd.to_numeric, errors='coerce')        

    if descriptor_name == 'ECFP4_bit' or descriptor_name == 'ECFP4_count' or descriptor_name == 'MACCS' or descriptor_name == 'mfp0' or descriptor_name == 'mfp3':
        train_X = [list(x) for x in train_X]
        val_X = [list(x) for x in val_X]

    else:
        features = train_X.columns 
        nans = train_X.isna().sum() + val_X.isna().sum()
        nans = nans[nans > 0]
        if len(nans) > 0:
            logging.info(f"Removing {len(nans)} features with NaN values")
            train_X = train_X.drop(columns=nans.index)
        features = train_X.columns

    if descriptor_name == 'RDKit_PhysChem' or descriptor_name == 'MDFP':
        try:
            df_train = df_train.drop(columns=['NumRotatableBonds'])
        except KeyError:
            pass
        df_train['NumRotatableBonds'] = df_train['molblock'].apply(lambda x: Descriptors.NumRotatableBonds(Chem.MolFromMolBlock(x)))
        try:
            df_val = df_val.drop(columns=['NumRotatableBonds'])
        except KeyError:
            df_val['NumRotatableBonds'] = df_val['molblock'].apply(lambda x: Descriptors.NumRotatableBonds(Chem.MolFromMolBlock(x)))
    if scale:
        scaler = StandardScaler()
        train_X = pd.DataFrame(scaler.fit_transform(train_X), columns=features, index=df_train.index)
        val_X = pd.DataFrame(scaler.transform(val_X), columns=features, index=df_val.index)
    if descriptor_name == 'crippen_atoms':
        train_X = train_X.rename(columns=dict(zip(list(EMPTY_DICT.keys()), range(110))))
        val_X = val_X.rename(columns=dict(zip(list(EMPTY_DICT.keys()), range(110))))
    return train_X, val_X