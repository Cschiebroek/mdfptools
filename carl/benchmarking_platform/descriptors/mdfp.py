import pandas as pd
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_mdfp_features(df_1, conn):
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

    # Execute the query
    cur = conn.cursor()
    cur.execute(query)
    data = cur.fetchall()

    # Convert the data to a DataFrame
    df = pd.DataFrame(data, columns=['molregno', 'conf_id', 'vp_log10_pa', 'mdfp', 'molblock', 'md_experiment_uuid', 'confgen_uuid'])

    # Filter the DataFrame based on specific md_experiment_uuid
    allowed_md_experiment_uuids = [
        "fc57851e-b654-4338-bcdd-faa28ec66253",
        '7a5837f2-e4ad-4e17-a2c3-6e5e956f938b',
        '26dee5cf-c401-4924-9c43-6e5f8f311763'
    ]
    df = df[df['md_experiment_uuid'].isin(allowed_md_experiment_uuids)]
    df = df.dropna(subset=['mdfp'])

    # Extract the 'mdfp' vector from JSON and handle potential errors
    def extract_mdfp_vector(mdfp_str):
        try:
            return json.loads(mdfp_str)['mdfp'] if mdfp_str else None
        except (json.JSONDecodeError, TypeError) as e:
            logging.error(f"Error decoding JSON: {e}")
            return None

    df['mdfp_vec'] = df['mdfp'].apply(extract_mdfp_vector)
    
    # Define MDFP feature names
    mdfp_features_all = [
        'NumHeavyAtoms', 'NumRotatableBonds', 'NumN', 'NumO', 'NumF', 'NumP', 'NumS', 'NumCl', 'NumBr', 'NumI',
        'water_intra_crf_mean', 'water_intra_crf_std', 'water_intra_crf_median', 'water_intra_lj_mean',
        'water_intra_lj_std', 'water_intra_lj_median', 'water_total_crf_mean', 'water_total_crf_std',
        'water_total_crf_median', 'water_total_lj_mean', 'water_total_lj_std', 'water_total_lj_median',
        'water_intra_ene_mean', 'water_intra_ene_std', 'water_intra_ene_median', 'water_total_ene_mean',
        'water_total_ene_std', 'water_total_ene_median', 'water_rgyr_mean', 'water_rgyr_std',
        'water_rgyr_median', 'water_sasa_mean', 'water_sasa_std', 'water_sasa_median'
    ]

    # Extract MDFP features and add them to the DataFrame
    mdfp_vecs_lists = df['mdfp_vec'].tolist()
    for i, mdfp_feature in enumerate(mdfp_features_all):
        df[mdfp_feature] = pd.Series([vec[i] if vec else None for vec in mdfp_vecs_lists])

    # Merge with the original DataFrame, on molregno
    df_combined = pd.merge(df_1, df, on='molregno', how='left')

    logging.info(f"Processed {len(df_combined)} records, merged MDFP features.")
    return df_combined
