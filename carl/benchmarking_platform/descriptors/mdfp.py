import pandas as pd
import logging
import json

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
    allowed_d_experiment_uuids = [
   "fc57851e-b654-4338-bcdd-faa28ec66253",
   '7a5837f2-e4ad-4e17-a2c3-6e5e956f938b',
   '26dee5cf-c401-4924-9c43-6e5f8f311763'
]
    df = df[df['md_experiment_uuid'].isin(allowed_d_experiment_uuids)]
    df = df.dropna(subset=['mdfp']) 

    # Extract the 'mdfp' key from the dictionary
    df['mdfp_vec'] = df['mdfp'].apply(lambda val: val['mdfp'] if val and 'mdfp' in val else None)
    mdfp_vecs = df['mdfp_vec'].tolist()
    #json loads to actual lsits
    mdfp_vecs_lists = [json.loads(mdfp_vec) for mdfp_vec in mdfp_vecs]
    # Define MDFP feature names
    mdfp_features_all = ['NumHeavyAtoms', 'NumRotatableBonds', 'NumN', 'NumO', 'NumF', 'NumP', 'NumS', 'NumCl', 'NumBr', 'NumI',
                         'water_intra_crf_mean', 'water_intra_crf_std', 'water_intra_crf_median', 'water_intra_lj_mean',
                         'water_intra_lj_std', 'water_intra_lj_median', 'water_total_crf_mean', 'water_total_crf_std',
                         'water_total_crf_median', 'water_total_lj_mean', 'water_total_lj_std', 'water_total_lj_median',
                         'water_intra_ene_mean', 'water_intra_ene_std', 'water_intra_ene_median', 'water_total_ene_mean',
                         'water_total_ene_std', 'water_total_ene_median', 'water_rgyr_mean', 'water_rgyr_std',
                         'water_rgyr_median', 'water_sasa_mean', 'water_sasa_std', 'water_sasa_median']

    for i, mdfp_feature in enumerate(mdfp_features_all):
        df[mdfp_feature] = [mdfp_vec[i] if mdfp_vec else None for mdfp_vec in mdfp_vecs_lists]

    # Ensure molregno is unique in df_1
    # df_1 = df_1.drop(columns=[col for col in df_1.columns if col in df.columns], errors='ignore')

    # Merge with the original DataFrame, on molregno, keeping only those that match, and no duplicate columns
    df_combined = pd.merge(df_1, df, on='molregno', how='left', suffixes=('', '_y'))

    return df_combined