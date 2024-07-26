import pandas as pd

df = pd.read_csv('/localhome/cschiebroek/MDFP_VP/mdfptools/carl/data_curation/cs_mdfps_schema_experimental_data.csv')
#keep only vp < 5
df = df[df['vp_log10_pa'] < 5]
print(len(df))
confids = df['conf_id'].tolist()
with open('gas_phase.txt','w') as file:
    for confid in confids:
        command = f'{confid} 5166be97-ef21-4cc5-bee1-719c7b9e3397'
        file.write(command + '\n')
file.close()