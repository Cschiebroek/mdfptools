import Parameteriser
from Simulator import SolutionSimulator
from Composer import SolutionComposer
import mdtraj as md
import pickle
import sys
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import rdmolops
import psycopg2
import os
import uuid
import json
import numpy as np
confid = sys.argv[1]
Md_Experiment_uuid = sys.argv[2]

print('Confid: ', confid)
print('Md_Experiment_uuid: ', Md_Experiment_uuid)


print('Connecting to database...')
hostname = 'scotland'
dbname = 'cs_mdfps'
username = 'cschiebroek'
cn = psycopg2.connect(host=hostname, dbname=dbname, user=username)
cur = cn.cursor()
#retrieve conformer from database by confid
cur.execute("SELECT * FROM conformers WHERE conf_id = %s", (confid,))
d = cur.fetchall()
print('Data fetched, creating rdkit molobject...')
mol = Chem.MolFromMolBlock(d[0][3])
print('Molobject created, parameterising...')
rdk_pmd = Parameteriser.SolutionParameteriser.via_rdkit(mol = mol)
topo_filename = f"topologies/{Md_Experiment_uuid}/{confid}.pickle"
pickle.dump(rdk_pmd, open(topo_filename, "wb"))

print('Topology saved, simulating...')
traj_path = f"trajectories/{Md_Experiment_uuid}"
SolutionSimulator.via_openmm(rdk_pmd, file_name = confid, file_path = traj_path, 
                             platform = "CUDA", num_steps = 5000 * 500)
print('Simulation finished, composing mdfp...')
traj = md.load(f"trajectories/{Md_Experiment_uuid}/{confid}.h5")
smiles = Chem.MolToSmiles(mol)
mdfp = SolutionComposer.run(traj, rdk_pmd,smiles=smiles)
mdfp = mdfp.get_mdfp()
mdfp_dict = {'mdfp':str(mdfp)}
print('Mdfp composed, saving to database...')
Mdfp_Experiment_uuid = Md_Experiment_uuid
mdfp_conf_uuid = uuid.uuid4()
cur.execute("insert into cs_mdfps_schema.mdfp_experiment_data values (%s, %s, %s,%s,%s)",(str(confid), str(Mdfp_Experiment_uuid), json.dumps(mdfp_dict),str(mdfp_conf_uuid),Md_Experiment_uuid))
print(mdfp)
cn.commit()
print('Data saved, closing connection...')
cn.close()
print('Connection closed, exiting...')
