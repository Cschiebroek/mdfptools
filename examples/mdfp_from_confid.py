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
Md_Experiment_uuid = sys.argv[1]
if not os.path.exists(f'topologies/{Md_Experiment_uuid}'):
    os.makedirs(f'topologies/{Md_Experiment_uuid}')
if not os.path.exists(f'trajectories/{Md_Experiment_uuid}'):
    os.makedirs(f'trajectories/{Md_Experiment_uuid}')
if not os.path.exists(f'fingerprints/{Md_Experiment_uuid}'):
    os.makedirs(f'fingerprints/{Md_Experiment_uuid}')

hostname = 'scotland'
dbname = 'cs_mdfps'
username = 'cschiebroek'
cn = psycopg2.connect(host=hostname, dbname=dbname, user=username)
cur = cn.cursor()
#retrieve conformer from database by confid
confid = sys.argv[1]
cur.execute(f"SELECT * FROM conformers WHERE conf_id = {confid}")
d = cur.fetchall()
mol = Chem.MolFromMolBlock(d[0][3])
Md_Experiment_uuid = sys.argv[1]
rdk_pmd = Parameteriser.SolutionParameteriser.via_rdkit(mol = mol)
pickle.dump(rdk_pmd, open(f"topologies/{Md_Experiment_uuid}/{confid}.pickle", "wb"))
exit()
SolutionSimulator.via_openmm(rdk_pmd, file_name = confid, file_path = f"trajectories/{Md_Experiment_uuid}", 
                             platform = "CUDA", num_steps = 5000 * 500)
traj = md.load(f"trajectories/{Md_Experiment_uuid}/{confid}.h5")
smiles = Chem.MolToSmiles(mol)
mdfp = SolutionComposer.run(traj, rdk_pmd,smiles=smiles)
# mdfp = list(mdfp.get_mdfp())
pickle.dump(mdfp, open(f"fingerprints/{Md_Experiment_uuid}/{confid}.pickle", "wb"))
Mdfp_Experiment_uuid = Md_Experiment_uuid
curs.execute("insert into cs_mdfps_schema.mdfp_experiment_metadata values (%s, %s, %s)",(str(confid), str(Mdfp_Experiment_uuid), str(mdfp)))
print(mdfp)
