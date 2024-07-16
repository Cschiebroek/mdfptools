#mdfptools
from Parameteriser import LiquidParameteriser
from Simulator import LiquidSimulator
from Composer import LiquidComposer

#rdkit
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdmolfiles import MolFromMolFile

#database
import psycopg2
import uuid
import json

#other
import mdtraj as md
import pickle
import sys


confid = sys.argv[1]
Md_Experiment_uuid = sys.argv[2]


print('Confid: ', confid)
print('Md_Experiment_uuid: ', Md_Experiment_uuid)
mol = MolFromMolFile(f'mols_3d/{confid}.mol')
smiles = Chem.MolToSmiles(mol)
print('Molobject created, parameterising...')
rdk_pmd = LiquidParameteriser.via_rdkit(smiles = smiles)

topo_filename = f"topologies/{Md_Experiment_uuid}/{confid}.pickle"
pickle.dump(rdk_pmd, open(topo_filename, "wb"))
#load topology
rdk_pmd = pickle.load(open(topo_filename, "rb"))
print('Topology saved, simulating...')
traj_path = f"trajectories/{Md_Experiment_uuid}"
LiquidSimulator.via_openmm(rdk_pmd, file_name = confid, file_path = traj_path,
                             platform = "CUDA", num_steps = 5000 * 500)
print('Simulation finished, composing mdfp...')
traj = md.load(f"trajectories/{Md_Experiment_uuid}/{confid}.h5")
smiles = Chem.MolToSmiles(mol)
mdfp = LiquidComposer.run(traj, rdk_pmd,smiles=smiles)
mdfp = mdfp.get_mdfp()
mdfp_dict = {'mdfp':str(mdfp)}
print(mdfp)
print('Mdfp composed, saving to database...')
Mdfp_Experiment_uuid = Md_Experiment_uuid
mdfp_conf_uuid = uuid.uuid4()

print('Connecting to database...')
hostname = 'scotland'
dbname = 'cs_mdfps'
username = 'cschiebroek'
cn = psycopg2.connect(host=hostname, dbname=dbname, user=username)
cur = cn.cursor()
#retrieve conformer from database by confid
cur.execute("insert into cs_mdfps_schema.mdfp_experiment_data values (%s, %s, %s,%s,%s)",(str(confid), str(Mdfp_Experiment_uuid), json.dumps(mdfp_dict),str(mdfp_conf_uuid),Md_Experiment_uuid))
print(mdfp)
cn.commit()
print('Data saved, closing connection...')
cn.close()
print('Connection closed, exiting...')
