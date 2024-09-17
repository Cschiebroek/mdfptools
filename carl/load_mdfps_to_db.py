import Parameteriser as Parameteriser
from Composer import SolutionComposer
import mdtraj as md
import pickle
from rdkit import Chem
import psycopg2
import uuid
import json
from rdkit.Chem.rdmolfiles import MolFromMolFile

print('Connecting to database...')
hostname = 'lebanon'
dbname = 'cs_mdfps'
username = 'cschiebroek'
cn = psycopg2.connect(host=hostname, dbname=dbname, user=username)
cur = cn.cursor()

confid_list = [16786,
17270,
 17271,
 17273,
 17274,
 17276,
 17278,
 17277,
 17280,
 17281,
 17282,
 17288,
 17289,
 17290,
 17291,
 17292,
 17293,
 17295,
 17297,
 17298,
 17304,
 17306,
 17307,
 17311,
 17313,
 17314,
 17315,
 17317,
 17318,
 17320,
 17322,
 17326,
 17335,
 17336,
 17341,
 17342,
 17343,
 17345,
 17347]
Md_Experiment_uuid = '13d08336-fb79-4042-83ce-af906193ff20'
for confid in confid_list:
    print('Confid: ', confid)
    mol = MolFromMolFile(f'mols_3d/{confid}.mol')
    topo_filename = f"topologies/{Md_Experiment_uuid}/{confid}.pickle"
    rdk_pmd = pickle.load(open(topo_filename, "rb"))
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
