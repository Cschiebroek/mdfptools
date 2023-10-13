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
molid = sys.argv[1]

from rdkit.Chem.rdmolfiles import MolFromMolFile
print('Internal Mol ID: ', molid)
mol = MolFromMolFile(f'mols_3d_biogen/{molid}.mol')
print('Molobject created, parameterising...')
rdk_pmd = Parameteriser.SolutionParameteriser.via_rdkit(mol = mol)
topo_filename = f"topologies/biogen/{molid}.pickle"
pickle.dump(rdk_pmd, open(topo_filename, "wb"))

print('Topology saved, simulating...')
traj_path = f"trajectories/biogen"
SolutionSimulator.via_openmm(rdk_pmd, file_name = molid, file_path = traj_path,
                             platform = "CUDA", num_steps = 5000 * 500)
print('Simulation finished, composing mdfp...')
traj = md.load(f"trajectories/biogen/{molid}.h5")
smiles = Chem.MolToSmiles(mol)
mdfp = SolutionComposer.run(traj, rdk_pmd,smiles=smiles)
mdfp = mdfp.get_mdfp()
#save mdfp to pickle
mdfp_filename = f"mdfps/biogen/{molid}.pickle"
pickle.dump(mdfp, open(mdfp_filename, "wb"))