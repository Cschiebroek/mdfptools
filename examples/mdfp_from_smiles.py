import Parameteriser
from Simulator import SolutionSimulator
from Composer import SolutionComposer
import mdtraj as md
import pickle
import sys
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import rdmolops


hash_code = sys.argv[1]
input_smiles = sys.argv[2]

seed = 0xf00d
mol = Chem.MolFromSmiles(input_smiles, sanitize=False)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, enforceChirality=True, randomSeed=seed)
rdmolops.AssignStereochemistryFrom3D(mol)
used_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
mol.SetProp("_Name", used_smiles)
mol.UpdatePropertyCache(strict=False)

print(Chem.MolToMolBlock(mol),file=open(f'{hash_code}_{seed}.mol','w+'))

rdk_pmd = Parameteriser.SolutionParameteriser.via_rdkit(mol = mol)
pickle.dump(rdk_pmd, open(f"topologies/{hash_code}.pickle", "wb"))
SolutionSimulator.via_openmm(rdk_pmd, file_name = hash_code, file_path = "trajectories/", 
                             platform = "CUDA", num_steps = 5000 * 500)
traj = md.load(f"trajectories/{hash_code}.h5")
mdfp = SolutionComposer.run(traj, rdk_pmd,smiles=used_smiles)
mdfp = list(mdfp.get_mdfp())
pickle.dump(mdfp, open(f"fingerprints/{hash_code}.pickle", "wb"))
print(mdfp)
