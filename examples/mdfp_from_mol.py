import Parameteriser
from Simulator import SolutionSimulator
from Composer import SolutionComposer
import mdtraj as md
import pickle
import sys

hash_code = sys.argv[1]
smiles = sys.argv[2]
#better, get from dict!

rdk_pmd = Parameteriser.SolutionParameteriser.via_rdkit(smiles)
pickle.dump(rdk_pmd, open(f"topologies/{hash_code}.pickle", "wb"))
SolutionSimulator.via_openmm(rdk_pmd, file_name = hash_code, file_path = "trajectories/", 
                             platform = "CUDA", num_steps = 5000 * 500)
traj = md.load(f"trajectories/{hash_code}.h5")
mdfp = SolutionComposer.run(traj, rdk_pmd,smiles=smiles)
mdfp = list(mdfp.get_mdfp())
pickle.dump(mdfp, open(f"fingerprints/{hash_code}.pickle", "wb"))
print(mdfp)
