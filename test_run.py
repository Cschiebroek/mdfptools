from carl.liquid_phase_experiments import Parameteriser
from carl.Simulator import SolutionSimulator

smiles = 'c1ccccc1'
rdk_pmd = Parameteriser.LiquidParameteriser.via_rdkit(smiles = smiles)

print('Topology saved, simulating...')
traj_path = f"./"
confid = 'test'
SolutionSimulator.via_openmm(rdk_pmd, file_name = confid, file_path = traj_path,
                             platform = "CUDA", num_steps = 5000 * 500)