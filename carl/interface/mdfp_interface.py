#mdfptools
#add .. to path
import sys
sys.path.append("..")
from Simulator import InterfaceSimulator
import pickle
from simtk import unit



confid = 2412
tmp = 600
t_ns = 5
file_name = f'{confid}_{tmp}K_{t_ns}ns'

topo_filename = f"topologies/24e3946b-fb2c-47bf-9965-1682bb0d63c9/{confid}.pickle"
rdk_pmd = pickle.load(open(topo_filename, "rb"))
print('Topology saved, simulating...')
traj_path = f"trajectories/"

InterfaceSimulator.via_openmm(rdk_pmd, file_name = file_name, file_path = traj_path,
                             platform = "CUDA", num_steps = 5000 * 100*t_ns,T_kelvin=tmp,equil_steps=5000)

