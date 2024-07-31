#mdfptools
#add .. to path
# import sys
# sys.path.append("..")
from Simulator import InterfaceSimulator
import pickle
import sys

#pass argument
T = sys.argv[1]
confid = 2412
topo_filename = f"topologies/24e3946b-fb2c-47bf-9965-1682bb0d63c9/{confid}.pickle"
rdk_pmd = pickle.load(open(topo_filename, "rb"))
print('Topology saved, simulating...')
traj_path = f"trajectories/"
InterfaceSimulator.via_openmm(rdk_pmd, file_name = confid, file_path = traj_path,
                             platform = "CUDA", num_steps = 5000 * 500,T_kelvin=T)

