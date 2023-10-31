from Extractor import WaterExtractor
import mdtraj as md
import pickle
from scipy.stats import ks_2samp
from tqdm import tqdm
import numpy as np
import sys
#get first argument
molregno = int(sys.argv[1])

with open("molregno_to_confid_one.pickle","rb") as f:
    molregno_to_confid_one = pickle.load(f)
with open("molregno_to_confid_many.pickle","rb") as f:
    molregno_to_confid_many = pickle.load(f)
    
def get_distributions(confid):
    try:
        traj = md.load(f"trajectories/fc57851e-b654-4338-bcdd-faa28ec66253/{confid}.h5")
        rdk_pmd = pickle.load(open(f"topologies/fc57851e-b654-4338-bcdd-faa28ec66253/{confid}.pickle", "rb"))
    except OSError:
        traj = md.load(f"trajectories/e0f120fb-efa9-4c88-a964-e7b99253027c/{confid}.h5")
        rdk_pmd = pickle.load(open(f"topologies/e0f120fb-efa9-4c88-a964-e7b99253027c/{confid}.pickle", "rb"))
        
    energie_dict = WaterExtractor.extract_energies(mdtraj_obj=traj,parmed_obj=rdk_pmd)
    rgyr_dict = WaterExtractor.extract_rgyr(mdtraj_obj=traj,parmed_obj=rdk_pmd)
    sasa_dict = WaterExtractor.extract_sasa(mdtraj_obj=traj,parmed_obj=rdk_pmd)
    water_rgyr = rgyr_dict["water_rgyr"]
    water_sasa = sasa_dict["water_sasa"]
    water_sasa = [item for sublist in water_sasa for item in sublist]
    water_intra_crf = energie_dict["water_intra_crf"]
    water_intra_lj = energie_dict["water_intra_lj"]
    water_total_crf = energie_dict["water_total_crf"]
    water_total_lj = energie_dict["water_total_lj"]
    water_intra_ene = energie_dict["water_intra_ene"]
    water_total_ene = energie_dict["water_total_ene"]

    return [water_rgyr,water_sasa,water_intra_crf,water_intra_lj,water_total_crf,water_total_lj,water_intra_ene,water_total_ene]
def get_k2s_stats(nested_list1,nested_list2):
    stats = []
    for l1,l2 in zip(nested_list1,nested_list2):
        l1 = np.array(l1)
        l2 = np.array(l2)
        stat = ks_2samp(l1,l2)
        stats.append(stat)
    return stats

ref_conf = molregno_to_confid_one[molregno]
test_confs = molregno_to_confid_many[molregno]
nested_list_ref = get_distributions(ref_conf)
stats_list = []
for conf in test_confs:
    nested_list_test = get_distributions(conf)
    stats = get_k2s_stats(nested_list_ref,nested_list_test)
    stats_list.append(stats)
    
with open(f"stats_{molregno}.pickle","wb") as f:
    pickle.dump(stats_list,f)