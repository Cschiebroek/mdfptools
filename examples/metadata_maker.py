import sys
import openff.toolkit
import openmm
import git

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdDepictor
rdDepictor.SetPreferCoordGen(True)
import os
import json
import uuid

import rdkit
print(rdkit.__version__)

import lwreg
from lwreg import standardization_lib
from lwreg import utils
import getpass

# Metadata values
Md_Experiment_uuid = sys.argv[1]
ff_name = "openff_unconstrained-2.1.0.offxml"
ff_version = openff.toolkit.__version__
simulation_type = "tMD water solution"
md_engine = "openMM"
version = openmm.__version__
steps_time = 5.0
Git_repo_name = "mdfptools"
repo = git.Repo(search_parent_directories=True)
Git_commit_hash = repo.head.object.hexsha
#print all metadata
print('Metadata values:')
print('------------------------------------')
print("Md_Experiment_uuid: ", Md_Experiment_uuid)
print("ff_name: ", ff_name)
print("ff_version: ", ff_version)
print("simulation_type: ", simulation_type)
print("md_engine: ", md_engine)
print("version: ", version)
print("steps_time: ", steps_time)
print("Git_repo_name: ", Git_repo_name)
print("Git_commit_hash: ", Git_commit_hash)
print('------------------------------------\n')

# Save metadata to table mdfp_carl.MD_experiments_metadata 
#string symbol for %
percentage = "%"
#string symbol for 
print(f'Are you 100% sure you want to save this metadata to the database?\n')
pw = getpass.getpass()
config = lwreg.utils.defaultConfig()
# set the name of the database we'll work with:
config['dbtype'] = 'postgresql'
config['dbname'] = 'cs_mdfps'
config['host'] = 'scotland'
config['user'] = 'cschiebroek'
config['password'] = pw # password is saved in our .pgpass
# we don't want to standardize the molecules:
config['standardization'] = standardization_lib.RemoveHs()
# we want to store conformers
config['registerConformers'] = True
cn = utils._connect(config)
cur = cn.cursor()
# cur.execute('insert into cs_mdfps_schema.experimental_data values (%s , %s, %s, %s, %s)',(str(molregno),str(0),json.dumps({}),str(VP),json.dumps(metadata)))

 