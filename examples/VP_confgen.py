#! /bin/env python
from rdkit import Chem
import gzip
from rdkit.Chem import rdDistGeom
import pickle
import rdkit
import os
print(f'RDKit version: {rdkit.__version__}')


import time
numConfs=100
def generate_confs(data,whichJob,nJobs):
    etkdg = rdDistGeom.ETKDGv3()
    etkdg.randomSeed = 0xa700f
    etkdg.verbose = False
    etkdg.numThreads = 4
    etkdg.trackFailures = True
    etkdg.useRandomCoords = True
    etkdg.pruneRmsThresh = 0.1
    outfn = f'./confgen/11093a30-b6d0-4e3f-a22b-8dcad60d6a11_100.block{whichJob}.pkl.gz'
    if os.path.exists(outfn):
        with gzip.open(outfn) as inf:
            accum = pickle.load(inf)
    else:
        accum = []
    seen = [x[0] for x in accum]
    for i,(nm,smi) in enumerate(data):
        if i%nJobs != whichJob or nm in seen:
            continue
        m = Chem.MolFromSmiles(smi)
        if not m:
            tpl = (nm,None, 'parse fail')
        else:
            m = Chem.AddHs(m)
            t1 = time.time()
            try:
                rdDistGeom.EmbedMultipleConfs(m,numConfs = numConfs, params = etkdg)
            except:
                tpl = (nm, None, 'conformers fail')
            else:
                tpl = (nm,m,etkdg.GetFailureCounts(),time.time()-t1)
        accum.append(tpl)
        seen.append(nm)
        if not len(accum)%5:
            print(f"DONE {len(accum)}")
            with gzip.open(outfn,'wb+') as outf:
                pickle.dump(accum,outf)

    if len(accum):
        with gzip.open(outfn,'wb+') as outf:
            pickle.dump(accum,outf)

import click
import logging
import sys

@click.command()
@click.option('--num_jobs', default=500)
@click.argument('job_index', type=int)
def run_COD(num_jobs,job_index):
    infile = './VP_molregno_smiles.pkl.gz'
    with gzip.open(infile,'rb') as inf:
        ind =pickle.load(inf)
    print(len(ind))
    generate_confs(ind,job_index,num_jobs)


if __name__ == '__main__':
    run_COD()

