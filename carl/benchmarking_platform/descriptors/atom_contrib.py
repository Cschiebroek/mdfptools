import pickle
from copy import deepcopy
import pandas as pd
import logging 

EMPTY_DICT = {'[CH4]': 0,
 '[CH3]C': 0,
 '[CH2](C)C': 0,
 '[CH](C)(C)C': 0,
 '[C](C)(C)(C)C': 0,
 '[CH3][N,O,P,S,F,Cl,Br,I]': 0,
 '[CH2X4]([N,O,P,S,F,Cl,Br,I])[A;!#1]': 0,
 '[CH1X4]([N,O,P,S,F,Cl,Br,I])([A;!#1])[A;!#1]': 0,
 '[CH0X4]([N,O,P,S,F,Cl,Br,I])([A;!#1])([A;!#1])[A;!#1]': 0,
 '[C]=[!C;A;!#1]': 0,
 '[CH2]=C': 0,
 '[CH1](=C)[A;!#1]': 0,
 '[CH0](=C)([A;!#1])[A;!#1]': 0,
 '[C](=C)=C': 0,
 '[CX2]#[A;!#1]': 0,
 '[CH3]c': 0,
 '[CH3]a': 0,
 '[CH2X4]a': 0,
 '[CHX4]a': 0,
 '[CH0X4]a': 0,
 '[cH0]-[A;!C;!N;!O;!S;!F;!Cl;!Br;!I;!#1]': 0,
 '[c][#9]': 0,
 '[c][#17]': 0,
 '[c][#35]': 0,
 '[c][#53]': 0,
 '[cH]': 0,
 '[c](:a)(:a):a': 0,
 '[c](:a)(:a)-a': 0,
 '[c](:a)(:a)-C': 0,
 '[c](:a)(:a)-N': 0,
 '[c](:a)(:a)-O': 0,
 '[c](:a)(:a)-S': 0,
 '[c](:a)(:a)=[C,N,O]': 0,
 '[C](=C)(a)[A;!#1]': 0,
 '[C](=C)(c)a': 0,
 '[CH1](=C)a': 0,
 '[C]=c': 0,
 '[CX4][A;!C;!N;!O;!P;!S;!F;!Cl;!Br;!I;!#1]': 0,
 '[#6]': 0,
 '[#1][#6,#1]': 0,
 '[#1]O[CX4,c]': 0,
 '[#1]O[!C;!N;!O;!S]': 0,
 '[#1][!C;!N;!O]': 0,
 '[#1][#7]': 0,
 '[#1]O[#7]': 0,
 '[#1]OC=[#6,#7,O,S]': 0,
 '[#1]O[O,S]': 0,
 '[#1]': 0,
 '[NH2+0][A;!#1]': 0,
 '[NH+0]([A;!#1])[A;!#1]': 0,
 '[NH2+0]a': 0,
 '[NH1+0]([!#1;A,a])a': 0,
 '[NH+0]=[!#1;A,a]': 0,
 '[N+0](=[!#1;A,a])[!#1;A,a]': 0,
 '[N+0]([A;!#1])([A;!#1])[A;!#1]': 0,
 '[N+0](a)([!#1;A,a])[A;!#1]': 0,
 '[N+0](a)(a)a': 0,
 '[N+0]#[A;!#1]': 0,
 '[NH3,NH2,NH;+,+2,+3]': 0,
 '[n+0]': 0,
 '[n;+,+2,+3]': 0,
 '[NH0;+,+2,+3]([A;!#1])([A;!#1])([A;!#1])[A;!#1]': 0,
 '[NH0;+,+2,+3](=[A;!#1])([A;!#1])[!#1;A,a]': 0,
 '[NH0;+,+2,+3](=[#6])=[#7]': 0,
 '[N;+,+2,+3]#[A;!#1]': 0,
 '[N;-,-2,-3]': 0,
 '[N;+,+2,+3](=[N;-,-2,-3])=N': 0,
 '[#7]': 0,
 '[o]': 0,
 '[OH,OH2]': 0,
 '[O]([A;!#1])[A;!#1]': 0,
 '[O](a)[!#1;A,a]': 0,
 '[O]=[#7,#8]': 0,
 '[OX1;-,-2,-3][#7]': 0,
 '[OX1;-,-2,-2][#16]': 0,
 '[O;-0]=[#16;-0]': 0,
 '[O-]C(=O)': 0,
 '[OX1;-,-2,-3][!#1;!N;!S]': 0,
 '[O]=c': 0,
 '[O]=[CH]C': 0,
 '[O]=C(C)([A;!#1])': 0,
 '[O]=[CH][N,O]': 0,
 '[O]=[CH2]': 0,
 '[O]=[CX2]=O': 0,
 '[O]=[CH]c': 0,
 '[O]=C([C,c])[a;!#1]': 0,
 '[O]=C(c)[A;!#1]': 0,
 '[O]=C([!#1;!#6])[!#1;!#6]': 0,
 '[#8]': 0,
 '[#9-0]': 0,
 '[#17-0]': 0,
 '[#35-0]': 0,
 '[#53-0]': 0,
 '[#9,#17,#35,#53;-]': 0,
 '[#53;+,+2,+3]': 0,
 '[+;#3,#11,#19,#37,#55]': 0,
 '[#15]': 0,
 '[S;-,-2,-3,-4,+1,+2,+3,+5,+6]': 0,
 '[S-0]=[N,O,P,S]': 0,
 '[S;A]': 0,
 '[s;a]': 0,
 '[#3,#11,#19,#37,#55]': 0,
 '[#4,#12,#20,#38,#56]': 0,
 '[#5,#13,#31,#49,#81]': 0,
 '[#14,#32,#50,#82]': 0,
 '[#33,#51,#83]': 0,
 '[#34,#52,#84]': 0,
 '[#21,#22,#23,#24,#25,#26,#27,#28,#29,#30]': 0,
 '[#39,#40,#41,#42,#43,#44,#45,#46,#47,#48]': 0,
 '[#72,#73,#74,#75,#76,#77,#78,#79,#80]': 0}

ORDER = ['C', 'H', 'N', 'O', 'F', 'Cl', 'Br', 'I', 'Ha', 'P', 'S', 'Me']



with open('descriptors/atom_contribs.pkl', 'rb') as f:
    PATTS = pickle.load(f)

def get_Crippen_atom_contribs(mol):
    nAtoms = mol.GetNumAtoms()
    atomContribs = [(0.,0.)]*nAtoms
    doneAtoms=[0]*nAtoms
    nAtomsFound=0
    done = False
    match_dict = {}
    for cha in ORDER:
        pattVect = PATTS[cha]
        for sma,patt,logp,mr in pattVect:
            for match in mol.GetSubstructMatches(patt,False,False):
                firstIdx = match[0]
                if not doneAtoms[firstIdx]:
                    doneAtoms[firstIdx]=1
                    atomContribs[firstIdx] = (logp,mr)
                    match_dict[match[0]] = sma
                    nAtomsFound+=1
                    if nAtomsFound>=nAtoms:
                        done=True
                        break
            if done:
                break
    fp_dict = deepcopy(EMPTY_DICT)
    for key in match_dict.values():
        fp_dict[key] += 1
    fp_dict
    fingerprint_vec = list(fp_dict.values())
    assert sum(fingerprint_vec) == nAtoms, "Mismatch between atoms and fingerprint vector"
    return fingerprint_vec
from rdkit import Chem

#now add call to df; add columns to df. column name should be the smarts
def add_crippen_atom_counts_to_df(df):
    df['crippen_atoms'] = df['molblock'].apply(lambda x: get_Crippen_atom_contribs(Chem.MolFromMolBlock(x)))
    df = pd.concat([df.drop(['crippen_atoms'], axis=1), df['crippen_atoms'].apply(pd.Series)], axis=1)
    df = df.rename(columns=dict(zip(range(110), list(EMPTY_DICT.keys()))))
    print(len(EMPTY_DICT.keys()))
    return df

    
