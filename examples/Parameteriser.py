import tempfile
from functools import partialmethod


from simtk import unit  # Unit handling for OpenMM
from openmm import *
from openmm import app
from openmm.app import *
from openmm.app import PDBFile


from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField

import parmed
from rdkit import Chem
import pickle
import shutil
import os
import numpy as np


from utils import approximate_volume_by_density

##############################################################


class BaseParameteriser():
    """

    .. warning :: The base class should not be used directly
    """
    na_ion_pmd = None
    cl_ion_pmd = None
    system_pmd = None


    @classmethod
    def via_rdkit(cls):
        """
        Abstract method
        """
        raise NotImplementedError

    @classmethod
    def pmd_generator(cls):
        """
        Abstract method
        """
        raise NotImplementedError

    @classmethod
    def _rdkit_setter(cls, smiles, seed = 0xf00d,**kwargs):
        """
        Prepares an rdkit molecule with 3D coordinates.

        Parameters
        ------------
        smiles : str
            SMILES string of the solute moleulce

        Returns
        ---------
        mol : rdkit.Chem.Mol
        """
        from rdkit.Chem import AllChem
        from rdkit.Chem import rdmolops


        seed = 0xf00d
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, enforceChirality=True, randomSeed=seed)
        rdmolops.AssignStereochemistryFrom3D(mol)
        used_smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
        mol.SetProp("_Name", used_smiles)
        mol.UpdatePropertyCache(strict=False)

        # print(Chem.MolToMolBlock(mol),file=open(f'{hash_code}_{seed}.mol','w+')) #TODO save when you know how to name it


        return mol



    @classmethod
    def _get_forcefield(cls, **kwargs):
        if "ff_path" in kwargs:
            try:
                return ForceField(kwargs['ff_path'], allow_cosmetic_attributes=True)
            except Exception as e:
                print("Specified forcefield cannot be found. Fallback to default forcefield: OpenFF 2.1.0 force field release (Sage)")
        return ForceField("openff_unconstrained-2.1.0.offxml")

    @classmethod
    def _rdkit_parameteriser(cls, mol, **kwargs):
        # from openforcefield.utils.toolkits import RDKitToolkitWrapper, ToolkitRegistry
        """
        Creates a parameterised system from rdkit molecule

        Parameters
        ----------
        mol : rdkit.Chem.Mol
        """
        try:
            molecule = Molecule.from_rdkit(mol, allow_undefined_stereo=cls.allow_undefined_stereo)
            from openff.toolkit.utils import AmberToolsToolkitWrapper
            molecule.assign_partial_charges(toolkit_registry=AmberToolsToolkitWrapper(),partial_charge_method="am1bcc")

        except Exception as e:
            raise ValueError("Charging Failed : {}".format(e))  # TODO

        return cls._off_handler(molecule, **kwargs)

    @classmethod
    def _off_handler(cls, molecule, **kwargs):
        forcefield = cls._get_forcefield(**kwargs)
        topology = Topology.from_molecules(molecule)
        openmm_system = forcefield.create_openmm_system(topology, charge_from_molecules=[molecule])

     
        tmp_dir = tempfile.mkdtemp()
        # We need all molecules as both pdb files (as packmol input)
        # and mdtraj.Trajectory for restoring bonds later.
        pdb_filename = tempfile.mktemp(suffix=".pdb", dir=tmp_dir)




        molecule.to_file(pdb_filename, "pdb")
        omm_top = PDBFile(pdb_filename).topology
        ligand_pmd = parmed.openmm.topsystem.load_topology(omm_top, openmm_system, molecule._conformers[0])

        return pdb_filename, ligand_pmd


    @classmethod
    def save(cls, file_name, file_path="./", **kwargs):
        """
        Save to file the parameterised system.

        Parameters
        ------------
        file_name : str
            No file type postfix is necessary
        file_path : str
            Default to current directory

        Returns
        --------
        path : str
            The absolute path where the trajectory is written to.
        """
        path = '{}/{}.pickle'.format(file_path, file_name)
        pickle_out = open(path, "wb")
        pickle.dump(cls.system_pmd, pickle_out)
        pickle_out.close()

        return os.path.abspath(path)




class SolutionParameteriser(BaseParameteriser):
    """
    Parameterisation of solution box, i.e. one copy of solute molecule surronded by water.

    Parameters
    --------------
    solvent_pmd : parmed.structure
        Parameterised tip3p water as parmed object
    """

    solvent_pmd = None


    @classmethod
    def run(cls, smiles=None, mol=None,seed = 0xf00d,*, solvent_smiles=None, allow_undefined_stereo=False, num_solvent=100, density=None, default_padding=1.25*unit.nanometer, box_scaleup_factor=1.5, **kwargs):
        """
        Parameterisation perfromed via rdkit.

        Parameters
        --------------------
        smiles : str
            SMILES string of the solute molecule
        solvent_smiles : str
            SMILES string of the solvent molecule, default is None, only relevant if the solute is not water.
        allow_undefined_stereo : bool
            Flag passed to OpenForceField `Molecule` object during parameterisation. When set to False an error is returned if SMILES have no/ambiguous steroechemistry. Default to False here as a sanity check for user.
        num_solvent : int
            The number of solvent molecules added into the system, only relevant if the solvent is not water. The default value is 100, but it is left for the user to determine the number of solvent molecule really needed to surrond the solute and create a big enough system.
        density : simtk.unit.quantity.Quantity
            Density of the solvent, default is None, only relevant if the solvent is not water
        default_padding : simtk.unit
            Dictates amount of water surronding the solute. Default is 1.25 nanometers, only relevant if water is the solvent.
        box_scaleup_factor : float
            Dicatates the packed volume with respect to the volume estimated from density. Default is 1.5, only relevant if the solvent is not water


        Returns
        ------------------
        system_pmd : parmed.structure
            The parameterised system as parmed object
        """
        # TODO currently only supports one solute molecule
        # sanity checks

        if smiles is None and mol is None:
            raise ValueError("smiles or mol must be provided")
        if smiles is not None and mol is not None:
            raise ValueError("smiles and mol cannot be both provided")
            
        cls.allow_undefined_stereo = allow_undefined_stereo
        cls.default_padding = default_padding.value_in_unit(unit.nanometer)
        cls.solvent_smiles = solvent_smiles
        cls.box_scaleup_factor = box_scaleup_factor
        if solvent_smiles is not None and density is None:
            raise ValueError("Density missing for the solvent {}".format(solvent_smiles))
        if density is not None:
            if type(density) is not unit.quantity.Quantity:
                raise ValueError("density needs to have unit")
            if solvent_smiles is None:
                raise ValueError("Solvent SMILES missing.")


        if mol is None and smiles is not None:
            mol = cls._rdkit_setter(smiles,seed, **kwargs)
        if mol is not None and smiles is None:
            smiles = Chem.MolToSmiles(mol)
        cls.smiles = smiles

        # mol = cls._rdkit_charger(mol)
        cls.pdb_filename, cls.ligand_pmd = cls._rdkit_parameteriser(mol, **kwargs)
        if solvent_smiles:
            mol = cls._rdkit_setter(solvent_smiles,seed, **kwargs)
            cls.solvent_pdb_filename, cls.solvent_pmd = cls._rdkit_parameteriser(mol, **kwargs)


        if cls.solvent_pmd is None:
            try:
                cls.solvent_pmd = parmed.load_file("tip3p.prmtop") #TODO get this path instead of giving absolute path


            except ValueError:
                raise ValueError("Water file cannot be located")
        if solvent_smiles is None:
            cls._via_helper_water(**kwargs)
        else:
            cls._via_helper_other_solvent(density, num_solvent, **kwargs)

        return cls._add_counter_charges(**kwargs)

    @classmethod
    def _via_helper_other_solvent(cls, density, num_solvent, **kwargs):
        from openmoltools import packmol
        density = density.value_in_unit(unit.gram / unit.milliliter)

        
        box_size = approximate_volume_by_density([cls.smiles, cls.solvent_smiles], [
                                                     1, num_solvent], density=density, 		box_scaleup_factor=cls.box_scaleup_factor, box_buffer=cls.default_padding)

        packmol_out = packmol.pack_box([cls.pdb_filename, cls.solvent_pdb_filename],
                                       [1, num_solvent], box_size=box_size)

        cls.system_pmd = cls.ligand_pmd + (cls.solvent_pmd * num_solvent)
        cls.system_pmd.positions = packmol_out.openmm_positions(0)
        cls.system_pmd.box_vectors = packmol_out.openmm_boxes(0)
        try:
            # TODO should maybe delete the higher parent level? i.e. -2?
            shutil.rmtree("/".join(cls.pdb_filename.split("/")[:-1]))
            shutil.rmtree("/".join(cls.solvent_pdb_filename.split("/")[:-1]))
            del cls.ligand_pmd, cls.solvent_pmd
        except Exception as e:
            print("Error due to : {}".format(e))

        cls.system_pmd.title = cls.smiles
        return cls.system_pmd

    @classmethod
    def _via_helper_water(cls, **kwargs):
        """
        Helper function for via_rdkit

        Returns
        ------------------
        system_pmd : parmed.structure
            The parameterised system as parmed object
        """
        from pdbfixer import PDBFixer  # for solvating

        fixer = PDBFixer(cls.pdb_filename)
        if "padding" not in kwargs:
            fixer.addSolvent(padding=cls.default_padding)
        else:
            fixer.addSolvent(padding=float(kwargs["padding"]))

        tmp_dir = tempfile.mkdtemp()
        cls.pdb_filename = tempfile.mktemp(suffix=".pdb", dir=tmp_dir)
        with open(cls.pdb_filename, "w") as f:
            PDBFile.writeFile(fixer.topology, fixer.positions, f)
        complex = parmed.load_file(cls.pdb_filename)

        solvent = complex["(:HOH)"]
        num_solvent = len(solvent.residues)

        solvent_pmd = cls.solvent_pmd * num_solvent
        solvent_pmd.positions = solvent.positions

        cls.system_pmd = cls.ligand_pmd + solvent_pmd
        cls.system_pmd.box_vectors = complex.box_vectors

        try:
            shutil.rmtree("/".join(cls.pdb_filename.split("/")[:-1]))
            del cls.ligand_pmd
        except:
            pass

        cls.system_pmd.title = cls.smiles
        return cls.system_pmd

    @classmethod
    def _add_counter_charges(cls, **kwargs):
        """in case the solute molecule has a net charge, 
        counter charge are added to the system in the form of ions,
        Na+ or Cl-, in order to keep charge neutrality.
        """

        solute_charge = int(Chem.GetFormalCharge(Chem.MolFromSmiles(cls.smiles)))
        if solute_charge == 0:
            return cls.system_pmd

        if solute_charge > 0:  # add -ve charge
            if cls.cl_ion_pmd is None:
                cls.cl_ion_pmd = parmed.load_file("../mdfptools/data/cl.prmtop")
            ion_pmd = cls.cl_ion_pmd * solute_charge

        elif solute_charge < 0:  # add +ve charge
            if cls.na_ion_pmd is None:
                cls.na_ion_pmd = parmed.load_file("../mdfptools/data/na.prmtop")
            ion_pmd = cls.na_ion_pmd * abs(solute_charge)

        # replace the last few solvent molecules and replace them by the ions
        ion_pmd.coordinates = np.array([np.mean(cls.system_pmd[":{}".format(
            len(cls.system_pmd.residues) - i)].coordinates, axis=0) for i in range(abs(solute_charge))])
        cls.system_pmd = cls.system_pmd[":1-{}".format(len(cls.system_pmd.residues) - abs(solute_charge))]
        cls.system_pmd += ion_pmd

        return cls.system_pmd

    via_rdkit = partialmethod(run)



