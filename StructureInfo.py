"""
Definition of the StructureInfo class
"""

import os

from pyrosetta import *
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.core.select.residue_selector import ChainSelector, OrResidueSelector

from utils import *
from mpnn import *


class StructureInfo:
    """
    A class to handle binder and target structure information from a given PDB file or Pose object.

    Attributes:
        pose (Pose): The pose representation of the structure.
        binder_chains (list): List of chain IDs representing the binder.
        target_chains (list): List of chain IDs representing the target.
        interface_analyzer (InterfaceAnalyzerMover): InterfaceAnalyzerMover object to analyze the pose.
    """

    def __init__(self, input_structure: str, binder_chains: list, target_chains: list, name: str=None):
        self.pdb_file = name 
        self.pose = self._load_pose(input_structure)
        self.interface_str = ''.join(binder_chains) + "_" + ''.join(target_chains)
        self.interface_analyzer = InterfaceAnalyzerMover()
        self.interface_analyzer.set_pack_separated(True)
        self.interface_analyzer.set_interface(self.interface_str)

        self._binder_chains = binder_chains
        self._target_chains = target_chains
        self.chain_ids = self._get_chain_ids()
        self.chain_lengths = self.get_chain_lengths()
        
        self._mpnn_pdb_input_path = None
        print("Select interface residues for ProteinMPNN")
        if len(binder_chains) > 0 and len(target_chains) > 0:
            _, _, self.binder_if_set, _  = select_interface_residues(self.pose, self.binder_chains, self.target_chains)
        else:
            self.binder_if_set = rosetta.utility.vector1_bool() 
        self._probs_per_chain = None
        
        self._design_args = {
            'save_probs': 1,
            'pdb_path': self._mpnn_pdb_input_path,
            'sampling_temp': "0.1",
            'seed': 42,
            'suppress_print': 1,
        }

    def _load_pose(self, input_structure: str):
        """
        Load the pose from a PDB file path or use an existing Pose object.
        """
        if isinstance(input_structure, str):
            pose = pose_from_pdb(input_structure)
            self.pdb_file = os.path.basename(pose.pdb_info().name())
            return pose
        elif isinstance(input_structure, Pose):
            assert self.pdb_file != None, "When initializing with a pose, you have to define a name for a potential output pdb file."
            if self.pdb_file[-4:] != '.pdb':
                self.pdb_file += '.pdb'
            pose = input_structure.clone()
            # Add proper pdb_info() object to pose
            if not pose.pdb_info():
                pose.pdb_info(rosetta.core.pose.PDBInfo(pose))
            pose.pdb_info().name(self.pdb_file)
            return pose
        else:
            raise ValueError("input_structure must be either a file path (string) or a Pose object")
        

    def _get_chain_ids(self) -> list:
        """Extract chain IDs in the order they appear in the PDB"""
        chain_ids = []
        for i in range(1, self.pose.total_residue() + 1):
            chain_letter = self.pose.pdb_info().chain(i)
            if chain_letter not in chain_ids:
                chain_ids.append(chain_letter)
        return chain_ids

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, value):
        if isinstance(value, Pose):
            self._pose = value.clone()  # Store a clone to avoid direct modification
        else:
            raise ValueError("pose must be a Pose object")

    @property
    def binder_chains(self):
        return self._binder_chains

    @binder_chains.setter
    def binder_chains(self, value):
        if isinstance(value, list) and all(isinstance(chain, str) for chain in value):
            self._binder_chains = value
        else:
            raise ValueError("binder_chains must be a list of strings")

    @property
    def target_chains(self):
        return self._target_chains

    @target_chains.setter
    def target_chains(self, value):
        if isinstance(value, list) and all(isinstance(chain, str) for chain in value):
            self._target_chains = value
        else:
            raise ValueError("target_chains must be a list of strings")

    @property
    def mpnn_pdb_input_path(self):
        return self._mpnn_pdb_input_path

    @mpnn_pdb_input_path.setter
    def mpnn_pdb_input_path(self, value):
        if isinstance(value, str):
            self._mpnn_pdb_input_path = value
            print("Update design args")
            self.design_args['pdb_path'] = self._mpnn_pdb_input_path
        else:
            raise ValueError("mpnn_pdb_input_path must be a string")

    @property
    def design_args(self):
        return self._design_args

    @design_args.setter
    def design_args(self, value):
        if isinstance(value, dict):
            self._design_args = value
        else:
            raise ValueError("design_args must be a dictionary")

    @property
    def binder_if_set(self):
        return self._binder_if_set

    @binder_if_set.setter
    def binder_if_set(self, value):
        if isinstance(value, rosetta.utility.vector1_bool):
            self._binder_if_set = value
        else:
            raise ValueError("binder_if_set must be a vector1_bool")

    @property
    def probs_per_chain(self):
        return self._probs_per_chain
    
    @probs_per_chain.setter
    def probs_per_chain(self, value):
        if isinstance(value, dict):
            self._probs_per_chain = value
        else:
            raise ValueError("probs_per_chain must be of type np.array")

    def get_chain_lengths(self):
        # Initialize a dictionary to store chain lengths
        chain_lengths = {}

        # Iterate over all residues to calculate chain lengths
        for i in range(1, self.pose.total_residue() + 1):  # PyRosetta is 1-indexed
            chain_id = self.pose.pdb_info().chain(i)
            
            if chain_id not in chain_lengths:
                chain_lengths[chain_id] = 0  # Initialize chain length if not present
            
            chain_lengths[chain_id] += 1  # Increment the count for this chain
        return chain_lengths


    def run_interface_analysis(self):
        """Apply interface analysis"""
        self.interface_analyzer.apply(self.pose)


    def setup_mpnn(self):
        """Create input files for ProteinMPNN"""
        chains_to_design = ' '.join(self.chain_ids)

        # Check if the binder interface residue set was already set
        # This might be desireable for customized design.
        # Otherwise use automatic detection.
        print("Setup Selectors")
        
        # Construct residue selector for the binder
        binder_selectors = [ChainSelector(chain) for chain in self.binder_chains]

        if len(binder_selectors) > 1:
            combined_binder_selector = OrResidueSelector(*binder_selectors)
        else:
            combined_binder_selector = binder_selectors[0]

        # Fix all positions that are in the binder but not part of the binding interface    
        print("Fix all positions that are in the binder but not part of the binding interface")
        fixed_positions = [str(i+1) for i,v in enumerate(combined_binder_selector.apply(self.pose)) if not self.binder_if_set[i+1]]   
        
        # Create input folder and file for MPNN
        # input_path = f"inputs_pdbs/{self.pdb_file[:-4]}"
        # os.makedirs(input_path, exist_ok=True)
        # pdb_input_path = f"{input_path}/pose_{self.pdb_file}" 
        # self.pose.dump_pdb(pdb_input_path)
        # print(pdb_input_path, 'created.')

        # Prepare json input files for MPNN
        print("Prepare mpnn files")
        pdb_input_path = prepare_mpnn(self.pose, chains_to_design, fixed_positions)
        
        print("Set pdb input to", pdb_input_path)
        self.mpnn_pdb_input_path = pdb_input_path

    def run_mpnn(self):
        if self.mpnn_pdb_input_path == None:
            print("Setup MPNN")
            self.setup_mpnn()
        self.probs_per_chain = run_mpnn(self._design_args, self.chain_lengths)