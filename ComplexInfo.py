"""A small class that serves as configuration and to store some interface info"""

from pyrosetta import *
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover


class ComplexInfo():
    """ Small class to congigure a protein complex for analysis and design"""

    def __init__(self, input_pdb: str, binder_chains: list, target_chains: list, name=None):
        self.input_pdb = input_pdb
        self.binder_chains = binder_chains
        self.target_chains = target_chains
        self.name = name
        scorefxn = get_fa_scorefxn()
        
        pose = pose_from_pdb(self.input_pdb)
        self.score = scorefxn(pose) 
        
        self.interface_analysis = None

    def run_interface_analysis(self) -> None:
        # init('-detect_disulf false')
        pose = pose_from_pdb(self.input_pdb)
        interface_str = f"{''.join(self.binder_chains)}_{''.join(self.target_chains)}"

        # Create InterfaceAnalyzerMover
        interface_analyzer = InterfaceAnalyzerMover()
        interface_analyzer.set_pack_separated(True)
        interface_analyzer.set_interface(interface_str)
        interface_analyzer.apply(pose)
        self.interface_analysis = interface_analyzer

    def get_interface_stats(self) -> dict:
        assert self.interface_analysis != None, "No interface analysis found"

        return {
            'dG':           round(self.interface_analysis.get_interface_dG(), 2),
            'area':         round(self.interface_analysis.get_interface_delta_sasa(),2),
            'num_res':      self.interface_analysis.get_num_interface_residues(),
            'unsats':       self.interface_analysis.get_interface_delta_hbond_unsat(),
            'pymol_unsats': self.interface_analysis.get_pymol_sel_hbond_unsat(),
            'hbond_E':      round(self.interface_analysis.get_total_Hbond_E(),2),
            'residue_set':  self.interface_analysis.get_interface_set(),
        }
    
    def get_pose(self):
        return pose_from_file(self.input_pdb)