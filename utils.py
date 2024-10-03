# Standard library imports
from types import SimpleNamespace
import multiprocessing

# Third-party imports (BioPython and NumPy)
from Bio import PDB
from Bio.Data import IUPACData
import numpy as np

# PyRosetta imports
import pyrosetta
from pyrosetta import rosetta
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.core.pack.task import TaskFactory
from pyrosetta.rosetta.core.pack.task.operation import (
    RestrictToRepacking,
    OperateOnResidueSubset,
    PreventRepackingRLT,
    NoRepackDisulfides,
    IncludeCurrent,
    InitializeFromCommandline,
)
from pyrosetta.rosetta.core.select.movemap import MoveMapFactory, move_map_action
from pyrosetta.rosetta.core.select.residue_selector import (
    NotResidueSelector,
    ChainSelector,
    InterGroupInterfaceByVectorSelector,
    NeighborhoodResidueSelector,
    ResidueIndexSelector,
    AndResidueSelector,
    OrResidueSelector,
)

# Concurrency imports
from concurrent.futures import ProcessPoolExecutor, as_completed


def select_interface_residues(pose, receptor_chains, binder_chains):
    # Set up Selectors for the receptor chains and the binder chains
    receptor_selectors = [ChainSelector(chain) for chain in receptor_chains]
    binder_selectors = [ChainSelector(chain) for chain in binder_chains]

    # Combine receptor chains using OrResidueSelector if there's more than one
    combined_receptor_selector = receptor_selectors[0]
    if len(receptor_selectors) > 1:
        combined_receptor_selector = OrResidueSelector(*receptor_selectors)

    # Combine binder chains using OrResidueSelector if there's more than one
    combined_binder_selector = binder_selectors[0]
    if len(binder_selectors) > 1:
        combined_binder_selector = OrResidueSelector(*binder_selectors)

    # Residue positions for receptors and binders
    receptor_set = combined_receptor_selector.apply(pose)
    binder_set = combined_binder_selector.apply(pose)

    # Set up Interface Selector
    interface_selector = InterGroupInterfaceByVectorSelector()
    interface_selector.group1_selector(
        combined_receptor_selector
    )  # Set receptor as group 1
    interface_selector.group2_selector(
        combined_binder_selector
    )  # Set binder as group 2

    interface_set = interface_selector.apply(pose)

    # Interface Residue Selectors for receptor and binder
    receptor_if_selector = AndResidueSelector(
        interface_selector, combined_receptor_selector
    )
    binder_if_selector = AndResidueSelector(
        interface_selector, combined_binder_selector
    )

    receptor_if_set = receptor_if_selector.apply(pose)
    binder_if_set = binder_if_selector.apply(pose)

    # Convert selected residues to PDB format
    receptor_if_residues = [i for i, m in enumerate(receptor_if_set, 1) if m == 1]
    binder_if_residues = [i for i, m in enumerate(binder_if_set, 1) if m == 1]

    return receptor_if_residues, binder_if_residues, receptor_if_set, binder_if_set


def create_args(overrides=None):
    """
    Constructs and returns the arguments for the ProteinMPNN main function.
    Allows overriding default values with a dictionary of arguments.

    Parameters:
        overrides (dict, optional): Dictionary of arguments to override the defaults.

    Returns:
        SimpleNamespace: A namespace object with all arguments.
    """
    # Step 1: Define the default arguments
    args = SimpleNamespace(
        suppress_print=0,  # 0 for False, 1 for True
        ca_only=False,  # Parse CA-only structures and use CA-only models
        path_to_model_weights="",  # Path to model weights folder
        model_name="v_48_020",  # Model name (different versions)
        use_soluble_model=False,  # Use weights trained on soluble proteins only
        seed=0,  # Random seed (0 means random)
        save_score=0,  # Save score to npy files (0 for False, 1 for True)
        save_probs=0,  # Save predicted probabilities per position
        score_only=0,  # Score input backbone-sequence pairs (0 for False, 1 for True)
        path_to_fasta="",  # Input sequence in fasta format
        conditional_probs_only=0,  # Output conditional probabilities
        conditional_probs_only_backbone=0,  # Conditional probabilities given backbone
        unconditional_probs_only=0,  # Output unconditional probabilities in one forward pass
        backbone_noise=0.00,  # Standard deviation of Gaussian noise to add to backbone atoms
        num_seq_per_target=1,  # Number of sequences to generate per target
        batch_size=1,  # Batch size (higher for bigger GPUs)
        max_length=200000,  # Max sequence length
        sampling_temp="0.1",  # Sampling temperature for amino acids
        out_folder=".",  # Output folder for sequences
        pdb_path="",  # Path to a single PDB to be designed
        pdb_path_chains="",  # Chains to design for a single PDB
        jsonl_path=None,  # Path to folder with parsed PDB into jsonl
        chain_id_jsonl="",  # Path to chain ID jsonl
        fixed_positions_jsonl="",  # Path to fixed positions jsonl
        omit_AAs="X",  # Specify amino acids to omit from the sequence
        bias_AA_jsonl="",  # Path to dictionary for AA composition bias
        bias_by_res_jsonl="",  # Path to dictionary with per position bias
        omit_AA_jsonl="",  # Path to amino acids omitted for specific chains
        pssm_jsonl="",  # Path to dictionary with pssm
        pssm_multi=0.0,  # Value between [0.0, 1.0] for PSSM usage
        pssm_threshold=0.0,  # Restrict per-position amino acids
        pssm_log_odds_flag=0,  # PSSM log odds flag
        pssm_bias_flag=0,  # PSSM bias flag
        tied_positions_jsonl="",  # Path to dictionary with tied positions
    )

    # Step 2: If overrides are provided, update the default values
    if overrides:
        for key, value in overrides.items():
            if hasattr(args, key):
                setattr(args, key, value)
            else:
                raise KeyError(f"Argument '{key}' is not a valid argument")

    return args


import numpy as np


def read_npz_file(file_path):
    """
    Reads an .npz file and returns its contents as a dictionary.

    Parameters:
        file_path (str): Path to the .npz file.

    Returns:
        dict: A dictionary where keys are variable names and values are the corresponding arrays.
    """
    # Load the .npz file
    npz_file = np.load(file_path)

    # Convert the file into a dictionary to access its contents easily
    npz_data = {key: npz_file[key] for key in npz_file.files}

    # Close the file after reading
    npz_file.close()

    return npz_data


def resnums2pdb(pose, res_list, num_only=False):
    if num_only:
        return [pose.pdb_info().number(res) for res in res_list]
    return [pose.pdb_info().pose2pdb(res) for res in res_list]


def get_chain_start_positions_and_lengths(input_pdb):
    # Create a parser to parse the PDB file
    parser = PDB.PDBParser(QUIET=True)

    # Parse the structure from the input PDB file
    structure = parser.get_structure("structure", input_pdb)

    # Dictionaries to store start positions and lengths
    start_positions = {}
    lengths = {}

    # Iterate over each chain in the structure
    for model in structure:
        for chain in model:
            # Get the chain ID
            chain_id = chain.get_id()

            # Get all residues in the chain
            residues = list(chain.get_residues())

            # Get the start residue index (first residue's index)
            start_residue_id = residues[0].get_id()[1]

            # Get the length of the chain (number of residues)
            chain_length = len(residues)

            # Store the start position and length for the current chain
            start_positions[chain_id] = start_residue_id
            lengths[chain_id] = chain_length

    return start_positions, lengths


def split_probs_by_chain(probs: np.array, lengths: dict, chain_order: list):
    # Check if the sum of lengths matches the length of probs
    total_length = sum(lengths.values())
    if total_length != len(probs):
        raise ValueError("The sum of lengths must equal the length of probs.")

    # Initialize an empty dictionary to store the slices
    slices_dict = {}

    # Track the starting index for each slice
    start_index = 0

    # Loop through the chain order to create slices
    for chain_id in chain_order:
        # Ensure that the chain exists in the lengths dictionary
        if chain_id not in lengths:
            raise ValueError(f"Chain {chain_id} not found in the lengths dictionary.")

        # Get the length for the current chain
        length = lengths[chain_id]

        # Slice the list according to the current length
        slices_dict[chain_id] = probs[start_index : start_index + length]

        # Update the starting index for the next slice
        start_index += length

    return slices_dict


def split_residue_set_by_chain(res_set, lengths: dict, chain_order: list):
    # transform residue set to np array to hijack the split_probs function :)
    np_array = np.array([res_set[i] for i in range(1, len(res_set) + 1)], dtype=int)
    split_bools = split_probs_by_chain(np_array, lengths, chain_order)
    return split_bools


def renumber_pdb_old(input_pdb, start_number=1):

    output_pdb = f"{input_pdb[:-4]}_renum.pdb"

    new_residue_number = start_number
    current_residue_number = None

    with open(input_pdb, "r") as infile, open(output_pdb, "w") as outfile:
        for line in infile:
            # Only process lines that start with "ATOM" or "HETATM"
            if line.startswith(("ATOM", "HETATM")):
                # Extract the relevant fields: residue number (22-26)
                residue_number = line[22:27].strip()

                # Renumber only when we encounter a new residue
                if residue_number != current_residue_number:
                    current_residue_number = residue_number
                    new_residue_number_str = f"{new_residue_number:>4} "
                    new_residue_number += 1

                # Replace the old residue number with the new one (in positions 22-26)
                new_line = line[:22] + new_residue_number_str + line[27:]
                outfile.write(new_line)
            else:
                # Write non-ATOM/HETATM lines (e.g., TER, HEADER, etc.) as is
                #outfile.write(line)
                pass

    return output_pdb


def renumber_pdb(input_pdb, start_number=1):
    output_pdb = f"{input_pdb[:-4]}_renum.pdb"

    new_residue_number = start_number

    with open(input_pdb, "r") as infile, open(output_pdb, "w") as outfile:
        for line in infile:
            # Only process lines that start with "ATOM" or "HETATM"
            if line.startswith(("ATOM", "HETATM")):
                # Extract the atom name (13-15)
                atom_name = line[12:16].strip()

                # Check for the nitrogen backbone atom ("N"), which marks the start of a new residue
                if atom_name == "N":
                    # Increment the residue number for every new amino acid based on the "N" atom
                    new_residue_number_str = f"{new_residue_number:>4} "
                    new_residue_number += 1

                # Replace the old residue number with the new one (in positions 22-26)
                new_line = line[:22] + new_residue_number_str + line[27:]
                outfile.write(new_line)
            else:
                # Write non-ATOM/HETATM lines (e.g., TER, HEADER, etc.) as is
                #outfile.write(line)
                pass

    return output_pdb




def get_chain_lengths(pose):
    # Initialize a dictionary to store chain lengths
    chain_lengths = {}

    # Iterate over all residues to calculate chain lengths
    for i in range(1, pose.total_residue() + 1):  # PyRosetta is 1-indexed
        chain_id = pose.pdb_info().chain(i)

        if chain_id not in chain_lengths:
            chain_lengths[chain_id] = 0  # Initialize chain length if not present

        chain_lengths[chain_id] += 1  # Increment the count for this chain
    return chain_lengths


def calculate_interface_dG(pose, chain_1, chain_2):
    """
    Calculate the interface ΔG using InterfaceAnalyzerMover.
    """
    iam = InterfaceAnalyzerMover(f"{chain_1}_{chain_2}", False)
    iam.apply(pose)
    interface_dG = iam.get_interface_dG()
    return interface_dG


def mutate_and_relax(input_pdb, chain_id, residue_number, mutation3, pack_radius=8.0):
    """
    Mutate a residue at the interface, relax surrounding 8 Å, and repack the entire interface.

    Args:
        pose: PyRosetta pose object.
        chain_id: Chain ID where the mutation is located.
        residue_number: Residue number to mutate (PDB numbering).
        mutation: Amino acid to mutate to (e.g., 'A' for alanine).
        pack_radius: Radius around the mutation site to relax (default is 8.0 Å).

    Returns:
        Relaxed and repacked PyRosetta pose object.
    """

    pose = pyrosetta.pose_from_pdb(input_pdb)

    # Step 1: Mutate the residue
    print("mutate")
    pose_residue = pose.pdb_info().pdb2pose(chain_id, residue_number)
    # mutate_residue(pose, pose_residue, mutation)
    mutation = rosetta.protocols.simple_moves.MutateResidue(residue_number, mutation3)
    mutation.apply(pose)
    pose.update_residue_neighbors()  # Manually update the neighbor graph for the pose

    # Step 2: Select residues around the mutation site (8 Å radius)
    relax_selector = NeighborhoodResidueSelector()
    relax_selector.set_focus(str(pose_residue))  # Focus on mutated residue
    relax_selector.set_distance(pack_radius)  # 8 Å radius around the mutation site
    relax_selector.set_include_focus_in_subset(
        True
    )  # Include the mutated residue itself
    not_relax_selector = NotResidueSelector(relax_selector)

    # Select interface residues using InterfaceByVectorSelector
    interface_selector = InterGroupInterfaceByVectorSelector()

    # Set up Interface Selector
    interface_selector = InterGroupInterfaceByVectorSelector()
    interface_selector.group1_selector(ChainSelector("A"))  # Set receptor as group 1
    interface_selector.group2_selector(ChainSelector("E"))  # Set binder as group 2
    not_IF = NotResidueSelector(interface_selector)
    no_pack = AndResidueSelector(not_IF, not_relax_selector)

    # Step 3: TaskFactory to repack around mutation and interface
    tf = TaskFactory()
    prevent_repacking = OperateOnResidueSubset(PreventRepackingRLT(), no_pack)
    tf.push_back(RestrictToRepacking())
    tf.push_back(prevent_repacking)  # Prevent repacking outside the selected region

    # Set up the task factory for mutating residues
    tf = TaskFactory()
    # These are pretty standard
    tf.push_back(InitializeFromCommandline())
    tf.push_back(IncludeCurrent())
    tf.push_back(NoRepackDisulfides())

    # Fix all sidechains but IF
    # tf.push_back(OperateOnResidueSubset(PreventRepackingRLT(), not_IF))
    # tf.push_back(OperateOnResidueSubset(RestrictToRepackingRLT(), interface_selector))

    # Step 4: Apply FastRelax
    print("Setup relax")
    movemap = MoveMapFactory()
    movemap.add_bb_action(
        move_map_action(False), relax_selector
    )  # Allow backbone flexibility within 8 Å
    movemap.add_chi_action(
        move_map_action(True), relax_selector
    )  # Allow sidechain flexibility within 8 Å
    movemap.add_chi_action(
        move_map_action(True), interface_selector
    )  # Allow sidechain flexibility within 8 Å
    movemap.add_bb_action(move_map_action(False), no_pack)  # Fix backbone outside 8 Å
    movemap.add_chi_action(
        move_map_action(False), no_pack
    )  # Fix sidechains outside 8 Å

    print("RELAX")
    relax = FastRelax()
    scorefxn = pyrosetta.get_fa_scorefxn()
    relax.set_scorefxn(scorefxn)
    relax.constrain_relax_to_start_coords(True)
    relax.set_movemap_factory(movemap)
    # relax.set_task_factory(tf)
    relax.apply(pose)
    # pose.update_residue_neighbors()  # Manually update the neighbor graph for the pose

    return pose


def calculate_dG(input_pdb, chain_1, chain_2, chain_id, residue_number, mutation):
    """
    Calculate the interface ΔΔG upon mutation at a specific residue in a protein interface.
    """
    pyrosetta.init("-detect_disulf false")
    pose = mutate_and_relax(input_pdb, chain_id, residue_number, mutation)

    # Calculate the interface ΔG for mutant pose
    mutant_dG = calculate_interface_dG(pose, chain_1, chain_2)
    return [mutation, mutant_dG, pose.sequence()]


def mutate_to_all_amino_acids_parallel(
    input_pdb,
    chain_id,
    residue_number,
    chain_1,
    chain_2,
    num_cores=4,
    amino_acids="ADEFGHIKLMNPQRSTVWY",
):
    """
    Mutate a residue at a given position to all 20 standard amino acids, relax each mutant,
    and calculate the ΔΔG relative to the wild-type structure. Runs the calculation in parallel on the specified number of cores.
    """

    pyrosetta.init("-detect_disulf false")
    pose = pyrosetta.pose_from_pdb(input_pdb)
    # Get the wild-type residue name
    wt_residue = pose.residue(
        pose.pdb_info().pdb2pose(chain_id, residue_number)
    ).name3()

    # Add wt residue to mutation list if not present
    wt_name1 = pose.residue(
        pose.pdb_info().pdb2pose(chain_id, residue_number)
    ).name1()
    if wt_name1 not in amino_acids:
        amino_acids += wt_name1

    # Step 1: Calculate the interface ΔG for the wild-type pose (set this as the reference ΔG = 0)
    # wt_dG = calculate_interface_dG(pose, chain_1, chain_2)
    # print(f"Wild-type interface ΔG: {wt_dG}")

    # Dictionary to store the ΔΔG for each amino acid
    ddG_results = {}
    ddG_results_raw = {}

    # Step 2: Use ProcessPoolExecutor to parallelize the mutations and ΔΔG calculations
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for aa in amino_acids:
            # Submit the mutation and ΔΔG calculation to the pool
            futures.append(
                executor.submit(
                    calculate_dG,
                    input_pdb,
                    chain_1,
                    chain_2,
                    chain_id,
                    residue_number,
                    IUPACData.protein_letters_1to3[aa].upper(),
                )
            )

        # Collect the results as they complete
        for future in as_completed(futures):
            aa, ddG, seq = future.result()
            ddG_results_raw[aa] = {'ddG': ddG, 'sequence': seq}

    for aa, ddG_raw in ddG_results_raw.items():
        ddG_results[aa] = {'ddG': ddG_raw['ddG'] - ddG_results_raw[wt_residue]['ddG'], 'sequence': ddG_raw['sequence']}
        print(f"ΔΔG for {aa} mutant: {ddG_results[aa]['ddG']:.2f} REU")
    for aa, ddG_raw in ddG_results_raw.items():
        print(f"Seq for {aa} mutant: {ddG_raw['sequence']}")


    return ddG_results
