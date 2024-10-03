import numpy as np
import sys, os
import importlib.util
import subprocess
from utils import *

# Step 1: Define the path to the script folder
protein_mpnn_folder = "/Users/raemisch/opt/ProteinMPNN/"

# Step 2: Add the folder to sys.path so the scripts can find each other
if protein_mpnn_folder not in sys.path:
    sys.path.append(protein_mpnn_folder)

spec = importlib.util.spec_from_file_location(
    "protein_mpnn_run", "/Users/raemisch/opt/ProteinMPNN/protein_mpnn_run.py"
)
protein_mpnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protein_mpnn)


##############################################################################################
# MPNN Preparation
# I don't like having to use the commandline tools. WIll change that some time in the future.

output_dir = "mpnn_output"

path_for_parsed_chains = f"{output_dir}/parsed_pdbs.jsonl"
path_for_assigned_chains = f"{output_dir}/assigned_pdbs.jsonl"
path_for_fixed_positions = f"{output_dir}/fixed_pdbs.jsonl"

FIXED_ARGS = {
    "out_folder": output_dir,
    "jsonl_path": path_for_parsed_chains,
    "chain_id_jsonl": path_for_assigned_chains,
    "fixed_positions_jsonl": path_for_fixed_positions,
}


def prepare_mpnn(pose, chains_to_design, fixed_positions):
    # Fix what is not part of the interface
    # chains_to_design="H L R" # Order as in pdb file!!!!!

    # chains_to_design = "H L R"
    # fixed_positions = [str(i+1) for i,v in enumerate(chainR_sel.apply(pose)) if not R_IF_set[i+1]]

    # Convert fixed_positions in rosetta numbering into the relevant string for MPNN

    pdb_info = pose.pdb_info()

    chain_list = [[] for _ in range(pose.num_chains())]

    for pos in [int(i) for i in fixed_positions]:
        chain_ID = pdb_info.chain(pos)
        if chain_ID in chains_to_design:
            chain_index = pose.chain(pos)
            chain_start = pose.conformation().chain_begin(chain_index)
            chain_end = pose.conformation().chain_end(chain_index)
            # Compute the local residue index within the chain
            chain_residue_index = pos - chain_start + 1
            chain_list[chain_index - 1].append(chain_residue_index)

    for chain_nr, chain in enumerate(chain_list):
        print(f"Fixed {len(chain)} positions in chain {chain_nr}")

    # Create string for fixed positions script
    fixed_positions_str_list = [
        " ".join([str(i) for i in chain]) for chain in chain_list if len(chain) > 0
    ]
    fixed_positions_str = ", ".join(fixed_positions_str_list)
    print(fixed_positions_str)

    # Create input folder and file for MPNN

    pdb_file = os.path.basename(pose.pdb_info().name())
    input_path = f"inputs_pdbs/{pdb_file[:-4]}"

    os.makedirs(input_path, exist_ok=True)
    pdb_input_path = f"{input_path}/pose_{pdb_file}"
    pose.dump_pdb(pdb_input_path)
    print(pdb_input_path, "created.")

    output_dir = "mpnn_output"

    # Run the first Python script
    subprocess.run(
        [
            "python",
            f"{protein_mpnn_folder}/helper_scripts/parse_multiple_chains.py",
            "--input_path",
            input_path,
            "--output_path",
            path_for_parsed_chains,
        ]
    )

    # Run the second Python script
    subprocess.run(
        [
            "python",
            f"{protein_mpnn_folder}/helper_scripts/assign_fixed_chains.py",
            "--input_path",
            path_for_parsed_chains,
            "--output_path",
            path_for_assigned_chains,
            "--chain_list",
            chains_to_design,
        ]
    )

    # Run the third Python script
    subprocess.run(
        [
            "python",
            f"{protein_mpnn_folder}/helper_scripts/make_fixed_positions_dict.py",
            "--input_path",
            path_for_parsed_chains,
            "--output_path",
            path_for_fixed_positions,
            "--chain_list",
            chains_to_design,
            "--position_list",
            fixed_positions_str,
        ]
    )

    return pdb_input_path


def run_mpnn(args, chain_lengths):
    # A dictionary of arguments to override
    # args = {
    #     'save_probs': 1,
    #     'pdb_path': pdb_input_path,
    #     'out_folder': output_dir,
    #     'jsonl_path': path_for_parsed_chains,
    #     'chain_id_jsonl': path_for_assigned_chains,
    #     'fixed_positions_jsonl': path_for_fixed_positions,
    #     'sampling_temp': "0.1",
    #     'seed': 42

    # }

    args = {**FIXED_ARGS, **args}
    # Combine args with default MPNN args and run
    args = create_args(args)
    protein_mpnn.main(args)

    # start_positions, chain_lengths = get_chain_start_positions_and_lengths(args.pdb_path)
    # Individual MPNN results

    name = os.path.basename(args.pdb_path)[:-4]
    npz_data = read_npz_file(f"{output_dir}/probs/{name}.npz")

    # What is in there?
    print("keys:       ", list(npz_data.keys()))
    print("probs shape:", npz_data["probs"][0].shape)

    probs = npz_data["probs"][0]
    mask = npz_data["mask"][0]
    chain_order = npz_data["chain_order"][0]
    probs_per_chain = split_probs_by_chain(probs, chain_lengths, chain_order)

    return probs_per_chain


def derive_sequence_from_probs(combined_probs, original_sequence, method="argmax", temperature=0.3, nseq=10):
    """
    Derives a sequence from combined probabilities, using original amino acids where probabilities are all 0.

    Args:
        combined_probs (numpy.ndarray): 2D array of shape (L, 21), representing probabilities for each position.
        original_sequence (str): The original sequence to fill positions with zero probabilities.
        method (str): "argmax" (deterministic) or "sampling" (probabilistic) method to derive the sequence.
        temperature (float): Temperature for scaling probabilities during sampling (only relevant for "sampling").

    Returns:
        str: The derived amino acid sequence.
    """
    alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
    
    # Handle NaNs, infinities, and negative values
    combined_probs = np.nan_to_num(combined_probs, nan=0.0, posinf=0.0, neginf=0.0)
    combined_probs[combined_probs < 0] = 0
    
    # Normalize probabilities and identify zero-sum rows
    row_sums = np.sum(combined_probs, axis=1, keepdims=True)
    zero_sum_rows = (row_sums == 0).flatten()
    combined_probs = np.divide(combined_probs, row_sums, where=row_sums != 0)

    if method == "argmax":
        indices = np.argmax(combined_probs, axis=1)
    elif method == "sampling":
        scaled_probs = np.exp(np.log(combined_probs + 1e-8) / temperature)  # Adding a small value to avoid log(0)
        scaled_probs /= scaled_probs.sum(axis=1, keepdims=True)
        indices = [np.random.choice(len(alphabet), p=scaled_probs[i]) for i in range(len(combined_probs))]
    else:
        raise ValueError("Method must be 'argmax' or 'sampling'")
    
    # Construct the final sequence, using original_sequence where probs were zero
    return ''.join(original_sequence[i] if zero_sum_rows[i] else alphabet[indices[i]] for i in range(len(combined_probs)))

