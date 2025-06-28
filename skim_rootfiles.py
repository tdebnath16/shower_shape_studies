import uproot
import awkward as ak
import pandas as pd
from multiprocessing import Pool

# Function to load and filter signal tree
def load_and_filter_signal_tree(tree, filter_pt=5, eta_range=(1.6, 2.8), cl_pt_threshold=5):
    # Load gen variables
    df_gen = ak.to_dataframe(tree.arrays(
        library="ak",
        filter_name=["gen_n", "gen_eta", "gen_phi", "gen_pt", "gen_energy", "gen_status", "gen_pdgid",
                     "genpart_exeta", "genpart_exphi", "event"]
    ))
    # Load cl3d variables
    df_cl3d = ak.to_dataframe(tree.arrays(
        library="ak",
        filter_name=["*cl3d*", "event"]
    ))
    # Apply filters to gen DataFrame
    df_gen_filtered = df_gen[(df_gen['gen_pt'] > filter_pt) & 
                              (abs(df_gen['gen_eta']) > eta_range[0]) & 
                              (abs(df_gen['gen_eta']) < eta_range[1])]
    # Apply filters to cl3d DataFrame
    df_cl3d_filtered = df_cl3d[(abs(df_cl3d['cl3d_eta']) > eta_range[0]) & 
                                (abs(df_cl3d['cl3d_eta']) < eta_range[1]) &
                                (df_cl3d['cl3d_pt'] > cl_pt_threshold)]
    return df_gen_filtered, df_cl3d_filtered

# Worker function for multiprocessing
def process_file(file_info):
    try:
        file_path, bg_folder, tree_name = file_info
        root_file = uproot.open(file_path)
        tree = root_file[f"{bg_folder}/{tree_name}"]

        # Load and filter gen and cl3d objects
        df_gen, df_cl3d = load_and_filter_signal_tree(tree)

        print(f"Processed {file_path}")
        return df_gen, df_cl3d

    except Exception as e:
        print(f"Error processing file {file_info[0]}: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Function to process files in parallel
def process_files_parallel(filelist_path, bg_folder, tree_name, output_dir, num_processes=20):
    # Ensure output directory exists
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Read the file list
    with open(filelist_path, "r") as f:
        file_list = [line.strip() for line in f.readlines()]

    # Prepare file info tuples for multiprocessing
    file_info_list = [(file_path, bg_folder, tree_name) for file_path in file_list]

    # Use multiprocessing Pool for parallel processing
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_file, file_info_list)

    # Separate gen and cl3d DataFrames from results
    gen_dfs, cl3d_dfs = zip(*results)

    # Combine all the DataFrames into single ones
    combined_gen_df = pd.concat(gen_dfs, ignore_index=True)
    combined_cl3d_df = pd.concat(cl3d_dfs, ignore_index=True)

    # Save the combined DataFrames to output files
    gen_output_path = f"{output_dir}/testQCD300toInfgen_filtered.h5"
    cl3d_output_path = f"{output_dir}/testQCD300toInfcl3d_filtered.h5"
    combined_gen_df.to_hdf(gen_output_path, key="gen", mode="w")
    combined_cl3d_df.to_hdf(cl3d_output_path, key="cl3d", mode="w")

    print(f"gen data saved to {gen_output_path}")
    print(f"cl3d data saved to {cl3d_output_path}")

# Set the paths
bg_folder = "l1tHGCalTriggerNtuplizer"
tree_name = "HGCalTriggerNtuple"
output_dir = "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples"

# Process the files in parallel (using 20 processes)
process_files_parallel("filelists/filelistQCD_EMenriched_pT300toInf.txt", bg_folder, tree_name, output_dir, num_processes=20)