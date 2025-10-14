import uproot
import awkward as ak
import pandas as pd
from multiprocessing import Pool
import numpy as np

def _filter_cl3d_to_df(tree, eta_name, pt_name, eta_min=1.6, eta_max=2.8, pt_thr=20):
    # Read just what we need
    arr = tree.arrays([eta_name, pt_name, "event"], library="ak")

    # Replace None with sentinel and build mask safely
    eta = ak.without_parameters(arr[eta_name])
    pt  = ak.without_parameters(arr[pt_name])

    eta = ak.fill_none(eta, np.nan)
    pt  = ak.fill_none(pt,  np.nan)

    # Flatten to 1D to avoid jagged alignment surprises
    eta_f = ak.to_numpy(ak.flatten(eta, axis=None))
    pt_f  = ak.to_numpy(ak.flatten(pt,  axis=None))
    # Broadcast event to CL3D multiplicity, then flatten
    evt_b = ak.broadcast_to(arr["event"], eta)
    evt_f = ak.to_numpy(ak.flatten(evt_b, axis=None))

    # Build mask in numpy so NaNs drop out cleanly
    m = np.isfinite(eta_f) & np.isfinite(pt_f)
    m &= (np.abs(eta_f) > eta_min) & (np.abs(eta_f) < eta_max) & (pt_f > pt_thr)

    kept = int(m.sum())
    print(f"{eta_name.replace('_eta','')}: kept {kept} clusters (pt>{pt_thr}, {eta_min}<|eta|<{eta_max})")

    if kept == 0:
        return pd.DataFrame()

    # Construct DF from filtered flat arrays
    out = pd.DataFrame({
        "event": evt_f[m],
        pt_name: pt_f[m],
        eta_name: eta_f[m],
    })
    # Optional: add more CL3D columns later by pulling them and flattening with the same mask
    return out
# Function to load and filter signal tree
def load_and_filter_signal_tree(tree, filter_pt=5, eta_range=(1.6, 2.8), cl_pt_threshold=20):
    # Load gen variables
    df_gen = ak.to_dataframe(tree.arrays(
        library="ak",
        filter_name=["gen_n", "gen_eta", "gen_phi", "gen_pt", "gen_energy", "gen_status", "gen_pdgid",
                     "genpart_exeta", "genpart_exphi", "event"]
    ))
    # Load cl3d variables
    df_cl3d_p016Tri = ak.to_dataframe(tree.arrays( library="ak", filter_name=["*cl3d_p016Tri*", "event"]))
    print(df_cl3d_p016Tri)
    df_cl3d_p03Tri = ak.to_dataframe(tree.arrays( library="ak", filter_name=["*cl3d_p03Tri*", "event"]))
    df_cl3d_p045Tri = ak.to_dataframe(tree.arrays( library="ak", filter_name=["*cl3d_p045Tri*", "event"]))
    # Apply filters to gen DataFrame
    df_gen_filtered = df_gen[(df_gen['gen_pt'] > filter_pt) &
                              (abs(df_gen['gen_eta']) > eta_range[0]) & 
                              (abs(df_gen['gen_eta']) < eta_range[1])]
    # Apply filters to cl3d DataFrame
    df_cl3d_p016Tri_filtered = df_cl3d_p016Tri[
                                (df_cl3d_p016Tri['cl3d_p016Tri_pt'] > cl_pt_threshold)]
    df_cl3d_p03Tri_filtered = df_cl3d_p03Tri[(abs(df_cl3d_p03Tri['cl3d_p03Tri_eta']) > eta_range[0]) & 
                                (abs(df_cl3d_p03Tri['cl3d_p03Tri_eta']) < eta_range[1]) &
                                (df_cl3d_p03Tri['cl3d_p03Tri_pt'] > cl_pt_threshold)]
    df_cl3d_p045Tri_filtered = df_cl3d_p045Tri[(abs(df_cl3d_p045Tri['cl3d_p045Tri_eta']) > eta_range[0]) & 
                                (abs(df_cl3d_p045Tri['cl3d_p045Tri_eta']) < eta_range[1]) &
                                (df_cl3d_p045Tri['cl3d_p045Tri_pt'] > cl_pt_threshold)]
    return df_gen_filtered#, df_cl3d_p016Tri, df_cl3d_p03Tri_filtered, df_cl3d_p045Tri_filtered

# Worker function for multiprocessing
def process_file(file_info):
    try:
        file_path, bg_folder, tree_name = file_info
        root_file = uproot.open(file_path)
        tree = root_file[f"{bg_folder}/{tree_name}"]

        # Load and filter gen and cl3d objects
        df_cl3dp016 = _filter_cl3d_to_df(tree, "cl3d_p016Tri_eta", "cl3d_p016Tri_pt",
                                        eta_min=1.6, eta_max=2.8, pt_thr=20)
        df_cl3dp03  = _filter_cl3d_to_df(tree, "cl3d_p03Tri_eta",  "cl3d_p03Tri_pt",
                                    eta_min=1.6, eta_max=2.8, pt_thr=20)
        df_cl3dp045 = _filter_cl3d_to_df(tree, "cl3d_p045Tri_eta", "cl3d_p045Tri_pt",
                                    eta_min=1.6, eta_max=2.8, pt_thr=20)

        print(f"Processed {file_path}")
        return df_cl3dp016, df_cl3dp03, df_cl3dp045

    except Exception as e:
        print(f"Error processing file {file_info[0]}: {e}")
        return pd.DataFrame(), pd.DataFrame()

# Function to process files in parallel
def process_files_parallel(filelist_path, bg_folder, tree_name, output_dir, num_processes):
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
    cl3d_p016_dfs, cl3d_p03_dfs, cl3d_p045_dfs = zip(*results)

    # Combine all the DataFrames into single ones
    #combined_gen_df = pd.concat(gen_dfs, ignore_index=True)
    combined_cl3dp016_df = pd.concat(cl3d_p016_dfs, ignore_index=True)
    combined_cl3dp03_df = pd.concat(cl3d_p03_dfs, ignore_index=True)
    combined_cl3dp045_df = pd.concat(cl3d_p045_dfs, ignore_index=True)
    # Save the combined DataFrames to output files
    #gen_output_path = f"{output_dir}/photonPU200_newalgogen_filtered.h5"
    cl3dp016_output_path = f"{output_dir}/photonPU200_newalgocl3dp016_filtered.h5"
    cl3dp03_output_path = f"{output_dir}/photonPU200_newalgocl3dp03_filtered.h5"
    cl3dp045_output_path = f"{output_dir}/photonPU200_newalgocl3dp045_filtered.h5"
    #combined_gen_df.to_hdf(gen_output_path, key="gen", mode="w")
    combined_cl3dp016_df.to_hdf(cl3dp016_output_path, key="cl3d_p016", mode="w")
    combined_cl3dp03_df.to_hdf(cl3dp03_output_path, key="cl3d_p03", mode="w")
    combined_cl3dp045_df.to_hdf(cl3dp045_output_path, key="cl3d_p045", mode="w")
    #print(f"gen data saved to {gen_output_path}")
    print(f"cl3d data saved to {cl3dp016_output_path}")
    print(f"cl3d data saved to {cl3dp03_output_path}")
    print(f"cl3d data saved to {cl3dp045_output_path}")
# Set the paths
bg_folder = "l1tHGCalTriggerNtuplizer"
tree_name = "HGCalTriggerNtuple"
output_dir = "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples"

# Process the files in parallel (using 20 processes)
process_files_parallel("filelists/photonPU200_newalgo.txt", bg_folder, tree_name, output_dir, num_processes=20)