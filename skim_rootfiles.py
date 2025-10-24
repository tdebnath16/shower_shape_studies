import uproot
import awkward as ak
import pandas as pd
from multiprocessing import Pool
import analysis as ana


# Function to load and filter signal tree
def load_and_filter_signal_tree(tree, filter_pt=20, eta_range=(1.6, 2.8), cl_pt_threshold=20):
    # Load gen variables
    df_gen = ak.to_dataframe(tree.arrays(
        library="ak",
        filter_name=["gen_n", "gen_eta", "gen_phi", "gen_pt", "gen_energy", "gen_status", "gen_pdgid", "event"]))
    df_genpart = ak.to_dataframe(tree.arrays(
        library="ak",
        filter_name=["genpart_exeta", "genpart_exphi", "genpart_gen", "genpart_reachedEE", "event"]))
    # Load cl3d_p016 variables
    df_cl3d_Ref = ak.to_dataframe(tree.arrays(library="ak",filter_name=["cl3d_Ref_energy", "cl3d_Ref_eta", "cl3d_Ref_pt", 
                                                                         "cl3d_Ref_phi", "cl3d_Ref_showerlength", "cl3d_Ref_coreshowerlength",
                                                                         "cl3d_Ref_firstlayer", "cl3d_Ref_maxlayer", "cl3d_Ref_emaxe",
                                                                         "cl3d_Ref_varrr", "cl3d_Ref_varee", "cl3d_Ref_varzz", "cl3d_Ref_varpp",
                                                                         "cl3d_Ref_hoe", "cl3d_Ref_meanz", "cl3d_Ref_eot",
                                                                         "cl3d_Ref_first1layers", "cl3d_Ref_first3layers", "cl3d_Ref_first5layers",
                                                                         "cl3d_Ref_firstHcal1layers", "cl3d_Ref_firstHcal3layers", "cl3d_Ref_firstHcal5layers",
                                                                         "cl3d_Ref_last1layers", "cl3d_Ref_last3layers", "cl3d_Ref_last5layers",
                                                                         "cl3d_Ref_emax1layers", "cl3d_Ref_emax3layers", "cl3d_Ref_emax5layers",
                                                                         "cl3d_Ref_ebm0", "cl3d_Ref_ebm1", "cl3d_Ref_hbm", "event"]))
    df_cl3d_p0113 = ak.to_dataframe(tree.arrays(library="ak",filter_name=["cl3d_p0113Tri_energy", "cl3d_p0113Tri_eta", "cl3d_p0113Tri_pt", 
                                                                         "cl3d_p0113Tri_phi", "cl3d_p0113Tri_showerlength", "cl3d_p0113Tri_coreshowerlength",
                                                                         "cl3d_p0113Tri_firstlayer", "cl3d_p0113Tri_maxlayer", "cl3d_p0113Tri_emaxe",
                                                                         "cl3d_p0113Tri_varrr", "cl3d_p0113Tri_varee", "cl3d_p0113Tri_varzz", "cl3d_p0113Tri_varpp",
                                                                         "cl3d_p0113Tri_hoe", "cl3d_p0113Tri_meanz", "cl3d_p0113Tri_eot",
                                                                         "cl3d_p0113Tri_first1layers", "cl3d_p0113Tri_first3layers", "cl3d_p0113Tri_first5layers",
                                                                         "cl3d_p0113Tri_firstHcal1layers", "cl3d_p0113Tri_firstHcal3layers", "cl3d_p0113Tri_firstHcal5layers",
                                                                         "cl3d_p0113Tri_last1layers", "cl3d_p0113Tri_last3layers", "cl3d_p0113Tri_last5layers",
                                                                         "cl3d_p0113Tri_emax1layers", "cl3d_p0113Tri_emax3layers", "cl3d_p0113Tri_emax5layers",
                                                                         "cl3d_p0113Tri_ebm0", "cl3d_p0113Tri_ebm1", "cl3d_p0113Tri_hbm", "event"]))
    df_cl3d_p016 = ak.to_dataframe(tree.arrays(library="ak",filter_name=["cl3d_p016Tri_energy", "cl3d_p016Tri_eta", "cl3d_p016Tri_pt", 
                                                                         "cl3d_p016Tri_phi", "cl3d_p016Tri_showerlength", "cl3d_p016Tri_coreshowerlength",
                                                                         "cl3d_p016Tri_firstlayer", "cl3d_p016Tri_maxlayer", "cl3d_p016Tri_emaxe",
                                                                         "cl3d_p016Tri_varrr", "cl3d_p016Tri_varee", "cl3d_p016Tri_varzz", "cl3d_p016Tri_varpp",
                                                                         "cl3d_p016Tri_hoe", "cl3d_p016Tri_meanz", "cl3d_p016Tri_eot",
                                                                         "cl3d_p016Tri_first1layers", "cl3d_p016Tri_first3layers", "cl3d_p016Tri_first5layers",
                                                                         "cl3d_p016Tri_firstHcal1layers", "cl3d_p016Tri_firstHcal3layers", "cl3d_p016Tri_firstHcal5layers",
                                                                         "cl3d_p016Tri_last1layers", "cl3d_p016Tri_last3layers", "cl3d_p016Tri_last5layers",
                                                                         "cl3d_p016Tri_emax1layers", "cl3d_p016Tri_emax3layers", "cl3d_p016Tri_emax5layers",
                                                                         "cl3d_p016Tri_ebm0", "cl3d_p016Tri_ebm1", "cl3d_p016Tri_hbm", "event"]))
    df_cl3d_p03 = ak.to_dataframe(tree.arrays(library="ak",filter_name=["cl3d_p03Tri_energy", "cl3d_p03Tri_eta", "cl3d_p03Tri_pt", 
                                                                         "cl3d_p03Tri_phi", "cl3d_p03Tri_showerlength", "cl3d_p03Tri_coreshowerlength",
                                                                         "cl3d_p03Tri_firstlayer", "cl3d_p03Tri_maxlayer", "cl3d_p03Tri_emaxe",
                                                                         "cl3d_p03Tri_varrr", "cl3d_p03Tri_varee", "cl3d_p03Tri_varzz", "cl3d_p03Tri_varpp",
                                                                         "cl3d_p03Tri_hoe", "cl3d_p03Tri_meanz", "cl3d_p03Tri_eot",
                                                                         "cl3d_p03Tri_first1layers", "cl3d_p03Tri_first3layers", "cl3d_p03Tri_first5layers",
                                                                         "cl3d_p03Tri_firstHcal1layers", "cl3d_p03Tri_firstHcal3layers", "cl3d_p03Tri_firstHcal5layers",
                                                                         "cl3d_p03Tri_last1layers", "cl3d_p03Tri_last3layers", "cl3d_p03Tri_last5layers",
                                                                         "cl3d_p03Tri_emax1layers", "cl3d_p03Tri_emax3layers", "cl3d_p03Tri_emax5layers",
                                                                         "cl3d_p03Tri_ebm0", "cl3d_p03Tri_ebm1", "cl3d_p03Tri_hbm", "event"]))
    df_cl3d_p045 = ak.to_dataframe(tree.arrays(library="ak",filter_name=["cl3d_p045Tri_energy", "cl3d_p045Tri_eta", "cl3d_p045Tri_pt", 
                                                                         "cl3d_p045Tri_phi", "cl3d_p045Tri_showerlength", "cl3d_p045Tri_coreshowerlength",
                                                                         "cl3d_p045Tri_firstlayer", "cl3d_p045Tri_maxlayer", "cl3d_p045Tri_emaxe",
                                                                         "cl3d_p045Tri_varrr", "cl3d_p045Tri_varee", "cl3d_p045Tri_varzz", "cl3d_p045Tri_varpp",
                                                                         "cl3d_p045Tri_hoe", "cl3d_p045Tri_meanz", "cl3d_p045Tri_eot",
                                                                         "cl3d_p045Tri_first1layers", "cl3d_p045Tri_first3layers", "cl3d_p045Tri_first5layers",
                                                                         "cl3d_p045Tri_firstHcal1layers", "cl3d_p045Tri_firstHcal3layers", "cl3d_p045Tri_firstHcal5layers",
                                                                         "cl3d_p045Tri_last1layers", "cl3d_p045Tri_last3layers", "cl3d_p045Tri_last5layers",
                                                                         "cl3d_p045Tri_emax1layers", "cl3d_p045Tri_emax3layers", "cl3d_p045Tri_emax5layers",
                                                                         "cl3d_p045Tri_ebm0", "cl3d_p045Tri_ebm1", "cl3d_p045Tri_hbm", "event"]))
    # Apply filters to gen DataFrame
    df_gen_filtered = df_gen[(df_gen['gen_pt'] > filter_pt) & 
                             #(df_gen['gen_pdgid'] == 22) & 
                             #(df_gen['gen_status'] == 1) & 
                             (abs(df_gen['gen_eta']) > eta_range[0]) & 
                             (abs(df_gen['gen_eta']) < eta_range[1]) ]
    df_genpart_filtered = df_genpart[(df_genpart['genpart_reachedEE'] == 2) & 
                              (df_genpart['genpart_gen'] == 1) &
                              (abs(df_genpart['genpart_exeta']) > eta_range[0]) & 
                              (abs(df_genpart['genpart_exeta']) < eta_range[1])]
    df_gen_merged = ana.load_and_filter_hdf(df_gen_filtered, df_genpart)
    # Apply filters to cl3d DataFrame
    df_cl3d_Ref_filtered = df_cl3d_Ref[(df_cl3d_Ref['cl3d_Ref_pt'] > cl_pt_threshold) & 
                                (abs(df_cl3d_Ref['cl3d_Ref_eta']) > eta_range[0]) & 
                                (abs(df_cl3d_Ref['cl3d_Ref_eta']) < eta_range[1])]
    df_cl3d_p0113_filtered = df_cl3d_p0113[(df_cl3d_p0113['cl3d_p0113Tri_pt'] > cl_pt_threshold) & 
                                (abs(df_cl3d_p0113['cl3d_p0113Tri_eta']) > eta_range[0]) & 
                                (abs(df_cl3d_p0113['cl3d_p0113Tri_eta']) < eta_range[1])]
    df_cl3d_p016_filtered = df_cl3d_p016[(df_cl3d_p016['cl3d_p016Tri_pt'] > cl_pt_threshold) & 
                                (abs(df_cl3d_p016['cl3d_p016Tri_eta']) > eta_range[0]) & 
                                (abs(df_cl3d_p016['cl3d_p016Tri_eta']) < eta_range[1])]
    df_cl3d_p03_filtered = df_cl3d_p03[(df_cl3d_p03['cl3d_p03Tri_pt'] > cl_pt_threshold) & 
                                (abs(df_cl3d_p03['cl3d_p03Tri_eta']) > eta_range[0]) & 
                                (abs(df_cl3d_p03['cl3d_p03Tri_eta']) < eta_range[1])]
    df_cl3d_p045_filtered = df_cl3d_p045[(df_cl3d_p045['cl3d_p045Tri_pt'] > cl_pt_threshold) & 
                                (abs(df_cl3d_p045['cl3d_p045Tri_eta']) > eta_range[0]) & 
                                (abs(df_cl3d_p045['cl3d_p045Tri_eta']) < eta_range[1])]
    
    return df_gen_merged, df_cl3d_Ref_filtered, df_cl3d_p0113_filtered, df_cl3d_p016_filtered, df_cl3d_p03_filtered, df_cl3d_p045_filtered

# Worker function for multiprocessing
def process_file(file_info):
    try:
        file_path, bg_folder, tree_name = file_info
        root_file = uproot.open(file_path)
        tree = root_file[f"{bg_folder}/{tree_name}"]

        # Load and filter gen and cl3d objects
        df_gen, df_cl3d_Ref, df_cl3d_p0113, df_cl3d_p016, df_cl3d_p03, df_cl3d_p045 = load_and_filter_signal_tree(tree)

        print(f"Processed {file_path}")
        return df_gen, df_cl3d_Ref, df_cl3d_p0113, df_cl3d_p016, df_cl3d_p03, df_cl3d_p045

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
    gen_dfs, cl3d_Ref_dfs,  cl3d_p0113_dfs, cl3d_p016_dfs, cl3d_p03_dfs, cl3d_p045_dfs= zip(*results)

    # Combine all the DataFrames into single ones
    combined_gen_df = pd.concat(gen_dfs, ignore_index=True)
    combined_cl3d_Ref_df = pd.concat(cl3d_Ref_dfs, ignore_index=True)
    combined_cl3d_p0113_df = pd.concat(cl3d_p0113_dfs, ignore_index=True)
    combined_cl3d_p016_df = pd.concat(cl3d_p016_dfs, ignore_index=True)
    combined_cl3d_p03_df = pd.concat(cl3d_p03_dfs, ignore_index=True)
    combined_cl3d_p045_df = pd.concat(cl3d_p045_dfs, ignore_index=True)

    # Save the combined DataFrames to output files
    gen_output_path = f"{output_dir}/pionPU200_newalgogen_filtered.h5"
    cl3d_Ref_output_path = f"{output_dir}/pionPU200_newalgocl3d_Ref_filtered.h5"
    cl3d_p0113_output_path = f"{output_dir}/pionPU200_newalgocl3d_p0113_filtered.h5"
    cl3d_p016_output_path = f"{output_dir}/pionPU200_newalgocl3d_p016_filtered.h5"
    cl3d_p03_output_path = f"{output_dir}/pionPU200_newalgocl3d_p03_filtered.h5"
    cl3d_p045_output_path = f"{output_dir}/pionPU200_newalgocl3d_p045_filtered.h5"
    combined_gen_df.to_hdf(gen_output_path, key="gen", mode="w")
    combined_cl3d_Ref_df.to_hdf(cl3d_Ref_output_path, key="cl3d_Ref", mode="w")
    combined_cl3d_p0113_df.to_hdf(cl3d_p0113_output_path, key="cl3d_p0113", mode="w")
    combined_cl3d_p016_df.to_hdf(cl3d_p016_output_path, key="cl3d_p016", mode="w")
    combined_cl3d_p03_df.to_hdf(cl3d_p03_output_path, key="cl3d_p03", mode="w")
    combined_cl3d_p045_df.to_hdf(cl3d_p045_output_path, key="cl3d_p045", mode="w")

# Set the paths
bg_folder = "l1tHGCalTriggerNtuplizer"
tree_name = "HGCalTriggerNtuple"
output_dir = "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples"

# Process the files in parallel (using 20 processes)
process_files_parallel("filelists/pionsPU200_newalgo.txt", bg_folder, tree_name, output_dir, num_processes=30)