import uproot
import awkward as ak
import pandas as pd
import numpy as np

# Filepath to the list of .root files
filelist_path = "filelist.txt"

# Function to load and filter the tree data
def load_and_filter_tree(tree):
    df = ak.to_dataframe(tree.arrays(
        library="ak",
        filter_name=["gen_n", "gen_eta", "gen_phi", "gen_pt", 
                     "genpart_exeta", "genpart_exphi", "*cl3d*", "event"]
    ))
    df_filtered = df[(np.abs(df['cl3d_eta']) > 1.6) & 
                     (np.abs(df['cl3d_eta']) < 2.8) &
                     (df['cl3d_pt'] > 5)]
    return df_filtered

# Function to process the files and save as a single .h5 file
def process_files(filelist_path, bg_folder, tree_name, output_file):
    # Initialize an empty list to accumulate all filtered data
    all_filtered_data = []

    # Read the file list
    with open(filelist_path, "r") as f:
        file_list = f.readlines()

    for file_path in file_list:
        file_path = file_path.strip()  # Clean up any whitespace/newlines

        # Open the ROOT file
        root_file = uproot.open(file_path)
        bg_tree_path = f"{bg_folder}/{tree_name}"
        bg_tree = root_file[bg_tree_path]
        bg_df_filtered = load_and_filter_tree(bg_tree)

        # Add the filtered data to the accumulated list
        all_filtered_data.append(bg_df_filtered)

        print(f"Processed {file_path}")

    # Concatenate all the filtered dataframes into a single dataframe
    final_df = pd.concat(all_filtered_data, ignore_index=True)

    # Save the combined DataFrame to a single .h5 file
    final_df.to_hdf(output_file, key='data', mode='w')

    print(f"All data saved to {output_file}")

# Set the paths
bg_folder = "l1tHGCalTriggerNtuplizer"
tree_name = "HGCalTriggerNtuple"
output_file = "/grid_mnt/data__data.polcms/cms/debnath/CMSSW_14_0_0_pre1/src/shower_shape_studies/combined_data.h5"  # Output single .h5 file

# Process the files listed in filelist.txt and save everything into one .h5 file
process_files(filelist_path, bg_folder, tree_name, output_file)
