import uproot
import awkward as ak
import pandas as pd

# File paths for signal and background files
bg_file_path = "/grid_mnt/data__data.polcms/cms/debnath/CMSSW_14_0_0_pre1/src/neutrino_bg.h5"  # Background data (already filtered)
bg_file = pd.read_hdf(bg_file_path)  # Background data (already filtered)
bg_folder = "l1tHGCalTriggerNtuplizer"  # This won't be used since we already have the background data in the form of a DataFrame

# Tree name to access within each file
tree_name = "HGCalTriggerNtuple"

# Build paths to the trees for ROOT file
signal_tree_path = f"{signal_folder}/{tree_name}"

# Access the signal tree in ROOT (uproot)
signal_tree = signal_file[signal_tree_path]

# Function to load and filter the signal tree data (for ROOT file)
def load_and_filter_signal_tree(tree, filter_pt, eta_range=(1.6, 2.8), cl_pt_threshold=5):
    df = ak.to_dataframe(tree.arrays(
        library="ak",
        filter_name=["gen_n", "gen_eta", "gen_phi", "gen_pt", 
                     "genpart_exeta", "genpart_exphi", "*cl3d*", "event"]
    ))
    df_filtered = df[(df['gen_pt'] > filter_pt) & 
                     (df['gen_eta'] > eta_range[0]) & 
                     (df['gen_eta'] < eta_range[1]) &
                     (df['cl3d_eta'] > eta_range[0]) & 
                     (df['cl3d_eta'] < eta_range[1]) &
                     (df['cl3d_pt'] > cl_pt_threshold)]
    return df_filtered

# Apply the filter to the signal data (only signal data needs filtering)
signal_df_filtered = load_and_filter_signal_tree(signal_tree, filter_pt=20)

# Since the background data is already filtered, no additional filtering is needed for it.
bg_df_filtered = bg_file  # Assuming the background DataFrame is already filtered

# Concatenate signal and background data into one DataFrame
filtered_data = pd.concat([signal_df_filtered, bg_df_filtered])

# Save the filtered data to a .h5 file
output_file = "combined_filtered_data.h5"
filtered_data.to_hdf(output_file, key='data', mode='w')

print(f"Filtered data saved to {output_file}")
