import uproot
import numpy as np
import awkward as ak
import pandas as pd
from multiprocessing import Pool
import analysis as ana
import os
import argparse as arg

algos = {
    "Ref":   "cl3d_Ref",
    "p0113": "cl3d_p0113Tri",
    "p016":  "cl3d_p016Tri",
    "p03":   "cl3d_p03Tri",
    "p045":  "cl3d_p045Tri",
}
cl3d_columns = [
    "energy", "eta", "pt", "phi","showerlength", "coreshowerlength",
    "firstlayer", "maxlayer", "emaxe", "varrr", "varee", "varzz", "varpp",
    "hoe", "meanz", "eot", "first1layers", "first3layers", "first5layers",
    "firstHcal1layers", "firstHcal3layers", "firstHcal5layers", "last1layers", "last3layers", "last5layers",
    "emax1layers", "emax3layers", "emax5layers", "ebm0", "ebm1", "hbm",
]
cl3d_hw_columns = [
    "hw_e", "hw_e_em", "hw_eta", "hw_phi", "hw_showerLength", "hw_coreShowerLength",
    "hw_firstLayer", "hw_lastLayer", "hw_sigma_roz", "hw_sigma_eta", "hw_sigma_z", "hw_sigma_phi",
    "hw_hoe", "hw_z", "hw_fractionInCE_E", "hw_fractionInCoreCE_E", "hw_fractionInEarlyCE_E", '_hw_nTC',
]
Sample_cfg = {
    "photonPU200": {
        "tag": "newalgo",
        "eta_range": (1.6, 2.8),
        "gen": {"pt_min": 20.0,
                "pt_max": 100.0},
        "genpart": {
            "reachedEE": 2,
            "gen_not": -1,
            "sign_match_to_gen": True,
            "exeta_in_gen_eta_window": True,  # apply same eta window on genpart_exeta
        },
        "cl3d": {
            "pt_min": 5.0,
            "pt_max": 100.0,
            "eta_in_window": True,
        },
    },
    "qcdPU200":  {
        "tag": "newalgo",
        "eta_range": (1.6, 2.8),
        "gen": {
            "pt_min": 20.0,
            "pt_max": 100.0,
            "status": 1,
            "pdgid": 22, 
        },
        "genpart": {
            "reachedEE": 2,
            "gen_not": -1,
            "sign_match_to_gen": True,
            "exeta_in_gen_eta_window": True,  # apply same eta window on genpart_exeta
        },
        "cl3d": {
            "pt_min": 5.0,
            "pt_max": 100.0,
            "eta_in_window": True,
        },
    },
    "pionPU200": {
        "tag": "newalgo",
        "eta_range": (1.6, 2.8),
        "gen": {
            "pt_min": 20.0,
            "pt_max": 100.0,
            "status": 1,
            #"pdg_id": 22,
        },
        "genpart": {
            "reachedEE": 2,
            "gen_not": -1,
            "sign_match_to_gen": True,
            "exeta_in_gen_eta_window": True,  # apply same eta window on genpart_exeta
        },
        "cl3d": {
            "pt_min": 5.0,
            "pt_max": 100.0,
            "eta_in_window": True,
        },
    },
    "PU200": {
        "tag": "newalgo",
        "eta_range": (1.6, 2.8),
        "cl3d": {
            "pt_min": 5.0,
            "pt_max": 100.0,
            "eta_in_window": True,
        },
    },
}

def _branches_for_prefix(prefix: str) -> list[str]:
    return [f"{prefix}_{s}" for s in (cl3d_columns + cl3d_hw_columns)] + ["event"]

def load_gen_dfs(tree) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_gen = ak.to_dataframe(tree.arrays(library="ak", filter_name=["gen_n", "gen_eta", "gen_phi", "gen_pt", "gen_energy", "gen_status", "gen_pdgid", "event"]))
    df_genpart = ak.to_dataframe(tree.arrays(library="ak", filter_name=["genpart_exeta", "genpart_exphi", "genpart_gen", "genpart_pid", "genpart_reachedEE", "event"]))
    return df_gen, df_genpart

def in_endcap_window(eta, eta0=1.6, eta1=2.8):
    a = np.abs(eta.astype(float))
    return (a > eta0) & (a < eta1)

def apply_gen_cuts(df_gen: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    eta0, eta1 = cfg["eta_range"]
    gen_cfg = cfg.get("gen", {})
    m = (df_gen["gen_pt"] > gen_cfg.get("pt_min", -1e9)) & (df_gen["gen_pt"] < gen_cfg.get("pt_max", 1e18)) & in_endcap_window(df_gen["gen_eta"], eta0, eta1)
    if "pdgid" in gen_cfg:
        m &= (df_gen["gen_pdgid"] == gen_cfg["pdgid"])
    if "status" in gen_cfg:
        m &= (df_gen["gen_status"] == gen_cfg["status"])
    return df_gen[m]

def apply_genpart_cuts(df_genpart: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    eta0, eta1 = cfg["eta_range"]
    gp_cfg = cfg.get("genpart", {})
    m = pd.Series(True, index=df_genpart.index)
    if "reachedEE" in gp_cfg:
        m &= (df_genpart["genpart_reachedEE"] == gp_cfg["reachedEE"])
    if "gen_not" in gp_cfg:
        m &= (df_genpart["genpart_gen"] != gp_cfg["gen_not"])
    if gp_cfg.get("exeta_in_gen_eta_window", False):
        m &= in_endcap_window(df_genpart["genpart_exeta"], eta0, eta1)
    return df_genpart[m]

def load_and_filter(tree, sample_key: str):
    cfg = Sample_cfg[sample_key]
    eta0, eta1 = cfg["eta_range"]
    has_gen = ("gen" in cfg) or ("genpart" in cfg)
    if has_gen:
        df_gen, df_genpart = load_gen_dfs(tree)
        df_gen_f = apply_gen_cuts(df_gen, cfg)
        df_genpart_f = apply_genpart_cuts(df_genpart, cfg)
        df_gen_merged = ana.load_and_filter_hdf(df_gen_f, df_genpart_f)
        # +EE ↔ +EE and -EE ↔ -EE matching 
        gp_cfg = cfg.get("genpart", {})
        if gp_cfg.get("sign_match_to_gen", True):
            # require both etas with same sign
            m_sign = (((df_gen_merged["gen_eta"] * df_gen_merged["genpart_exeta"]) > 0.0))
            df_gen_merged = df_gen_merged[m_sign]
    else:
        #For PU200: no gen info needed
        df_gen_merged = pd.DataFrame()
    
    cl_cfg = cfg.get("cl3d", {})
    pt_min = cl_cfg.get("pt_min", -1e9)
    pt_max = cl_cfg.get("pt_max",  1e18)
    out_cl = {}
    for algo_name, prefix in algos.items():
        branches = _branches_for_prefix(prefix)
        df_cl = ak.to_dataframe(tree.arrays(library="ak", filter_name=branches))
        pt_col  = f"{prefix}_pt"
        eta_col = f"{prefix}_eta"
        m = (df_cl[pt_col] > pt_min)
        if cl_cfg.get("eta_in_window", True):
            m &= in_endcap_window(df_cl[eta_col], eta0, eta1)
        out_cl[algo_name] = df_cl[m]
    return df_gen_merged, out_cl

def process_file(file_info):
    file_path, bg_folder, tree_name, sample_key = file_info
    try:
        with uproot.open(file_path) as root_file:
            tree = root_file[f"{bg_folder}/{tree_name}"]
            df_gen, cl3d_dict = load_and_filter(tree, sample_key)
        print(f"Processed {file_path}")
        return df_gen, cl3d_dict
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        # return empty shells with correct structure
        return pd.DataFrame(), {k: pd.DataFrame() for k in algos.keys()}

def process_files_parallel(filelist_path, bg_folder, tree_name, output_dir, sample_key, num_processes=20):
    os.makedirs(output_dir, exist_ok=True)
    cfg = Sample_cfg[sample_key]
    tag = cfg.get("tag", "filtered")
    with open(filelist_path, "r") as f:
        file_list = [line.strip() for line in f if line.strip()]
    file_info_list = [(fp, bg_folder, tree_name, sample_key) for fp in file_list]
    with Pool(processes=num_processes) as pool:
        results = pool.map(process_file, file_info_list)

    # combine gen
    gen_dfs = [r[0] for r in results]
    combined_gen_df = pd.concat(gen_dfs, ignore_index=True) if gen_dfs else pd.DataFrame()

    # combine cl3d per algo
    combined_cl = {}
    for algo in algos.keys():
        algo_dfs = [r[1][algo] for r in results]
        combined_cl[algo] = pd.concat(algo_dfs, ignore_index=True)

    # save
    base = f"{sample_key}_{tag}"
    gen_output_path = os.path.join(output_dir, f"{base}_gen.h5")
    combined_gen_df.to_hdf(gen_output_path, key="gen", mode="w") if gen_dfs else pd.DataFrame()

    for algo, df in combined_cl.items():
        out_path = os.path.join(output_dir, f"{base}_cl3d_{algo}.h5")
        df.to_hdf(out_path, key=f"cl3d_{algo}", mode="w")

    print("Saved:")
    print("  ", gen_output_path)
    for algo in algos.keys():
        print("  ", os.path.join(output_dir, f"{base}_cl3d_{algo}.h5"))

def main():
    parser = arg.ArgumentParser(description="Filter HGCAL ntuples and dump to HDF5.")
    parser.add_argument("--sample", "-s", required=True, choices=list(Sample_cfg.keys()), help="Sample key to process (must exist in Sample_cfg).")
    parser.add_argument("--filelist", "-f", default='filelists/PU200_newalgo.txt', help="Path to filelist txt (one ROOT file path per line).")
    parser.add_argument("--bg-folder", default="l1tHGCalTriggerNtuplizer")
    parser.add_argument("--tree-name", default="HGCalTriggerNtuple")
    parser.add_argument("--outdir", "-o", default="/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples", help="Output directory for HDF5 files.")
    parser.add_argument("--nproc", "-j", type=int, default=20, help="Number of worker processes.")
    args = parser.parse_args()
    filelist = args.filelist
    print("\nConfig:")
    print("  sample   =", args.sample)
    print("  filelist =", filelist)
    print("  outdir   =", args.outdir)
    print("  nproc    =", args.nproc)
    process_files_parallel(filelist_path=filelist, bg_folder=args.bg_folder, tree_name=args.tree_name, output_dir=args.outdir, sample_key=args.sample, num_processes=args.nproc)

if __name__ == "__main__":
    main()