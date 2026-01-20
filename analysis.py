import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak
import seaborn as sns
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_sample_weight
import shutil
import datetime
from sklearn.datasets import make_hastie_10_2
from pandas.api.types import is_integer_dtype
import logging
import sys
import mplhep as mh

def delta_r(eta1, phi1, eta2, phi2):
    delta_eta = np.abs(eta1 - eta2)
    delta_phi = np.abs(phi1 - phi2)
    delta_phi = np.where(delta_phi > np.pi, 2 * np.pi - delta_phi, delta_phi)  # Adjust phi to be within [-pi, pi]
    return np.sqrt(delta_eta**2 + delta_phi**2)

def filtering(df, prefix, thr):
    df=df.copy()
    df['delta_r'] = delta_r(df[rf"cl3d_{prefix}_eta"], df[rf"cl3d_{prefix}_phi"], df['gen_eta'], df['gen_phi'])
    df = df[df['delta_r'] <=thr]
    df = (df.sort_values(by=["event", f"cl3d_{prefix}_energy"],ascending=[True, False] ))  # event fixed, energy high → low).reset_index(drop=True
    eta_cl = df[rf"cl3d_{prefix}_eta"]
    eta_ref = df[rf"gen_eta"]   # change if needed
    mask = (eta_cl * eta_ref) > 0   # same sign (both + or both -)
    df = df.loc[mask].reset_index(drop=True)
    df = df.sort_values(["event", "genpart_gen", f'cl3d_{prefix}_energy'], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["event", "genpart_gen"], keep="first")
    df = (df.sort_values(["event", f'cl3d_{prefix}_energy'], ascending=[True, False]).groupby("event", as_index=False).head(2).reset_index(drop=True))
    return df

def filtering_photon(df, prefix, thr):
    required_columns = [f"cl3d_{prefix}_eta", f"cl3d_{prefix}_phi", "genpart_exeta", "genpart_exphi", f"cl3d_{prefix}_energy", "event", "gen_pt",]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    df = df.copy()
    df['delta_r'] = delta_r(df[rf"cl3d_{prefix}_eta"], df[rf"cl3d_{prefix}_phi"], df['genpart_exeta'], df['genpart_exphi'])
    m_sign = (df[f"cl3d_{prefix}_eta"] * df["genpart_exeta"]) > 0.0
    df = df[m_sign]
    # build per-row ΔR threshold
    if prefix in ["Ref", "p0113Tri", "p016Tri"]:
        dr_thr = thr
        m_dr = df["delta_r"] < dr_thr
    elif prefix in [ "p03Tri", "p045Tri"]:
        # gen-eta dependent
        dr_thr_arr = np.where(df["genpart_exeta"].to_numpy() > 2.4, 0.1, 0.20)
        m_dr = df["delta_r"].to_numpy() < dr_thr_arr
    else:
        raise ValueError(f"Unknown prefix '{prefix}' for ΔR threshold rules")
    df_filtered = df[m_dr]
    df = df.sort_values(["event", "genpart_gen", f'cl3d_{prefix}_energy'], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["event", "genpart_gen"], keep="first")
    df = (df.sort_values(["event", f'cl3d_{prefix}_energy'], ascending=[True, False]).groupby("event", as_index=False).head(2).reset_index(drop=True))
    return df

# Function to load and filter the tree data (for ROOT file)
def load_and_filter_tree(tree, filter_pt = 20, eta_range=(1.6, 2.8), cl_pt_threshold=5):
    df_gen = ak.to_dataframe(tree.arrays(
        library="ak",
        filter_name=["gen_n", "gen_eta", "gen_phi", "gen_pt", 
                     "genpart_exeta", "genpart_exphi", "event"]
    ))
    df_cl3d = ak.to_dataframe(tree.arrays(
        library="ak",
        filter_name=["*cl3d*", "event"]
    ))
    df_gen_filtered = df_gen[(df_gen['gen_pt'] > filter_pt) & 
                     (abs(df_gen['gen_eta']) > eta_range[0]) & 
                     (abs(df_gen['gen_eta']) < eta_range[1])]
    df_cl3d_filtered = df_cl3d[(abs(df_cl3d['cl3d_eta']) > eta_range[0]) & 
                     (abs(df_cl3d['cl3d_eta']) < eta_range[1]) &
                     (df_cl3d['cl3d_pt'] > cl_pt_threshold)]
    merged_df = pd.merge(
        df_gen_filtered,
        df_cl3d_filtered,
        on="event",
        how="inner",  # Keep only rows where the event ID exists in both
        suffixes=('_gen', '_cl3d')  # Differentiate common column names
    )
    return merged_df

# Function to load and filter the tree data (for HDF file)
def load_and_filter_hdf(df1, df2):
    merged_df = pd.merge(
        df1,
        df2,
        on="event",
        how="inner",  # Keep only rows where the event ID exists in both
        #suffixes=('_gen', '_cl3d')  # Differentiate common column names
    )
    return merged_df
def load_and_filter_hdf_path(df_gen_path, df_cl3d_path):
    if df_gen_path == 'None':
        df_cl3d = pd.read_hdf(df_cl3d_path)
        return df_cl3d 
 
    else:
        df_gen = pd.read_hdf(df_gen_path)
        df_cl3d = pd.read_hdf(df_cl3d_path)
        merged_df = pd.merge(
            df_gen,
            df_cl3d,
            on="event",
            how="inner",  # Keep only rows where the event ID exists in both
            #suffixes=('_gen', '_cl3d')  # Differentiate common column names
        )
        return merged_df

def calculate_partial_auc(y_true, y_pred_probs, tpr_min, tpr_max):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    valid_idx = (tpr >= tpr_min) & (tpr <= tpr_max)
    partial_auc = auc(fpr[valid_idx], tpr[valid_idx])
    return partial_auc

def plot_roc_curve(y_true, y_pred_probs, threshold, plots_dir):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='b', label='ROC curve (AUC = {:.2f})'.format(auc(fpr, tpr)))
    valid_idx = tpr > threshold
    plt.fill_between(fpr[valid_idx], tpr[valid_idx], color='orange', alpha=0.3, label=f'Signal efficiency > {threshold}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Diagonal line (no discrimination)
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.yscale('log')
    plt.xscale('log')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    filename = os.path.join(plots_dir,f"roc_curve.png")
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.show()
    plt.close()

def plot_delta_r_3d_two_dfs(df, label, plots_dir, eta_col='cl3d_eta', energy_col='cl3d_energy', delta_r_col='delta_r', 
                            colors=('blue', 'red'), cmap = 'plasma', figsize=(12, 8)):
    # Ensure required columns are present
    required_columns = [eta_col, delta_r_col, energy_col]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    
    # Create the plot
    plt.figure(figsize=figsize)
    scatter = plt.scatter(
        df[eta_col], df[delta_r_col], c=df[energy_col], cmap=cmap, alpha=0.7, edgecolor='k', s=20
    )
    cbar = plt.colorbar(scatter, label=f"Energy [GeV]")
    # Set axis labels and title
    plt.xlabel("$\eta$ [rad]")
    plt.ylabel("$\Delta$ R [rad]")
    plt.title(f"$\Delta$ R vs $\eta$ for {label}")
    plt.tight_layout()
    filename = os.path.join(plots_dir,f"delta_r_vs_etaenergy_{label.replace(' ', '_')}.png")
    plt.savefig(filename, dpi=300)
    print(f"Saved plot as: {filename}")
    plt.show()

def plot_histograms(df_signal, df_bg1, df_bg2, df_bg3, variables, label_signal, label_bg1, label_bg2, label_bg3, plots_dir, var_latex_map, prefix, num_bins=40, cl3d_pt_range=(20, 200), figsize=(8, 4)):
    df_signal_filtered = df_signal[(df_signal[f'cl3d_{prefix}_pt'] >= cl3d_pt_range[0]) & (df_signal[f'cl3d_{prefix}_pt'] <= cl3d_pt_range[1])]
    df_bg1_filtered = df_bg1[(df_bg1[f'cl3d_{prefix}_pt'] >= cl3d_pt_range[0]) & (df_bg1[f'cl3d_{prefix}_pt'] <= cl3d_pt_range[1])]
    df_bg2_filtered = df_bg2[(df_bg2[f'cl3d_{prefix}_pt'] >= cl3d_pt_range[0]) & (df_bg2[f'cl3d_{prefix}_pt'] <= cl3d_pt_range[1])]
    df_bg3_filtered = df_bg3[(df_bg3[f'cl3d_{prefix}_pt'] >= cl3d_pt_range[0]) & (df_bg3[f'cl3d_{prefix}_pt'] <= cl3d_pt_range[1])]
    for var in variables:
        plt.figure(figsize=figsize)
        if df_signal_filtered[var].dtype in ['int64', 'int32']:
            min_value = min(df_signal_filtered[var].min(), df_bg1_filtered[var].min(), df_bg2_filtered[var].min(), df_bg3_filtered[var].min())
            max_value = max(df_signal_filtered[var].max(), df_bg1_filtered[var].max(), df_bg2_filtered[var].max(), df_bg3_filtered[var].max())
            bin_edges = np.arange(min_value - 0.5, max_value + 1.5, 1)
        else:
            min_value = min(df_signal_filtered[var].min(), df_bg1_filtered[var].min(), df_bg2_filtered[var].min(), df_bg3_filtered[var].min())
            max_value = max(df_signal_filtered[var].max(), df_bg1_filtered[var].max(), df_bg2_filtered[var].max(), df_bg3_filtered[var].max())
            bin_width = (max_value - min_value) / num_bins
            bin_edges = np.arange(min_value - bin_width / 2, max_value + bin_width / 2, bin_width)
        plt.hist(df_signal_filtered[var], histtype='step', bins=bin_edges, color='b', linewidth=1.5, label=label_signal, density=True)
        plt.hist(df_bg1_filtered[var], histtype='step', bins=bin_edges, color='g', linewidth=1.5, label=label_bg1, density=True)
        plt.hist(df_bg2_filtered[var], histtype='step', bins=bin_edges, color='r', linewidth=1.5, label=label_bg2, density=True)
        plt.hist(df_bg3_filtered[var], histtype='step', bins=bin_edges, color='o', linewidth=1.5, label=label_bg3, density=True)
        plt.title("Cluster " + f"{var_latex_map.get(var, var)} Histogram", fontsize=14)
        plt.xlabel(var_latex_map.get(var, var), fontsize=12)
        plt.ylabel('Normalized Frequency', fontsize=12)
        plt.legend()
        plt.tight_layout()
        fig = plt.gcf()
        fig.text(0.01, 0.98, r"$\bf{CMS}$  $\it{Simulation}$", ha="left", va="top", fontsize=15)
        fig.text(0.98, 0.98, "14 TeV",                 ha="right", va="top", fontsize=14)
        plt.subplots_adjust(top=0.90)
        filename = os.path.join(plots_dir, f"{var}_histogram.png")
        plt.savefig(filename, dpi=300)
        print(f"Saved: {filename}")
        plt.show()
        plt.close()

def plot_qcd_histograms(df_signal, df_bg1, df_bg2, df_bg3, variables,num_bins=40, cl3d_pt_range=(0, 120), figsize=(8, 4)):
    df_signal_filtered = df_signal[(df_signal['cl3d_pt'] >= cl3d_pt_range[0]) & (df_signal['cl3d_pt'] <= cl3d_pt_range[1])]
    df_bg1_filtered = df_bg1[(df_bg1['cl3d_pt'] >= cl3d_pt_range[0]) & (df_bg1['cl3d_pt'] <= cl3d_pt_range[1])]
    df_bg2_filtered = df_bg2[(df_bg2['cl3d_pt'] >= cl3d_pt_range[0]) & (df_bg2['cl3d_pt'] <= cl3d_pt_range[1])]
    df_bg3_filtered = df_bg3[(df_bg3['cl3d_pt'] >= cl3d_pt_range[0]) & (df_bg3['cl3d_pt'] <= cl3d_pt_range[1])]

    for var in variables:
        plt.figure(figsize=figsize)

        # Determine bin edges based on data type
        min_value = min(
            df_signal_filtered[var].min(),
            df_bg1_filtered[var].min(),
            df_bg2_filtered[var].min(),
            df_bg3_filtered[var].min()
        )
        max_value = max(
            df_signal_filtered[var].max(),
            df_bg1_filtered[var].max(),
            df_bg2_filtered[var].max(),
            df_bg3_filtered[var].max()
        )

        if df_signal_filtered[var].dtype in ['int64', 'int32']:
            bin_edges = np.arange(min_value - 0.5, max_value + 1.5, 1)
        else:
            bin_width = (max_value - min_value) / num_bins
            bin_edges = np.arange(min_value - bin_width / 2, max_value + bin_width / 2, bin_width)

        # Plot all four histograms
        plt.hist(df_signal_filtered[var], histtype='step', bins=bin_edges, color='b', linewidth=1.5, label='QCD_pt20to30', density=True)
        plt.hist(df_bg1_filtered[var], histtype='step', bins=bin_edges, color='g', linewidth=1.5, label='QCD_pt30to50', density=True)
        plt.hist(df_bg2_filtered[var], histtype='step', bins=bin_edges, color='r', linewidth=1.5, label='QCD_pt50to80', density=True)
        plt.hist(df_bg3_filtered[var], histtype='step', bins=bin_edges, color='m', linewidth=1.5, label='QCD_pt80to120', density=True)

        # Add labels and title
        plt.title("Cluster " + f"{var_latex_map.get(var, var)} Histogram", fontsize=14)
        plt.xlabel(var_latex_map.get(var, var), fontsize=12)
        plt.ylabel('Normalized Frequency', fontsize=12)
        plt.legend()
        plt.tight_layout()

        filename = os.path.join(plots_dir, f"{var}_histogram_cl3d_pt_{cl3d_pt_range[0]}_{cl3d_pt_range[1]}.png")
        plt.savefig(filename, dpi=300)
        print(f"Saved: {filename}")
        plt.show()
        plt.close()

def quantize(feat, nbits, method, fmin, fmax):
    nbins = 2 ** nbits
    if method == 'uniform':
        bins = np.linspace(fmin, fmax, nbins + 1)
    elif method == 'percentile':
        bins = [np.percentile(feat, p) for p in np.linspace(0, 100, nbins + 1)]
    else:
        raise ValueError("Unsupported quantization method")
    return np.digitize(feat, bins, right=True)

def remove_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def train_quantized_multiclass(precision, depth, rounds, iteration, X_train, y_train, X_test, y_test):
    timer = time.time()
    print(f"[{iteration}] Training model: precision={precision}, depth={depth}, rounds={rounds}")

    # Step 1: Quantization
    qtrain = pd.DataFrame()
    qtest = pd.DataFrame()
    for feat in X_train.columns:
        fmin, fmax = X_train[feat].min(), X_train[feat].max()
        qtrain[feat] = quantize(X_train[feat], precision, 'uniform', fmin, fmax)
        qtest[feat] = quantize(X_test[feat], precision, 'uniform', fmin, fmax)

    # Step 2: Normalize
    max_range = 1 - 1 / (2 ** precision)
    scaler = MinMaxScaler(feature_range=(0, max_range))
    qtrain_scaled = pd.DataFrame(scaler.fit_transform(qtrain), columns=X_train.columns)
    qtest_scaled = pd.DataFrame(scaler.transform(qtest), columns=X_test.columns)

    # Step 3: XGBoost Training
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=4,
        max_depth=depth,
        n_estimators=rounds,
        learning_rate=0.05,
        eval_metric='mlogloss',
        use_label_encoder=False,
        n_jobs=8,
        verbosity=0
    )
    model.fit(qtrain_scaled, y_train)

    pred_probs = model.predict_proba(qtest_scaled)
    y_pred = np.argmax(pred_probs, axis=1)

    acc = (y_pred == y_test).sum() / len(y_test)
    macro_auc = roc_auc_score(y_test, pred_probs, multi_class='ovo')

    # Step 4: Conifer Config
    cfg = conifer.backends.vhdl.auto_config()
    path = f"hdlprojects/prj_vhdl_multiclass_{precision}_{depth}_{rounds}_{iteration}"
    if os.path.exists(path):
        remove_folder(path)
    cfg['OutputDir'] = path
    cfg['XilinxPart'] = 'xcvu13p-fhgb2104-2L-e'
    cfg['Precision'] = f"ap_fixed<{precision},0>"
    cfg['ClockPeriod'] = 3
    cfg['ProjectName'] = 'hgcal_multiclass'

    # Step 5: Conifer Conversion & Synthesis
    cnf_model = conifer.model(model.get_booster(), conifer.converters.xgboost,
                              conifer.backends.vhdl, cfg)
    cnf_model.compile()
    y_hdl = expit(cnf_model.decision_function(qtest_scaled))
    cnf_model.build(csim=True)

    # LUT Extraction
    report_path = os.path.join(cfg['OutputDir'], 'util.rpt')
    with open(report_path, 'r') as f:
        lines = f.readlines()
        LUT = int(lines[37].split('|')[2])

    duration = time.time() - timer
    print(f"Finished Iter {iteration}: Accuracy={acc:.4f}, AUC={macro_auc:.4f}, LUT={LUT}, Time={duration:.2f}s")

    os.chdir(base_dir)
    return (precision, depth, rounds, acc, macro_auc, LUT)


def variables_to_plot(prefix):
    variables = [f'cl3d_{prefix}_pt', f'cl3d_{prefix}_energy', f'cl3d_{prefix}_eta', f'cl3d_{prefix}_phi', 
       f'cl3d_{prefix}_emax1layers', f'cl3d_{prefix}_emax3layers', f'cl3d_{prefix}_showerlength', f'cl3d_{prefix}_coreshowerlength', 
       f'cl3d_{prefix}_firstlayer', f'cl3d_{prefix}_maxlayer', f'cl3d_{prefix}_varrr', f'cl3d_{prefix}_varzz', f'cl3d_{prefix}_varee', 
       f'cl3d_{prefix}_varpp', f'cl3d_{prefix}_emaxe', f'cl3d_{prefix}_hoe', f'cl3d_{prefix}_meanz', 
       f'cl3d_{prefix}_first1layers', f'cl3d_{prefix}_first3layers', f'cl3d_{prefix}_first5layers', 
       f'cl3d_{prefix}_firstHcal1layers', f'cl3d_{prefix}_firstHcal3layers',
       f'cl3d_{prefix}_firstHcal5layers', f'cl3d_{prefix}_last1layers', f'cl3d_{prefix}_last3layers',
       f'cl3d_{prefix}_last5layers', f'cl3d_{prefix}_eot', f'cl3d_{prefix}_ebm0', f'cl3d_{prefix}_ebm1']#, 'cl3d_Ref_hbm']
    return variables

def columns_for_training(prefix):
    columns = [
     f'cl3d_{prefix}_showerlength', f'cl3d_{prefix}_coreshowerlength', f'cl3d_{prefix}_firstlayer', 
     f'cl3d_{prefix}_eot', f'cl3d_{prefix}_firstHcal5layers', f'cl3d_{prefix}_first5layers', 
     f'cl3d_{prefix}_varrr', f'cl3d_{prefix}_varzz', f'cl3d_{prefix}_varee', f'cl3d_{prefix}_varpp', f'cl3d_{prefix}_meanz', 
     f'cl3d_{prefix}_last5layers', f'cl3d_{prefix}_emax5layers', 
     f'cl3d_{prefix}_ebm0', f'cl3d_{prefix}_ebm1', f'cl3d_{prefix}_hbm'
]
    return columns

def var_map(prefix):
    var_latex_map = {
    f'cl3d_{prefix}_pt': r'$p_T$ [GeV]', 
    f'cl3d_{prefix}_energy': 'Energy [GeV]', 
    f'cl3d_{prefix}_eta': r'$\eta$',
    f'cl3d_{prefix}_phi': r'$\phi$',
    f'cl3d_{prefix}_emax1layers': 'Emax1layers',
    f'cl3d_{prefix}_emax3layers': 'Emax3layers',
    f'cl3d_{prefix}_emax5layers': 'Emax5layers',
    f'cl3d_{prefix}_showerlength': 'Shower Length',
    f'cl3d_{prefix}_coreshowerlength': 'Core Shower Length',
    f'cl3d_{prefix}_firstlayer': 'First Layer',
    f'cl3d_{prefix}_hoe': 'CE-H/CE-E',
    f'cl3d_{prefix}_varrr': '$\sigma^2_{rr}$',
    f'cl3d_{prefix}_varzz': '$\sigma^2_{zz}$',
    f'cl3d_{prefix}_varee': '$\sigma^2_{\eta\eta}$',
    f'cl3d_{prefix}_varpp': '$\sigma^2_{\phi\phi}$',
    f'cl3d_{prefix}_meanz': '<z>',
    f'cl3d_{prefix}_first1layers': 'First1layer',
    f'cl3d_{prefix}_first3layers': 'First3layers',
    f'cl3d_{prefix}_first5layers': 'First5layers',
    f'cl3d_{prefix}_firstHcal1layers': 'FirstHcal1layer',
    f'cl3d_{prefix}_firstHcal3layers': 'First Hcal3layers',
    f'cl3d_{prefix}_firstHcal5layers': 'First Hcal5layers',
    f'cl3d_{prefix}_last1layers': 'Last1layer',
    f'cl3d_{prefix}_last3layers': 'Last3layers',
    f'cl3d_{prefix}_last5layers': 'Last5layers',
    f'cl3d_{prefix}_ebm0' : 'EBM0', 
    f'cl3d_{prefix}_ebm1' : 'EBM1',
    f'cl3d_{prefix}_hbm' : 'HBM',
    f'cl3d_{prefix}_eot' : 'E/Total E'}
    return var_latex_map

def var_map_suffix() -> dict:
    """Latex labels keyed by the *suffix* (last token after the second underscore)."""
    return {
        "pt"               : r"$p_T$ [GeV]",
        "energy"           : "Energy [GeV]",
        "eta"              : r"$\eta$",
        "phi"              : r"$\phi$",
        "emax1layers"      : "Emax1layers",
        "emax3layers"      : "Emax3layers",
        "emax5layers"      : "Emax5layers",
        "showerlength"     : "Shower Length",
        "coreshowerlength" : "Core Shower Length",
        "firstlayer"       : "First Layer",
        "hoe"              : "CE-H/CE-E",
        "varrr"            : r"$\sigma^2_{rr}$",
        "varzz"            : r"$\sigma^2_{zz}$",
        "varee"            : r"$\sigma^2_{\eta\eta}$",
        "varpp"            : r"$\sigma^2_{\phi\phi}$",
        "meanz"            : r"$\langle z \rangle$",
        "first1layers"     : "First1layer",
        "first3layers"     : "First3layers",
        "first5layers"     : "First5layers",
        "firstHcal1layers" : "First Hcal1layer",
        "firstHcal3layers" : "First Hcal3layers",
        "firstHcal5layers" : "First Hcal5layers",
        "last1layers"      : "Last1layer",
        "last3layers"      : "Last3layers",
        "last5layers"      : "Last5layers",
        "ebm0"             : "EBM0",
        "ebm1"             : "EBM1",
        "hbm"              : "HBM",
        "eot"              : "CE-E/Total E",
    }

from pandas.api.types import is_integer_dtype
def plot_across_five_lists(
    df_ref, df_p0113, df_p016, df_p03, df_p045,
    vars_ref, vars_p0113, vars_p016, vars_p03, vars_p045,
    label_ref="Ref", label_p0113="p=0.113", label_p016="p=0.16", label_p03="p=0.30", label_p045="p=0.45",
    plots_dir="plots_triangles", var_latex_map=None, num_bins=40, cl3d_pt_range=(20, 200),
    pt_col_ref="cl3d_Ref_pt", pt_col_p0113="cl3d_p0113Tri_pt", pt_col_p016="cl3d_p016Tri_pt",
    pt_col_p03="cl3d_p03Tri_pt", pt_col_p045="cl3d_p045Tri_pt",
    density=False, logy=False, weight_cols=None
):
    os.makedirs(plots_dir, exist_ok=True)
    if var_latex_map is None:
        var_latex_map = {}
    suffix_labels = var_map_suffix()
    # Build maps: suffix -> full col name (for each DF)
    def suffix(name): 
        return name.split("_", maxsplit=2)[-1] if name.count("_")>=2 else name

    by_suffix = {"ref":{}, "p0113":{}, "p016":{}, "p03":{}, "p045":{}}
    for c in vars_ref:  by_suffix["ref"][suffix(c)]  = c
    for c in vars_p0113: by_suffix["p0113"][suffix(c)] = c
    for c in vars_p016: by_suffix["p016"][suffix(c)] = c
    for c in vars_p03:  by_suffix["p03"][suffix(c)]  = c
    for c in vars_p045: by_suffix["p045"][suffix(c)] = c

    # Union of all suffixes to try plotting
    all_suffixes = list(dict.fromkeys(
        list(by_suffix["ref"].keys()) +
        list(by_suffix["p0113"].keys()) +
        list(by_suffix["p016"].keys()) +
        list(by_suffix["p03"].keys()) +
        list(by_suffix["p045"].keys())
    ))

    # Helper to get (series, weights) after pT window for a given variant
    def select(df, col, pt_col, wcol):
        if (col not in df.columns) or (pt_col not in df.columns): 
            return pd.Series(dtype=float), None
        m = (df[pt_col] >= cl3d_pt_range[0]) & (df[pt_col] <= cl3d_pt_range[1])
        ser = df.loc[m, col].dropna()
        w = (df.loc[m, wcol] if (wcol and wcol in df.columns) else None)
        if w is not None: w = w.loc[ser.index]
        return ser, w

    # Iterate each suffix (i.e., each physics variable)
    for suf in all_suffixes:
        cols = {
            "ref":  by_suffix["ref"].get(suf,  None),
            "p0113": by_suffix["p0113"].get(suf, None),
            "p016": by_suffix["p016"].get(suf, None),
            "p03":  by_suffix["p03"].get(suf,  None),
            "p045": by_suffix["p045"].get(suf, None),
        }

        # Gather data
        s_ref,  w_ref  = select(df_ref,  cols["ref"],  pt_col_ref,  (weight_cols or {}).get("ref"))
        s_p0113, w_p0113 = select(df_p0113, cols["p0113"], pt_col_p0113, (weight_cols or {}).get("p0113"))
        s_p016, w_p016 = select(df_p016, cols["p016"], pt_col_p016, (weight_cols or {}).get("p016"))
        s_p03,  w_p03  = select(df_p03,  cols["p03"],  pt_col_p03,  (weight_cols or {}).get("p03"))
        s_p045, w_p045 = select(df_p045, cols["p045"], pt_col_p045, (weight_cols or {}).get("p045"))

        series_list = [s for s in [s_ref, s_p0113, s_p016, s_p03, s_p045] if not s.empty]
        if not series_list:
            print(f"[skip] No data for '{suf}' after pT filter.")
            continue

        # Binning: integer bins if all are integer-like, else uniform numeric bins
        mins = [s.min() for s in series_list]
        maxs = [s.max() for s in series_list]
        all_int = all(is_integer_dtype(s) for s in series_list)
        gmin, gmax = float(np.min(mins)), float(np.max(maxs))
        if all_int and np.isfinite(gmin) and np.isfinite(gmax):
            bin_edges = np.arange(np.floor(gmin)-0.5, np.ceil(gmax)+1.5, 1.0)
        else:
            if gmin == gmax:
                gmin -= 0.5; gmax += 0.5
            bw = (gmax - gmin) / float(num_bins)
            bin_edges = np.arange(gmin - bw/2, gmax + bw/2 + 1e-12, bw)

        # Plot
        plt.figure(figsize=(8,4))
        if not s_ref.empty:
            plt.hist(s_ref.values,  bins=bin_edges, histtype="step", label=label_ref,  density=density, weights=(w_ref.values if w_ref is not None else None))
        if not s_p0113.empty:
            plt.hist(s_p0113.values, bins=bin_edges, histtype="step", label=label_p0113, density=density, weights=(w_p0113.values if w_p0113 is not None else None))
        if not s_p016.empty:
            plt.hist(s_p016.values, bins=bin_edges, histtype="step", label=label_p016, density=density, weights=(w_p016.values if w_p016 is not None else None))
        if not s_p03.empty:
            plt.hist(s_p03.values,  bins=bin_edges, histtype="step", label=label_p03,  density=density, weights=(w_p03.values if w_p03 is not None else None))
        if not s_p045.empty:
            plt.hist(s_p045.values, bins=bin_edges, histtype="step", label=label_p045, density=density, weights=(w_p045.values if w_p045 is not None else None))

        # Labels
        sample_full = next((c for c in [cols["ref"], cols["p0113"], cols["p016"], cols["p03"], cols["p045"]] if c is not None), None)
        suf_label   = suffix_labels.get(suf)  # e.g. 'pt' -> '$p_T$ [GeV]'
        x_label     = suf_label or (var_latex_map or {}).get(sample_full, suf)
        plt.xlabel(x_label, fontsize=12)
        plt.yscale('log')
        plt.ylabel("Normalized Frequency" if density else "Entries", fontsize=12)
        plt.legend()
        plt.tight_layout()
        fig = plt.gcf()
        fig.text(0.01, 0.98, r"$\bf{CMS}$  $\it{Simulation}$", ha="left", va="top", fontsize=15)
        fig.text(0.98, 0.98, "14 TeV",                 ha="right", va="top", fontsize=14)
        plt.subplots_adjust(top=0.90)
        out = os.path.join(plots_dir, f"{suf}_across_triangles.pdf")
        plt.savefig(out)
        print(f"Saved: {out}")
        plt.show()
        plt.close()
def curves_eff_vs_dr_in_2d_bins(
    df,
    prefix,
    eta_bins,
    pt_bins,
    dr_vals,
    use_abs_eta=True,
    gen_eta_col="gen_eta",
    gen_pt_col="gen_pt",
    outpath="eff_vs_deltaR_by_genEta_genPt.png",
    cms_label="Preliminary",
    com=14,
    legend_ncol=2,
    legend_fontsize=12,
):
    """
    Makes ONE plot: efficiency vs ΔR curves.
    Each curve corresponds to a 2D bin: (eta_bin, pt_bin).
    Efficiency = N_pass / N_total using unique (event, genpart_gen).
    """
    # storage: eff_map[ieta, ipt, idr]
    eff_map = np.full((len(eta_bins)-1, len(pt_bins)-1, len(dr_vals)), np.nan, dtype=float)
    n0_map  = np.zeros((len(eta_bins)-1, len(pt_bins)-1), dtype=int)

    # precompute arrays
    eta = df[gen_eta_col].to_numpy()
    pt  = df[gen_pt_col].to_numpy()
    eta_sel = np.abs(eta) if use_abs_eta else eta

    for ieta in range(len(eta_bins)-1):
        elo, ehi = eta_bins[ieta], eta_bins[ieta+1]

        m_eta = (eta_sel >= elo) & (eta_sel < ehi)

        for ipt in range(len(pt_bins)-1):
            plo, phi = pt_bins[ipt], pt_bins[ipt+1]

            m_pt = (pt >= plo) & (pt < phi)
            df_bin = df.loc[m_eta & m_pt]

            n0 = df_bin.drop_duplicates(pair_cols).shape[0]
            n0_map[ieta, ipt] = n0
            if n0 == 0:
                continue

            for idr, dr in enumerate(dr_vals):
                df_cut = filtering(df_bin, prefix, dr)  # <-- YOUR function
                n_pass = df_cut.drop_duplicates(pair_cols).shape[0]
                eff_map[ieta, ipt, idr] = n_pass / n0

    # ---- plot curves (same representation as yours)
    plt.figure(figsize=(12, 8))

    for ieta in range(len(eta_bins)-1):
        elo, ehi = eta_bins[ieta], eta_bins[ieta+1]
        for ipt in range(len(pt_bins)-1):
            plo, phi = pt_bins[ipt], pt_bins[ipt+1]

            n0 = n0_map[ieta, ipt]
            if n0 == 0:
                continue

            y = eff_map[ieta, ipt, :]
            # label includes BOTH bins
            if use_abs_eta:
                lab = rf"${elo:g}\leq |\eta^{{gen}}|<{ehi:g},\ {plo:g}\leq p_T^{{gen}}<{phi:g}$ (N={n0})"
            else:
                lab = rf"${elo:g}\leq \eta^{{gen}}<{ehi:g},\ {plo:g}\leq p_T^{{gen}}<{phi:g}$ (N={n0})"

            plt.plot(dr_vals, y, marker="o", linewidth=1.8, label=lab)

    plt.xlabel(r"$\Delta R$ threshold")
    plt.ylabel(r"$\epsilon$ (matched / gen)")
    plt.ylim(0, 1.05)

    plt.legend(ncol=legend_ncol, fontsize=16, frameon=True)
    mh.cms.label(cms_label, data=False, com=com)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.show()

def run_all_triangles_curves_2d(
    dfs, eta_bins, pt_bins, dr_vals,
    outdir="eff_vs_deltaR_2Dbins",
    use_abs_eta=True,
):
    os.makedirs(outdir, exist_ok=True)

    for label, (df, prefix) in dfs.items():
        curves_eff_vs_dr_in_2d_bins(
            df=df,
            prefix=prefix,
            eta_bins=eta_bins,
            pt_bins=pt_bins,
            dr_vals=dr_vals,
            use_abs_eta=use_abs_eta,
            outpath=os.path.join(outdir, f"eff_vs_deltaR_by_eta_pt_{label}.png"),
            legend_ncol=2,
            legend_fontsize=11,
        )
def wrap_phi(phi):
    return (phi + np.pi) % (2*np.pi) - np.pi

def _col(template, Prefix):
    return template.format(Prefix=Prefix)

def _bin_label(slice_var, lo, hi):
    if slice_var == "eta":
        return rf"{lo:.2f}<absEtaGen<{hi:.2f}"
    if slice_var == "pt":
        return rf"{lo:.0f}<ptGen<{hi:.0f}"
    if slice_var == "phi":
        return rf"{lo:.2f}<phiGen<{hi:.2f}"
    return rf"{lo}<x<{hi}"

def binned_mean_sem(x, y, bins):
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    ib = np.digitize(x, bins) - 1
    nb = len(bins) - 1
    xc = 0.5 * (np.asarray(bins[:-1]) + np.asarray(bins[1:]))

    mean = np.full(nb, np.nan)
    sem  = np.full(nb, np.nan)
    n    = np.zeros(nb, dtype=int)

    for b in range(nb):
        yy = y[ib == b]
        n[b] = yy.size
        if yy.size:
            mean[b] = np.mean(yy)
            sem[b]  = (np.std(yy, ddof=1) / np.sqrt(yy.size)) if yy.size > 1 else 0.0
    return xc, mean, sem, n



def plot_response_hists_one_by_one(
    dfs,                         # dict: label -> (df, Prefix)
    slice_var,                   # "eta" | "pt" | "phi"
    slice_bins,                  # edges for that variable
    gen_eta_col="genpart_exeta",
    gen_pt_col="gen_pt",
    gen_phi_col="genpart_exphi",
    pt_reco_col="cl3d_{Prefix}_pt",
    # optional global selections (applied in addition to slice)
    ptgen_sel=None,              # e.g. (20, 100)
    etagen_sel=None,             # e.g. (1.6, 2.8) on abs(eta)
    phigen_sel=None,             # e.g. (-pi, pi) on wrapped phi
    # histogram settings
    nbins=60,
    yscale="log",
    density=False,
    weights_col=None,
    # labels / output
    outdir="resp_hists",
    tag="",
    cms_label="Preliminary",
    right_label="PU200 photons",
    xlabel=r"$p_T^{reco}/p_T^{gen}$",
):
    os.makedirs(outdir, exist_ok=True)

    slice_bins = np.asarray(slice_bins)
    for i in range(len(slice_bins) - 1):
        lo, hi = slice_bins[i], slice_bins[i+1]

        fig, ax = plt.subplots(figsize=(12,8))

        for lab, (df, Prefix) in dfs.items():
            pt_col = _col(pt_reco_col, Prefix)
            if pt_col not in df.columns:
                print(f"[WARN] {lab}: missing {pt_col}, skipping")
                continue

            eta = df[gen_eta_col].to_numpy()
            ptg = df[gen_pt_col].to_numpy()
            phi = wrap_phi(df[gen_phi_col].to_numpy())
            ptr = df[pt_col].to_numpy()

            m = np.isfinite(eta) & np.isfinite(ptg) & np.isfinite(phi) & np.isfinite(ptr)

            # global selections
            if ptgen_sel is not None:
                m &= (ptg >= ptgen_sel[0]) & (ptg < ptgen_sel[1])
            if etagen_sel is not None:
                m &= (np.abs(eta) >= etagen_sel[0]) & (np.abs(eta) < etagen_sel[1])
            if phigen_sel is not None:
                m &= (phi >= phigen_sel[0]) & (phi < phigen_sel[1])

            # slice selection
            if slice_var == "eta":
                m &= (np.abs(eta) >= lo) & (np.abs(eta) < hi)
                bin_text = rf"${lo:.2f} < |\eta^{{gen}}| < {hi:.2f}$"
            elif slice_var == "pt":
                m &= (ptg >= lo) & (ptg < hi)
                bin_text = rf"${lo:.0f} < p_T^{{gen}} < {hi:.0f}\ \mathrm{{GeV}}$"
            elif slice_var == "phi":
                m &= (phi >= lo) & (phi < hi)
                bin_text = rf"${lo:.2f} < \phi^{{gen}} < {hi:.2f}$"
            else:
                raise ValueError("slice_var must be 'eta', 'pt', or 'phi'")

            if not np.any(m):
                continue

            resp = ptr[m] / ptg[m]
            w = None
            if weights_col is not None:
                w = df.loc[m, weights_col].to_numpy()

            ax.hist(
                resp,
                bins=nbins,
                histtype="step",
                linewidth=1.8,
                density=density,
                weights=w,
                label=lab,
            )

        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Counts" if not density else "A.U.")
        ax.grid(True, alpha=0.25)

        mh.cms.label(cms_label, data=False, rlabel=r"$14~\mathrm{TeV},\ \langle PU\rangle=200$")

        ax.legend(frameon=True, loc="upper right")
        fig.tight_layout()

        fname = f"resp_{slice_var}_{_bin_label(slice_var, lo, hi)}{tag}.png"
        fig.savefig(os.path.join(outdir, fname), bbox_inches="tight")
        plt.show()
        plt.close(fig)

def plot_delta_vs_gen(
    dfs,                          # dict: label -> (df, Prefix)
    which="eta",                  # "eta" or "phi"
    gen_eta_col="genpart_exeta",
    gen_phi_col="genpart_exphi",
    reco_eta_col="cl3d_{Prefix}_eta",
    reco_phi_col="cl3d_{Prefix}_phi",
    eta_bins=np.linspace(1.6, 2.8, 10),
    phi_bins=np.linspace(-np.pi, np.pi, 7),
    ptgen_sel=None,               # optional (lo,hi) on gen_pt
    gen_pt_col="gen_pt",
    outpath=None,
    cms_label="Preliminary",
):
    fig, ax = plt.subplots(figsize=(12,8))

    for lab, (df, Prefix) in dfs.items():
        if which == "eta":
            x = df[gen_eta_col].to_numpy()
            reco_col = _col(reco_eta_col, Prefix)
            if reco_col not in df.columns:
                print(f"[WARN] {lab}: missing {reco_col}, skipping")
                continue
            y = abs(df[reco_col].to_numpy() - x)
            bins = np.asarray(eta_bins)
            xlabel = r"$\eta^{gen}$"
            ylabel = r"$\Delta\eta  = |\eta^{reco} - \eta^{gen}$|"
            #hline = 0.0

        elif which == "phi":
            x = wrap_phi(df[gen_phi_col].to_numpy())
            reco_col = _col(reco_phi_col, Prefix)
            if reco_col not in df.columns:
                print(f"[WARN] {lab}: missing {reco_col}, skipping")
                continue
            y = wrap_phi(df[reco_col].to_numpy() - df[gen_phi_col].to_numpy())
            bins = np.asarray(phi_bins)
            xlabel = r"$\phi^{gen}$"
            ylabel = r"$\Delta\phi = (\phi^{reco}-\phi^{gen})$"
            #hline = 0.0
        else:
            raise ValueError("which must be 'eta' or 'phi'")

        m = np.isfinite(x) & np.isfinite(y)

        if ptgen_sel is not None:
            ptg = df[gen_pt_col].to_numpy()
            m &= np.isfinite(ptg) & (ptg >= ptgen_sel[0]) & (ptg < ptgen_sel[1])

        xc, ym, ye, n = binned_mean_sem(x[m], y[m], bins)
        ok = np.isfinite(ym)
        ax.errorbar(xc[ok], ym[ok], yerr=ye[ok], fmt="o", ms=5, capsize=2, label=lab)

    #ax.axhline(hline, lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    mh.cms.label(cms_label, data=False, rlabel=r"$14~\mathrm{TeV},\ \langle PU\rangle=200$")
    ax.legend(frameon=True)

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, bbox_inches="tight")
    plt.show()
    
def binned_rms_and_err(x, y, bins):
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    bins = np.asarray(bins)
    nb = len(bins) - 1
    xc = 0.5 * (bins[:-1] + bins[1:])
    xerr = 0.5 * (bins[1:] - bins[:-1])

    rms = np.full(nb, np.nan)
    er  = np.full(nb, np.nan)
    n   = np.zeros(nb, dtype=int)

    ib = np.digitize(x, bins) - 1
    for b in range(nb):
        yy = y[ib == b]
        n[b] = yy.size
        if yy.size >= 2:
            # RMS around mean (i.e. standard deviation)
            s = np.std(yy, ddof=1)
            rms[b] = s
            # approx error on std (Gaussian assumption)
            er[b] = s / np.sqrt(2*(yy.size - 1))
        elif yy.size == 1:
            rms[b] = 0.0
            er[b] = np.nan
    return xc, xerr, rms, er, n

def plot_sigma_delta_vs_gen(
    dfs,                           # dict: label -> (df, Prefix)
    which="phi",                   # "eta" or "phi"
    # gen columns
    gen_eta_col="genpart_exeta",
    gen_phi_col="genpart_exphi",
    gen_pt_col="gen_pt",
    # reco columns templates
    reco_eta_col="cl3d_{Prefix}_eta",
    reco_phi_col="cl3d_{Prefix}_phi",
    # binning
    eta_bins=np.linspace(1.6, 2.8, 13),          # for |eta|
    phi_bins=np.linspace(-np.pi, np.pi, 13),     # for phi
    use_abs_eta=True,
    # selection
    ptgen_sel=(20, 100),
    # cosmetics
    ylabel_eta=r"$\sigma(\Delta\eta)$",
    ylabel_phi=r"$\sigma(\Delta\phi)$",
    cms_label="Preliminary",
    right_label="PU200 photons",
    outpath=None,
):
    fig, ax = plt.subplots(figsize=(12, 8))

    for lab, (df, Prefix) in dfs.items():
        if which == "eta":
            reco_col = _col(reco_eta_col, Prefix)
            if reco_col not in df.columns:
                print(f"[WARN] {lab}: missing {reco_col}, skipping")
                continue
            eta_gen = df[gen_eta_col].to_numpy()
            eta_rec = df[reco_col].to_numpy()
            x = np.abs(eta_gen) if use_abs_eta else eta_gen
            y = eta_rec - eta_gen
            bins = np.asarray(eta_bins)
            xlabel = r"$|\eta^{gen}|$" if use_abs_eta else r"$\eta^{gen}$"
            ylabel = ylabel_eta

        elif which == "phi":
            reco_col = _col(reco_phi_col, Prefix)
            if reco_col not in df.columns:
                print(f"[WARN] {lab}: missing {reco_col}, skipping")
                continue
            phi_gen = wrap_phi(df[gen_phi_col].to_numpy())
            phi_rec = wrap_phi(df[reco_col].to_numpy())
            x = phi_gen
            y = wrap_phi(phi_rec - phi_gen)
            bins = np.asarray(phi_bins)
            xlabel = r"$\phi^{gen}$"
            ylabel = ylabel_phi

        else:
            raise ValueError("which must be 'eta' or 'phi'")

        ptg = df[gen_pt_col].to_numpy()
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(ptg)
        if ptgen_sel is not None:
            m &= (ptg >= ptgen_sel[0]) & (ptg < ptgen_sel[1])

        xc, xerr, sig, sigerr, n = binned_rms_and_err(x[m], y[m], bins)
        ok = np.isfinite(sig)

        ax.errorbar(
            xc[ok], sig[ok],
            xerr=xerr[ok], yerr=sigerr[ok],
            fmt="s", ms=5, capsize=2,
            label=lab
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    mh.cms.label(cms_label, data=False, rlabel=r"$14~\mathrm{TeV},\ \langle PU\rangle=200$")
    ax.legend(frameon=True)

    fig.tight_layout()
    if outpath:
        fig.savefig(outpath, bbox_inches="tight")
    plt.show()

def binned_stats(x, y, bins):
    x = np.asarray(x); y = np.asarray(y)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]

    bins = np.asarray(bins)
    nb = len(bins) - 1
    xc   = 0.5 * (bins[:-1] + bins[1:])
    xerr = 0.5 * (bins[1:]  - bins[:-1])

    mu  = np.full(nb, np.nan)
    mue = np.full(nb, np.nan)
    sig = np.full(nb, np.nan)
    sige= np.full(nb, np.nan)
    rel = np.full(nb, np.nan)
    rele= np.full(nb, np.nan)
    n   = np.zeros(nb, dtype=int)

    ib = np.digitize(x, bins) - 1
    for b in range(nb):
        yy = y[ib == b]
        n[b] = yy.size
        if yy.size >= 2:
            m0 = np.mean(yy)
            s0 = np.std(yy, ddof=1)

            mu[b]  = m0
            sig[b] = s0

            # errors
            mue[b]  = s0 / np.sqrt(yy.size)                 # SEM
            sige[b] = s0 / np.sqrt(2*(yy.size - 1))         # error on std (approx)

            if np.isfinite(m0) and m0 != 0:
                rel[b] = s0 / m0
                rele[b] = rel[b] * np.sqrt((sige[b]/s0)**2 + (mue[b]/m0)**2)
        elif yy.size == 1:
            mu[b] = yy[0]
            sig[b] = 0.0

    return xc, xerr, mu, mue, sig, sige, rel, rele, n

def plot_ptresp_metric_vs(
    dfs,                            # dict: label -> (df, Prefix)
    xvar="pt",                      # "pt" | "eta" | "phi"
    bins=None,
    metric="mean",                  # "mean" | "sigma" | "rel"
    # columns
    gen_pt_col="gen_pt",
    gen_eta_col="genpart_exeta",
    gen_phi_col="genpart_exphi",
    pt_reco_col="cl3d_{Prefix}_pt",
    # selections
    ptgen_sel=(20, 100),
    abs_eta_range=None,             # e.g. (1.6, 2.8)
    # cosmetics
    cms_label="Preliminary",
    right_label=r"$14~\mathrm{TeV},\ \langle PU\rangle=200$",
    title=None,
    outpath=None,
    legend_loc="upper right",
):
    if bins is None:
        raise ValueError("Provide 'bins'.")

    fig, ax = plt.subplots(figsize=(12, 8))

    # axis labels
    if xvar == "pt":
        xlabel = r"$p_T^{gen}$ [GeV]"
    elif xvar == "eta":
        xlabel = r"$|\eta^{gen}|$"
    elif xvar == "phi":
        xlabel = r"$\phi^{gen}$"
    else:
        raise ValueError("xvar must be 'pt', 'eta', or 'phi'")

    if metric == "mean":
        ylabel = r"$\langle p_T^{cluster}/p_T^{gen}\rangle$"
    elif metric == "sigma":
        ylabel = r"$\sigma(p_T^{cluster}/p_T^{gen})$"
    elif metric == "rel":
        ylabel = r"$(\sigma/\mu)_{\mathrm{eff}}$"
    else:
        raise ValueError("metric must be 'mean', 'sigma', or 'rel'")

    for lab, (df, Prefix) in dfs.items():
        pt_col = _col(pt_reco_col, Prefix)
        if pt_col not in df.columns:
            print(f"[WARN] {lab}: missing {pt_col}, skipping")
            continue

        ptg  = df[gen_pt_col].to_numpy()
        etag = df[gen_eta_col].to_numpy()
        phig = wrap_phi(df[gen_phi_col].to_numpy())
        ptr  = df[pt_col].to_numpy()

        resp = ptr / ptg

        m = np.isfinite(ptg) & np.isfinite(etag) & np.isfinite(phig) & np.isfinite(resp)

        if ptgen_sel is not None:
            m &= (ptg >= ptgen_sel[0]) & (ptg < ptgen_sel[1])
        if abs_eta_range is not None:
            m &= (np.abs(etag) >= abs_eta_range[0]) & (np.abs(etag) < abs_eta_range[1])

        if xvar == "pt":
            x = ptg
        elif xvar == "eta":
            x = np.abs(etag)
        elif xvar == "phi":
            x = phig

        xc, xerr, mu, mue, sig, sige, rel, rele, n = binned_stats(x[m], resp[m], bins)

        if metric == "mean":
            y, ye = mu, mue
        elif metric == "sigma":
            y, ye = sig, sige
        else:  # rel
            y, ye = rel, rele

        ok = np.isfinite(y)
        ax.errorbar(
            xc[ok], y[ok],
            xerr=xerr[ok], yerr=ye[ok],
            fmt="o", ms=5, capsize=2,
            label=lab
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)

    mh.cms.label(cms_label, data=False, rlabel=right_label)
    ax.legend(frameon=True, loc=legend_loc)

    fig.tight_layout()
    if outpath:
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
        fig.savefig(outpath, bbox_inches="tight")
    plt.show()

def plot_phi_bias_and_resolution_in_etabins(
    dfs,                                # dict: label -> (df, Prefix)
    eta_bins=np.linspace(1.6, 2.8, 13),  # bins in |etaGen|
    gen_eta_col="genpart_exeta",
    gen_phi_col="genpart_exphi",
    gen_pt_col="gen_pt",
    reco_phi_col="cl3d_{Prefix}_phi",
    ptgen_sel=(20, 100),
    cms_label="Preliminary",
    right_label=r"$14~\mathrm{TeV},\ \langle PU\rangle=200$",
    outdir="phi_in_etabins",
    tag="",
):
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    for lab, (df, Prefix) in dfs.items():
        reco_col = _col(reco_phi_col, Prefix)
        if reco_col not in df.columns:
            print(f"[WARN] {lab}: missing {reco_col}, skipping")
            continue

        etag = df[gen_eta_col].to_numpy()
        phig = wrap_phi(df[gen_phi_col].to_numpy())
        phir = wrap_phi(df[reco_col].to_numpy())
        ptg  = df[gen_pt_col].to_numpy()

        x = np.abs(etag)
        dphi = wrap_phi(phir - phig)

        m = np.isfinite(x) & np.isfinite(dphi) & np.isfinite(ptg)
        if ptgen_sel is not None:
            m &= (ptg >= ptgen_sel[0]) & (ptg < ptgen_sel[1])

        xc, xerr, mu, mue, n = binned_mean_sem(x[m], dphi[m], eta_bins)
        ok = np.isfinite(mu)
        ax.errorbar(xc[ok], mu[ok], xerr=xerr[ok], yerr=mue[ok],
                    fmt="o", ms=5, capsize=2, label=lab)
    ax.set_xlabel(r"$|\eta^{gen}|$")
    ax.set_ylabel(r"$\Delta\phi = |(\phi^{reco}-\phi^{gen})|$")
    ax.set_xlim(eta_bins[0], eta_bins[-1])
    ax.grid(True, alpha=0.25)
    hep.cms.label(cms_label, data=False, rlabel=right_label)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"mean_dphi_vs_absEtaGen{tag}.png"), bbox_inches="tight")
    plt.show()

    #phi
    fig, ax = plt.subplots(figsize=(12, 8))
    for lab, (df, Prefix) in dfs.items():
        reco_col = _col(reco_phi_col, Prefix)
        if reco_col not in df.columns:
            continue

        etag = df[gen_eta_col].to_numpy()
        phig = wrap_phi(df[gen_phi_col].to_numpy())
        phir = wrap_phi(df[reco_col].to_numpy())
        ptg  = df[gen_pt_col].to_numpy()

        x = np.abs(etag)
        dphi = wrap_phi(phir - phig)

        m = np.isfinite(x) & np.isfinite(dphi) & np.isfinite(ptg)
        if ptgen_sel is not None:
            m &= (ptg >= ptgen_sel[0]) & (ptg < ptgen_sel[1])

        xc, xerr, sig, sige, n = binned_rms_and_err(x[m], dphi[m], eta_bins)
        ok = np.isfinite(sig)
        ax.errorbar(xc[ok], sig[ok], xerr=xerr[ok], yerr=sige[ok],
                    fmt="o", ms=5, capsize=2, label=lab)

    ax.set_xlabel(r"$|\eta^{gen}|$")
    ax.set_ylabel(r"$\sigma(\Delta\phi)$")
    ax.set_xlim(eta_bins[0], eta_bins[-1])
    ax.grid(True, alpha=0.25)
    mh.cms.label(cms_label, data=False, rlabel=r"$14~\mathrm{TeV},\ \langle PU\rangle=200$")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"sigma_dphi_vs_absEtaGen{tag}.png"), bbox_inches="tight")
    plt.show()

def plot_eta_response_and_resolution_vs_ptgen(
    dfs,                               # dict: label -> (df, Prefix)
    pt_bins,                           # edges for ptgen bins
    gen_pt_col="gen_pt",
    gen_eta_col="genpart_exeta",
    reco_eta_col="cl3d_{Prefix}_eta",
    ptgen_sel=None,                    # optional extra selection (lo,hi)
    abs_eta_range=None,                # optional selection in |etaGen|, e.g. (1.6,2.8)
    cms_label="Preliminary",
    right_label=r"$14~\mathrm{TeV},\ \langle PU\rangle=200$",
    outdir="eta_vs_ptgen",
    tag="",
):
    os.makedirs(outdir, exist_ok=True)
    pt_bins = np.asarray(pt_bins)
    # vs pt_gen
    fig, ax = plt.subplots(figsize=(12, 8))
    for lab, (df, Prefix) in dfs.items():
        eta_col = _col(reco_eta_col, Prefix)
        if eta_col not in df.columns:
            print(f"[WARN] {lab}: missing {eta_col}, skipping")
            continue

        ptg  = df[gen_pt_col].to_numpy()
        etag = df[gen_eta_col].to_numpy()
        etar = df[eta_col].to_numpy()

        dEta = etar - etag

        m = np.isfinite(ptg) & np.isfinite(etag) & np.isfinite(dEta)
        if ptgen_sel is not None:
            m &= (ptg >= ptgen_sel[0]) & (ptg < ptgen_sel[1])
        if abs_eta_range is not None:
            m &= (np.abs(etag) >= abs_eta_range[0]) & (np.abs(etag) < abs_eta_range[1])

        xc, xerr, mu, mue, n = binned_mean_sem(ptg[m], dEta[m], pt_bins)
        ok = np.isfinite(mu)
        ax.errorbar(xc[ok], mu[ok], xerr=xerr[ok], yerr=mue[ok],
                    fmt="o", ms=5, capsize=2, label=lab)

    ax.set_xlabel(r"$p_T^{gen}$ [GeV]")
    ax.set_ylabel(r"$\langle \Delta\eta \rangle = \langle \eta^{reco}-\eta^{gen}\rangle$")
    ax.grid(True, alpha=0.25)
    mh.cms.label(cms_label, data=False, rlabel=right_label)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"mean_dEta_vs_ptGen{tag}.png"), bbox_inches="tight", dpi=200)
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8))
    for lab, (df, Prefix) in dfs.items():
        eta_col = _col(reco_eta_col, Prefix)
        if eta_col not in df.columns:
            continue

        ptg  = df[gen_pt_col].to_numpy()
        etag = df[gen_eta_col].to_numpy()
        etar = df[eta_col].to_numpy()

        dEta = etar - etag

        m = np.isfinite(ptg) & np.isfinite(etag) & np.isfinite(dEta)
        if ptgen_sel is not None:
            m &= (ptg >= ptgen_sel[0]) & (ptg < ptgen_sel[1])
        if abs_eta_range is not None:
            m &= (np.abs(etag) >= abs_eta_range[0]) & (np.abs(etag) < abs_eta_range[1])

        xc, xerr, sig, sige, n = binned_rms_and_err(ptg[m], dEta[m], pt_bins)
        ok = np.isfinite(sig)
        ax.errorbar(xc[ok], sig[ok], xerr=xerr[ok], yerr=sige[ok],
                    fmt="o", ms=5, capsize=2, label=lab)

    ax.set_xlabel(r"$p_T^{gen}$ [GeV]")
    ax.set_ylabel(r"$\sigma(\Delta\eta)$")
    ax.grid(True, alpha=0.25)
    mh.cms.label(cms_label, data=False, rlabel=right_label)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"sigma_dEta_vs_ptGen{tag}.png"), bbox_inches="tight", dpi=200)
    plt.show()
    plt.close(fig)