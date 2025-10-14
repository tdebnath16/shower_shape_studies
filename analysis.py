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
import conifer
import shutil
import datetime
from sklearn.datasets import make_hastie_10_2
import logging
import sys

def delta_r(eta1, phi1, eta2, phi2):
    delta_eta = np.abs(eta1 - eta2)
    delta_phi = np.abs(phi1 - phi2)
    delta_phi = np.where(delta_phi > np.pi, 2 * np.pi - delta_phi, delta_phi)  # Adjust phi to be within [-pi, pi]
    return np.sqrt(delta_eta**2 + delta_phi**2)

def filter_by_delta_r(df, prefix, delta_r_threshold):
    """Filter DataFrame to keep only the highest-energy match per event within the delta R threshold."""
    required_columns = [f"{prefix}_eta", f"{prefix}_phi", 'genpart_exeta', 'genpart_exphi', f"{prefix}_energy", 'event']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    df = df.copy()
    #df = df[df['cl3d_ienergy'] > 0]
    df['delta_r'] = delta_r(df[f"{prefix}_eta"], df[f"{prefix}_phi"], df['genpart_exeta'], df['genpart_exphi'])
    df_filtered = df[df['delta_r'] < delta_r_threshold]
    df_sorted = df_filtered.sort_values(by=['event', f"{prefix}_energy", 'delta_r'], ascending=[True, False, True])
    df_best_match = df_sorted.groupby('event').first().reset_index()
    return df_best_match

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
def load_and_filter_hdf(df_gen_path, df_cl3d_path):
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
            min_value = min(df_signal_filtered[var].min(), df_bg1_filtered[var].min(), df_bg2_filtered[var].min())
            max_value = max(df_signal_filtered[var].max(), df_bg1_filtered[var].max(), df_bg2_filtered[var].max())
            bin_edges = np.arange(min_value - 0.5, max_value + 1.5, 1)
        else:
            min_value = min(df_signal_filtered[var].min(), df_bg1_filtered[var].min(), df_bg2_filtered[var].min())
            max_value = max(df_signal_filtered[var].max(), df_bg1_filtered[var].max(), df_bg2_filtered[var].max())
            bin_width = (max_value - min_value) / num_bins
            bin_edges = np.arange(min_value - bin_width / 2, max_value + bin_width / 2, bin_width)
        plt.hist(df_signal_filtered[var], histtype='step', bins=bin_edges, color='b', linewidth=1.5, label=label_signal, density=True)
        plt.hist(df_bg1_filtered[var], histtype='step', bins=bin_edges, color='g', linewidth=1.5, label=label_bg1, density=True)
        plt.hist(df_bg2_filtered[var], histtype='step', bins=bin_edges, color='r', linewidth=1.5, label=label_bg2, density=True)
        plt.hist(df_bg3_filtered[var], histtype='step', bins=bin_edges, color='black', linewidth=1.5, label=label_bg3, density=True)
        plt.title("Cluster " + f"{var_latex_map.get(var, var)} Histogram", fontsize=14)
        plt.xlabel(var_latex_map.get(var, var), fontsize=12)
        plt.ylabel('Normalized Frequency', fontsize=12)
        plt.legend()
        plt.tight_layout()

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