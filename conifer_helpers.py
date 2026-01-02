import numpy as np
import pandas as pd
import analysis as ana
import os, time
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, label_binarize
from pandas.api.types import is_float_dtype, is_integer_dtype
from math import ceil, log2
from scipy.special import softmax
import conifer
from typing import Optional, Dict, Mapping, Union
from pathlib import Path
import os, datetime, glob, re, xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

def maxbits(series: pd.Series, maxbit: int) -> int:
    if is_float_dtype(series):
        return int(maxbit)
    uniq = pd.unique(series.dropna())
    nval = max(1, len(uniq))
    bits_needed = 1 if nval <= 2 else ceil(log2(nval))
    return int(min(maxbit, max(1, bits_needed)))

def _learn_edges(x: np.ndarray, nbits: int, method: str, fmin=None, fmax=None):
    nbins = 2**nbits
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.array([0, 1], dtype=float)
    lo = np.min(x) if fmin is None else float(fmin)
    hi = np.max(x) if fmax is None else float(fmax)
    if not (hi > lo):
        return np.array([lo, lo+1e-9], dtype=float)
    if method == 'uniform':
        edges = np.linspace(lo, hi, nbins+1, endpoint=True)
    elif method == 'percentile':
        qs = np.linspace(0, 100, nbins+1)
        edges = np.percentile(x, qs)
        edges = np.maximum.accumulate(edges)
        if not (edges[-1] > edges[0]):
            edges = np.linspace(lo, hi, nbins+1, endpoint=True)
    else:
        raise ValueError("method must be 'uniform' or 'percentile'")
    return edges

def _digitize_codes(x: np.ndarray, edges: np.ndarray, keep_nan_code: int = -1):
    codes = np.full(x.shape, keep_nan_code, dtype=np.int32)
    finite = np.isfinite(x)
    if finite.any():
        idx = np.digitize(x[finite], edges, right=True)  # 0..nbins
        nbins = len(edges) - 1
        idx = np.clip(idx, 1, nbins) - 1               # 0..nbins-1
        codes[finite] = idx.astype(np.int32, copy=False)
    return codes

def fit_quantizers(
    X_train: pd.DataFrame,
    maxbit: int = 8,
    float_method: str = 'percentile',
    int_method: str = 'uniform',
    per_feature_bits: Optional[Mapping[str, int]] = None,   # <- fix here
):
    qs: Dict[str, Dict[str, object]] = {}
    for c in X_train.columns:
        s = X_train[c]
        # choose bits
        nbits = per_feature_bits[c] if (per_feature_bits is not None and c in per_feature_bits) else maxbits(s, maxbit)
        # choose method
        method = int_method if is_integer_dtype(s) else float_method
        # coerce to float for robust finite handling
        x = pd.to_numeric(s, errors='coerce').to_numpy(dtype='float64', copy=False)
        edges = _learn_edges(x, nbits=nbits, method=method)
        qs[c] = {'nbits': nbits, 'edges': edges, 'method': method}
    return qs
def transform_quantizers(X: pd.DataFrame, qspec: dict, keep_nan_code: int = -1) -> pd.DataFrame:
    out = {}
    for c, spec in qspec.items():
        x = pd.to_numeric(X[c], errors='coerce').to_numpy(dtype='float64', copy=False)
        out[c] = _digitize_codes(x, spec['edges'], keep_nan_code=keep_nan_code)
    Q = pd.DataFrame(out, index=X.index)[X.columns]  # preserve column order
    return Q

def count_total_splits_and_leaves(booster: xgb.Booster) -> int:
    df = booster.trees_to_dataframe()
    n_splits = int((df['Feature'] != 'Leaf').sum())
    n_leaves = int((df['Feature'] == 'Leaf').sum())
    return n_splits, n_leaves

def timed_decision_function(model, X_np):
    t0 = time.perf_counter()
    logits = model.decision_function(X_np)
    dt = time.perf_counter() - t0
    return np.asarray(logits), dt

def _softmax(logits):
    logits = np.asarray(logits, dtype=float)
    logits -= logits.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)

def _parse_luts_from_report(report_path):
    """
    Extract 'LUT as Logic' and 'LUT as Memory' (Used) from a Vivado util.rpt.
    Returns (lut_logic, lut_memory) as ints, or (None, None) if not found.
    """
    lut_logic = None
    lut_memory = None

    if not os.path.exists(report_path):
        return lut_logic, lut_memory

    with open(report_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        if ("LUT as Logic" in line or "LUT as Memory" in line) and "|" in line:
            # Split columns by '|' and strip whitespace
            parts = [s.strip() for s in line.split("|") if s.strip()]
            if not parts:
                continue
            label = parts[0]  # e.g. "LUT as Logic" or "LUT as Memory"
            if len(parts) >= 2:
                try:
                    used = int(parts[1])
                except ValueError:
                    continue
                if label == "LUT as Logic":
                    lut_logic = used
                elif label == "LUT as Memory":
                    lut_memory = used
        if (lut_logic is not None) and (lut_memory is not None):
            break
    return lut_logic, lut_memory

def auc_photon_vs_each(y_true: np.ndarray, y_score: np.ndarray, photon_class: int = 0) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    classes = np.unique(y_true)
    aucs = []
    for k in classes:
        if k == photon_class:
            continue
        m = (y_true == photon_class) | (y_true == k)
        if m.sum() < 2:
            continue
        y_bin = (y_true[m] == photon_class).astype(int)
        # Use the photon probability as the score
        scores = y_score[m, photon_class]
        # Need both positives and negatives
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        fpr, tpr, _ = roc_curve(y_bin, scores)
        aucs.append(auc(fpr, tpr))
    return float(np.mean(aucs)) if aucs else np.nan

def plot_roc_curves_sw_hdl(
    y_true, y_score_sw, y_score_hdl, class_names, title, out_path, photon_class=0
):
    """
    Plot ROC curves for Photon-vs-X, overlaying SW (dashed) and HDL (solid).
    Saves as <out_path>.pdf/.png
    """
    y_true = np.asarray(y_true)
    y_score_sw  = np.asarray(y_score_sw)
    y_score_hdl = np.asarray(y_score_hdl)
    photon_name = class_names[photon_class]

    plt.figure(figsize=(7.2, 6.2))
    ax = plt.gca()
    aucs_sw, aucs_hdl = [], []

    for i, name in enumerate(class_names):
        if i == photon_class: 
            continue
        m = (y_true == photon_class) | (y_true == i)
        if m.sum() < 2:
            continue
        y_bin = (y_true[m] == photon_class).astype(int)

        # Photon probability is the score
        s_sw  = y_score_sw[m,  photon_class]
        s_hdl = y_score_hdl[m, photon_class]

        # Need both positives & negatives
        if (y_bin.sum() == 0) or (y_bin.sum() == len(y_bin)):
            continue

        fpr_sw,  tpr_sw,  _ = roc_curve(y_bin, s_sw)
        fpr_hdl, tpr_hdl, _ = roc_curve(y_bin, s_hdl)

        auc_sw  = auc(fpr_sw,  tpr_sw)
        auc_hdl = auc(fpr_hdl, tpr_hdl)

        aucs_sw.append(auc_sw)
        aucs_hdl.append(auc_hdl)

        # Plot HDL solid, SW dashed
        ax.plot(fpr_hdl, tpr_hdl, lw=2.0, label=f"{photon_name} vs {name} (HDL AUC={auc_hdl:.3f})")
        ax.plot(fpr_sw,  tpr_sw,  lw=1.8, ls="--", label=f"{photon_name} vs {name} (SW  AUC={auc_sw:.3f})")

    mean_sw  = np.mean(aucs_sw)  if aucs_sw  else float("nan")
    mean_hdl = np.mean(aucs_hdl) if aucs_hdl else float("nan")

    ax.plot([0,1],[0,1], "--", color="gray", alpha=0.7, lw=1.2)
    ax.set_xscale('log')#(0.0, 1.0); ax.set_ylim(0.0, 1.02)
    ax.set_yscale('log')
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title}\nMean Photon-vs-X AUC — HDL={mean_hdl:.3f}, SW={mean_sw:.3f}", pad=10)

    # CMS header
    plt.text(0.02, 1.02, r"$\bf{CMS}$  $\it{Simulation}$",
             ha="left", va="bottom", transform=ax.transAxes, fontsize=12)
    plt.text(0.98, 1.02, "14 TeV",
             ha="right", va="bottom", transform=ax.transAxes, fontsize=13)

    plt.legend(loc="lower right", frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path + ".pdf", bbox_inches="tight")
    plt.savefig(out_path + ".png", dpi=220, bbox_inches="tight")
    plt.close()

def plot_score_distributions_sw_hdl(
    y_true, y_score_sw, y_score_hdl, class_names, out_path, photon_class=0, bins=40
):
    """
    For each non-photon class X: histogram the photon probability p(photon)
    for samples in {Photon, X}, overlaying SW (dashed edge) and HDL (solid edge).
    Saves a multi-panel figure.
    """
    y_true = np.asarray(y_true)
    y_score_sw  = np.asarray(y_score_sw)
    y_score_hdl = np.asarray(y_score_hdl)
    photon_name = class_names[photon_class]

    # Collect non-photon classes present
    others = [i for i in range(len(class_names)) if i != photon_class]

    n = len(others)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2*nrows), squeeze=False)
    axes = axes.ravel()

    for ax, i in zip(axes, others):
        name = class_names[i]
        m = (y_true == photon_class) | (y_true == i)
        if m.sum() < 2:
            ax.set_axis_off()
            continue

        y_bin = (y_true[m] == photon_class).astype(int)
        s_sw  = y_score_sw[m,  photon_class]
        s_hdl = y_score_hdl[m, photon_class]

        # Plot histograms (density)
        ax.hist(s_sw[y_bin==1],  bins=bins, density=True, histtype="step", lw=1.8, label=f"{photon_name} (SW)",  alpha=0.9)
        ax.hist(s_hdl[y_bin==1], bins=bins, density=True, histtype="step", lw=2.2, label=f"{photon_name} (HDL)", alpha=0.9)

        ax.hist(s_sw[y_bin==0],  bins=bins, density=True, histtype="step", lw=1.8, ls="--", label=f"{name} (SW)",  alpha=0.9)
        ax.hist(s_hdl[y_bin==0], bins=bins, density=True, histtype="step", lw=2.2, ls="-",  label=f"{name} (HDL)", alpha=0.9)

        ax.set_xlabel("Score = P(photon)")
        ax.set_ylabel("Density")
        ax.set_title(f"{photon_name} vs {name}: score distributions")

        # Optional: vertical lines at 0.5 threshold
        ax.axvline(0.5, color="gray", lw=1.0, ls=":")

        ax.legend(loc="best", frameon=False, fontsize=9)

    # Turn off any unused axes
    for j in range(len(others), len(axes)):
        axes[j].set_axis_off()

    fig.suptitle("Photon vs X — SW vs HDL score distributions", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path + ".pdf", bbox_inches="tight")
    fig.savefig(out_path + ".png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def train_quantized_multiclass(precision, depth, rounds, iteration, X_train, y_train, X_test, y_test):
    t0 = time.time()
    print(f"[{iteration}] Training model: precision={precision}, depth={depth}, rounds={rounds}")

    # ensure labels are 0..(K-1) 
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
    n_classes = len(le.classes_)

    # Quantization (uniform, bins from TRAIN only) 
    qtrain = pd.DataFrame(index=X_train.index)
    qtest  = pd.DataFrame(index=X_test.index)
    for feat in X_train.columns:
        fmin, fmax = X_train[feat].min(), X_train[feat].max()
        # ana.quantize should return integer bin indices in [0 .. 2**precision-1]
        qtrain[feat] = ana.quantize(X_train[feat], precision, 'uniform', fmin, fmax)
        qtest[feat]  = ana.quantize(X_test[feat],  precision, 'uniform', fmin, fmax)

    # Normalize to [0, 1) for fixed-point ap_fixed<precision,0> 
    max_range = 1.0 - 1.0/(2**precision-1)
    scaler = MinMaxScaler(feature_range=(0.0, max_range))
    qtrain_scaled = pd.DataFrame(scaler.fit_transform(qtrain), columns=X_train.columns, index=X_train.index)
    qtest_scaled  = pd.DataFrame(scaler.transform(qtest),     columns=X_test.columns,  index=X_test.index)

    # Train XGBoost (multi-class, softprob) 
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=n_classes,
        max_depth=depth,
        n_estimators=rounds,
        learning_rate=0.001,
        n_jobs=8,
    )
    model.fit(qtrain_scaled, y_train_enc)
    booster = model.get_booster()
    #booster.set_attr(objective="multi:softprob")

    # SW metrics
    booster.set_param({'objective': 'multi:softprob', 'num_class': n_classes})
    prob_sw = booster.predict(xgb.DMatrix(qtest_scaled), output_margin=False)
    ypred_sw = np.argmax(prob_sw, axis=1)
    acc_sw = accuracy_score(y_test_enc, ypred_sw)
    # roc_auc_score needs probability matrix and label vector
    try:
        auc_sw = auc_photon_vs_each(y_test_enc, prob_sw, photon_class=0)
    except ValueError:
        # If a class is absent in the test fold, AUC may fail; fall back to NaN
        auc_sw = np.nan

    # Conifer (VHDL backend for VU13P) ----
    cfg = conifer.backends.vhdl.auto_config()
    proj_dir = 'conifer_VHDL_4I_lessrounds/prj_{}'.format(int(datetime.datetime.now().timestamp()))
    os.makedirs(proj_dir, exist_ok=True)
    cfg['OutputDir']   = proj_dir
    cfg['XilinxPart']  = 'xcvu13p-fhgb2104-2LV-e'#VU13P
    cfg['Precision']='ap_fixed<{},{}>'.format(precision, 4) 
    print(cfg['Precision'])
    booster = model.get_booster()
    cnf_model = conifer.converters.convert_from_xgboost(booster, cfg)
    cnf_model.write()
    cnf_model.compile()

    # CSIM logits on test set (float32 contiguous array is safest)
    X_csim = np.ascontiguousarray(qtest_scaled.to_numpy(dtype=np.float32))
    logits_hdl = cnf_model.decision_function(X_csim)
    logits_hdl = np.asarray(logits_hdl, dtype=float)
    prob_hdl = _softmax(logits_hdl)
    ypred_hdl = np.argmax(prob_hdl, axis=1)

    # Build (csim) to generate reports; some flows create util.rpt on compile+build
    cnf_model.build(csim=False, synth=True, vsynth=True)

    # ---- Step 5: HDL metrics + LUTs ----
    acc_hdl = accuracy_score(y_test_enc, ypred_hdl)
    try:
        auc_hdl = auc_photon_vs_each(y_test_enc, prob_hdl, photon_class=0)
    except ValueError:
        auc_hdl = np.nan

    lut_logic, lut_memory = _parse_luts_from_report(os.path.join(cfg['OutputDir'], 'util.rpt'))
    splits, leaves = count_total_splits_and_leaves(booster)
    dur = time.time() - t0
    print(f"[{iteration}] Done: acc_sw={acc_sw:.4f}, auc_sw={auc_sw:.4f}, "
          f"acc_hdl={acc_hdl:.4f}, auc_hdl={auc_hdl:.4f}, LUT_logic={lut_logic}, LUT_memory={lut_memory}, splits ={splits}, leaves = {leaves}, time={dur:.2f}s"
          )
    return (precision, depth, rounds, acc_sw, auc_sw, acc_hdl, auc_hdl, lut_logic, lut_memory, splits, leaves, dur)
