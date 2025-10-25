import numpy as np
import pandas as pd
import analysis as ana
import os, time
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pandas.api.types import is_float_dtype, is_integer_dtype
from math import ceil, log2
from scipy.special import softmax
import conifer
from typing import Optional, Dict, Mapping, Union
from pathlib import Path
import os, glob, re, xml.etree.ElementTree as ET

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

def count_total_splits(booster: xgb.Booster) -> int:
    df = booster.trees_to_dataframe()
    return int((df['Feature'] != 'Leaf').sum())

def timed_decision_function(model, X_np):
    t0 = time.perf_counter()
    logits = model.decision_function(X_np)
    dt = time.perf_counter() - t0
    return np.asarray(logits), dt

def parse_hls_reports(output_dir: Union[str, Path]):
    out = dict(LUT=np.nan, FF=np.nan, BRAM=np.nan, DSP=np.nan,
               LatencyMin=np.nan, LatencyMax=np.nan, Interval=np.nan)
    outdir = Path(output_dir)
    rpt_candidates = list(outdir.glob("**/syn/report/*csynth.rpt"))
    if not rpt_candidates:
        return out

    rpt = rpt_candidates[0].read_text(errors='ignore')

    def grab(pattern, cast=float):
        m = re.search(pattern, rpt, flags=re.MULTILINE)
        return cast(m.group(1)) if m else np.nan

    out['LUT']   = grab(r"^\s*Total\s+LUTs\s*:\s*([\d,]+)", lambda s:int(s.replace(",","")))
    out['FF']    = grab(r"^\s*Total\s+FFs\s*:\s*([\d,]+)",  lambda s:int(s.replace(",","")))
    out['BRAM']  = grab(r"^\s*Total\s+BRAM_18K\s*:\s*([\d,]+)", lambda s:int(s.replace(",","")))
    out['DSP']   = grab(r"^\s*Total\s+DSPs\s*:\s*([\d,]+)",  lambda s:int(s.replace(",","")))
    out['LatencyMin'] = grab(r"Latency\s*\(cycles\)\s*min\s*=\s*([0-9]+)")
    out['LatencyMax'] = grab(r"Latency\s*\(cycles\)\s*max\s*=\s*([0-9]+)")
    out['Interval']   = grab(r"Interval\s*\(II\)\s*=\s*([0-9]+)")
    return out

def _softmax(logits):
    logits = np.asarray(logits, dtype=float)
    logits -= logits.max(axis=1, keepdims=True)  # numerical stability
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)

def _parse_lut_from_report(report_path):
    """Try to extract LUT usage from util.rpt (VHDL backend). Returns int or None."""
    if not os.path.exists(report_path):
        return None
    with open(report_path, "r") as f:
        lines = f.readlines()
    try:
        parts = lines[37].split("|")
        return int(parts[2])
    except Exception:
        pass
    for line in lines:
        if "LUT" in line and "|" in line:
            try:
                fields = [s.strip() for s in line.split("|") if s.strip()]
                for tok in fields:
                    if tok.isdigit():
                        return int(tok)
            except Exception:
                continue
    return None

def inspect_hls_reports(output_dir: str):
    print(f"[inspect] OutputDir = {output_dir}")
    report_dirs = []
    for root, dirs, files in os.walk(output_dir):
        if root.endswith(os.path.join('solution1','syn','report')):
            report_dirs.append(root)
    if not report_dirs:
        print("[inspect] No solution1/syn/report dirs found.")
        return

    for d in report_dirs:
        print(f"[inspect] Report dir: {d}")
        files = sorted(glob.glob(os.path.join(d, '*')))
        for f in files:
            print("  -", os.path.basename(f))

        # XML: dump <Resources> attributes
        for xml in glob.glob(os.path.join(d, '*_csynth.xml')):
            print(f"[inspect] XML: {os.path.basename(xml)}")
            try:
                root = ET.parse(xml).getroot()
                for res in root.iter('Resources'):
                    print("    Resources attribs:", dict(res.attrib))
            except Exception as e:
                print("    (XML parse error)", e)

        # RPT: grep resource lines and show exact labels
        for rpt in glob.glob(os.path.join(d, '*_csynth.rpt')):
            print(f"[inspect] RPT: {os.path.basename(rpt)}")
            txt = open(rpt, 'r', errors='ignore').read()
            # print a few lines around "Area" section if present
            area_idx = txt.lower().find('area')
            if area_idx != -1:
                snippet = txt[max(0, area_idx-300): area_idx+800]
                print("    --- Area snippet ---")
                print(snippet)
                print("    --------------------")
            # extract label:value pairs (very permissive)
            for pat in [r'(CLB\s+LUTs)\s*:\s*([\d,]+)',
                        r'(LUTs?)\s*:\s*([\d,]+)',
                        r'(LUT\s+as\s+Logic)\s*:\s*([\d,]+)',
                        r'(FFs?)\s*:\s*([\d,]+)',
                        r'(CLB\s+Registers)\s*:\s*([\d,]+)',
                        r'(DSP48\w*)\s*:\s*([\d,]+)',
                        r'(BRAM_?18K)\s*:\s*([\d,]+)',
                        r'(BRAMs?)\s*:\s*([\d,]+)',
                        r'(URAM\w*)\s*:\s*([\d,]+)']:
                for m in re.finditer(pat, txt, flags=re.IGNORECASE):
                    print(f"    {m.group(1)} -> {m.group(2)}")

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
    max_range = 1.0 - 1.0/(2**precision)
    scaler = MinMaxScaler(feature_range=(0.0, max_range))
    qtrain_scaled = pd.DataFrame(scaler.fit_transform(qtrain), columns=X_train.columns, index=X_train.index)
    qtest_scaled  = pd.DataFrame(scaler.transform(qtest),     columns=X_test.columns,  index=X_test.index)

    # Train XGBoost (multi-class, softprob) 
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        max_depth=depth,
        n_estimators=rounds,
        learning_rate=0.001,
        eval_metric='mlogloss',
        n_jobs=8,
        verbosity=1
    )
    model.fit(qtrain_scaled, y_train_enc)

    # SW metrics
    prob_sw = model.predict_proba(qtest_scaled)                      # shape (N, K)
    ypred_sw = np.argmax(prob_sw, axis=1)
    acc_sw = accuracy_score(y_test_enc, ypred_sw)
    # roc_auc_score needs probability matrix and label vector
    try:
        auc_sw = roc_auc_score(y_test_enc, prob_sw, multi_class='ovo')
    except ValueError:
        # If a class is absent in the test fold, AUC may fail; fall back to NaN
        auc_sw = np.nan

    # Conifer (VHDL backend for VU13P) ----
    cfg = conifer.backends.vhdl.auto_config()
    proj_dir = f"hdlprojects/prj_vhdl_multiclass_{precision}_{depth}_{rounds}_{iteration}"
    if os.path.exists(proj_dir):
        ana.remove_folder(proj_dir)
    os.makedirs(proj_dir, exist_ok=True)
    cfg['OutputDir']   = proj_dir
    cfg['XilinxPart']  = 'xcvu13p-fhgb2104-2L-e'#VU13P
    cfg['Precision']   = f"ap_fixed<{precision},0>"   # inputs are [0,1)
    cfg['ClockPeriod'] = 3
    cfg['ProjectName'] = 'hgcal_multiclass'
    booster = model.get_booster()
    cnf_model = conifer.converters.convert_from_xgboost(booster, cfg)
    cnf_model.compile()

    # CSIM logits on test set (float32 contiguous array is safest)
    X_csim = np.ascontiguousarray(qtest_scaled.to_numpy(dtype=np.float32))
    logits_hdl = cnf_model.decision_function(X_csim)
    logits_hdl = np.asarray(logits_hdl, dtype=float)
    prob_hdl = softmax(logits_hdl)
    ypred_hdl = np.argmax(prob_hdl, axis=1)

    # Build (csim) to generate reports; some flows create util.rpt on compile+build
    cnf_model.build(csim=True)

    # ---- Step 5: HDL metrics + LUTs ----
    acc_hdl = accuracy_score(y_test_enc, ypred_hdl)
    try:
        auc_hdl = roc_auc_score(y_test_enc, prob_hdl, multi_class='ovo')
    except ValueError:
        auc_hdl = np.nan

    lut = _parse_lut_from_report(os.path.join(cfg['OutputDir'], 'util.rpt'))
    splits = count_total_splits(booster)
    dur = time.time() - t0
    print(f"[{iteration}] Done: acc_sw={acc_sw:.4f}, auc_sw={auc_sw:.4f}, "
          f"acc_hdl={acc_hdl:.4f}, auc_hdl={auc_hdl:.4f}, LUT={lut}, splits ={splits}, time={dur:.2f}s")
    return (precision, depth, rounds, acc_sw, auc_sw, acc_hdl, auc_hdl, lut, splits, dur)
