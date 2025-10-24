import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import awkward as ak
import seaborn as sns
import analysis as ana
import conifer_helpers as cf_helper
import os, time, datetime, shutil, json, shap, pathlib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize, MinMaxScaler, LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.datasets import make_hastie_10_2
from pandas.api.types import is_float_dtype, is_integer_dtype
from math import ceil, log2
from scipy.special import softmax
import conifer
from typing import Optional, Dict, Mapping, Union
from pathlib import Path

triangle = ["Ref", "p0113Tri", "p016Tri", "p03Tri", "p045Tri"]
cl3d_colname = {
    "Ref":   "cl3d_Ref",
    "p0113Tri": "cl3d_p0113Tri",
    "p016Tri":  "cl3d_p016Tri",
    "p03Tri":   "cl3d_p03Tri",
    "p045Tri":  "cl3d_p045Tri",
}
dr = {"photon": 0.05, "qcd": 0.5, "pion": 0.1}
photon_cl3d = {
    "Gen": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/photonPU200_newalgogen_filtered.h5",
    "Ref": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/photonPU200_newalgocl3d_Ref_filtered.h5",
    "p0113Tri": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/photonPU200_newalgocl3d_p0113_filtered.h5",
    "p016Tri": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/photonPU200_newalgocl3d_p016_filtered.h5",
    "p03Tri": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/photonPU200_newalgocl3d_p03_filtered.h5",
    "p045Tri": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/photonPU200_newalgocl3d_p045_filtered.h5",
}
QCD_cl3d = {
    "Gen": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/QCDPU200_newalgogen_filtered.h5",
    "Ref": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/QCDPU200_newalgocl3d_Ref_filtered.h5",
    "p0113Tri": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/QCDPU200_newalgocl3d_p0113_filtered.h5",
    "p016Tri": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/QCDPU200_newalgocl3d_p016_filtered.h5",
    "p03Tri": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/QCDPU200_newalgocl3d_p03_filtered.h5",
    "p045Tri": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/QCDPU200_newalgocl3d_p045_filtered.h5",
}
pion_cl3d = {
    "Gen": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/pionPU200_newalgogen_filtered.h5",
    "Ref": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/pionPU200_newalgocl3d_Ref_filtered.h5",
    "p0113Tri": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/pionPU200_newalgocl3d_p0113_filtered.h5",
    "p016Tri": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/pionPU200_newalgocl3d_p016_filtered.h5",
    "p03Tri": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/pionPU200_newalgocl3d_p03_filtered.h5",
    "p045Tri": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/pionPU200_newalgocl3d_p045_filtered.h5",
}
PU_cl3d = {
    "Ref": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/PU200_newalgocl3d_Ref_filtered.h5",
    "p0113Tri": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/PU200_newalgocl3d_p0113_filtered.h5",
    "p016Tri": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/PU200_newalgocl3d_p016_filtered.h5",
    "p03Tri": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/PU200_newalgocl3d_p03_filtered.h5",
    "p045Tri": "/data/data.polcms/cms/debnath/HGCAL/CMSSW_14_0_5/src/shower_shape_studies/samples/PU200_newalgocl3d_p045_filtered.h5",
}
parser = argparse.ArgumentParser()
parser.add_argument("--tri", choices=triangle, default="p016Tri", help="Triangle size / cl3d variant to use (default: p016).")
parser.add_argument("--prec-min", type=int, default=5)
parser.add_argument("--prec-max", type=int, default=14)
parser.add_argument("--depth-min", type=int, default=2)
parser.add_argument("--depth-max", type=int, default=5)
parser.add_argument("--rounds-min", type=int, default=5)
parser.add_argument("--rounds-max", type=int, default=20)
args = parser.parse_args()

def load_variant(tri):
    cl3d_col = cl3d_colname[tri]
    photon_df = ana.load_and_filter_hdf_path(photon_cl3d["Gen"], photon_cl3d[tri])
    qcd_df    = ana.load_and_filter_hdf_path(QCD_cl3d["Gen"],    QCD_cl3d[tri])
    pion_df   = ana.load_and_filter_hdf_path(pion_cl3d["Gen"],   pion_cl3d[tri])
    pu_df     = pd.read_hdf(PU_cl3d[tri])
    photon_df = ana.filter_by_delta_r(photon_df, cl3d_col, dr["photon"])
    qcd_df    = ana.filter_by_delta_r(qcd_df,    cl3d_col, dr["qcd"])
    pion_df   = ana.filter_by_delta_r(pion_df,   cl3d_col, dr["pion"])
    photon_df["label"] = 0
    pu_df["label"]     = 1
    qcd_df["label"]    = 2
    pion_df["label"]   = 3
    return {
        "cl3d_col": cl3d_col,
        "photon": photon_df,
        "qcd": qcd_df,
        "pion": pion_df,
        "pu": pu_df,
    }
variant = load_variant(args.tri)
cl3d_col = variant["cl3d_col"]
photon_df_filtered = variant["photon"]
qcd_df_filtered    = variant["qcd"]
pion_df_filtered   = variant["pion"]
PU_df              = variant["pu"]

# Setup for BDT training
all_dfs = [qcd_df_filtered, photon_df_filtered, PU_df, pion_df_filtered]
for i in range(len(all_dfs)):
    all_dfs[i] = all_dfs[i][ana.columns_for_training(args.tri) + ['label']]
df_combined = pd.concat(all_dfs, ignore_index=True)
X = df_combined[ana.columns_for_training(args.tri)]
y = df_combined['label']
sample_weights = compute_sample_weight(class_weight='balanced', y=y)
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(X, y, sample_weights, test_size=0.2, stratify=y, random_state=42)
# certain columns to be discrete ints before q-learning
discrete_base = ["showerlength", "coreshowerlength", "firstlayer"]
discrete_cols = [f"{cl3d_col}_{name}" for name in discrete_base]
for c in (set(discrete_cols) & set(X_train.columns)):
    # nullable int is fine; we coerce to float internally when learning edges
    X_train.loc[:, c] = X_train[c].astype('Int64')
    X_test.loc[:, c]  = X_test[c].astype('Int64')

# optionally specify per-feature bit budgets (otherwise computed via maxbits)
per_feature_bits = None  # e.g., {'cl3d_sigmaRR': 6, 'cl3d_emaxe': 5, ...} ---- this is to be cross checked!

# fit on training set only
qspec = cf_helper.fit_quantizers(X_train, maxbit = 14, float_method = 'percentile', int_method = 'uniform', per_feature_bits = per_feature_bits)
# transform train/test -> integer codes; cast to float32 for models
Q_train = cf_helper.transform_quantizers(X_train, qspec).astype('float32')
Q_test  = cf_helper.transform_quantizers(X_test,  qspec).astype('float32')

# results table (SW = XGBoost in Python; HDL = Conifer-generated model)
result_cols = ["precision","depth","rounds","acc_sw","auc_sw","acc_hdl","auc_hdl","LUT","splits","time_s"]
result_table = pd.DataFrame(columns=result_cols)
base_dir = os.getcwd()
iteration = 0
for prec in range(args.prec_min, args.prec_max, 1):
    for depth in range(args.depth_min, args.depth_max, 1):
        for rounds in range(args.rounds_min, args.rounds_max, 1):
            row = cf_helper.train_quantized_multiclass(prec, depth, rounds, iteration, Q_train, y_train, Q_test, 
                                                       y_test)
            result_table.loc[len(result_table)] = row
            result_table.to_csv("conifer_multiclass_opti.csv", index=False)
            iteration += 1