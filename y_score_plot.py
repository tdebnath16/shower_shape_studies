#!/usr/bin/env python3
import argparse, pandas as pd, numpy as np
import matplotlib.pyplot as plt

def load_filtered(csv_path, precision, depth, rounds):
    df = pd.read_csv(csv_path)
    mask = (df["precision"]==precision) & (df["depth"]==depth) & (df["rounds"]==rounds)
    return df[mask].copy()

def plot_roc_overlay(df, out_prefix):
    # Keep only ROC rows
    d = df[df["section"]=="roc"].copy()
    if d.empty:
        print("No ROC rows found for this config.")
        return
    pairs = sorted(d[["pair_pos","pair_neg"]].drop_duplicates().itertuples(index=False, name=None))
    plt.figure(figsize=(7.2, 6.2)); ax = plt.gca()

    auc_sw_all, auc_hdl_all = [], []
    for (pp, pn) in pairs:
        dd = d[(d["pair_pos"]==pp) & (d["pair_neg"]==pn)].copy()
        # SW
        sw = dd[dd["source"]=="SW"].sort_values("idx_or_sid")
        fpr_sw = sw["fpr"].to_numpy(dtype=float)
        tpr_sw = sw["tpr"].to_numpy(dtype=float)
        auc_sw = np.trapz(tpr_sw, fpr_sw)
        auc_sw_all.append(auc_sw)
        ax.plot(fpr_sw, tpr_sw, ls="--", lw=1.8, label=f"{pp} vs {pn} (SW AUC={auc_sw:.3f})")
        # HDL
        hd = dd[dd["source"]=="HDL"].sort_values("idx_or_sid")
        fpr_hdl = hd["fpr"].to_numpy(dtype=float)
        tpr_hdl = hd["tpr"].to_numpy(dtype=float)
        auc_hdl = np.trapz(tpr_hdl, fpr_hdl)
        auc_hdl_all.append(auc_hdl)
        ax.plot(fpr_hdl, tpr_hdl, lw=2.2, label=f"{pp} vs {pn} (HDL AUC={auc_hdl:.3f})")

    mean_sw  = np.mean(auc_sw_all)  if auc_sw_all  else float("nan")
    mean_hdl = np.mean(auc_hdl_all) if auc_hdl_all else float("nan")

    ax.plot([0,1],[0,1],"--", lw=1.2, alpha=0.7, color="gray")
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC (Photon-vs-X) — HDL={mean_hdl:.3f}, SW={mean_sw:.3f}", pad=10)
    plt.text(0.02, 1.02, r"$\bf{CMS}$  $\it{Simulation}$", transform=ax.transAxes, ha="left", va="bottom", fontsize=12)
    plt.text(0.98, 1.02, "14 TeV", transform=ax.transAxes, ha="right", va="bottom", fontsize=13)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_prefix + ".pdf", bbox_inches="tight")
    plt.savefig(out_prefix + ".png", dpi=220, bbox_inches="tight")
    plt.close()

def plot_score_panels(df, out_prefix, bins=40):
    # Keep only SCORE rows
    d = df[df["section"]=="score"].copy()
    if d.empty:
        print("No SCORE rows found for this config.")
        return
    pairs = sorted(d[["pair_pos","pair_neg"]].drop_duplicates().itertuples(index=False, name=None))
    n = len(pairs); ncols = 2; nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4.2*nrows), squeeze=False)
    axes = axes.ravel()

    for ax, (pp, pn) in zip(axes, pairs):
        dd = d[(d["pair_pos"]==pp) & (d["pair_neg"]==pn)].copy()
        # Photon=1, Other=0
        sw = dd[dd["source"]=="SW"]
        hd = dd[dd["source"]=="HDL"]

        for label, subset, style in [
            (f"{pp} (SW)",  sw[sw["y_true_bin"]==1], dict(lw=1.8, ls="--")),
            (f"{pp} (HDL)", hd[hd["y_true_bin"]==1], dict(lw=2.2, ls="-")),
            (f"{pn} (SW)",  sw[sw["y_true_bin"]==0], dict(lw=1.8, ls="--")),
            (f"{pn} (HDL)", hd[hd["y_true_bin"]==0], dict(lw=2.2, ls="-")),
        ]:
            ax.hist(subset["score_photon"].to_numpy(dtype=float),
                    bins=bins, density=True, histtype="step", label=label, **style)

        ax.axvline(0.5, color="gray", lw=1.0, ls=":")
        ax.set_xlabel("Score = P(photon)")
        ax.set_ylabel("Density")
        ax.set_title(f"{pp} vs {pn} — SW vs HDL")

        ax.legend(loc="best", frameon=False, fontsize=9)

    for j in range(len(pairs), len(axes)):
        axes[j].set_axis_off()

    fig.suptitle("Photon vs X — score distributions (SW vs HDL)", y=0.995)
    fig.tight_layout()
    fig.savefig(out_prefix + ".pdf", bbox_inches="tight")
    fig.savefig(out_prefix + ".png", dpi=220, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV produced by append_photon_ovo_to_csv")
    ap.add_argument("--precision", type=int, required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    df = load_filtered(args.csv, args.precision, args.depth, args.rounds)
    tag = f"prec{args.precision}_d{args.depth}_r{args.rounds}"

    plot_roc_overlay(df,   out_prefix=f"{args.outdir}/roc_overlay_{tag}")
    plot_score_panels(df,  out_prefix=f"{args.outdir}/score_panels_{tag}")

if __name__ == "__main__":
    main()
