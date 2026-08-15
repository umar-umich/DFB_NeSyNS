"""Consolidate A1 into the cross-source response matrix (a paper deliverable)."""
import pathlib

import pandas as pd

ROOT = pathlib.Path("analysis/discern_v2/A1_complementarity")
SRC = ["DF40", "CDFv3", "CDFv2", "DFEval24", "DFDC", "DFDCP", "DFD"]
OPS = ["P0-DS", "P1a", "P1b", "P1c", "P1d", "P2a", "P3a", "P4"]

auc, resc, harm, overlap = {}, {}, {}, {}
for s in SRC:
    tf = ROOT / s / "threshold_free_auroc.csv"
    rh = ROOT / s / "rescue_harm_overall.csv"
    eo = ROOT / s / "error_overlap.csv"
    if tf.exists():
        d = pd.read_csv(tf).set_index("operator")["auroc"]
        auc[s] = d
    if rh.exists():
        d = pd.read_csv(rh).set_index("operator")
        resc[s], harm[s] = d["rescue"], d["harm"]
    if eo.exists():
        d = pd.read_csv(eo)
        d = d[d["op_i"] == "P0-DS"].set_index("op_j")["jaccard"]
        overlap[s] = d


def show(title, table, fmt="{:.3f}"):
    df = pd.DataFrame(table).reindex(OPS)
    print(f"\n### {title}")
    print(df.to_string(float_format=lambda v: fmt.format(v)))
    return df


a = show("Video-level AUROC per source (threshold-free)", auc)
print("\nmean AUROC across the 7 OOD sources:")
print(a.mean(axis=1).sort_values(ascending=False).to_string(float_format=lambda v: f"{v:.4f}"))

show("Rescue vs P0 (fraction of videos, frozen threshold)", resc)
show("Harm vs P0", harm)
o = show("Error overlap with P0 (Jaccard; lower = more complementary)", overlap)
print("\nmean error overlap with P0 across sources (lower is better):")
print(o.mean(axis=1).dropna().sort_values().to_string(float_format=lambda v: f"{v:.4f}"))
