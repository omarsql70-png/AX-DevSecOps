#!/usr/bin/env python3
"""summarize.py: turn the raw CSVs of axd_experiments.py into the paper tables.
    python summarize.py results/
Writes table2_zoo.csv ... table5_defects.csv, sensitivity.csv and prints them."""
import json, sys, types
import numpy as np, pandas as pd
sys.path.insert(0, __import__("os").path.dirname(__file__) or ".")
from axd_experiments import gate_checks, fail_fast, GATE_ORDER

d = sys.argv[1] if len(sys.argv) > 1 else "results"
args = types.SimpleNamespace(**json.load(open(f"{d}/run_info.json"))["args"])
ms = lambda s: f"{s.mean():.3f} ± {s.std(ddof=1):.3f}" if len(s) > 1 else f"{s.mean():.3f}"
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

z = pd.read_csv(f"{d}/zoo_raw.csv")
rows = []
for m, g in z.groupby("model", sort=False):
    r, c = g[g.split == "random"], g[g.split == "chrono"]
    g3 = ((c.macro_f1 >= args.min_f1) & (c.fpr <= args.max_fpr)).sum()
    rows.append({"model": m, "F1_random": ms(r.macro_f1), "F1_chrono": ms(c.macro_f1), "FPR_chrono": ms(c.fpr),
                 "drop_pts": ms(100 * (r.macro_f1.values - c.macro_f1.values)), "size_kb": ms(c.size_kb), "p95_ms": ms(c.p95_ms),
                 "G3_pass_datasets": f"{g3}/{len(c)}"})
t2 = pd.DataFrame(rows); t2.to_csv(f"{d}/table2_zoo.csv", index=False); print("TABLE II\n", t2, "\n")

e = pd.read_csv(f"{d}/explanations_raw.csv")
rows = []
for m, g in e.groupby("model", sort=False):
    top1 = g.top_features.str.split(";").str[0].value_counts()
    rows.append({"model": m, "k80": ms(g.k80), "S10": ms(g.stability), "del_drop": ms(g.deletion_drop), "shap_ms": ms(g.explain_ms),
                 "top_feature_mode": f"{top1.index[0]} ({top1.iloc[0]}/{len(g)})", "src_port_rank": ",".join(map(str, g.src_port_rank))})
t3 = pd.DataFrame(rows); t3.to_csv(f"{d}/table3_explanations.csv", index=False); print("TABLE III\n", t3, "\n")

s = pd.read_csv(f"{d}/sprints_raw.csv")
rows = []
for (cand, sp), g in s.groupby(["candidate", "sprint"], sort=False):
    fails = {k: int((g[f"{k}_pass"] == False).sum()) for k in GATE_ORDER}
    rows.append({"candidate": cand, "sprint": sp, "n_feat": ms(g.n_features), "audit_removed": ";".join(sorted(set(g.removed.fillna("")) - {""})) or "-",
                 "datasets_flagging": int((g.removed.fillna("") != "").sum()),
                 "F1_random": ms(g.macro_f1_random), "F1_chrono": ms(g.macro_f1_chrono), "FPR_chrono": ms(g.fpr_chrono),
                 "size_mb": ms(g.size_kb / 1024), "k80": ms(g.k80), "S10": ms(g.stability), "fit_s": ms(g.fit_s), "xai_s": ms(g.xai_s),
                 "released": f"{int(g.release.sum())}/{len(g)}", **{f"{k}_fail": v for k, v in fails.items()}})
t4 = pd.DataFrame(rows); t4.to_csv(f"{d}/table4_sprints.csv", index=False); print("TABLE IV\n", t4.T, "\n")

D = pd.read_csv(f"{d}/defects_raw.csv")
def ff_rows(df, t=1.0):
    out = []
    for _, r in df.iterrows():
        m = r.to_dict()
        if isinstance(m.get("flagged"), float): m["flagged"] = ""
        g = gate_checks(m, args, t); out.append((g, fail_fast(g)))
    return out
rows = []
for case, g in D.groupby("case", sort=False):
    res = ff_rows(g)
    row = {"case": case, "runs": len(g), "blocked": sum(not f["release"] for _, f in res)}
    for k in GATE_ORDER:
        row[k] = sum(f[k] is False for _, f in res); row[k + "_nr"] = sum(f[k] is None for _, f in res)
    row["F1_chrono"] = ms(g.macro_f1_chrono); row["size_mb"] = ms(g.size_kb / 1024)
    row["solo_max"] = ms(g.solo_max_f1.dropna()) if g.solo_max_f1.notna().any() else "n/r"
    row["top_share"] = ms(g.top_share.dropna()) if g.top_share.notna().any() else "n/r"
    row["S10"] = ms(g.stability.dropna()) if g.stability.notna().any() else "n/r"
    row["k80"] = ms(g.k80.dropna()) if g.k80.notna().any() else "n/r"
    rows.append(row)
t5 = pd.DataFrame(rows); t5.to_csv(f"{d}/table5_defects.csv", index=False); print("TABLE V\n", t5.T, "\n")

rows = []
for t in [0.9, 1.0, 1.1]:
    row = {"tightness": t}
    for case, g in D.groupby("case", sort=False):
        res = ff_rows(g, t); row[case] = f"{sum(not f['release'] for _, f in res)}/{len(g)}"
    rows.append(row)
sens = pd.DataFrame(rows); sens.to_csv(f"{d}/sensitivity.csv", index=False); print("BLOCKED RUNS vs THRESHOLD TIGHTNESS\n", sens.T)
