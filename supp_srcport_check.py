#!/usr/bin/env python3
"""supp_srcport_check.py: does source port act as a shortcut for each candidate?
Trains XGB and RF with and without Src Port on the five simulated data sets (chrono and random splits).
    python supp_srcport_check.py results/
"""
import sys, os
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_iot_flows as sim, axd_experiments as ax
out = sys.argv[1] if len(sys.argv) > 1 else "results"
rows = []
for ds in [7, 8, 9, 10, 11]:
    f, y, ok, cl, b = ax.prepare(sim.make(45000, seed=ds), None, 50); names = list(f.columns); X = f.values.astype(float)
    sp = {"random": ax.split_random(y, 0), "chrono": ax.split_chrono(y, ok)}
    for model in ["XGB", "RF"]:
        for tag, idx in [("with", list(range(len(names)))), ("without", [i for i, n in enumerate(names) if n != "Src Port"])]:
            r = {"data_seed": ds, "model": model, "src_port": tag}
            for s, (tr, te) in sp.items():
                m = ax.make_models(0)[model].fit(X[tr][:, idx], y[tr])
                d = ax.det_metrics(y[te], m.predict(X[te][:, idx]), m.predict_proba(X[te][:, idx]), b)
                r[f"f1_{s}"], r[f"fpr_{s}"] = d["macro_f1"], d["fpr"]
            rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(f"{out}/supp_srcport_check.csv", index=False)
