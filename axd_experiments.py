#!/usr/bin/env python3
"""
axd_experiments.py

Experiment driver for the AX-DevSecOps paper. It produces every number in the
Results section (Tables II-V) and the supplementary CSVs.

    python axd_experiments.py --sim-seeds 7,8,9,10,11 --out results/   # simulator, 5 data sets
    python axd_experiments.py --data /path/to/AD_Flow-based-features --out results_real/

What it does
  * Five simulated data sets (--sim-seeds) or one real data set (--data); every table is reported
    per data set and summarized by summarize.py.
  * Fail-fast gates: G2a duplicate overlap, G4 edge budget, G3 detection, then the attribution
    checks G2b (shortcut audit) and G5 (explanation quality), which are not run once a cheaper
    gate has blocked the release.
  * Explanations that drive decisions (audit ranking, SHAP pruning, stability) use training flows only.
  * Sprint loop for XGB and RF; seeded-defect experiment on the XGB release candidate.
  * All gate metrics are stored, so threshold sensitivity can be recomputed offline.

Edge numbers (latency) are measured on whatever machine runs the script. Gate G1
(SAST / dependency / image scanning) is a pipeline concern and is not part of this script.

Requires: numpy, pandas, scikit-learn, xgboost, shap, matplotlib.
"""
import argparse
import glob
import io
import json
import os
import re
import sys
import time
import warnings
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

LABEL_CANDIDATES = ["Label", "label", "Attack", "attack", "Class", "class", "Attack_type", "attack_type"]
TIME_CANDIDATES = ["Timestamp", "timestamp", "Time", "time"]
# columns that identify a flow or a host rather than describe behaviour
ID_PATTERN = re.compile(r"flow.?id|src.?ip|dst.?ip|source.?ip|destination.?ip|timestamp|src.?port|source.?port", re.I)
BENIGN_PATTERN = re.compile(r"benign|normal", re.I)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- data
def synthetic_df(n=5000, seed=0):
    """Tiny stand-in with the same shape as the real data. Smoke test only."""
    rng = np.random.default_rng(seed)
    classes = ["Benign", "DDoS", "DoS", "Spoofing", "BruteForce", "Web"]
    y = rng.choice(len(classes), size=n, p=[0.45, 0.2, 0.15, 0.1, 0.06, 0.04])
    X = rng.normal(size=(n, 30)) + y[:, None] * rng.normal(0.4, 0.1, size=(1, 30))
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(30)])
    df["Src Port"] = (y * 9000 + rng.integers(0, 3000, size=n)).astype(float)  # planted shortcut
    df["Timestamp"] = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 10**6, size=n)), unit="s")
    df["Src IP"] = "10.0.0." + pd.Series(rng.integers(1, 250, size=n)).astype(str)
    df["Label"] = [classes[i] for i in y]
    return df


def load_csvs(data_dir, max_rows_per_file):
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not files:
        raise SystemExit(f"No CSV files found under {data_dir}")
    frames = []
    for f in files:
        d = pd.read_csv(f, low_memory=False, nrows=max_rows_per_file)
        d.columns = [c.strip() for c in d.columns]
        frames.append(d)
        log(f"loaded {os.path.basename(f)}: {len(d):,} rows")
    return pd.concat(frames, ignore_index=True)


def prepare(df, label_col, min_class_count, dedup=True):
    df = df.copy()
    label_col = label_col or next((c for c in LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        raise SystemExit(f"Could not find a label column; pass --label-col. Columns: {list(df.columns)[:15]}...")
    time_col = next((c for c in TIME_CANDIDATES if c in df.columns), None)
    t = None
    if time_col is not None:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        t = (ts - pd.Timestamp("1970-01-01")).dt.total_seconds()
    y_raw = df[label_col].astype(str).str.strip()
    feats = df.drop(columns=[c for c in [label_col, time_col] if c is not None]).select_dtypes(include=[np.number])
    feats = feats.replace([np.inf, -np.inf], np.nan)
    keep = feats.notna().all(axis=1)
    feats, y_raw = feats[keep], y_raw[keep]
    t = t[keep] if t is not None else None
    n0 = len(feats)
    if dedup:
        dup = pd.concat([feats, y_raw.rename("__y")], axis=1).duplicated(keep="first")
        feats, y_raw = feats[~dup], y_raw[~dup]
        t = t[~dup] if t is not None else None
    counts = y_raw.value_counts()
    small = counts[counts < min_class_count].index.tolist()
    if small:
        log(f"dropping classes with < {min_class_count} flows: {small}")
        m = ~y_raw.isin(small)
        feats, y_raw = feats[m], y_raw[m]
        t = t[m] if t is not None else None
    classes = sorted(y_raw.unique())
    y = y_raw.map({c: i for i, c in enumerate(classes)}).values.astype(int)
    benign_ids = [i for i, c in enumerate(classes) if BENIGN_PATTERN.search(c)]
    if not benign_ids:
        raise SystemExit(f"No benign class recognised among {classes}; edit BENIGN_PATTERN.")
    log(f"{n0:,} clean rows -> {len(feats):,} after de-duplication; {len(classes)} classes; benign id {benign_ids[0]}")
    order_key = t.values if (t is not None and np.isfinite(t.values).any()) else np.arange(len(feats), dtype=float)
    return feats.reset_index(drop=True), y, np.nan_to_num(order_key, nan=0.0), classes, benign_ids[0]


def split_random(y, seed, test_frac=0.2):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
    tr, te = next(sss.split(np.zeros(len(y)), y))
    return tr, te


def split_chrono(y, order_key, test_frac=0.2):
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        idx = idx[np.argsort(order_key[idx], kind="stable")]
        cut = int(round(len(idx) * (1 - test_frac)))
        tr.append(idx[:cut]); te.append(idx[cut:])
    return np.concatenate(tr), np.concatenate(te)


def cap_train(tr, y, cap, seed):
    if cap is None or len(tr) <= cap:
        return tr
    keep, _ = train_test_split(tr, train_size=cap, stratify=y[tr], random_state=seed)
    return np.sort(keep)


# ------------------------------------------------------------------------- models
def make_models(seed, fast=False):
    m = {
        "DT": DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=seed),
        "RF": RandomForestClassifier(n_estimators=25 if fast else 60, max_depth=12,
                                     class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced")),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                                                             max_iter=20 if fast else 60, random_state=seed)),
    }
    try:
        from xgboost import XGBClassifier
        m["XGB"] = XGBClassifier(n_estimators=50 if fast else 100, max_depth=6, tree_method="hist",
                                 n_jobs=-1, random_state=seed, verbosity=0)
    except Exception:
        pass
    return m


TREE_MODELS = {"DT", "RF", "XGB"}


def det_metrics(y_true, y_pred, proba, benign_id):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    yb, pb = (y_true != benign_id), (y_pred != benign_id)
    fp = np.sum(~yb & pb); tn = np.sum(~yb & ~pb)
    out = {"acc": accuracy_score(y_true, y_pred), "macro_p": p, "macro_r": r, "macro_f1": f,
           "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
           "fpr": fp / max(fp + tn, 1),
           "bin_f1": f1_score(yb, pb, zero_division=0)}
    try:
        out["bin_auc"] = roc_auc_score(yb, 1.0 - proba[:, benign_id])
    except Exception:
        out["bin_auc"] = np.nan
    return out


def edge_metrics(model, X_test, n_single=200, batch=5000):
    buf = io.BytesIO(); joblib.dump(model, buf, compress=0)
    lat = []
    for i in range(min(n_single, len(X_test))):
        t0 = perf_counter(); model.predict(X_test[i:i + 1]); lat.append((perf_counter() - t0) * 1000)
    xb = X_test[:batch]
    t0 = perf_counter(); model.predict(xb); dt = perf_counter() - t0
    return {"size_kb": buf.tell() / 1024, "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "throughput_fps": len(xb) / max(dt, 1e-9)}


def fit_eval(name, model, Xtr, ytr, Xte, yte, benign_id):
    t0 = perf_counter(); model.fit(Xtr, ytr); fit_s = perf_counter() - t0
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)
    if proba.shape[1] != (max(ytr.max(), yte.max()) + 1):  # class missing in train: pad
        full = np.zeros((len(Xte), max(ytr.max(), yte.max()) + 1)); full[:, model.classes_] = proba; proba = full
    d = det_metrics(yte, pred, proba, benign_id); d["fit_s"] = fit_s
    return d




# ---------------------------------------------------------------------------- XAI
def shap_abs(model, Xs):
    sv = shap.TreeExplainer(model).shap_values(Xs, check_additivity=False)
    if isinstance(sv, list):
        return np.stack([np.abs(s) for s in sv], axis=0).mean(axis=0)
    sv = np.asarray(sv)
    return np.abs(sv).mean(axis=2) if sv.ndim == 3 else np.abs(sv)


def k_at_share(imp, share=0.8):
    c = np.cumsum(np.sort(imp)[::-1]) / max(imp.sum(), 1e-12)
    return int(np.searchsorted(c, share) + 1)


def stability(factory, Xtr, ytr, Xs, k=10, reps=4, seed=0, boot_cap=40000):
    """Mean pairwise Jaccard of top-k SHAP sets over bootstrap retrains (training data only)."""
    rng = np.random.default_rng(seed); tops = []
    for r in range(reps):
        idx = rng.choice(len(Xtr), size=min(len(Xtr), boot_cap), replace=True)
        m = factory(seed + 100 + r).fit(Xtr[idx], ytr[idx])
        tops.append(set(np.argsort(-shap_abs(m, Xs).mean(axis=0))[:k]))
    js = [len(a & b) / len(a | b) for i, a in enumerate(tops) for b in tops[i + 1:]]
    return float(np.mean(js)) if js else np.nan


def deletion_drop(model, Xte, yte, top_idx, train_median):
    base = f1_score(yte, model.predict(Xte), average="macro", zero_division=0)
    Xm = Xte.copy(); Xm[:, top_idx] = train_median[top_idx]
    return base - f1_score(yte, model.predict(Xm), average="macro", zero_division=0)


def explain_latency(model, X, n=30):
    ex = shap.TreeExplainer(model); lat = []
    for i in range(min(n, len(X))):
        t0 = perf_counter(); ex.shap_values(X[i:i + 1], check_additivity=False); lat.append((perf_counter() - t0) * 1000)
    return float(np.median(lat))


def solo_probe(Xtr, ytr, Xte, yte, top_idx, depth=6):
    """Best macro-F1 reachable from ONE feature alone. A single column that separates all classes is a shortcut."""
    best, who = 0.0, None
    for j in top_idx:
        m = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr[:, [j]], ytr)
        f = f1_score(yte, m.predict(Xte[:, [j]]), average="macro", zero_division=0)
        if f > best:
            best, who = f, j
    return best, who


def xai_bundle(factory, model, Xtr, ytr, Xte, yte, names, shap_n, seed, k=10, reps=4):
    """Attributions that drive decisions (ranking, audit, pruning, stability) use training flows only.
    Test labels are used only to score deletion fidelity and the solo probe."""
    rng = np.random.default_rng(seed)
    xs = Xtr[rng.choice(len(Xtr), size=min(shap_n, len(Xtr)), replace=False)]
    imp = shap_abs(model, xs).mean(axis=0)
    order = np.argsort(-imp); top = order[:k]
    flagged = [names[i] for i in top if ID_PATTERN.search(names[i])]
    solo_f1, solo_j = solo_probe(Xtr, ytr, Xte, yte, order[:5])
    sub = rng.choice(len(Xte), size=min(20000, len(Xte)), replace=False)
    return {"imp": imp, "k80": k_at_share(imp), "top_share": float(imp[order[0]] / max(imp.sum(), 1e-12)),
            "flagged": flagged, "solo_max_f1": solo_f1, "solo_feature": names[solo_j] if solo_j is not None else None,
            "stability": stability(factory, Xtr, ytr, xs, k=k, reps=reps, seed=seed),
            "deletion_drop": deletion_drop(model, Xte[sub], yte[sub], top, np.median(Xtr, axis=0)),
            "explain_ms": explain_latency(model, Xte), "top_features": [names[i] for i in top]}


# --------------------------------------------------------------------------- gates
GATE_ORDER = ["G2a", "G4", "G3", "G2b", "G5"]  # cheapest first (size and latency need no test labels); G2a/G2b are the data and attribution halves of G2


def gate_checks(m, th, t=1.0):
    """All gate decisions from stored metrics. t > 1 tightens every threshold, t < 1 loosens it."""
    g = {"G2a": bool(m["dup_overlap"] == 0),
         "G3": bool(m["macro_f1_chrono"] >= min(th.min_f1 * t, 0.999) and m["fpr_chrono"] <= th.max_fpr / t),
         "G4": bool(m["size_kb"] <= th.max_size_kb / t and m["p95_ms"] <= th.max_p95_ms / t)}
    if m.get("stability") is not None and not (isinstance(m.get("stability"), float) and np.isnan(m["stability"])):
        flagged = m["flagged"] if isinstance(m["flagged"], list) else [f for f in str(m["flagged"] or "").split(";") if f]
        g["G2b"] = bool(not flagged and m["top_share"] <= th.max_top_share / t and m["solo_max_f1"] <= min(th.max_solo_f1 / t, 0.999))
        g["G5"] = bool(m["stability"] >= min(th.min_stab * t, 1.0) and m["k80"] <= th.max_k80 / t)
    else:
        g["G2b"], g["G5"] = None, None
    return g


def fail_fast(g):
    """What the pipeline reports: a gate after the first failure is 'not run' (None)."""
    out, blocked = {}, False
    for k in GATE_ORDER:
        out[k] = None if blocked else g[k]
        if g[k] is False:
            blocked = True
    out["release"] = all(g[k] is True for k in GATE_ORDER)
    return out


def dup_overlap(Xtr, Xte):
    a = pd.util.hash_pandas_object(pd.DataFrame(Xtr), index=False).values
    b = pd.util.hash_pandas_object(pd.DataFrame(Xte), index=False).values
    return int(np.isin(b, a).sum())


def run_config(feat_idx, names, X, y, splits, benign_id, args, seed, factory, y_train=None):
    """Train a candidate on every split given and collect what the gates need (chrono split).
    Attribution metrics are skipped only when the model already busts the size budget (G4):
    explaining a 290 MB forest is exactly the cost fail-fast exists to avoid."""
    res = {"n_features": len(feat_idx)}
    yy = y if y_train is None else y_train
    for tag, (tr, te) in splits.items():
        Xtr, Xte = X[tr][:, feat_idx], X[te][:, feat_idx]
        m = factory(seed); d = fit_eval("cand", m, Xtr, yy[tr], Xte, y[te], benign_id)
        res[f"macro_f1_{tag}"], res[f"fpr_{tag}"] = d["macro_f1"], d["fpr"]
        if tag == "chrono":
            res["fit_s"] = d["fit_s"]
            res.update(edge_metrics(m, Xte)); res["dup_overlap"] = dup_overlap(Xtr, Xte)
            if res["size_kb"] <= args.max_size_kb:
                tx = perf_counter()
                xb = xai_bundle(factory, m, Xtr, yy[tr], Xte, y[te], [names[i] for i in feat_idx], args.shap_n, seed, reps=args.stab_reps)
                res["xai_s"] = perf_counter() - tx
                res.update({k: xb[k] for k in ["imp", "k80", "top_share", "flagged", "solo_max_f1", "solo_feature",
                                                "stability", "deletion_drop", "explain_ms", "top_features"]})
    return res


def flat(r):
    o = {k: v for k, v in r.items() if k != "imp"}
    for k in ("flagged", "top_features"):
        if k in o and isinstance(o[k], list):
            o[k] = ";".join(o[k])
    return o


# ------------------------------------------------------------------------ one data set
def run_dataset(df, args, ds):
    n_raw = len(df)
    feats, y, order_key, classes, benign_id = prepare(df, args.label_col, args.min_class_count, dedup=not args.no_dedup)
    names = list(feats.columns); X = feats.values.astype(np.float64)
    info = {"data_seed": ds, "rows_raw": n_raw, "rows_used": len(y), "n_features": len(names), "classes": classes}
    sp = {"random": split_random(y, 0), "chrono": split_chrono(y, order_key)}
    sp = {k: (cap_train(a, y, args.train_cap, 0), b) for k, (a, b) in sp.items()}
    info["n_train"], info["n_test"] = len(sp["chrono"][0]), len(sp["chrono"][1])
    fac = {n: (lambda s, n=n: make_models(s, args.fast)[n]) for n in make_models(0, args.fast)}

    # Table II: model zoo
    zoo = []
    for name in fac:
        for tag, (tr, te) in sp.items():
            m = fac[name](0); d = fit_eval(name, m, X[tr], y[tr], X[te], y[te], benign_id)
            if tag == "chrono":
                d.update(edge_metrics(m, X[te]))
            zoo.append({"data_seed": ds, "model": name, "split": tag, **d})
        log(f"[ds {ds}] zoo {name} done")

    # Table IV (and III): sprint loop for both ensemble candidates
    sprints, expl = [], []
    all_idx = np.arange(len(names))
    for cand in ["XGB", "RF"]:
        s0 = run_config(all_idx, names, X, y, sp, benign_id, args, 0, fac[cand])
        flagged = list(s0["flagged"])
        idx1 = np.array([i for i in all_idx if names[i] not in flagged])
        s1 = run_config(idx1, names, X, y, sp, benign_id, args, 0, fac[cand]) if flagged else s0
        idx2 = idx1[np.argsort(-s1["imp"])[:args.prune_k]]
        s2 = run_config(idx2, names, X, y, sp, benign_id, args, 0, fac[cand])
        for tag, s in [("S0", s0), ("S1", s1), ("S2", s2)]:
            g = gate_checks(s, args)
            sprints.append({"data_seed": ds, "candidate": cand, "sprint": tag, "removed": ";".join(flagged) if tag != "S0" else "",
                            **flat(s), **{f"{k}_pass": v for k, v in g.items()}, "release": fail_fast(g)["release"]})
        expl.append({"data_seed": ds, "model": cand, **{k: flat(s0)[k] for k in ["k80", "stability", "deletion_drop", "explain_ms", "top_features", "top_share"]},
                     "src_port_rank": (s0["top_features"].index("Src Port") + 1) if "Src Port" in s0["top_features"] else ">10"})
        if cand == "XGB":
            xgb_s1_idx = idx1
        log(f"[ds {ds}] sprints {cand} done; flagged {flagged}")
    tr, te = sp["chrono"]
    m = fac["DT"](0).fit(X[tr], y[tr])
    xb = xai_bundle(fac["DT"], m, X[tr], y[tr], X[te], y[te], names, args.shap_n, 0, reps=args.stab_reps)
    expl.append({"data_seed": ds, "model": "DT", **{k: flat(xb)[k] for k in ["k80", "stability", "deletion_drop", "explain_ms", "top_features", "top_share"]},
                 "src_port_rank": (xb["top_features"].index("Src Port") + 1) if "Src Port" in xb["top_features"] else ">10"})

    # Table V: seeded defects on the XGB release candidate (S1 features), chrono split only
    defects = []
    base_idx = list(xgb_s1_idx)
    def with_col(col, nm):
        return np.hstack([X, col[:, None]]), names + [nm]
    cases = ["clean control", "label copy", "noisy label proxy", "15% label flips", "oversized forest", "2% training set", "train/test duplicates"]
    for case in cases:
        for rep in range(args.defect_reps):
            rng = np.random.default_rng(1000 * ds + 17 * rep + cases.index(case))
            Xd, nd, idx, ytr_over, f = X, names, list(base_idx), None, fac["XGB"]
            tr, te = sp["chrono"]
            if case == "label copy":
                Xd, nd = with_col(y + rng.normal(0, 0.05, len(y)), "aux_score"); idx = idx + [len(nd) - 1]
            elif case == "noisy label proxy":
                proxy = np.where(rng.random(len(y)) < args.proxy_p, y, rng.integers(0, len(classes), len(y))) + rng.normal(0, 0.3, len(y))
                Xd, nd = with_col(proxy, "aux_score"); idx = idx + [len(nd) - 1]
            elif case == "15% label flips":
                ytr_over = y.copy(); msk = rng.random(len(y)) < 0.15
                ytr_over[msk] = rng.integers(0, len(classes), msk.sum())
            elif case == "oversized forest":
                f = lambda s: RandomForestClassifier(n_estimators=100 if args.fast else 400, n_jobs=-1, random_state=s)
            elif case == "2% training set":
                tr = np.sort(rng.choice(tr, size=max(int(0.02 * len(tr)), 200), replace=False))
            elif case == "train/test duplicates":
                tr = np.concatenate([tr, rng.choice(te, size=int(0.05 * len(te)), replace=False)])
            r = run_config(np.array(idx), nd, Xd, y, {"chrono": (tr, te)}, benign_id, args, rep, f, y_train=ytr_over)
            g = gate_checks(r, args); ff = fail_fast(g)
            defects.append({"data_seed": ds, "case": case, "rep": rep, **flat(r),
                            **{f"{k}_pass": v for k, v in g.items()}, **{f"{k}_ff": v for k, v in ff.items()}})
            log(f"[ds {ds}] defect '{case}' rep{rep}: {ff}")
    return info, zoo, expl, sprints, defects


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data"); ap.add_argument("--out", default="results")
    ap.add_argument("--sim-seeds", help="comma list; regenerate sim_iot_flows.py once per seed")
    ap.add_argument("--sim-n", type=int, default=45000)
    ap.add_argument("--synthetic", action="store_true", help="smoke test on fake data; results are meaningless")
    ap.add_argument("--label-col"); ap.add_argument("--max-rows-per-file", type=int, default=None)
    ap.add_argument("--min-class-count", type=int, default=50); ap.add_argument("--train-cap", type=int, default=300000)
    ap.add_argument("--shap-n", type=int, default=300); ap.add_argument("--stab-reps", type=int, default=4)
    ap.add_argument("--prune-k", type=int, default=16); ap.add_argument("--defect-reps", type=int, default=2)
    ap.add_argument("--proxy-p", type=float, default=0.5, help="share of flows where the noisy proxy equals the label")
    ap.add_argument("--no-dedup", action="store_true"); ap.add_argument("--fast", action="store_true")
    # gate thresholds (team policy; choose them BEFORE looking at test results)
    ap.add_argument("--min-f1", type=float, default=0.80); ap.add_argument("--max-fpr", type=float, default=0.15)
    ap.add_argument("--max-size-kb", type=float, default=20000); ap.add_argument("--max-p95-ms", type=float, default=50)
    ap.add_argument("--min-stab", type=float, default=0.6); ap.add_argument("--max-k80", type=int, default=16)
    ap.add_argument("--max-top-share", type=float, default=0.5)
    ap.add_argument("--max-solo-f1", type=float, default=0.9, help="G2: one feature alone must not reach this macro-F1")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    if args.sim_seeds:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import sim_iot_flows as sim
        sources = [(int(s), (lambda s=int(s): sim.make(args.sim_n, seed=s))) for s in args.sim_seeds.split(",")]
    elif args.synthetic:
        log("*** SYNTHETIC SMOKE TEST: numbers below are NOT results ***"); sources = [(0, synthetic_df)]
    elif args.data:
        sources = [(0, lambda: load_csvs(args.data, args.max_rows_per_file))]
    else:
        raise SystemExit("Give --sim-seeds, --data DIR or --synthetic")

    infos, Z, E, S, D = [], [], [], [], []
    for ds, make_df in sources:
        info, zoo, expl, sprints, defects = run_dataset(make_df(), args, ds)
        infos.append(info); Z += zoo; E += expl; S += sprints; D += defects
        for nm, rows in [("zoo_raw", Z), ("explanations_raw", E), ("sprints_raw", S), ("defects_raw", D)]:
            pd.DataFrame(rows).to_csv(f"{args.out}/{nm}.csv", index=False)  # checkpoint after every data set
    import platform, sklearn, xgboost
    json.dump({"python": platform.python_version(), "sklearn": sklearn.__version__, "shap": shap.__version__,
               "xgboost": xgboost.__version__, "cpu": platform.processor() or platform.machine(),
               "datasets": infos, "args": vars(args)}, open(f"{args.out}/run_info.json", "w"), indent=2)
    log(f"done. Raw tables in {args.out}/; run summarize.py for the paper tables")


if __name__ == "__main__":
    main()
