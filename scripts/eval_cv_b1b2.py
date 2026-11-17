#!/usr/bin/env python3
"""
eval_cv_b1b2.py

Frame-grouped 5-fold CV study of (1) what the AE adds over CP / ilastik for the 4 FA classes and
(2) whether B2 labels define the embedding / classifier better than B1 labels on the *representative* B1 test set.

For each fold: test = patches in held-out frames (B1 test set = B1 labels; B2 test set = B2 labels).
Probe training sources (frames outside the fold): B1, B2, B2m (B2 size-matched to B1), B1+B2 (B2 priority).
Representations: CP, ilastik, CP+ilastik, ConAE (unsupervised AE), AE_{b1,b2,b12} (SupCon-AE trained on that
label arm with the fold's labels hidden; seed 0 only), and CP+ilastik+AE_arm.
Metrics pooled over folds: balanced accuracy, macro-F1, per-class recall; cluster-bootstrap (by frame) CI for seed 0.

Outputs: results/cv_b1b2/{oof.csv, summary.csv, summary_seeds.csv}
"""
import argparse, sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_common import *

OUT = Path(__file__).resolve().parents[1] / "results" / "cv_b1b2"
SRCS, TESTS = ["B1", "B2", "B2m", "B1+B2"], ["B1", "B2"]
ap = argparse.ArgumentParser(); ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); ap.add_argument("--boot", type=int, default=300)
args = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)

b1, b2 = load_fa4()
cp = pd.read_csv(FEAT / "cellprofiler" / "ds1.csv").set_index("filename")
il = pd.read_csv(FEAT / "ilastik" / "ds1.csv").set_index("filename")
cl = pd.read_csv(RUN / "stage2_rebuild_conae" / "latents.csv").set_index("filename")
cl = cl[[c for c in cl.columns if c.startswith("z_")]]
FIXED = {"CP": cp, "IL": il, "CP+IL": pd.concat([cp, il], axis=1), "ConAE": cl}


def ae_latents(arm, fold):
    p = RUN / "cv_ae" / f"{arm}_f{fold}" / "latents.csv"
    if not p.exists():
        return None
    d = pd.read_csv(p).set_index("filename")
    return d[[c for c in d.columns if c.startswith("z_")]]


def probe(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))


rows = []
for seed in args.seeds:
    folds = make_folds(seed); rng = np.random.RandomState(seed)
    for k in range(N_FOLDS):
        T = {g for g, f in folds.items() if f == k}
        srcs = train_sources(b1, b2, T, rng)
        tests = {"B1": b1[b1.g.isin(T)], "B2": b2[b2.g.isin(T)]}
        reps = dict(FIXED)
        if seed == 0:
            for arm in ("b1", "b2", "b12"):
                z = ae_latents(arm, k)
                if z is not None:
                    reps[f"AE_{arm}"] = z
                    reps[f"CP+IL+AE_{arm}"] = pd.concat([cp, il], axis=1).join(z, how="inner")
        for rep, X in reps.items():
            for sname, tr in srcs.items():
                tr = tr[tr.fn.isin(X.index)]
                for tname, te in tests.items():
                    te = te[te.fn.isin(X.index)]
                    if len(te) == 0:
                        continue
                    pred = probe(X.loc[tr.fn].values, tr.label.values, X.loc[te.fn].values)
                    rows.append(pd.DataFrame({"seed": seed, "fold": k, "rep": rep, "src": sname, "test": tname,
                                              "fn": te.fn.values, "g": te.g.values, "y": te.label.values, "pred": pred}))
    print("seed", seed, "done", flush=True)
oof = pd.concat(rows, ignore_index=True); oof.to_csv(OUT / "oof.csv", index=False)


def metrics(d):
    cls = [c for c in FA4 if c in set(d.y)]
    r = {"n": len(d), "bal_acc": balanced_accuracy_score(d.y, d.pred), "macro_f1": f1_score(d.y, d.pred, average="macro", labels=cls)}
    for c, key in zip(FA4, ["nasc", "fcomp", "fadh", "fibr"]):
        r["rec_" + key] = recall_score(d.y == c, d.pred == c) if (d.y == c).any() else np.nan
    return r


per_seed = pd.DataFrame([{**dict(zip(["seed", "rep", "src", "test"], key)), **metrics(d)} for key, d in oof.groupby(["seed", "rep", "src", "test"])])
per_seed.to_csv(OUT / "summary_seeds.csv", index=False)

rng = np.random.RandomState(123); out = []
for key, d in oof[oof.seed == 0].groupby(["rep", "src", "test"]):
    frames = d.g.unique(); by = {g: x for g, x in d.groupby("g")}
    bs = []
    for _ in range(args.boot):
        s = pd.concat([by[g] for g in rng.choice(frames, len(frames))])
        bs.append(balanced_accuracy_score(s.y, s.pred))
    out.append({**dict(zip(["rep", "src", "test"], key)), **metrics(d), "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
summ = pd.DataFrame(out); summ.to_csv(OUT / "summary.csv", index=False)
pd.set_option("display.width", 220)
print(summ[summ.test == "B1"].pivot(index="rep", columns="src", values="bal_acc").round(3))
print(summ[summ.test == "B2"].pivot(index="rep", columns="src", values="bal_acc").round(3))
ms = per_seed.groupby(["rep", "src", "test"]).bal_acc.agg(["mean", "std"]).round(3)
print(ms.loc[["CP", "IL", "CP+IL", "ConAE"]].unstack("test").to_string())
