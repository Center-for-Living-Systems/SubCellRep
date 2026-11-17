#!/usr/bin/env python3
"""
eval_image_efficiency_b1_full.py

Image efficiency test on Margaret's full label set (labels_vinc_20260521.csv).
1340 patches across 89 frames (50 control + 39 ycomp), converted to 2-class.
'Uncertain' patches excluded (~30), leaving ~1310 usable patches.

Experiments:
  1. image_budget: K ∈ K_VALUES training images, N_REPEATS random repeats per K.
     For each repeat, vary label budget within the training pool.
  2. loio: leave-one-image-out — train on all-1 images, test on 1.

Features: CellProfiler and ilastik (pre-calculated on DS1).
Classifier: LightGBM or LogReg (--clf lgbm / logreg).

Output: eval_results/image_efficiency_b1_full_{clf}.csv
"""
from __future__ import annotations

import argparse
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

REPO     = Path(__file__).resolve().parent.parent
DATA     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LAB      = DATA / "labelling"
FEAT     = DATA / "ae_results/features"
CP_CSV   = FEAT / "cellprofiler/ds1.csv"
IL_CSV   = FEAT / "ilastik/ds1.csv"
LABEL_CSV = LAB / "labels_vinc_20260521.csv"
OUT_DIR  = REPO / "eval_results"
PRED_DIR = REPO / "eval_results" / "patch_preds"

SEED = 42

# Multi-class → binary mapping
ADHESION_CLASSES  = {"focal adhesion", "Nascent Adhesion", "fibrillar adhesion", "focal complex"}
NOAD_CLASSES      = {"No adhesion"}
EXCLUDE_CLASSES   = {"Uncertain"}

K_VALUES      = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70]
N_REPEATS     = 20
LABEL_BUDGETS = [5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, "all"]

def _build_clf(clf_name: str):
    if clf_name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            min_child_samples=1, class_weight="balanced",
            n_jobs=4, random_state=42, verbose=-1,
        )
    elif clf_name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0,
            solver="lbfgs", random_state=42,
        )
    raise ValueError(f"Unknown clf: {clf_name}")


def uid_to_fn(uid: str) -> str:
    """control-f0001x0592y0560ps32.tif → control_f0001x0592y0560ps32.tif"""
    return uid.replace("-f", "_f", 1)


def parse_image(uid: str) -> tuple[str, int]:
    m = re.match(r'^(control|ycomp)-f(\d+)', uid)
    if not m:
        raise ValueError(f"Cannot parse image from uid: {uid!r}")
    return m.group(1), int(m.group(2))


def to_binary(cls: str) -> str | None:
    if cls in ADHESION_CLASSES:
        return "adhesion"
    if cls in NOAD_CLASSES:
        return "No adhesion"
    return None


def load_labels() -> pd.DataFrame:
    df = pd.read_csv(LABEL_CSV)
    df["binary_label"] = df["classification"].apply(to_binary)
    df = df[df["binary_label"].notna()].copy()
    df["fn"] = df["unique_ID"].apply(uid_to_fn)
    df[["cond", "frame_idx"]] = pd.DataFrame(
        df["unique_ID"].apply(parse_image).tolist(), index=df.index
    )
    df["image_key"] = df["cond"] + "-f" + df["frame_idx"].astype(str).str.zfill(2)
    print(f"Loaded {len(df)} patches (after Uncertain exclusion) from "
          f"{df['image_key'].nunique()} images")
    print(f"  adhesion: {(df.binary_label=='adhesion').sum()}  "
          f"No adhesion: {(df.binary_label=='No adhesion').sum()}")
    return df


def load_features(feat: str) -> pd.DataFrame:
    csv = CP_CSV if feat == "cp" else IL_CSV
    df = pd.read_csv(csv)
    feat_cols = [c for c in df.columns if c != "filename"]
    return df[["filename"] + feat_cols].set_index("filename")


def strat_sample(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Stratified random sample of n from pool (by binary_label)."""
    groups = {lbl: grp.reset_index(drop=True)
              for lbl, grp in pool.groupby("binary_label")}
    total = len(pool)
    alloc = {}
    for lbl, grp in groups.items():
        alloc[lbl] = max(1, int(round(n * len(grp) / total)))
        alloc[lbl] = min(alloc[lbl], len(grp))
    diff = n - sum(alloc.values())
    if diff != 0:
        largest = max(groups, key=lambda l: len(groups[l]))
        alloc[largest] = min(len(groups[largest]), max(1, alloc[largest] + diff))
    parts = []
    for lbl, grp in groups.items():
        k = alloc[lbl]
        idx = rng.choice(len(grp), size=k, replace=False)
        parts.append(grp.iloc[idx])
    return pd.concat(parts, ignore_index=True)


def score(clf, X_te, y_te) -> float:
    if len(X_te) == 0 or len(np.unique(y_te)) < 2:
        return float("nan")
    y_pred = clf.predict(X_te)
    return round(balanced_accuracy_score(y_te, y_pred), 6)


def run_eval(feat: str, clf_name: str, labels: pd.DataFrame, feats_all: pd.DataFrame,
             save_preds: bool = False) -> tuple[pd.DataFrame, list]:
    images = sorted(labels["image_key"].unique())
    n_images = len(images)
    print(f"\n--- feature={feat}  clf={clf_name}  total_images={n_images} ---")

    rows = []
    patch_rows = []
    rng_master = np.random.default_rng(SEED)

    # ── Exp 1: image budget ───────────────────────────────────────────────────
    for k in K_VALUES:
        if k >= n_images:
            print(f"  k={k} >= n_images={n_images}, skipping")
            continue
        print(f"  k={k} training images …", flush=True)

        for rep in range(N_REPEATS):
            rng = np.random.default_rng(rng_master.integers(0, 2**32))
            train_imgs = rng.choice(images, size=k, replace=False).tolist()
            test_imgs  = [im for im in images if im not in train_imgs]

            tr_pool = labels[labels["image_key"].isin(train_imgs)].copy()
            te_df   = labels[labels["image_key"].isin(test_imgs)].copy()

            X_te_raw = feats_all.reindex(te_df["fn"]).values
            y_te_raw = (te_df["binary_label"] == "adhesion").astype(int).values
            te_ok    = ~np.isnan(X_te_raw).any(axis=1)
            X_te_raw, y_te_raw = X_te_raw[te_ok], y_te_raw[te_ok]

            if len(y_te_raw) == 0 or len(np.unique(y_te_raw)) < 2:
                continue

            for budget in LABEL_BUDGETS:
                n_pool = len(tr_pool)
                if budget == "all":
                    tr_sub = tr_pool
                else:
                    n = int(budget)
                    if n > n_pool:
                        continue
                    tr_sub = strat_sample(tr_pool, n, rng)

                if len(tr_sub) == 0 or len(tr_sub["binary_label"].unique()) < 2:
                    continue

                X_tr_raw = feats_all.reindex(tr_sub["fn"]).values
                y_tr_raw = (tr_sub["binary_label"] == "adhesion").astype(int).values
                tr_ok    = ~np.isnan(X_tr_raw).any(axis=1)
                X_tr, y_tr = X_tr_raw[tr_ok], y_tr_raw[tr_ok]
                if len(X_tr) == 0 or len(np.unique(y_tr)) < 2:
                    continue

                sc = StandardScaler().fit(X_tr)
                model = _build_clf(clf_name)
                model.fit(sc.transform(X_tr), y_tr)
                ba = score(model, sc.transform(X_te_raw), y_te_raw)

                rows.append(dict(
                    experiment="image_budget", feature=feat, clf=clf_name,
                    k_images=k, repeat=rep,
                    held_out_image=None,
                    n_train_pool=n_pool,
                    label_budget=budget,
                    n_train=int(tr_ok.sum()),
                    n_test_images=len(test_imgs),
                    n_test_patches=int(te_ok.sum()),
                    bal_acc=ba,
                ))

    # ── Exp 2: LOIO ───────────────────────────────────────────────────────────
    print(f"  LOIO (n={n_images}) …", flush=True)
    for held_out in images:
        train_imgs = [im for im in images if im != held_out]
        tr_pool    = labels[labels["image_key"].isin(train_imgs)].copy()
        te_df      = labels[labels["image_key"] == held_out].copy()

        X_te_raw = feats_all.reindex(te_df["fn"]).values
        y_te_raw = (te_df["binary_label"] == "adhesion").astype(int).values
        te_ok    = ~np.isnan(X_te_raw).any(axis=1)
        X_te_raw, y_te_raw = X_te_raw[te_ok], y_te_raw[te_ok]
        if len(y_te_raw) == 0:
            continue

        for budget in LABEL_BUDGETS:
            rng = np.random.default_rng(SEED)
            n_pool = len(tr_pool)
            if budget == "all":
                tr_sub = tr_pool
            else:
                n = int(budget)
                if n > n_pool:
                    continue
                tr_sub = strat_sample(tr_pool, n, rng)

            if len(tr_sub) == 0 or len(tr_sub["binary_label"].unique()) < 2:
                continue

            X_tr_raw = feats_all.reindex(tr_sub["fn"]).values
            y_tr_raw = (tr_sub["binary_label"] == "adhesion").astype(int).values
            tr_ok    = ~np.isnan(X_tr_raw).any(axis=1)
            X_tr, y_tr = X_tr_raw[tr_ok], y_tr_raw[tr_ok]
            if len(X_tr) == 0 or len(np.unique(y_tr)) < 2:
                continue

            sc = StandardScaler().fit(X_tr)
            model = _build_clf(clf_name)
            model.fit(sc.transform(X_tr), y_tr)
            ba = score(model, sc.transform(X_te_raw), y_te_raw)

            if save_preds and budget == "all":
                prob = model.predict_proba(sc.transform(X_te_raw))[:, 1]
                te_fns = te_df["fn"].values[te_ok]
                y_pred_arr = model.predict(sc.transform(X_te_raw))
                for fn_i, yt, yp, pb in zip(te_fns, y_te_raw, y_pred_arr, prob):
                    patch_rows.append(dict(
                        fn=fn_i, y_true=int(yt), y_pred=int(yp), prob_adhesion=float(pb),
                        feature=feat, clf=clf_name, held_out_image=held_out,
                    ))

            rows.append(dict(
                experiment="loio", feature=feat, clf=clf_name,
                k_images=n_images - 1,
                repeat=0,
                held_out_image=held_out,
                n_train_pool=n_pool,
                label_budget=budget,
                n_train=int(tr_ok.sum()),
                n_test_images=1,
                n_test_patches=int(te_ok.sum()),
                bal_acc=ba,
            ))

    return pd.DataFrame(rows), patch_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="+", default=["cp", "ilastik"])
    ap.add_argument("--clf",   nargs="+", default=["lgbm"], choices=["lgbm", "logreg"])
    ap.add_argument("--save_preds", action="store_true",
                    help="Save per-patch LOIO predictions (budget=all only) to eval_results/patch_preds/")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    labels = load_labels()

    for clf_name in args.clf:
        all_dfs = []
        all_patch_rows = []
        for feat in args.feats:
            feats_all = load_features(feat)
            df, patch_rows = run_eval(feat, clf_name, labels, feats_all,
                                      save_preds=args.save_preds)
            all_dfs.append(df)
            all_patch_rows.extend(patch_rows)
            print(f"  {feat} {clf_name}: {len(df)} rows")

        if not all_dfs:
            print("No results.")
            continue

        result = pd.concat(all_dfs, ignore_index=True)
        out_csv = OUT_DIR / f"image_efficiency_b1_full_{clf_name}.csv"
        result.to_csv(out_csv, index=False)
        print(f"\nSaved → {out_csv}  ({len(result)} rows)")

        if args.save_preds and all_patch_rows:
            PRED_DIR.mkdir(parents=True, exist_ok=True)
            pred_csv = PRED_DIR / f"image_eff_b1_full_{clf_name}.csv"
            pd.DataFrame(all_patch_rows).to_csv(pred_csv, index=False)
            print(f"Saved patch preds → {pred_csv}  ({len(all_patch_rows)} rows)")

        # Quick summary
        for feat in args.feats:
            sub = result[(result.feature == feat) & (result.experiment == "image_budget") &
                         (result.label_budget == "all")]
            by_k = sub.groupby("k_images")["bal_acc"].mean().round(3)
            print(f"\n{feat} image_budget all-labels:")
            print(by_k.to_string())
            loio = result[(result.feature == feat) & (result.experiment == "loio") &
                          (result.label_budget == "all")]
            if not loio.empty:
                print(f"  LOIO mean: {loio.bal_acc.mean():.3f} ± {loio.bal_acc.std():.3f}")


if __name__ == "__main__":
    main()
