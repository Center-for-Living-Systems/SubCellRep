#!/usr/bin/env python3
"""
eval_image_efficiency_b1b2v2.py

Image efficiency test on B1 / B2v2 matched label sets with nested label efficiency.

Split unit: image (frame).  Four frames with ≤3 patches are permanently excluded.

Experiments
-----------
1. LOIO (leave-one-image-out):  hold each of 22 valid images out one by one;
   train on all patches from the remaining 21 images — at multiple label budgets.
2. Image budget sweep:  K = 1..19 training images, 20 random repeats per K,
   nested label efficiency within each image budget.

Features  : CellProfiler and ilastik (pre-calculated on DS1)
Classifier: LGBM or LogReg (--clf lgbm / logreg)
Annotators: B1  (vinc_b1_matched_ds1_v2.csv)
            B2v2 (vinc_b2v2_matched_ds1.csv)

Output CSV
----------
  eval_results/image_efficiency_b1b2v2_{clf}.csv

  Columns:
    experiment, annotator, feature, clf,
    k_images, repeat, held_out_image,
    n_train_pool, label_budget, n_train,
    n_test_images, n_test_patches,
    bal_acc
"""
from __future__ import annotations

import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO     = Path(__file__).resolve().parent.parent
DATA     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
LAB      = DATA / "labelling"
FEAT     = DATA / "ae_results/features"
CP_CSV   = FEAT / "cellprofiler/ds1.csv"
IL_CSV   = FEAT / "ilastik/ds1.csv"
B1_CSV   = LAB / "vinc_b1_matched_ds1_v2.csv"
B2V2_CSV = LAB / "vinc_b2v2_matched_ds1.csv"
OUT_DIR  = REPO / "eval_results"
PRED_DIR = REPO / "eval_results" / "patch_preds"

# ── Constants ─────────────────────────────────────────────────────────────────
SEED = 42

# Permanently excluded images (≤3 patches) — tuple (cond, frame_int)
EXCLUDED = {("control", 6), ("control", 7), ("control", 19), ("ycomp", 0)}

# K = number of training images; 20 random repeats each
K_VALUES      = [1, 2, 3, 5, 7, 10, 13, 16, 19]
N_REPEATS     = 20
LABEL_BUDGETS = [5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, "all"]


# ── Helpers ───────────────────────────────────────────────────────────────────
def uid_to_fn(uid: str) -> str:
    """Convert unique_ID (dash-f) to feature CSV filename (underscore-f)."""
    return uid.replace("-f", "_f", 1)


def parse_image(uid: str) -> tuple[str, int]:
    """Return (cond, frame_int) from a unique_ID string."""
    import re
    m = re.match(r'^(control|ycomp)-f(\d+)', uid)
    if not m:
        raise ValueError(f"Cannot parse image from uid: {uid!r}")
    return m.group(1), int(m.group(2))


def image_key(uid: str) -> str:
    """Human-readable image key: 'control_f0001' style."""
    cond, fr = parse_image(uid)
    return f"{cond}_f{fr}"


def _build_clf(clf_name: str, seed: int):
    if clf_name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=200, learning_rate=0.05, num_leaves=15,
            random_state=seed, verbose=-1, n_jobs=1,
        )
    elif clf_name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0,
            solver="lbfgs", random_state=seed,
        )
    raise ValueError(f"Unknown clf: {clf_name}")


def fit_clf(clf_name: str, X_tr: np.ndarray, y_tr: np.ndarray,
            X_te: np.ndarray, y_te: np.ndarray,
            seed: int, return_preds: bool = False):
    sc = StandardScaler().fit(X_tr)
    clf = _build_clf(clf_name, seed)
    clf.fit(sc.transform(X_tr), y_tr)
    X_te_sc = sc.transform(X_te)
    y_pred = clf.predict(X_te_sc)
    ba = float(balanced_accuracy_score(y_te, y_pred))
    if return_preds:
        y_proba = clf.predict_proba(X_te_sc)[:, 1]
        return ba, y_pred, y_proba
    return ba


# ── Data loading ──────────────────────────────────────────────────────────────
def load_features() -> dict[str, pd.DataFrame]:
    """Load CP and ilastik feature tables, indexed by filename."""
    cp = pd.read_csv(CP_CSV).set_index("filename")
    il = pd.read_csv(IL_CSV).set_index("filename")
    return {"cp": cp, "ilastik": il}


def load_annotator(csv_path: Path, feat_tables: dict) -> dict:
    """
    Load an annotator CSV, join features, parse image keys, exclude bad images.

    Returns a dict with keys:
      patches : DataFrame with columns [filename, label_bin, image_key, *feat_cols]
      images  : sorted list of valid image_keys
      feat_cols : dict {feat_name: list of feature column names}
    """
    ann = pd.read_csv(csv_path)
    ann["filename"] = ann["unique_ID"].apply(uid_to_fn)
    ann["image_key"] = ann["unique_ID"].apply(image_key)

    # Exclude the 4 permanently excluded images
    def _excluded(uid: str) -> bool:
        try:
            return parse_image(uid) in EXCLUDED
        except ValueError:
            return False

    ann = ann[~ann["unique_ID"].apply(_excluded)].copy()
    ann["label_bin"] = (ann["label"] == "adhesion").astype(int)

    result_by_feat: dict[str, pd.DataFrame] = {}
    feat_cols_by: dict[str, list[str]] = {}

    for feat_name, feat_df in feat_tables.items():
        merged = ann.join(feat_df, on="filename", how="inner")
        cols = feat_df.columns.tolist()
        result_by_feat[feat_name] = merged[["filename", "image_key", "label_bin"] + cols].copy()
        feat_cols_by[feat_name] = cols

    # Valid images = images present after exclusion (use CP as reference)
    all_images = sorted(result_by_feat["cp"]["image_key"].unique())

    return {"patches": result_by_feat, "images": all_images, "feat_cols": feat_cols_by}


# ── Experiments ───────────────────────────────────────────────────────────────
def run_single(
    patches_df: pd.DataFrame,
    feat_cols: list[str],
    train_images: list[str],
    test_images: list[str],
    label_budget: int | str,
    rng: np.random.Generator,
    seed_clf: int,
    clf_name: str = "lgbm",
    return_preds: bool = False,
) -> dict | tuple | None:
    """
    Train on `train_images` patches (with label_budget subsampling) and test
    on `test_images` patches.  Returns result dict or None if infeasible.
    When return_preds=True, returns (result_dict, fns, y_pred, y_proba, y_te).
    """
    df_tr_pool = patches_df[patches_df["image_key"].isin(train_images)]
    df_te      = patches_df[patches_df["image_key"].isin(test_images)]

    if len(df_te) == 0 or len(df_tr_pool) == 0:
        return None

    n_pool = len(df_tr_pool)

    if label_budget == "all" or label_budget >= n_pool:
        df_tr = df_tr_pool
        n_train = n_pool
    else:
        idx = rng.choice(n_pool, size=label_budget, replace=False)
        df_tr = df_tr_pool.iloc[idx]
        n_train = label_budget

    X_tr = df_tr[feat_cols].values.astype(float)
    y_tr = df_tr["label_bin"].values
    X_te = df_te[feat_cols].values.astype(float)
    y_te = df_te["label_bin"].values

    if len(np.unique(y_tr)) < 2:
        return None

    result = dict(
        n_train_pool=n_pool,
        label_budget="all" if (label_budget == "all" or label_budget >= n_pool) else label_budget,
        n_train=n_train,
        n_test_images=len(test_images),
        n_test_patches=len(df_te),
    )

    if return_preds:
        ba, y_pred, y_proba = fit_clf(clf_name, X_tr, y_tr, X_te, y_te, seed_clf, return_preds=True)
        result["bal_acc"] = round(ba, 6)
        return result, df_te["filename"].values, y_pred, y_proba, y_te
    else:
        ba = fit_clf(clf_name, X_tr, y_tr, X_te, y_te, seed_clf)
        result["bal_acc"] = round(ba, 6)
        return result


def experiment_loio(
    ann_data: dict, feat_name: str,
    rng: np.random.Generator,
    clf_name: str = "lgbm",
    save_preds: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Leave-one-image-out: hold each valid image out, train on rest at all label budgets.

    Returns (rows, pred_rows).  pred_rows is non-empty only when save_preds=True;
    it contains per-patch predictions for the label_budget=="all" condition only.
    """
    patches_df = ann_data["patches"][feat_name]
    feat_cols  = ann_data["feat_cols"][feat_name]
    images     = ann_data["images"]

    rows: list[dict] = []
    pred_rows: list[dict] = []
    for i, hold_img in enumerate(images):
        train_imgs = [img for img in images if img != hold_img]
        test_imgs  = [hold_img]

        for lb in LABEL_BUDGETS:
            do_preds = save_preds and (lb == "all")
            res = run_single(
                patches_df, feat_cols, train_imgs, test_imgs,
                label_budget=lb, rng=rng, seed_clf=SEED + i,
                clf_name=clf_name, return_preds=do_preds,
            )
            if res is None:
                continue
            if do_preds:
                result_dict, fns, y_pred, y_proba, y_te = res
                for fn, yp, prob, yt in zip(fns, y_pred, y_proba, y_te):
                    pred_rows.append(dict(
                        fn=fn,
                        y_true=int(yt),
                        y_pred=int(yp),
                        prob_adhesion=float(prob),
                        held_out_image=hold_img,
                    ))
            else:
                result_dict = res
            rows.append(dict(
                experiment="loio",
                k_images=len(train_imgs),
                repeat=i,
                held_out_image=hold_img,
                **result_dict,
            ))
    return rows, pred_rows


def experiment_image_budget(
    ann_data: dict, feat_name: str,
    rng: np.random.Generator,
    clf_name: str = "lgbm",
) -> list[dict]:
    """Image budget sweep: K training images, N_REPEATS random selections, nested label eff."""
    patches_df = ann_data["patches"][feat_name]
    feat_cols  = ann_data["feat_cols"][feat_name]
    images     = np.array(ann_data["images"])
    n_images   = len(images)

    rows = []
    for k in K_VALUES:
        if k >= n_images:
            continue
        for rep in range(N_REPEATS):
            train_idx  = rng.choice(n_images, size=k, replace=False)
            train_imgs = images[train_idx].tolist()
            test_imgs  = [img for img in images if img not in train_imgs]

            for lb in LABEL_BUDGETS:
                res = run_single(
                    patches_df, feat_cols, train_imgs, test_imgs,
                    label_budget=lb, rng=rng, seed_clf=SEED + rep,
                    clf_name=clf_name,
                )
                if res is None:
                    continue
                rows.append(dict(
                    experiment="image_budget",
                    k_images=k,
                    repeat=rep,
                    held_out_image="",
                    **res,
                ))
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotators", nargs="+", default=["b1", "b2v2"],
                    choices=["b1", "b2v2"])
    ap.add_argument("--features", nargs="+", default=["cp", "ilastik"],
                    choices=["cp", "ilastik"])
    ap.add_argument("--clf", nargs="+", default=["lgbm"],
                    choices=["lgbm", "logreg"])
    ap.add_argument("--save_preds", action="store_true",
                    help="Save per-patch predictions (LOIO + label_budget==all only) "
                         "to eval_results/patch_preds/image_eff_b1b2v2_{clf}.csv")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading features ...", flush=True)
    feat_tables = load_features()
    print(f"  CP:      {feat_tables['cp'].shape}")
    print(f"  ilastik: {feat_tables['ilastik'].shape}")

    annotator_map = {
        "b1":   B1_CSV,
        "b2v2": B2V2_CSV,
    }

    for clf_name in args.clf:
        rng = np.random.default_rng(SEED)
        all_rows: list[dict] = []
        all_pred_rows: list[dict] = []

        for ann_name in args.annotators:
            print(f"\nAnnotator: {ann_name}  clf: {clf_name}", flush=True)
            ann_data = load_annotator(annotator_map[ann_name], feat_tables)
            print(f"  Valid images: {len(ann_data['images'])}  "
                  f"(CP patches: {len(ann_data['patches']['cp'])})", flush=True)
            print(f"  Images: {ann_data['images']}", flush=True)

            for feat_name in args.features:
                if feat_name not in ann_data["patches"]:
                    continue
                n_patches = len(ann_data["patches"][feat_name])
                print(f"  [{feat_name}] n_patches={n_patches}", flush=True)

                # LOIO
                print(f"    Running LOIO ...", flush=True)
                loio_rows, loio_pred_rows = experiment_loio(
                    ann_data, feat_name, rng, clf_name,
                    save_preds=args.save_preds,
                )
                for r in loio_rows:
                    r.update(annotator=ann_name, feature=feat_name, clf=clf_name)
                all_rows.extend(loio_rows)
                print(f"    LOIO: {len(loio_rows)} rows", flush=True)
                if args.save_preds and loio_pred_rows:
                    for r in loio_pred_rows:
                        r.update(annotator=ann_name, feature=feat_name, clf=clf_name)
                    all_pred_rows.extend(loio_pred_rows)
                    print(f"    LOIO patch preds: {len(loio_pred_rows)} rows", flush=True)

                # Image budget sweep
                print(f"    Running image budget sweep (K={K_VALUES}) ...", flush=True)
                budget_rows = experiment_image_budget(ann_data, feat_name, rng, clf_name)
                for r in budget_rows:
                    r.update(annotator=ann_name, feature=feat_name, clf=clf_name)
                all_rows.extend(budget_rows)
                print(f"    Image budget: {len(budget_rows)} rows", flush=True)

        # Assemble main output
        col_order = [
            "experiment", "annotator", "feature", "clf",
            "k_images", "repeat", "held_out_image",
            "n_train_pool", "label_budget", "n_train",
            "n_test_images", "n_test_patches",
            "bal_acc",
        ]
        out_df = pd.DataFrame(all_rows)[col_order]
        out_path = OUT_DIR / f"image_efficiency_b1b2v2_{clf_name}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved {len(out_df)} rows → {out_path}")
        print(out_df.groupby(["experiment", "annotator", "feature"])["bal_acc"].agg(["count", "mean", "std"]).round(3).to_string())

        # Save per-patch predictions (LOIO + all-budget only)
        if args.save_preds and all_pred_rows:
            PRED_DIR.mkdir(parents=True, exist_ok=True)
            pred_col_order = ["fn", "y_true", "y_pred", "prob_adhesion",
                              "annotator", "feature", "clf", "held_out_image"]
            pred_df = pd.DataFrame(all_pred_rows)[pred_col_order]
            pred_path = PRED_DIR / f"image_eff_b1b2v2_{clf_name}.csv"
            pred_df.to_csv(pred_path, index=False)
            print(f"Saved {len(pred_df)} patch prediction rows → {pred_path}")


if __name__ == "__main__":
    main()
