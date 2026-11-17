#!/usr/bin/env python3
"""
eval_image_efficiency_cp_combo.py

Image-efficiency test for DS1 using combinations of CP s32, CP s128, and
distance-to-boundary features — modelled on eval_image_efficiency_b1b2v2.py.

Feature sets tested:
  cp_s32, cp_s128, dist,
  cp_s32+s128, cp_s32+dist, cp_s32+s128+dist

Experiments:
  LOIO          : leave-one-image-out, multiple label budgets
  image_budget  : K = 1..19 training images, 20 repeats, nested label eff.

Annotators: B1 (vinc_b1_matched_ds1_v2.csv), B2v2 (vinc_b2v2_matched_ds1.csv)
Classifiers: lgbm, logreg

Outputs:
  eval_results/image_efficiency_cp_combo_{clf}.csv
    Columns: experiment, annotator, feature, clf,
             k_images, repeat, held_out_image,
             n_train_pool, label_budget, n_train,
             n_test_images, n_test_patches, bal_acc

  eval_results/patch_preds/image_eff_cp_combo_{clf}.csv  (LOIO + label_budget=all)
    Columns: fn, y_true, y_pred, prob_adhesion,
             annotator, feature, clf, held_out_image

Usage:
  python scripts/eval_image_efficiency_cp_combo.py --clf lgbm logreg
  python scripts/eval_image_efficiency_cp_combo.py --clf lgbm --feat cp_s32+s128+dist
"""
from __future__ import annotations
import argparse, re, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths

REPO     = Path(__file__).resolve().parent.parent
DATA     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
SPACES   = Path("/net/spaces/clinic/CLS/lding/data/fa_data_analysis")
LAB      = DATA / "labelling"
B1_CSV   = LAB / "vinc_b1_matched_ds1_v2.csv"
B2V2_CSV = LAB / "vinc_b2v2_matched_ds1.csv"
OUT_DIR  = REPO / "eval_results"
PRED_DIR = REPO / "eval_results" / "patch_preds"

FEAT_CSVS = {
    "cp_s16":        SPACES / "ae_results/features/cp_scale/ds1_s16.csv",
    "cp_s32":        SPACES / "ae_results/features/cp_scale/ds1_s32.csv",
    "cp_s64":        SPACES / "ae_results/features/cp_scale/ds1_s64.csv",
    "cp_s128":       SPACES / "ae_results/features/cp_scale/ds1_s128.csv",
    "dist":          SPACES / "ae_results/features/dist_features/ds1.csv",
    "cp_s16_ch2":    SPACES / "ae_results/features/cp_scale/ds1_ch2_s16.csv",
    "cp_s32_ch2":    SPACES / "ae_results/features/cp_scale/ds1_ch2_s32.csv",
    "cp_s64_ch2":    SPACES / "ae_results/features/cp_scale/ds1_ch2_s64.csv",
    "cp_s128_ch2":   SPACES / "ae_results/features/cp_scale/ds1_ch2_s128.csv",
    "cp_s16_ch3":    SPACES / "ae_results/features/cp_scale/ds1_ch3_s16.csv",
    "cp_s32_ch3":    SPACES / "ae_results/features/cp_scale/ds1_ch3_s32.csv",
    "cp_s64_ch3":    SPACES / "ae_results/features/cp_scale/ds1_ch3_s64.csv",
    "cp_s128_ch3":   SPACES / "ae_results/features/cp_scale/ds1_ch3_s128.csv",
    "il_s32":        SPACES / "ae_results/features/ilastik_scale/ds1_s32.csv",
    "supcon_z":      SPACES / "ae_results/features/supcon/ds1_z.csv",
    "supcon_p":      SPACES / "ae_results/features/supcon/ds1_p.csv",
}

_META_DROP = {"filename", "condition", "condition_name", "frame_idx",
              "canvas_cx", "canvas_cy", "group", "ps"}

FEAT_COMBOS = {
    # --- baselines (single feature source) ---
    "cp_s32":              ["cp_s32"],
    "cp_s128":             ["cp_s128"],
    "dist":                ["dist"],
    # --- previously run multi-source combos ---
    "cp_s32+s128":         ["cp_s32", "cp_s128"],
    "cp_s32+dist":         ["cp_s32", "dist"],
    "cp_s32+s128+dist":    ["cp_s32", "cp_s128", "dist"],
    # --- cp_s32 paired with every other single feature ---
    "cp_s16+cp_s32":       ["cp_s16", "cp_s32"],
    "cp_s32+cp_s64":       ["cp_s32", "cp_s64"],
    "cp_s32+cp_s16_ch2":   ["cp_s32", "cp_s16_ch2"],
    "cp_s32+cp_s32_ch2":   ["cp_s32", "cp_s32_ch2"],
    "cp_s32+cp_s64_ch2":   ["cp_s32", "cp_s64_ch2"],
    "cp_s32+cp_s128_ch2":  ["cp_s32", "cp_s128_ch2"],
    "cp_s32+cp_s16_ch3":   ["cp_s32", "cp_s16_ch3"],
    "cp_s32+cp32ch3":      ["cp_s32", "cp_s32_ch3"],
    "cp_s32+cp_s64_ch3":   ["cp_s32", "cp_s64_ch3"],
    "cp_s32+cp_s128_ch3":  ["cp_s32", "cp_s128_ch3"],
    # --- supcon / ilastik combos ---
    "supcon_z":                  ["supcon_z"],
    "il_s32":                    ["il_s32"],
    "cp_s32+supcon_z":           ["cp_s32", "supcon_z"],
    "cp_s32+il_s32":             ["cp_s32", "il_s32"],
    "supcon_z+il_s32":           ["supcon_z", "il_s32"],
    "cp_s32+supcon_z+il_s32":    ["cp_s32", "supcon_z", "il_s32"],
}

# ---------------------------------------------------------------------------
# Experiment constants (mirrors eval_image_efficiency_b1b2v2.py)

SEED = 42
EXCLUDED = {("control", 6), ("control", 7), ("control", 19), ("ycomp", 0)}
K_VALUES      = [1, 2, 3, 5, 7, 10, 13, 16, 19]
N_REPEATS     = 20
LABEL_BUDGETS = [5, 10, 20, 50, 100, 150, 200, 250, 300, 400, 500,
                 600, 700, 800, 900, 1000, 1100, 1200, "all"]

# ---------------------------------------------------------------------------
# Helpers

def uid_to_fn(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def parse_image(uid: str) -> tuple[str, int]:
    m = re.match(r'^(control|ycomp)-f(\d+)', uid)
    if not m:
        raise ValueError(f"Cannot parse image from uid: {uid!r}")
    return m.group(1), int(m.group(2))


def image_key(uid: str) -> str:
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


def fit_predict(clf_name: str, X_tr, y_tr, X_te, y_te, seed: int):
    sc = StandardScaler().fit(X_tr)
    clf = _build_clf(clf_name, seed)
    clf.fit(sc.transform(X_tr), y_tr)
    X_te_sc = sc.transform(X_te)
    y_pred  = clf.predict(X_te_sc)
    y_proba = clf.predict_proba(X_te_sc)[:, 1]
    ba = float(balanced_accuracy_score(y_te, y_pred))
    return ba, y_pred, y_proba

# ---------------------------------------------------------------------------
# Feature loading

def load_base_features() -> dict[str, pd.DataFrame]:
    tables = {}
    for name, path in FEAT_CSVS.items():
        df = pd.read_csv(path)
        feat_cols = [c for c in df.columns if c not in _META_DROP]
        tables[name] = df[["filename"] + feat_cols].set_index("filename")
        print(f"  {name}: {tables[name].shape}", flush=True)
    return tables


def build_combo_table(base: dict[str, pd.DataFrame], combo_key: str) -> pd.DataFrame:
    parts = [base[k].add_prefix(f"{k}__") for k in FEAT_COMBOS[combo_key]]
    return pd.concat(parts, axis=1)


def load_annotator(csv_path: Path, feat_tables: dict[str, pd.DataFrame]) -> dict:
    ann = pd.read_csv(csv_path)
    ann["filename"]  = ann["unique_ID"].apply(uid_to_fn)
    ann["image_key"] = ann["unique_ID"].apply(image_key)

    def _excluded(uid: str) -> bool:
        try:
            return parse_image(uid) in EXCLUDED
        except ValueError:
            return False

    ann = ann[~ann["unique_ID"].apply(_excluded)].copy()
    ann["label_bin"] = (ann["label"] == "adhesion").astype(int)

    result_by_feat: dict[str, pd.DataFrame] = {}
    feat_cols_by: dict[str, list[str]] = {}
    for feat_key, feat_df in feat_tables.items():
        merged = ann.join(feat_df, on="filename", how="inner")
        cols = feat_df.columns.tolist()
        result_by_feat[feat_key] = merged[["filename", "image_key", "label_bin"] + cols].copy()
        feat_cols_by[feat_key] = cols

    ref_key = next(iter(feat_tables))
    all_images = sorted(result_by_feat[ref_key]["image_key"].unique())
    return {"patches": result_by_feat, "images": all_images, "feat_cols": feat_cols_by}


# ---------------------------------------------------------------------------
# Single train/test run

def run_single(patches_df, feat_cols, train_images, test_images,
               label_budget, rng, seed_clf, clf_name,
               return_preds=False):
    df_tr_pool = patches_df[patches_df["image_key"].isin(train_images)]
    df_te      = patches_df[patches_df["image_key"].isin(test_images)]
    if len(df_te) == 0 or len(df_tr_pool) == 0:
        return None

    n_pool = len(df_tr_pool)
    if label_budget == "all" or label_budget >= n_pool:
        df_tr   = df_tr_pool
        n_train = n_pool
    else:
        idx     = rng.choice(n_pool, size=label_budget, replace=False)
        df_tr   = df_tr_pool.iloc[idx]
        n_train = label_budget

    X_tr = df_tr[feat_cols].values.astype(float)
    y_tr = df_tr["label_bin"].values
    X_te = df_te[feat_cols].values.astype(float)
    y_te = df_te["label_bin"].values

    if len(np.unique(y_tr)) < 2:
        return None

    ba, y_pred, y_proba = fit_predict(clf_name, X_tr, y_tr, X_te, y_te, seed_clf)

    result = dict(
        n_train_pool=n_pool,
        label_budget=("all" if (label_budget == "all" or label_budget >= n_pool)
                      else label_budget),
        n_train=n_train,
        n_test_images=len(test_images),
        n_test_patches=len(df_te),
        bal_acc=round(ba, 6),
    )
    if return_preds:
        return result, df_te["filename"].values, y_pred, y_proba, y_te
    return result, None, None, None, None


# ---------------------------------------------------------------------------
# Experiments

def experiment_loio(ann_data, feat_key, rng, clf_name) -> tuple[list, list]:
    patches_df = ann_data["patches"][feat_key]
    feat_cols  = ann_data["feat_cols"][feat_key]
    images     = ann_data["images"]
    rows, pred_rows = [], []

    for i, hold_img in enumerate(images):
        train_imgs = [img for img in images if img != hold_img]
        for lb in LABEL_BUDGETS:
            do_preds = (lb == "all")
            result_tuple = run_single(
                patches_df, feat_cols, train_imgs, [hold_img],
                label_budget=lb, rng=rng, seed_clf=SEED + i,
                clf_name=clf_name, return_preds=do_preds,
            )
            if result_tuple is None:
                continue
            res, fns, y_pred, y_proba, y_te = result_tuple
            rows.append(dict(experiment="loio", k_images=len(train_imgs),
                             repeat=i, held_out_image=hold_img, **res))
            if do_preds and fns is not None:
                for fn, yp, prob, yt in zip(fns, y_pred, y_proba, y_te):
                    pred_rows.append(dict(fn=fn, y_true=int(yt), y_pred=int(yp),
                                         prob_adhesion=float(prob),
                                         held_out_image=hold_img))
    return rows, pred_rows


def experiment_image_budget(ann_data, feat_key, rng, clf_name) -> list:
    patches_df = ann_data["patches"][feat_key]
    feat_cols  = ann_data["feat_cols"][feat_key]
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
                result_tuple = run_single(
                    patches_df, feat_cols, train_imgs, test_imgs,
                    label_budget=lb, rng=rng, seed_clf=SEED + rep,
                    clf_name=clf_name, return_preds=False,
                )
                if result_tuple is None:
                    continue
                res, *_ = result_tuple
                rows.append(dict(experiment="image_budget", k_images=k,
                                 repeat=rep, held_out_image="", **res))
    return rows


# ---------------------------------------------------------------------------
# Main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotators", nargs="+", default=["b1", "b2v2"],
                    choices=["b1", "b2v2"])
    ap.add_argument("--feat", nargs="+", choices=list(FEAT_COMBOS),
                    default=list(FEAT_COMBOS))
    ap.add_argument("--clf", nargs="+", choices=["lgbm", "logreg"],
                    default=["lgbm", "logreg"])
    ap.add_argument("--append", action="store_true",
                    help="Append to existing output CSV instead of overwriting")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading base features ...", flush=True)
    base = load_base_features()

    annotator_map = {"b1": B1_CSV, "b2v2": B2V2_CSV}

    for clf_name in args.clf:
        rng = np.random.default_rng(SEED)
        all_rows: list[dict] = []
        all_pred_rows: list[dict] = []

        for ann_name in args.annotators:
            print(f"\nAnnotator: {ann_name}  clf: {clf_name}", flush=True)

            # Build combo tables and pass all to load_annotator for joining
            combo_tables = {k: build_combo_table(base, k) for k in args.feat}
            ann_data = load_annotator(annotator_map[ann_name], combo_tables)
            print(f"  Valid images: {len(ann_data['images'])}", flush=True)

            for feat_key in args.feat:
                n_patches = len(ann_data["patches"][feat_key])
                print(f"  [{feat_key}] n_patches={n_patches}", flush=True)

                print(f"    LOIO ...", flush=True)
                loio_rows, loio_pred_rows = experiment_loio(
                    ann_data, feat_key, rng, clf_name)
                for r in loio_rows:
                    r.update(annotator=ann_name, feature=feat_key, clf=clf_name)
                all_rows.extend(loio_rows)
                for r in loio_pred_rows:
                    r.update(annotator=ann_name, feature=feat_key, clf=clf_name)
                all_pred_rows.extend(loio_pred_rows)
                print(f"    LOIO: {len(loio_rows)} rows, {len(loio_pred_rows)} patch preds",
                      flush=True)

                print(f"    Image budget sweep (K={K_VALUES}) ...", flush=True)
                budget_rows = experiment_image_budget(ann_data, feat_key, rng, clf_name)
                for r in budget_rows:
                    r.update(annotator=ann_name, feature=feat_key, clf=clf_name)
                all_rows.extend(budget_rows)
                print(f"    Image budget: {len(budget_rows)} rows", flush=True)

        col_order = [
            "experiment", "annotator", "feature", "clf",
            "k_images", "repeat", "held_out_image",
            "n_train_pool", "label_budget", "n_train",
            "n_test_images", "n_test_patches", "bal_acc",
        ]
        out_df = pd.DataFrame(all_rows)[col_order]
        out_path = OUT_DIR / f"image_efficiency_cp_combo_{clf_name}.csv"
        if args.append and out_path.exists():
            out_df = pd.concat([pd.read_csv(out_path), out_df], ignore_index=True)
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved {len(out_df)} rows → {out_path}")
        print(out_df.groupby(["experiment", "annotator", "feature"])["bal_acc"]
              .agg(["count", "mean", "std"]).round(3).to_string())

        if all_pred_rows:
            pred_col_order = ["fn", "y_true", "y_pred", "prob_adhesion",
                              "annotator", "feature", "clf", "held_out_image"]
            pred_df = pd.DataFrame(all_pred_rows)[pred_col_order]
            pred_path = PRED_DIR / f"image_eff_cp_combo_{clf_name}.csv"
            pred_df.to_csv(pred_path, index=False)
            print(f"Saved {len(pred_df)} patch pred rows → {pred_path}")


if __name__ == "__main__":
    main()
