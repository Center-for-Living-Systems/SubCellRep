#!/usr/bin/env python3
"""
eval_le_cp_combo.py

Label-efficiency budget study for B1/B2v2 matched set, testing combinations
of CP s32, CP s128, and distance features.

Feature sets tested:
  cp_s32           : CellProfiler 32×32 patch (50 feat)
  cp_s128          : CellProfiler 128×128 context crop (50 feat)
  dist             : distance-to-boundary features d00–d07 + equiv_diam (9 feat)
  cp_s32+s128      : s32 ∥ s128 (100 feat)
  cp_s32+dist      : s32 ∥ dist (59 feat)
  cp_s32+s128+dist : s32 ∥ s128 ∥ dist (109 feat)

Models same fold/strategy structure as eval_le_b1b2v2_budget.py.

Outputs (EVAL_DIR = eval_results/b1b2v2/):
  le_b1b2v2_budget_cp_combo_{clf}.csv
    Columns: feat, clf, strategy, budget, fold, repeat, test_batch,
             bal_acc, recall_ad, recall_noad, prec_ad, prec_noad,
             n_train, n_train_ad, n_test

  eval_results/patch_preds/le_budget_cp_combo_{clf}.csv  (budget=0/all only)
    Columns: fn, y_true, y_pred, prob_adhesion, feat, clf, strategy,
             budget, fold, repeat, test_batch

Usage:
  python scripts/eval_le_cp_combo.py --clf lgbm logreg
  python scripts/eval_le_cp_combo.py --clf lgbm --feat cp_s32+s128+dist
"""
from __future__ import annotations
import argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths

REPO     = Path(__file__).resolve().parent.parent
DATA     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
SPACES   = Path("/net/spaces/clinic/CLS/lding/data/fa_data_analysis")
EVAL_DIR = REPO / "eval_results" / "b1b2v2"
FS_DIR   = DATA / "labelling" / "le_b1b2v2_matched"
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

# Metadata columns to drop when loading feature CSVs (keep only feature values)
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
# Budget / strategy config (mirrors eval_le_b1b2v2_budget.py)

FOLDS      = list(range(5))
BUDGETS    = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 300, 340]
BUDGETS_MIXED = (BUDGETS + [380, 420, 460, 500, 560, 620, 680, 740, 800] + [0])
N_REPEATS  = 10
STRATEGIES = ["b1_only", "b2v2_only", "mixed"]

# ---------------------------------------------------------------------------

def _h2u(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def _build_clf(clf_name: str):
    if clf_name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=200, num_leaves=31, learning_rate=0.05,
            min_child_samples=1, class_weight="balanced",
            n_jobs=2, random_state=42, verbose=-1,
        )
    elif clf_name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            class_weight="balanced", max_iter=2000, C=1.0,
            solver="lbfgs", random_state=42,
        )
    raise ValueError(f"Unknown clf: {clf_name}")


def _score(y_true, y_pred) -> dict:
    return dict(
        bal_acc     = round(balanced_accuracy_score(y_true, y_pred), 6),
        recall_ad   = round(recall_score(y_true, y_pred, pos_label=1, zero_division=0), 6),
        recall_noad = round(recall_score(y_true, y_pred, pos_label=0, zero_division=0), 6),
        prec_ad     = round(precision_score(y_true, y_pred, pos_label=1, zero_division=0), 6),
        prec_noad   = round(precision_score(y_true, y_pred, pos_label=0, zero_division=0), 6),
    )


def _strat_sample(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    groups = {lbl: grp.reset_index(drop=True) for lbl, grp in pool.groupby("label")}
    total  = len(pool)
    alloc  = {}
    for lbl, grp in groups.items():
        alloc[lbl] = max(1, int(round(n * len(grp) / total)))
        alloc[lbl] = min(alloc[lbl], len(grp))
    diff = n - sum(alloc.values())
    if diff != 0:
        largest = max(groups, key=lambda l: len(groups[l]))
        alloc[largest] = min(len(groups[largest]), max(1, alloc[largest] + diff))
    return pd.concat(
        [grp.iloc[rng.choice(len(grp), size=alloc[lbl], replace=False)]
         for lbl, grp in groups.items()],
        ignore_index=True,
    )


# ---------------------------------------------------------------------------
# Feature loading

def load_base_features() -> dict[str, pd.DataFrame]:
    """Load each base feature CSV, return {name: DataFrame indexed by filename}."""
    tables = {}
    for name, path in FEAT_CSVS.items():
        df = pd.read_csv(path)
        feat_cols = [c for c in df.columns if c not in _META_DROP]
        tables[name] = df[["filename"] + feat_cols].set_index("filename")
        print(f"  {name}: {tables[name].shape}", flush=True)
    return tables


def build_combo_table(base: dict[str, pd.DataFrame], combo_key: str) -> pd.DataFrame:
    """Concatenate feature tables for a given combo, joining on filename."""
    parts = [base[k].add_prefix(f"{k}__") for k in FEAT_COMBOS[combo_key]]
    return pd.concat(parts, axis=1)


# ---------------------------------------------------------------------------
# Core eval loop (mirrors run_budget_eval in eval_le_b1b2v2_budget.py)

def run_budget_eval(
    feat_key: str,
    feat_table: pd.DataFrame,
    clf_name: str,
    budgets: list,
    n_repeats: int,
    strategies: list,
    seed: int = 0,
) -> tuple[pd.DataFrame, list]:

    fs_b1   = pd.read_csv(FS_DIR / "fold_splits_b1.csv")
    fs_b2v2 = pd.read_csv(FS_DIR / "fold_splits_b2v2.csv")
    fs_b1["fn"]   = fs_b1["unique_ID"].apply(_h2u)
    fs_b2v2["fn"] = fs_b2v2["unique_ID"].apply(_h2u)

    b1_lbl   = dict(zip(fs_b1["fn"],   fs_b1["label"]))
    b2v2_lbl = dict(zip(fs_b2v2["fn"], fs_b2v2["label"]))
    conflict_fns = {fn for fn in set(b1_lbl) & set(b2v2_lbl) if b1_lbl[fn] != b2v2_lbl[fn]}

    master_rng = np.random.default_rng(seed)
    rows = []
    patch_rows = []

    for fold in FOLDS:
        pool_b1        = fs_b1[fs_b1["fold"] != fold][["fn", "label"]].copy()
        pool_b2v2      = fs_b2v2[fs_b2v2["fold"] != fold][["fn", "label"]].copy()
        pool_b1_clean  = pool_b1[~pool_b1["fn"].isin(conflict_fns)]
        pool_b2v2_clean= pool_b2v2[~pool_b2v2["fn"].isin(conflict_fns)]
        te_b1          = fs_b1[fs_b1["fold"] == fold][["fn", "label"]].copy()
        te_b2v2        = fs_b2v2[fs_b2v2["fold"] == fold][["fn", "label"]].copy()

        print(f"  fold {fold}: pool_b1={len(pool_b1)}  pool_b2v2={len(pool_b2v2)}", flush=True)

        for budget in budgets:
            use_all = (budget == 0)
            reps    = 1 if use_all else n_repeats

            for rep in range(reps):
                rng = np.random.default_rng(master_rng.integers(0, 2**32))

                if use_all:
                    tr_sets = {
                        "b1_only":   pool_b1.copy(),
                        "b2v2_only": pool_b2v2.copy(),
                        "mixed":     pd.concat([pool_b1_clean, pool_b2v2_clean],
                                               ignore_index=True),
                    }
                else:
                    half = budget // 2
                    tr_sets = {
                        "b1_only":   _strat_sample(pool_b1,
                                         min(budget, len(pool_b1)),
                                         np.random.default_rng(rng.integers(0, 2**32))),
                        "b2v2_only": _strat_sample(pool_b2v2,
                                         min(budget, len(pool_b2v2)),
                                         np.random.default_rng(rng.integers(0, 2**32))),
                        "mixed":     pd.concat([
                            _strat_sample(pool_b1_clean,
                                          min(half, len(pool_b1_clean)),
                                          np.random.default_rng(rng.integers(0, 2**32))),
                            _strat_sample(pool_b2v2_clean,
                                          min(half, len(pool_b2v2_clean)),
                                          np.random.default_rng(rng.integers(0, 2**32))),
                        ], ignore_index=True),
                    }
                tr_sets = {k: v for k, v in tr_sets.items() if k in strategies}

                for strategy, tr_df in tr_sets.items():
                    train_fns = set(tr_df["fn"])
                    X_tr_raw  = feat_table.reindex(tr_df["fn"]).values
                    y_tr_raw  = np.array([1 if l == "adhesion" else 0
                                          for l in tr_df["label"]])
                    tr_ok = ~np.isnan(X_tr_raw).any(axis=1)
                    X_tr, y_tr = X_tr_raw[tr_ok], y_tr_raw[tr_ok]
                    if len(X_tr) == 0 or len(np.unique(y_tr)) < 2:
                        continue

                    sc  = StandardScaler().fit(X_tr)
                    clf = _build_clf(clf_name)
                    clf.fit(sc.transform(X_tr), y_tr)

                    for test_batch, te_full in [("b1", te_b1), ("b2v2", te_b2v2)]:
                        te = te_full[~te_full["fn"].isin(train_fns)].copy()
                        if len(te) == 0:
                            continue
                        X_te_raw = feat_table.reindex(te["fn"]).values
                        y_te_raw = np.array([1 if l == "adhesion" else 0
                                             for l in te["label"]])
                        te_ok = ~np.isnan(X_te_raw).any(axis=1)
                        X_te = sc.transform(X_te_raw[te_ok])
                        y_te = y_te_raw[te_ok]
                        if len(X_te) == 0:
                            continue

                        y_pred = clf.predict(X_te)
                        prob   = clf.predict_proba(X_te)[:, 1]

                        # Save per-patch predictions for budget=all
                        if use_all:
                            fns = te["fn"].values[te_ok]
                            for fn_i, yt_i, yp_i, pb_i in zip(fns, y_te, y_pred, prob):
                                patch_rows.append(dict(
                                    fn=fn_i, y_true=int(yt_i), y_pred=int(yp_i),
                                    prob_adhesion=float(pb_i),
                                    feat=feat_key, clf=clf_name,
                                    strategy=strategy, budget=budget,
                                    fold=fold, repeat=rep, test_batch=test_batch,
                                ))

                        rows.append(dict(
                            feat=feat_key, clf=clf_name,
                            strategy=strategy, budget=budget,
                            fold=fold, repeat=rep, test_batch=test_batch,
                            n_train=int(tr_ok.sum()), n_train_ad=int(y_tr.sum()),
                            n_test=int(te_ok.sum()),
                            **_score(y_te, y_pred),
                        ))

    return pd.DataFrame(rows), patch_rows


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf",  nargs="+", choices=["lgbm", "logreg"],
                    default=["lgbm", "logreg"])
    ap.add_argument("--feat", nargs="+", choices=list(FEAT_COMBOS),
                    default=list(FEAT_COMBOS),
                    help="Feature combo(s) to evaluate")
    ap.add_argument("--strategy", nargs="+", choices=STRATEGIES, default=STRATEGIES)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--append", action="store_true",
                    help="Append to existing output CSV instead of overwriting")
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading base features ...", flush=True)
    base = load_base_features()

    for clf_name in args.clf:
        all_rows = []
        all_pred_rows = []

        for feat_key in args.feat:
            print(f"\n{'='*60}", flush=True)
            print(f"feat={feat_key}  clf={clf_name}", flush=True)
            print(f"{'='*60}", flush=True)
            feat_table = build_combo_table(base, feat_key)

            strats_single = [s for s in args.strategy if s != "mixed"]
            strats_mixed  = [s for s in args.strategy if s == "mixed"]

            if strats_single:
                df_s, pr_s = run_budget_eval(
                    feat_key, feat_table, clf_name, BUDGETS, N_REPEATS,
                    strats_single, seed=args.seed,
                )
                all_rows.append(df_s)
                all_pred_rows.extend(pr_s)

            if strats_mixed:
                df_m, pr_m = run_budget_eval(
                    feat_key, feat_table, clf_name, BUDGETS_MIXED, N_REPEATS,
                    strats_mixed, seed=args.seed,
                )
                all_rows.append(df_m)
                all_pred_rows.extend(pr_m)

        if not all_rows:
            continue

        out_df = pd.concat(all_rows, ignore_index=True)
        out_path = EVAL_DIR / f"le_b1b2v2_budget_cp_combo_{clf_name}.csv"
        if args.append and out_path.exists():
            out_df = pd.concat([pd.read_csv(out_path), out_df], ignore_index=True)
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved {len(out_df)} rows → {out_path}")

        if all_pred_rows:
            pred_path = PRED_DIR / f"le_budget_cp_combo_{clf_name}.csv"
            pd.DataFrame(all_pred_rows).to_csv(pred_path, index=False)
            print(f"Saved {len(all_pred_rows)} patch pred rows → {pred_path}")

        # Quick summary
        print(out_df.groupby(["feat", "test_batch"])["bal_acc"]
              .agg(["count", "mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
