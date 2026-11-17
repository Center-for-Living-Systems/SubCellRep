#!/usr/bin/env python3
"""
eval_le_b1b2_budget.py

Mixed-label budget study: at a fixed total annotation budget N, does a 50:50
B1+B2 mix outperform B1-only or B2-only?

Training strategies (same total budget N):
  b1_only  — N labels drawn from B1 training pool (stratified by class)
  b2_only  — N labels drawn from B2 training pool (stratified by class)
  mixed    — N/2 from B1 + N/2 from B2 (each stratified)

Evaluation:
  Each strategy is tested separately on B1 test patches and B2 test patches.
  Leakage prevention: any training patch that appears in the test set is removed.

Outputs (EVAL_DIR):
  le_b1b2_budget_{feat}_{clf}.csv
  Columns: feat, clf, strategy, budget, fold, repeat, test_batch,
           bal_acc, recall_ad, recall_noad, prec_ad, prec_noad,
           n_train, n_train_ad, n_test

Usage
-----
  python scripts/eval_le_b1b2_budget.py --feat cp ilastik --clf lgbm
  python scripts/eval_le_b1b2_budget.py --feat cp         --clf lgbm logreg
"""
from __future__ import annotations
import argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler

DATA     = Path("/net/projects/CLS/lding/data/fa_data_analysis")
EVAL_DIR = DATA / "ae_results" / "features" / "eval_results"
FS_DIR   = DATA / "labelling" / "le_b1b2_matched"
CP_CSV   = DATA / "ae_results" / "features" / "cellprofiler" / "ds1.csv"
IL_CSV   = DATA / "ae_results" / "features" / "ilastik" / "ds1.csv"

FOLDS    = list(range(5))
BUDGETS  = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
BUDGETS_MIXED = (BUDGETS +
                 [300, 340, 380, 420, 460, 500] +
                 [0])   # 0 = sentinel for "all" (~519 labels)
N_REPEATS = 10
STRATEGIES = ["b1_only", "b2_only", "mixed"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _h2u(uid: str) -> str:
    return uid.replace("-f", "_f", 1)


def _strat_sample(pool: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Sample n rows from pool, stratified by 'label'. Guarantees ≥1 per class."""
    groups = {lbl: grp.reset_index(drop=True) for lbl, grp in pool.groupby("label")}
    total  = len(pool)
    alloc  = {}
    for lbl, grp in groups.items():
        alloc[lbl] = max(1, int(round(n * len(grp) / total)))
        alloc[lbl] = min(alloc[lbl], len(grp))
    # Adjust rounding drift so sum == n (modify largest class)
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


def _load_feats(feat: str) -> pd.DataFrame:
    csv = CP_CSV if feat == "cp" else IL_CSV
    df  = pd.read_csv(csv)
    feat_cols = [c for c in df.columns if c != "filename"]
    return df[["filename"] + feat_cols].set_index("filename")


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def run_budget_eval(feat: str, clf_name: str, budgets, n_repeats, seed: int = 0,
                    strategies: list | None = None) -> pd.DataFrame:
    if strategies is None:
        strategies = STRATEGIES
    feats_all = _load_feats(feat)

    fs_b1 = pd.read_csv(FS_DIR / "fold_splits_b1.csv")
    fs_b2 = pd.read_csv(FS_DIR / "fold_splits_b2.csv")
    fs_b1["fn"] = fs_b1["unique_ID"].apply(_h2u)
    fs_b2["fn"] = fs_b2["unique_ID"].apply(_h2u)

    # Conflict patches: present in both annotators with different labels
    # (used to exclude from mixed training pool)
    b1_lbl = dict(zip(fs_b1["fn"], fs_b1["label"]))
    b2_lbl = dict(zip(fs_b2["fn"], fs_b2["label"]))
    conflict_fns = {fn for fn in set(b1_lbl) & set(b2_lbl)
                    if b1_lbl[fn] != b2_lbl[fn]}

    master_rng = np.random.default_rng(seed)
    rows = []

    for fold in FOLDS:
        # ---- Build per-fold training pools ----
        pool_b1 = fs_b1[fs_b1["fold"] != fold][["fn","label"]].copy()
        pool_b2 = fs_b2[fs_b2["fold"] != fold][["fn","label"]].copy()
        # For mixed: remove conflict patches from both pools
        pool_b1_clean = pool_b1[~pool_b1["fn"].isin(conflict_fns)]
        pool_b2_clean = pool_b2[~pool_b2["fn"].isin(conflict_fns)]

        # ---- Build per-fold test sets (fixed across strategies) ----
        # Will trim per strategy to remove training patches below
        te_b1_full = fs_b1[fs_b1["fold"] == fold][["fn","label"]].copy()
        te_b2_full = fs_b2[fs_b2["fold"] == fold][["fn","label"]].copy()

        print(f"  fold {fold}: pool_b1={len(pool_b1)}  pool_b2={len(pool_b2)}")

        for budget in budgets:
            # budget == 0 is sentinel for "all available labels" (no subsampling, 1 repeat)
            use_all  = (budget == 0)
            reps     = 1 if use_all else n_repeats

            for rep in range(reps):
                rng = np.random.default_rng(master_rng.integers(0, 2**32))

                if use_all:
                    tr_sets = {
                        "b1_only": pool_b1.copy(),
                        "b2_only": pool_b2.copy(),
                        "mixed":   pd.concat([pool_b1_clean, pool_b2_clean],
                                             ignore_index=True),
                    }
                else:
                    half      = budget // 2
                    n_b1      = min(budget, len(pool_b1))
                    n_b2      = min(budget, len(pool_b2))
                    n_half_b1 = min(half,   len(pool_b1_clean))
                    n_half_b2 = min(half,   len(pool_b2_clean))
                    tr_sets = {
                        "b1_only": _strat_sample(pool_b1, n_b1, np.random.default_rng(rng.integers(0,2**32))),
                        "b2_only": _strat_sample(pool_b2, n_b2, np.random.default_rng(rng.integers(0,2**32))),
                        "mixed":   pd.concat([
                            _strat_sample(pool_b1_clean, n_half_b1, np.random.default_rng(rng.integers(0,2**32))),
                            _strat_sample(pool_b2_clean, n_half_b2, np.random.default_rng(rng.integers(0,2**32))),
                        ], ignore_index=True),
                    }
                # Filter to requested strategies
                tr_sets = {k: v for k, v in tr_sets.items() if k in strategies}

                for strategy, tr_df in tr_sets.items():
                    train_fns = set(tr_df["fn"])

                    # ---- Fit scaler + clf ----
                    X_tr_raw = feats_all.reindex(tr_df["fn"]).values
                    y_tr_raw = np.array([1 if l == "adhesion" else 0 for l in tr_df["label"]])
                    tr_ok    = ~np.isnan(X_tr_raw).any(axis=1)
                    X_tr, y_tr = X_tr_raw[tr_ok], y_tr_raw[tr_ok]

                    if len(X_tr) == 0 or len(np.unique(y_tr)) < 2:
                        continue

                    sc   = StandardScaler().fit(X_tr)
                    X_tr = sc.transform(X_tr)
                    clf  = _build_clf(clf_name)
                    clf.fit(X_tr, y_tr)

                    # ---- Evaluate on B1 and B2 test sets ----
                    for test_batch, te_full in [("b1", te_b1_full), ("b2", te_b2_full)]:
                        te = te_full[~te_full["fn"].isin(train_fns)].copy()
                        if len(te) == 0:
                            continue

                        X_te_raw = feats_all.reindex(te["fn"]).values
                        y_te_raw = np.array([1 if l == "adhesion" else 0
                                             for l in te["label"]])
                        te_ok    = ~np.isnan(X_te_raw).any(axis=1)
                        X_te, y_te = sc.transform(X_te_raw[te_ok]), y_te_raw[te_ok]

                        if len(X_te) == 0:
                            continue

                        y_pred = clf.predict(X_te)
                        s = _score(y_te, y_pred)
                        rows.append(dict(
                            feat=feat, clf=clf_name,
                            strategy=strategy, budget=budget,
                            fold=fold, repeat=rep, test_batch=test_batch,
                            n_train=int(tr_ok.sum()),
                            n_train_ad=int(y_tr.sum()),
                            n_test=int(te_ok.sum()),
                            **s,
                        ))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat",       nargs="+", choices=["cp","ilastik"],
                    default=["cp","ilastik"])
    ap.add_argument("--clf",        nargs="+", choices=["lgbm","logreg"],
                    default=["lgbm"])
    ap.add_argument("--strategy",   nargs="+", choices=STRATEGIES,
                    default=STRATEGIES,
                    help="Which strategies to run (default: all three)")
    ap.add_argument("--budgets",    nargs="+", type=int, default=None,
                    help="Override budget list (default: BUDGETS for single-source, "
                         "BUDGETS_MIXED for mixed)")
    ap.add_argument("--n_repeats",  type=int, default=N_REPEATS)
    ap.add_argument("--seed",       type=int, default=0)
    ap.add_argument("--append",     action="store_true",
                    help="Append to existing CSV instead of overwriting")
    args = ap.parse_args()

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    for feat in args.feat:
        for clf_name in args.clf:
            # Determine budget list
            if args.budgets is not None:
                budgets = args.budgets
            else:
                # Use extended list for mixed, standard list for single-source strategies
                # Run separately if strategies differ
                strats_single = [s for s in args.strategy if s != "mixed"]
                strats_mixed  = [s for s in args.strategy if s == "mixed"]
                all_rows = []

                if strats_single:
                    print(f"\n{'='*60}")
                    print(f"feat={feat}  clf={clf_name}  strategies={strats_single}  budgets={BUDGETS}")
                    print(f"{'='*60}")
                    df_s = run_budget_eval(feat, clf_name, BUDGETS, args.n_repeats,
                                           args.seed, strategies=strats_single)
                    all_rows.append(df_s)

                if strats_mixed:
                    print(f"\n{'='*60}")
                    print(f"feat={feat}  clf={clf_name}  strategies={strats_mixed}  budgets={BUDGETS_MIXED}")
                    print(f"{'='*60}")
                    df_m = run_budget_eval(feat, clf_name, BUDGETS_MIXED, args.n_repeats,
                                           args.seed, strategies=strats_mixed)
                    all_rows.append(df_m)

                if not all_rows:
                    continue
                df = pd.concat(all_rows, ignore_index=True)
                out = EVAL_DIR / f"le_b1b2_budget_{feat}_{clf_name}.csv"
                if args.append and out.exists():
                    df = pd.concat([pd.read_csv(out), df], ignore_index=True)
                df.to_csv(out, index=False)
                print(f"\nSaved → {out}  ({len(df)} rows)")
                for te in ["b1","b2"]:
                    print(f"\n  Test on {te}:")
                    sub = df[df["test_batch"]==te]
                    for strat in args.strategy:
                        m = sub[sub["strategy"]==strat].groupby("budget")["bal_acc"].mean()
                        tail = "  ".join(f"N={b}: {v:.3f}" for b, v in list(m.items())[-5:])
                        print(f"    {strat:10s}: … {tail}")
                continue

            # Explicit --budgets override: run all strategies with same list
            print(f"\n{'='*60}")
            print(f"feat={feat}  clf={clf_name}  budgets={budgets}")
            print(f"{'='*60}")
            df = run_budget_eval(feat, clf_name, budgets, args.n_repeats,
                                 args.seed, strategies=args.strategy)
            out = EVAL_DIR / f"le_b1b2_budget_{feat}_{clf_name}.csv"
            if args.append and out.exists():
                df = pd.concat([pd.read_csv(out), df], ignore_index=True)
            df.to_csv(out, index=False)
            print(f"\nSaved → {out}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
