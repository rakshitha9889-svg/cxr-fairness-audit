"""Step 4: fairness audit of the baseline on the test split (runs on a laptop, no GPU).

    python src/fairness_audit.py --preds-dir reports/baseline --out-dir reports/fairness

For each finding and each group (sex, age bin, view) it reports:
  - AUROC (ranking quality)
  - TPR (sensitivity) and FPR at ONE threshold per finding, chosen on the validation set
    so that overall sensitivity is ~80%. Missed findings (low TPR) are the dangerous error.
  - 95% confidence intervals from a patient-level bootstrap (resample patients, not images,
    because one patient has several X-rays).

Outputs:
  thresholds.csv        threshold per finding (from validation)
  subgroup_metrics.csv  one row per finding x group, with CIs
  gaps.csv              best-minus-worst group difference per finding and attribute, with CIs
                        (signed difference for a fixed pair, so the CI can include 0)
  auroc_by_sex.png, tpr_gap.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, false_positive_rate, true_positive_rate
from sklearn.metrics import roc_auc_score

FINDINGS = ["Cardiomegaly", "Effusion", "Atelectasis", "Pneumothorax", "Edema"]
ATTRIBUTES = ["sex", "age_bin", "view"]
MIN_POS = 10  # skip a group's metric if it has fewer positives than this (too noisy)


def threshold_at_sensitivity(y, p, target=0.80):
    """Highest threshold whose sensitivity on (y, p) is >= target."""
    pos = np.sort(p[y == 1])[::-1]
    k = int(np.ceil(target * len(pos))) - 1
    return float(pos[max(k, 0)])


def safe_auc(y, p):
    return roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan


def group_metrics(df, finding, attr, thr):
    y = df[finding].values
    p = df[f"prob_{finding}"].values
    yhat = (p >= thr).astype(int)
    mf = MetricFrame(
        metrics={"auroc": lambda yt, yp: safe_auc(yt, yp),
                 "n": lambda yt, yp: len(yt),
                 "n_pos": lambda yt, yp: int(np.sum(yt))},
        y_true=y, y_pred=p, sensitive_features=df[attr].values)
    rates = MetricFrame(metrics={"tpr": true_positive_rate, "fpr": false_positive_rate},
                        y_true=y, y_pred=yhat, sensitive_features=df[attr].values)
    out = mf.by_group.join(rates.by_group)
    out.loc[out["n_pos"] < MIN_POS, ["auroc", "tpr"]] = np.nan
    return out


def fast_group_metrics(df, finding, attr, thr):
    """Same numbers as group_metrics, in plain numpy (fast enough for 1,000 bootstrap rounds)."""
    y = df[finding].values
    p = df[f"prob_{finding}"].values
    yhat = p >= thr
    out = {}
    for grp, idx in df.groupby(attr).indices.items():
        yt, pt, ht = y[idx], p[idx], yhat[idx]
        n_pos = yt.sum()
        n_neg = len(yt) - n_pos
        out[grp] = {
            "auroc": safe_auc(yt, pt) if n_pos >= MIN_POS else np.nan,
            "tpr": ht[yt == 1].mean() if n_pos >= MIN_POS else np.nan,
            "fpr": ht[yt == 0].mean() if n_neg > 0 else np.nan,
        }
    return out


def patient_bootstrap(df, n_boot, seed):
    """Yield bootstrap samples that resample whole patients."""
    rng = np.random.default_rng(seed)
    groups = df.groupby("patient_id").indices
    ids = np.array(list(groups.keys()))
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([groups[i] for i in pick])
        yield df.iloc[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", default="reports/baseline")
    ap.add_argument("--out-dir", default="reports/fairness")
    ap.add_argument("--target-sensitivity", type=float, default=0.80)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    val = pd.read_csv(Path(args.preds_dir) / "val_preds.csv")
    test = pd.read_csv(Path(args.preds_dir) / "test_preds.csv")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. One threshold per finding, chosen on validation (never on test).
    thr = {f: threshold_at_sensitivity(val[f].values, val[f"prob_{f}"].values, args.target_sensitivity)
           for f in FINDINGS}
    pd.Series(thr, name="threshold").to_csv(out / "thresholds.csv")

    # 2. Point estimates on the test split.
    rows = []
    for f in FINDINGS:
        for a in ATTRIBUTES:
            g = group_metrics(test, f, a, thr[f])
            for grp, r in g.iterrows():
                rows.append({"finding": f, "attribute": a, "group": grp, **r.to_dict()})
    point = pd.DataFrame(rows)

    # 3. Patient-level bootstrap for confidence intervals.
    print(f"Bootstrapping {args.n_boot} resamples (a few minutes on a laptop)...")
    boot_rows = []
    for b, sample in enumerate(patient_bootstrap(test, args.n_boot, args.seed)):
        for f in FINDINGS:
            for a in ATTRIBUTES:
                for grp, r in fast_group_metrics(sample, f, a, thr[f]).items():
                    boot_rows.append({"finding": f, "attribute": a, "group": grp, "b": b, **r})
    boot = pd.DataFrame(boot_rows)

    lo = boot.groupby(["finding", "attribute", "group"])[["auroc", "tpr", "fpr"]].quantile(0.025).add_suffix("_lo")
    hi = boot.groupby(["finding", "attribute", "group"])[["auroc", "tpr", "fpr"]].quantile(0.975).add_suffix("_hi")
    sub = point.set_index(["finding", "attribute", "group"]).join(lo).join(hi).reset_index()
    sub.round(4).to_csv(out / "subgroup_metrics.csv", index=False)

    # Gap = SIGNED difference between the best and the worst group (pair fixed on the full test set).
    # Its bootstrap CI can cross 0; if it does, the gap may be noise.
    gap_rows = []
    for (f, a), t in point.groupby(["finding", "attribute"]):
        row = {"finding": f, "attribute": a}
        for m in ["tpr", "auroc", "fpr"]:
            tt = t.dropna(subset=[m])
            if len(tt) < 2:
                row.update({f"{m}_gap": np.nan, f"{m}_high_group": "", f"{m}_low_group": "",
                            f"{m}_gap_lo": np.nan, f"{m}_gap_hi": np.nan})
                continue
            g_hi = tt.loc[tt[m].idxmax(), "group"]
            g_lo = tt.loc[tt[m].idxmin(), "group"]
            bb = boot[(boot["finding"] == f) & (boot["attribute"] == a)].pivot(index="b", columns="group", values=m)
            diff = (bb[g_hi] - bb[g_lo]).dropna()
            row.update({f"{m}_gap": tt[m].max() - tt[m].min(), f"{m}_high_group": g_hi, f"{m}_low_group": g_lo,
                        f"{m}_gap_lo": diff.quantile(0.025), f"{m}_gap_hi": diff.quantile(0.975)})
        gap_rows.append(row)
    gaps = pd.DataFrame(gap_rows)
    gaps["tpr_gap_significant"] = gaps["tpr_gap_lo"] > 0
    gaps["auroc_gap_significant"] = gaps["auroc_gap_lo"] > 0
    gaps.round(4).to_csv(out / "gaps.csv", index=False)

    # 4. Readable summary.
    print("\nThresholds (validation, ~{:.0%} sensitivity):".format(args.target_sensitivity))
    for f, t in thr.items():
        print(f"  {f:13s} {t:.3f}")
    names = {"sex": "sex", "age_bin": "age", "view": "view"}
    for a in ATTRIBUTES:
        print(f"\nBy {names[a]} (TPR = sensitivity, FPR = false alarms; 95% CI):")
        t = sub[sub["attribute"] == a]
        for _, r in t.iterrows():
            tpr = "   n/a (too few positives)" if np.isnan(r["tpr"]) else \
                f"{r['tpr']:.3f} [{r['tpr_lo']:.3f}, {r['tpr_hi']:.3f}]"
            print(f"  {r['finding']:13s} {str(r['group']):6s} TPR {tpr}  FPR {r['fpr']:.3f}  "
                  f"AUROC {r['auroc']:.3f}  positives {int(r['n_pos'])}/{int(r['n'])}")
    print("\nTPR gaps, best minus worst group (95% CI). REAL = CI does not include 0:")
    for _, r in gaps.sort_values("tpr_gap", ascending=False).iterrows():
        if np.isnan(r["tpr_gap"]):
            print(f"  {r['finding']:13s} {r['attribute']:8s} not enough positives in 2+ groups")
            continue
        flag = "REAL" if r["tpr_gap_significant"] else "could be noise"
        print(f"  {r['finding']:13s} {r['attribute']:8s} {str(r['tpr_high_group']):>5s} > {str(r['tpr_low_group']):5s} "
              f"by {r['tpr_gap']:.3f} [{r['tpr_gap_lo']:+.3f}, {r['tpr_gap_hi']:+.3f}]  {flag}   "
              f"(AUROC gap {r['auroc_gap']:.3f} [{r['auroc_gap_lo']:+.3f}, {r['auroc_gap_hi']:+.3f}])")

    # 5. Plots for the README.
    s = sub[sub["attribute"] == "sex"]
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.38
    x = np.arange(len(FINDINGS))
    for i, sex in enumerate(["F", "M"]):
        t = s[s["group"] == sex].set_index("finding").reindex(FINDINGS)
        err = [t["auroc"] - t["auroc_lo"], t["auroc_hi"] - t["auroc"]]
        ax.bar(x + (i - 0.5) * width, t["auroc"], width, yerr=err, capsize=3,
               label={"F": "Female", "M": "Male"}[sex])
    ax.set_xticks(x, FINDINGS)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUROC (95% CI)")
    ax.set_title("Test AUROC by sex")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "auroc_by_sex.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, a in enumerate(ATTRIBUTES):
        t = gaps[gaps["attribute"] == a].set_index("finding").reindex(FINDINGS)
        err = [t["tpr_gap"] - t["tpr_gap_lo"], t["tpr_gap_hi"] - t["tpr_gap"]]
        ax.bar(x + (i - 1) * 0.27, t["tpr_gap"], 0.27, yerr=err, capsize=3, label=a)
    ax.set_xticks(x, FINDINGS)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_ylabel("TPR: best minus worst group")
    ax.set_title("Sensitivity gap at the validation threshold (95% CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "tpr_gap.png", dpi=150)
    print(f"\nSaved tables and plots to {out}/")


if __name__ == "__main__":
    main()
