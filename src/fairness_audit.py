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
  gaps.csv              max-min gap per finding and attribute, with CIs
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

    def gaps_of(table):
        res = []
        for (f, a), t in table.groupby(["finding", "attribute"]):
            res.append({"finding": f, "attribute": a,
                        "auroc_gap": t["auroc"].max() - t["auroc"].min(),
                        "tpr_gap": t["tpr"].max() - t["tpr"].min(),
                        "fpr_gap": t["fpr"].max() - t["fpr"].min()})
        return pd.DataFrame(res)

    point_gaps = gaps_of(point)

    # 3. Patient-level bootstrap for confidence intervals.
    print(f"Bootstrapping {args.n_boot} resamples (a few minutes on a laptop)...")
    boot_rows, boot_gaps = [], []
    for b, sample in enumerate(patient_bootstrap(test, args.n_boot, args.seed)):
        rs = []
        for f in FINDINGS:
            for a in ATTRIBUTES:
                for grp, r in fast_group_metrics(sample, f, a, thr[f]).items():
                    rs.append({"finding": f, "attribute": a, "group": grp, "b": b, **r})
        rs = pd.DataFrame(rs)
        boot_rows.append(rs)
        bg = gaps_of(rs)
        bg["b"] = b
        boot_gaps.append(bg)
    boot = pd.concat(boot_rows)
    bgap = pd.concat(boot_gaps)

    def add_ci(point_df, boot_df, keys, cols):
        lo = boot_df.groupby(keys)[cols].quantile(0.025).add_suffix("_lo")
        hi = boot_df.groupby(keys)[cols].quantile(0.975).add_suffix("_hi")
        return point_df.set_index(keys).join(lo).join(hi).reset_index()

    sub = add_ci(point, boot, ["finding", "attribute", "group"], ["auroc", "tpr", "fpr"])
    gaps = add_ci(point_gaps, bgap, ["finding", "attribute"], ["auroc_gap", "tpr_gap", "fpr_gap"])
    sub.round(4).to_csv(out / "subgroup_metrics.csv", index=False)
    gaps.round(4).to_csv(out / "gaps.csv", index=False)

    # 4. Readable summary.
    print("\nThresholds (validation, ~{:.0%} sensitivity):".format(args.target_sensitivity))
    for f, t in thr.items():
        print(f"  {f:13s} {t:.3f}")
    print("\nTPR (sensitivity) by sex, with 95% CI:")
    s = sub[sub["attribute"] == "sex"]
    for _, r in s.iterrows():
        print(f"  {r['finding']:13s} {r['group']}: TPR {r['tpr']:.3f} [{r['tpr_lo']:.3f}, {r['tpr_hi']:.3f}]"
              f"  AUROC {r['auroc']:.3f}  positives {int(r['n_pos'])}")
    print("\nLargest gaps (TPR, with 95% CI):")
    for _, r in gaps.sort_values("tpr_gap", ascending=False).iterrows():
        print(f"  {r['finding']:13s} {r['attribute']:8s} TPR gap {r['tpr_gap']:.3f} "
              f"[{r['tpr_gap_lo']:.3f}, {r['tpr_gap_hi']:.3f}]   AUROC gap {r['auroc_gap']:.3f}")

    # 5. Plots for the README.
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
    ax.set_ylabel("Max - min TPR across groups")
    ax.set_title("Sensitivity gap at the validation threshold (95% CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "tpr_gap.png", dpi=150)
    print(f"\nSaved tables and plots to {out}/")


if __name__ == "__main__":
    main()
