"""Step 1-2: load NIH ChestX-ray14 metadata, pick a patient-level subset and split it
by patient (no patient appears in two splits).

Usage (on Kaggle or locally):
    python src/data_prep.py --data-dir /kaggle/input --max-images 25000

Outputs:
    data/splits/train.csv, val.csv, test.csv   (small CSVs, safe to commit)
    reports/prevalence_by_group.csv            (label prevalence by sex and age bin)
"""
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

FINDINGS = ["Cardiomegaly", "Effusion", "Atelectasis", "Pneumothorax", "Edema"]
AGE_BINS = [0, 40, 60, 200]
AGE_LABELS = ["<40", "40-60", ">60"]


def find_metadata(data_dir: Path) -> Path:
    """Find Data_Entry_2017.csv wherever the dataset was mounted."""
    hits = sorted(data_dir.rglob("Data_Entry_2017*.csv"))
    if not hits:
        raise FileNotFoundError(f"Data_Entry_2017.csv not found under {data_dir}")
    return hits[0]


def load_metadata(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={
        "Image Index": "image_index",
        "Finding Labels": "finding_labels",
        "Patient ID": "patient_id",
        "Patient Age": "age",
        "Patient Gender": "sex",
        "View Position": "view",
        "Follow-up #": "follow_up",
    })
    df = df[["image_index", "patient_id", "age", "sex", "view", "follow_up", "finding_labels"]]

    # The NIH file has a few impossible ages (e.g. 412); drop them and say how many.
    bad_age = (df["age"] < 0) | (df["age"] > 100)
    print(f"Dropping {bad_age.sum()} images with impossible ages")
    df = df[~bad_age].copy()

    df["age_bin"] = pd.cut(df["age"], bins=AGE_BINS, labels=AGE_LABELS, right=False).astype(str)
    labels = df["finding_labels"].str.split("|")
    for f in FINDINGS:
        df[f] = labels.apply(lambda ls, f=f: int(f in ls))
    return df


def patient_subset(df: pd.DataFrame, max_images: int, seed: int) -> pd.DataFrame:
    """Take whole patients at random until we reach max_images (never split a patient)."""
    if max_images <= 0 or max_images >= len(df):
        return df
    counts = df.groupby("patient_id").size().sample(frac=1.0, random_state=seed)
    keep = counts[counts.cumsum() <= max_images].index
    return df[df["patient_id"].isin(keep)].copy()


def split_by_patient(df: pd.DataFrame, seed: int, test_size=0.2, val_size=0.1):
    """70/10/20 split where every patient lands in exactly one split."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_idx, test_idx = next(gss.split(df, groups=df["patient_id"]))
    trainval, test = df.iloc[trainval_idx], df.iloc[test_idx]

    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=seed)
    train_idx, val_idx = next(gss2.split(trainval, groups=trainval["patient_id"]))
    return trainval.iloc[train_idx], trainval.iloc[val_idx], test


def prevalence_by_group(df: pd.DataFrame) -> pd.DataFrame:
    """Share of images with each finding, per sex and per age bin (explains later fairness gaps)."""
    parts = []
    for col in ["sex", "age_bin"]:
        g = df.groupby(col)[FINDINGS].mean().round(4)
        g.insert(0, "n_images", df.groupby(col).size())
        g.index = [f"{col}={v}" for v in g.index]
        parts.append(g)
    overall = df[FINDINGS].mean().round(4).to_frame("overall").T
    overall.insert(0, "n_images", len(df))
    return pd.concat([overall] + parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/kaggle/input")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--report-dir", default="reports")
    ap.add_argument("--max-images", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_metadata(find_metadata(Path(args.data_dir)))
    print(f"Loaded {len(df)} images from {df['patient_id'].nunique()} patients")
    df = patient_subset(df, args.max_images, args.seed)
    print(f"Subset: {len(df)} images from {df['patient_id'].nunique()} patients")

    train, val, test = split_by_patient(df, args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part.to_csv(out / f"{name}.csv", index=False)
        print(f"{name:5s}: {len(part):6d} images, {part['patient_id'].nunique():5d} patients")

    rep = Path(args.report_dir)
    rep.mkdir(parents=True, exist_ok=True)
    prev = prevalence_by_group(train)
    prev.to_csv(rep / "prevalence_by_group.csv")
    print("\nLabel prevalence in the training split:\n", prev.to_string())


if __name__ == "__main__":
    main()
