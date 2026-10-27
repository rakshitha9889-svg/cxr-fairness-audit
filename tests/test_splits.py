"""Checks that the splits are leakage-free. Run with:  pytest -q"""
from pathlib import Path

import pandas as pd
import pytest

SPLITS = Path(__file__).resolve().parents[1] / "data" / "splits"
FINDINGS = ["Cardiomegaly", "Effusion", "Atelectasis", "Pneumothorax", "Edema"]


@pytest.fixture(scope="module")
def splits():
    if not (SPLITS / "train.csv").exists():
        pytest.skip("Run src/data_prep.py first")
    return {n: pd.read_csv(SPLITS / f"{n}.csv") for n in ["train", "val", "test"]}


def test_no_patient_in_two_splits(splits):
    ids = {n: set(df["patient_id"]) for n, df in splits.items()}
    assert not ids["train"] & ids["val"]
    assert not ids["train"] & ids["test"]
    assert not ids["val"] & ids["test"]


def test_no_image_in_two_splits(splits):
    all_images = pd.concat([df["image_index"] for df in splits.values()])
    assert all_images.is_unique


def test_labels_are_binary(splits):
    for df in splits.values():
        assert set(df[FINDINGS].stack().unique()) <= {0, 1}


def test_split_sizes_roughly_70_10_20(splits):
    total = sum(len(df) for df in splits.values())
    assert 0.6 < len(splits["train"]) / total < 0.8
    assert 0.05 < len(splits["val"]) / total < 0.15
    assert 0.15 < len(splits["test"]) / total < 0.25
