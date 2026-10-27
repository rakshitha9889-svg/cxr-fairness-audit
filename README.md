# Fairness-audited chest X-ray AI

**Research project, not for clinical use.**

Multi-label chest X-ray classifier (5 findings) trained on NIH ChestX-ray14, audited for performance gaps across sex and age, externally validated on CheXpert.

## Status
- [x] Patient-level 70/10/20 split with leakage tests
- [ ] Baseline DenseNet-121
- [ ] Fairness audit
- [ ] Mitigation
- [ ] Grad-CAM
- [ ] Model card + demo
- [ ] External validation (CheXpert)
- [ ] Selective prediction

## Data split
Images are split **by patient**, not by image: the same patient has many X-rays, and a random image split leaks patients into the test set and inflates AUROC. `tests/test_splits.py` checks that no patient appears in two splits.

## Reproduce
```bash
pip install -r requirements.txt
python src/data_prep.py --data-dir /kaggle/input --max-images 25000
pytest -q
```
