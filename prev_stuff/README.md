# Offroad Semantic Segmentation — Iterative Benchmarking Journey

This repository documents our team’s end-to-end experimentation journey for offroad semantic segmentation across three major model attempts:

- `attempt_1_vanilla_unet`
- `attempt_2_resnet50_unet`
- `attempt_3_deeplabv3+_fnet`

The goal was not just to improve scores, but to build a **clear, reproducible, and judge-ready** workflow that explains what we tried, what failed, what improved, and why.

---

## 1) Project Summary

We trained and evaluated multiple segmentation pipelines on the offroad dataset to understand performance under difficult real-world conditions:

- strong class imbalance,
- rare/absent classes,
- large train-test distribution gap,
- scene complexity (camouflage, clutter, intra-class variability),
- and data quality issues.

Our documentation style follows a practical storyline:

**Problem → Fix → Results → Challenges → Future Work**

---

## 2) Repository Structure

```text
attempt_1_vanilla_unet/      # Baseline TensorFlow U-Net
attempt_2_resnet50_unet/     # Pretrained encoder + augmentation experiments
attempt_3_deeplabv3+_fnet/   # PyTorch DeepLabV3+ pipeline (latest)
HONRED_ONE'S_OBSERVATIONS.md # Detailed field notes on failure modes and data behavior
```

Expected dataset layout:

```text
root/
	data/
		Offroad_Segmentation_Training_Dataset/
		Offroad_Segmentation_testImages/
```

---

## 3) Experiment Timeline and What Changed

### Attempt 1 — Vanilla U-Net (`attempt_1_vanilla_unet`)

**What we did**
- Trained a vanilla U-Net baseline.
- Minimal augmentation and simpler training setup.

**Observed behavior**
- Could learn broad regions, but struggled with fine boundaries and rare classes.
- Signs of overfitting and texture bias.

**Outcome snapshot**
- Reported test mean IoU was in the ~0.25–0.26 range.

---

### Attempt 2 — ResNet50 U-Net + Analysis (`attempt_2_resnet50_unet`)

**What we improved**
- Switched to a pretrained encoder (ResNet50) to improve feature extraction.
- Added augmentation and deeper diagnostics (distribution and texture analysis).

**Key findings from analysis**
- Mean IoU implementation was valid.
- Pairing and label mapping were correct.
- Performance bottleneck came from data behavior, not metric bugs.

**Challenges uncovered**
- Severe class imbalance (several classes extremely underrepresented).
- Some classes absent/near-absent in test split.
- Strong domain shift between train and test distributions.

**Important data quality note**
- We identified that nearly **700 images were completely black / not rendered properly**.
- These low-information samples can degrade training signal and limit model generalization.

**Outcome snapshot (from logged run)**
- Final train Mean IoU ≈ **0.6328**
- Final val Mean IoU ≈ **0.5739**
- Final test Mean IoU ≈ **0.2530**

---

### Attempt 3 — DeepLabV3+ (EfficientNet-B3 encoder, PyTorch) (`attempt_3_deeplabv3+_fnet`)

**What we improved**
- Moved to DeepLabV3+ with stronger encoder and modern loss setup (weighted CE + Dice).
- Used mixed precision and cosine annealing schedule.
- Explicit class reweighting via inverse-log frequency.

**What we fixed from earlier attempts**
- Addressed preprocessing inconsistencies that previously hurt train/val behavior.
- Preserved original image geometry (avoiding unnecessary destructive resizing in core pipeline logic).

**Current status**
- Training and testing scripts are now separated and reproducible:
	- `attempt_3_deeplabv3+_fnet/train.py`
	- `attempt_3_deeplabv3+_fnet/test.py`

---

## 4) Benchmarking Protocol (Sample Dataset)

After model training is complete, establish your benchmark in the **same command prompt session** used for training, so runtime context is consistent.

### Practical benchmark flow
1. Run `train.py` for the selected attempt.
2. Evaluate immediately after training (for the latest PyTorch attempt, run `test.py`).
3. Record benchmark artifacts:
	 - predictions/qualitative outputs,
	 - loss metrics (when available in that attempt),
	 - IoU metrics (per-class and mean).

This benchmark becomes your baseline for future changes. Use it to:

- compare model variants,
- track improvements over time,
- identify weak classes/locations where the model fails.

---

## 5) Why IoU Can Stay Low Even After Fixes

A lower test IoU does not always mean the model is fundamentally broken. In this project, we observed:

1. **Complete domain shift** between parts of training data and test conditions.
2. **Severe class imbalance**, including rare/absent classes in evaluation splits.
3. **Data quality degradation** (black/unrendered images).

Together, these factors make Mean IoU especially harsh, because absent or weakly represented classes still affect the global average.

---

## 6) Documenting Challenges and How We Addressed Them

### Major challenge patterns
- Background clutter and camouflage in desert-like scenes.
- Intra-class variability (e.g., similar vegetation classes).
- Sequential-frame dataset bias (video-like continuity).

### What we changed to handle them
- Preferred random sampling over sequential batching to reduce temporal overfitting.
- Added stronger encoder features and targeted augmentation.
- Added distribution-level diagnostics to separate metric behavior from model bugs.
- Introduced class-aware weighting to improve minority-class learning pressure.

### Example log style for concise reporting
- **Task:** Model training on offroad segmentation
- **Initial signal:** Low IoU on rare classes
- **Issue:** Weak recall on sparse categories (e.g., Logs) under occlusion and class scarcity
- **Fix:** Rebalancing strategy + improved architecture + better preprocessing consistency
- **Result:** More stable validation behavior and clearer class-wise diagnostics

---

## 7) Documentation & Presentation Standards

To keep team output clear, reproducible, and professional:

- Use short, structured sections with headings and bullets.
- Keep technical depth, but write in plain language for first-time readers.
- Include visuals from training outputs (loss curves, IoU trends, confusion patterns, qualitative samples).
- Show both success cases and failure cases (before/after where possible).

### Suggested report story (max ~8 pages)

1. **Title** — Team, project, one-line objective
2. **Methodology** — Data pipeline, model attempts, key changes
3. **Results & Metrics (pages 3–4)** — IoU trends, comparisons, confusion insights
4. **Challenges & Solutions (pages 5–6)** — What failed, how it was fixed
5. **Conclusion & Future Work (page 7)** — What worked, next upgrades

---

## 8) Reproducibility Checklist

- [ ] Dataset extracted into expected folder structure
- [ ] Required environment/packages installed per attempt
- [ ] Training script executed and checkpoint saved
- [ ] Benchmark evaluation run immediately after training
- [ ] Benchmark table updated with loss/IoU + qualitative notes
- [ ] Failure cases documented with screenshots and corrective action

---

## 9) Final Takeaway

This project demonstrates that segmentation performance is shaped as much by **data quality and distribution realism** as by architecture choice. Our three-attempt pipeline creates a repeatable framework to benchmark honestly, diagnose failures, and iterate toward better generalization.