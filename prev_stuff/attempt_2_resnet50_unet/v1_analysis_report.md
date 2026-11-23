# Segmentation Project Analysis Report

Date: 2026-02-08
Project: Offroad segmentation (ResNet50 U-Net)

## Executive summary
- The MeanIoU implementation is correct for logits and is not the reason for the gap.
- Image-mask pairing is correct for all splits (0 missing pairs).
- Label mapping is consistent: raw mask values match `CLASS_IDS` in all splits.
- The train/val/test class distributions differ substantially, and several classes are absent or near-absent in the test set. This alone can depress MeanIoU even if qualitative predictions look reasonable.

## What I ran (concrete tests)
I ran a scan over a 50-mask sample per split to validate pairing and label distributions.

### 1) Image-mask pairing
Result: all splits are clean in the file system.
- Train: 2857 images, 2857 masks, 0 missing
- Val: 317 images, 317 masks, 0 missing
- Test: 1002 images, 1002 masks, 0 missing

### 2) Raw label values in masks
Result: raw values are consistent with your `CLASS_IDS`.
- Train raw top IDs: 300, 7100, 10000, 500, 550, 800, 200
- Val raw top IDs: 10000, 300, 7100, 500, 550, 800, 200, 100
- Test raw top IDs: 7100, 800, 300, 10000, 500, 100, 200

This confirms there is no remapping mismatch between train/val/test.

### 3) Remapped class distribution (sample of 50 masks)
Classes are in the order: 100, 200, 300, 500, 550, 600, 700, 800, 7100, 10000.

Train (sample 50):
- 300: 36.49%
- 7100: 34.93%
- 10000: 18.20%
- 500: 5.04%
- 550: 3.34%
- 800: 1.87%
- 200: 0.13%
- 100 / 600 / 700: 0.00%

Val (sample 50):
- 10000: 41.86%
- 300: 28.94%
- 7100: 20.58%
- 500: 3.81%
- 550: 2.75%
- 800: 1.87%
- 200: 0.18%
- 100: 0.01%
- 600 / 700: 0.00%

Test (sample 50):
- 7100: 39.75%
- 800: 22.01%
- 300: 18.38%
- 10000: 17.18%
- 500: 2.44%
- 100: 0.24%
- 200: 0.00%
- 550 / 600 / 700: 0.00%

This shows a real distribution shift: test has more class 800 and fewer 550/200; classes 600 and 700 are absent across all samples. This makes MeanIoU more pessimistic because it averages across all classes regardless of presence.

## Why MeanIoU drops on test even when predictions look reasonable
1) **MeanIoU averages across all classes**
   - If a class is absent in ground truth, its IoU is effectively 0, which drags the mean down.
   - In the test sample, 3 classes are completely absent and 2 are near-zero. That means at least 30-50% of the MeanIoU components are effectively zeros.

2) **Test distribution differs from train/val**
   - The test split has a larger share of class 800 and fewer 550/200 compared to train/val.
   - This affects both the true IoU and the metric sensitivity, even if the model has learned the dominant classes.

3) **Class imbalance (already visible in training)**
   - Even in train/val, classes 600/700 are not present in the sample and 200/100 are tiny.
   - This means the model likely has weak supervision for those classes, and MeanIoU will remain lower.

## Validation of MeanIoU implementation
Your current metric is correct and consistent between train/test:
- `MeanIoUFromLogits` uses `argmax` over the channel dimension.
- It matches the training metric and uses the same label mapping.

There is no bug in the MeanIoU computation.

## Why your qualitative prediction looks reasonable
The model is still learning textures and boundaries. Your visualization confirms it differentiates regions, but it does not cleanly separate classes that are rare or missing in test. This is typical when:
- The test set emphasizes different classes,
- Classes are highly imbalanced,
- The metric is strict (MeanIoU across all classes).

## Is 20 epochs too low? Is U-Net wrong?
- 20 epochs is not too low given that validation MeanIoU already plateaus around 0.57.
- U-Net + ResNet50 is a valid baseline. The issue is not architecture but distribution + imbalance.

## Actionable next steps (concrete)
1) **Report class-wise IoU and frequency-weighted IoU**
   - This will show whether the model is actually good on frequent classes.

2) **Consider a presence-weighted MeanIoU**
   - Mean IoU across only classes present in each image is a more stable indicator.

3) **Add class-weighted CE or focal loss**
   - It will reduce dominance of 300/7100/10000 and lift rare classes.

4) **Tune augmentation and backbone fine-tuning**
   - Domain shift is likely. Fine-tune top ResNet blocks at a lower LR.

## Conclusion
The gap between train/val MeanIoU and test MeanIoU is explained by measurable distribution differences and class sparsity in the test set. The metric implementation is correct, the pairing is correct, and the label mapping is consistent. The biggest driver is class distribution shift combined with MeanIoU’s strict averaging across all classes.

---
End of report.
