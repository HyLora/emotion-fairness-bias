## 📂 Fairness-Aware Emotion Classification Using AIF360

*Script: `fairness_audio_classification.py`*

> **Fairness-aware Emotion Classification from Audio using AIF360**
>
> This script uses the RAVDESS dataset to classify emotions (happy vs neutral) from audio signals. It introduces a controlled bias (by removing a fraction of male-happy samples), then applies the Reweighing method from IBM's AIF360 library to mitigate this bias. Fairness metrics and classification performance are compared before and after the mitigation.
>
> Developed by: Laura Capella
> Course: Logics for AI – Fairness Project

This repository investigates **bias mitigation in binary emotion classification** using **audio features** extracted from the **RAVDESS dataset**. A known imbalance is introduced in the data by removing a significant portion of "happy" samples from male speakers, and the **Reweighing algorithm** from IBM’s **AIF360** is applied to improve fairness without substantially sacrificing accuracy.

---

### 🧠 Objectives

* Perform **emotion classification** (Happy vs Neutral) from speech audio.
* Simulate **gender-based bias** by distorting the original distribution (70% removal of male-happy).
* Use **AIF360's Reweighing method** to mitigate observed bias.
* Measure and compare **accuracy**, **F1-score**, **ROC AUC**, and **fairness metrics** before and after mitigation.

---

### 🗂️ Dataset & Parameters

* **Source**: [RAVDESS Dataset](https://zenodo.org/record/1188976)
* **Audio Files**: `.wav` from 24 actors, sampled at 22 050 Hz
* **Emotions used**:

  * `Neutral` (`label = 0`)
  * `Happy` (`label = 1`)
* **Gender inference**: Actor ID parity

  * Even = Female (`gender = 0`)
  * Odd  = Male   (`gender = 1`)
* **Bias removal fraction**: `REMOVAL_FRACTION = 0.7` (70% of male-happy removed)

---

### 🔍 Feature Extraction

* **Technique**: MFCC (13 coefficients)
* **Library**: `librosa`
* **Processing**:

  1. Load audio at 22 050 Hz
  2. Compute MFCCs
  3. Average over time → 13-dimensional feature vector

---

### 📊 Pipeline Overview

1. **Data Extraction & Labeling**
2. **Bias Introduction**: remove subset of male-happy samples
3. **Visualization**: bar plots of (gender,label) distributions
4. **Preprocessing**: `StandardScaler` + train/test split (80/20, stratified)
5. **Baseline Model**: Logistic Regression
6. **Fairness Mitigation**: AIF360 `Reweighing`
7. **Evaluation**:

   * Accuracy, F1-score, ROC AUC
   * Fairness metrics: Disparate Impact, Statistical Parity Difference, Equal Opportunity Difference, Average Odds Difference
   * Confusion matrices (normalized overlays)

---

### 📈 Results Summary

| Metric                   | Baseline | Post-Reweighing |
| ------------------------ | -------- | --------------- |
| Accuracy                 | 0.XX     | 0.XX            |
| F1-score                 | 0.XX     | 0.XX            |
| ROC AUC                  | 0.XX     | 0.XX            |
| Disparate Impact         | 0.XX     | ✅ \~1.0         |
| Statistical Parity Diff. | 0.XX     | ✅ reduced       |
| Equal Opportunity Diff.  | 0.XX     | ✅ reduced       |
| Average Odds Difference  | 0.XX     | ✅ reduced       |

*Exact values available in script output.*

---

### 🛠️ Requirements

```bash
pip install numpy pandas matplotlib librosa scikit-learn aif360
```

---

### 📄 File Structure

```
🔹 Audio_Speech_Actors_01-24/     # RAVDESS audio files
🔹 fairness_audio_classification.py  # main script with docstring header
🔹 README.md
```

---

### 👩‍💼 Author

**Laura Capella**
Master's Degree in Human-Centered Artificial Intelligence
Department of Philosophy, University of Milan
Course: *Logics for AI*
Supervisor: Prof. Giuseppe Primiero
