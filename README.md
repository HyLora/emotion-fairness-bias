## 📂 Fairness-Aware Emotion Classification Using AIF360

This project investigates **bias mitigation in binary emotion classification** using **audio features** extracted from the **RAVDESS dataset**. A known imbalance is introduced in the data by removing a significant portion of "happy" samples from male speakers, and the **Reweighing algorithm** from IBM’s **AIF360** is applied to improve fairness without substantially sacrificing accuracy.

---

### 🧠 Objectives

* Perform **emotion classification** (Happy vs Neutral) from speech audio.
* Simulate **gender-based bias** by distorting the original distribution.
* Use **AIF360's Reweighing method** to mitigate observed bias.
* Measure and compare **accuracy** and **fairness metrics** before and after mitigation.

---

### 🗂️ Dataset

* **Source**: [RAVDESS Dataset](https://zenodo.org/record/1188976)
* **Input**: `.wav` audio files from 24 actors.
* **Emotions used**:

  * `Neutral` (`label = 0`)
  * `Happy` (`label = 1`)
* **Gender inference**: Actor ID parity

  * Even = Female (`gender = 0`)
  * Odd = Male (`gender = 1`)

---

### 🔍 Feature Extraction

* **Technique**: MFCC (Mel-Frequency Cepstral Coefficients)
* **Library**: `librosa`
* **Processing**:

  * 13 MFCCs extracted per audio
  * Averaged over time to form a 13-dimensional feature vector per sample

---

### ⚖️ Bias Introduction

To simulate a realistic fairness challenge:

* **70% of male "happy" samples** are **removed** to induce bias.
* This produces a **disproportionate representation** across the (gender, label) intersection.

The imbalance simulates algorithmic unfairness due to skewed training data.

---

### 📊 Preprocessing and Training

* **Standardization**: Features scaled with `StandardScaler`
* **Train/Test Split**: 80/20, stratified by label
* **Classifier**: Logistic Regression
* **Evaluation metrics**:

  * **Accuracy**
  * **F1-score**
  * **ROC AUC**
  * **Confusion Matrix**

---

### ♻️ Fairness Mitigation

**Tool**: IBM AIF360
**Method**: `Reweighing`

* Assigns weights to training instances based on their (gender, label) combination.
* Aims to **equalize the distribution** across privileged/unprivileged groups.

Training is repeated using the instance weights provided by the Reweighing algorithm.

---

### 📏 Fairness Metrics

Computed using `aif360.metrics.ClassificationMetric`:

| Metric                            | Description                                      |
| --------------------------------- | ------------------------------------------------ |
| **Disparate Impact**              | Ratio of favorable outcomes across groups        |
| **Statistical Parity Difference** | Difference in selection rates                    |
| **Equal Opportunity Difference**  | Difference in true positive rates                |
| **Average Odds Difference**       | Combined difference in true/false positive rates |

---

### 🧪 Results Summary

| Metric                   | Baseline | Post-Reweighing |
| ------------------------ | -------- | --------------- |
| Accuracy                 | 0.XX     | 0.XX            |
| F1-score                 | 0.XX     | 0.XX            |
| ROC AUC                  | 0.XX     | 0.XX            |
| Disparate Impact         | 0.XX     | ✅  \~1.0        |
| Statistical Parity Diff. | 0.XX     | ✅ reduced       |
| Equal Opportunity Diff.  | 0.XX     | ✅ reduced       |
| Average Odds Difference  | 0.XX     | ✅ reduced       |

*Exact values available in the output summary.*

---

### 📈 Visualizations

* **Distributions** of (gender, label) before and after induced bias
* **Confusion matrices** for baseline and mitigated models
* **Overlay of fairness metrics**

---

### 🛠️ Requirements

```bash
pip install numpy pandas matplotlib librosa scikit-learn aif360
```

---

### 📄 File Structure

```
👉 Audio_Speech_Actors_01-24/     # RAVDESS audio files
👉 fairness_audio_classification.py
👉 README.md
```

---

### 👩‍💼 Author

**Laura Capella**
Master's Degree in Human-Centered Artificial Intelligence
Department of Philosophy, University of Milan
Course: *Logics for AI*
Supervisor: Prof. Giuseppe Primiero
