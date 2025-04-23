# emotion-fairness-bias
Experimental report on bias mitigation in emotion classification using the RAVDESS dataset and IBM AIF360. Part of the 'Logics for AI' course at UniMi.
# Evaluating and Mitigating Bias in Emotion Classification

This project was developed as part of the "Logics for AI" course (Test 3) held by Professor Giuseppe Primiero at the University of Milan.

## 🎯 Objective

The goal is to evaluate and mitigate gender bias in an emotion classification task using the RAVDESS dataset and the IBM AIF360 fairness toolkit.

## 📁 Project Structure

- `data/`: https://zenodo.org/records/1188976 (link to the audio files from RAVDESS).
- `preprocessing/`: scripts to extract features and labels, and to infer gender from filenames.
- `models/`: model definition and training pipeline (e.g., logistic regression, SVM).
- `fairness/`: reweighing mitigation implementation using AIF360.
- `evaluation/`: script to compute accuracy, precision, recall, and fairness metrics (e.g., disparate impact).
- `notebooks/`: interactive exploratory analysis (optional).

## ⚙️ Dependencies

- Python 3.10+
- `librosa`, `scikit-learn`, `pandas`, `aif360`, `matplotlib`

Install all requirements with:

```bash
pip install -r requirements.txt
