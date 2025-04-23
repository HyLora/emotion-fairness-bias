# emotion-fairness-bias
Experimental report on bias mitigation in emotion classification using the RAVDESS dataset and IBM AIF360. Part of the 'Logics for AI' course at UniMi.
# Evaluating and Mitigating Bias in Emotion Classification

This project was developed as part of the "Logics for AI" course (Test 3) held by Professor Giuseppe Primiero at the University of Milan.

## 🎯 Objective

The goal is to evaluate and mitigate gender bias in an emotion classification task using the RAVDESS dataset and the IBM AIF360 fairness toolkit.

## 📁 Project Structure

- `data/`: contains the audio files from RAVDESS.
- `preprocessing/`: scripts to extract features and labels, and to infer gender from filenames.
- `models/`: model definition and training pipeline (e.g., logistic regression, SVM).
- `fairness/`: reweighing mitigation implementation using AIF360.
- `evaluation/`: script to compute accuracy, precision, recall, and fairness metrics (e.g., disparate impact).
- `notebooks/`: interactive exploratory analysis (optional).

## ⚙️ Dependencies

- Python 3.10+
- `librosa`, `scikit-learn`, `pandas`, `aif360`, `matplotlib`

Install all requirements with:

pip install -r requirements.txt

📊 Approach
Extract features and gender labels from RAVDESS audio filenames.

Train a baseline emotion classification model.

Evaluate performance across subgroups (male vs. female).

Apply reweighing mitigation from AIF360.

Re-evaluate and compare fairness metrics.

📌 Notes
The dataset is not included in the repository due to size. Please download it from: RAVDESS Dataset Link.

Gender information is inferred from filename structure.

🔗 Author
Laura Capella — Master's in Human-Centered AI
University of Milan

yaml
Copy
Edit

