import os
import numpy as np
import pandas as pd
import warnings
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import ClassificationMetric

warnings.filterwarnings("ignore")

# === PARAMETERS ===
DATA_DIR = './Audio_Speech_Actors_01-24'
SAMPLE_RATE = 22050
REMOVAL_FRACTION = 0.7  # fraction of male-happy samples to remove

# === HELPER FUNCTIONS ===
def get_emotion_label(filename):
    """Return 'happy' if emotion code == 3, 'neutral' if == 1, else None."""
    code = int(filename.split('-')[2])
    if code == 3:
        return 'happy'
    if code == 1:
        return 'neutral'
    return None

def get_gender_label(filename):
    """Return 0 for female (even actor ID), 1 for male (odd actor ID)."""
    actor_id = int(filename.split('-')[-1].split('.')[0])
    return 0 if actor_id % 2 == 0 else 1

def extract_mfcc_features(path):
    """Load audio and compute mean MFCC vector (13 coefficients)."""
    y, _ = librosa.load(path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

# === DATA EXTRACTION ===
X_list, y_list, gender_list = [], [], []
for root, _, files in os.walk(DATA_DIR):
    for fname in files:
        if not fname.endswith('.wav'):
            continue
        emo = get_emotion_label(fname)
        if emo is None:
            continue
        features = extract_mfcc_features(os.path.join(root, fname))
        X_list.append(features)
        y_list.append(1 if emo == 'happy' else 0)
        gender_list.append(get_gender_label(fname))

X = np.array(X_list)
y = np.array(y_list)
gender = np.array(gender_list)

# === INTRODUCE CONTROLLED IMBALANCE ===
df = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
df['label']  = y
df['gender'] = gender

print("\n=== Original (gender, label) proportions ===")
print(df.groupby(['gender','label']).size().div(len(df)))

# Remove a fraction of male-happy samples
mask = (df['gender'] == 1) & (df['label'] == 1)
drop_idx = df[mask].sample(frac=REMOVAL_FRACTION, random_state=42).index
df = df.drop(drop_idx)

print("\n=== After imbalance (gender, label) proportions ===")
print(df.groupby(['gender','label']).size().div(len(df)))

# === PREPROCESSING ===
features = df[[f'x{i}' for i in range(X.shape[1])]].values
labels   = df['label'].values
prot_attr= df['gender'].values

scaler = StandardScaler()
features = scaler.fit_transform(features)

X_train, X_test, y_train, y_test, g_train, g_test = train_test_split(
    features, labels, prot_attr,
    test_size=0.2, random_state=42, stratify=labels
)

# === AIF360 DATASET CONSTRUCTION ===
def make_binary_label_dataset(X, y, g):
    df_bld = pd.DataFrame(
        np.hstack((X, y.reshape(-1,1), g.reshape(-1,1))),
        columns=[f'x{i}' for i in range(X.shape[1])] + ['label','gender']
    )
    return BinaryLabelDataset(
        favorable_label=1,
        unfavorable_label=0,
        df=df_bld,
        label_names=['label'],
        protected_attribute_names=['gender']
    )

train_bld = make_binary_label_dataset(X_train, y_train, g_train)
test_bld  = make_binary_label_dataset(X_test,  y_test,  g_test)

# === BASELINE MODEL ===
clf_base = LogisticRegression(max_iter=1000)
clf_base.fit(X_train, y_train)
pred_base = clf_base.predict(X_test)
acc_base  = accuracy_score(y_test, pred_base)

test_pred_base = test_bld.copy()
test_pred_base.labels = pred_base.reshape(-1,1)

metric_base = ClassificationMetric(
    test_bld, test_pred_base,
    unprivileged_groups=[{'gender': 0}],
    privileged_groups=[{'gender': 1}]
)

print("\n=== BASELINE RESULTS ===")
print(f"Accuracy: {acc_base:.4f}")
print("Disparate Impact:          ", metric_base.disparate_impact())
print("Statistical Parity Diff.:  ", metric_base.statistical_parity_difference())
print("Equal Opportunity Diff.:   ", metric_base.equal_opportunity_difference())
print("Average Odds Diff.:        ", metric_base.average_odds_difference())

# === REWEIGHING PREPROCESSING ===
RW = Reweighing(
    unprivileged_groups=[{'gender': 0}],
    privileged_groups=[{'gender': 1}]
)
train_rw = RW.fit_transform(train_bld)

# Debug: inspect weights
weights = pd.Series(train_rw.instance_weights)
print("\n=== Instance Weights After Reweighing ===")
print(weights.describe())

wdf = pd.DataFrame({
    'gender': train_rw.protected_attributes.flatten(),
    'label':  train_rw.labels.flatten(),
    'weight': train_rw.instance_weights
})
print("\n=== Mean Weight by (gender, label) ===")
print(wdf.groupby(['gender','label'])['weight'].mean())

# === MODEL WITH WEIGHTS ===
# Extract only MFCC features (drop protected 'gender' column)
X_train_rw = train_rw.features[:, :X_train.shape[1]]
y_train_rw = train_rw.labels.ravel()
w_train_rw = train_rw.instance_weights

clf_rw = LogisticRegression(max_iter=1000)
clf_rw.fit(X_train_rw, y_train_rw, sample_weight=w_train_rw)
pred_rw = clf_rw.predict(X_test)
acc_rw  = accuracy_score(y_test, pred_rw)

test_pred_rw = test_bld.copy()
test_pred_rw.labels = pred_rw.reshape(-1,1)

metric_rw = ClassificationMetric(
    test_bld, test_pred_rw,
    unprivileged_groups=[{'gender': 0}],
    privileged_groups=[{'gender': 1}]
)

print("\n=== POST-REWEIGHING RESULTS ===")
print(f"Accuracy: {acc_rw:.4f}")
print("Disparate Impact:          ", metric_rw.disparate_impact())
print("Statistical Parity Diff.:  ", metric_rw.statistical_parity_difference())
print("Equal Opportunity Diff.:   ", metric_rw.equal_opportunity_difference())
print("Average Odds Diff.:        ", metric_rw.average_odds_difference())

# === ENHANCED CONFUSION MATRIX PLOTS ===
labels = ['Neutral', 'Happy']
cm_base = confusion_matrix(y_test, pred_base)
cm_rw   = confusion_matrix(y_test, pred_rw)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, cm, title in zip(axes, [cm_base, cm_rw], ['Baseline', 'Post‑Reweighing']):
    # row‑normalize to get percentages
    cm_pct = cm.astype(float) / cm.sum(axis=1)[:, None]

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap='viridis', colorbar=False, values_format='d')

    # overlay percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm_pct[i, j] * 100
            ax.text(j, i + 0.15, f"{pct:.1f}%", 
                    ha='center', va='center', color='white', fontsize=10)

    ax.set_title(title)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

plt.tight_layout()
plt.show()
