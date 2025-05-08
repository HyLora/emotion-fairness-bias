#!/usr/bin/env python3
# Fair Speech Emotion Recognition
# Author: Laura Capella
# Date: May 2025
# Course: Logics for AI - HCAI 2024-2025

import os
import sys
import argparse
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler  # Added for feature scaling
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns

# Define emotion labels mapping for clarity
EMOTION_MAP = {
    1: "neutral",      # label 01 in RAVDESS dataset
    2: "calm",         # label 02 in RAVDESS dataset
    3: "happy",        # label 03 in RAVDESS dataset
    4: "sad",          # label 04 in RAVDESS dataset
    5: "angry",        # label 05 in RAVDESS dataset
    6: "fearful",      # label 06 in RAVDESS dataset
    7: "disgust",      # label 07 in RAVDESS dataset
    8: "surprised"     # label 08 in RAVDESS dataset
}

# For binary classification in this study, we focus on neutral vs happy
PRIMARY_EMOTIONS = {1: "neutral", 3: "happy"}

# Define fairness threshold constants
DI_LOWER_BOUND = 0.8  # Lower bound for Disparate Impact
DI_UPPER_BOUND = 1.25  # Upper bound for Disparate Impact

def parse_arguments():
    """Parse command line arguments for the script."""
    parser = argparse.ArgumentParser(description='Fair Speech Emotion Recognition')
    parser.add_argument('--data_dir', type=str, default='Audio_Speech_Actors_01-24',
                        help='Directory containing audio files')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results and plots')
    parser.add_argument('--sample_rate', type=int, default=22050,
                        help='Sample rate for audio processing')
    parser.add_argument('--n_mfcc', type=int, default=13,
                        help='Number of MFCC features to extract')
    # Modified default bias_drop to a more moderate value
    parser.add_argument('--bias_drop', type=float, default=0.4,
                        help='Fraction of male-happy samples to drop for bias induction')
    # Added a new parameter for balancing female-neutral samples
    parser.add_argument('--balance_factor', type=float, default=0.2,
                        help='Fraction of female-neutral samples to drop for better balancing')
    # Modified default eta for PrejudiceRemover to be more moderate
    parser.add_argument('--eta', type=float, default=25.0,
                        help='Eta parameter for PrejudiceRemover (fairness-accuracy trade-off)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of cross-validation splits')
    parser.add_argument('--no_plots', action='store_true',
                        help='Disable plot generation')
    # Added regularization parameters for models
    parser.add_argument('--C', type=float, default=1.0,
                        help='Inverse of regularization strength for logistic regression')
    return parser.parse_args()

def setup_directories(output_dir):
    """Create necessary directories for outputs."""
    dirs = {
        'plots': os.path.join(output_dir, 'plots'),
        'data': os.path.join(output_dir, 'data')
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    return dirs

def extract_features_and_labels(data_dir, sample_rate, n_mfcc):
    """Extract MFCC features from audio files with proper error handling."""
    features, labels, genders = [], [], []
    processed_files = 0
    error_files = 0
    
    print(f"Extracting features from audio files in {data_dir}...")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    for actor_dir in os.listdir(data_dir):
        actor_path = os.path.join(data_dir, actor_dir)
        if not os.path.isdir(actor_path): 
            continue

        for file in os.listdir(actor_path):
            if file.endswith(".wav"):
                try:
                    # Parse filename for metadata (assuming RAVDESS naming convention)
                    parts = file.split("-")
                    if len(parts) < 3:
                        print(f"Warning: File {file} does not follow expected naming convention. Skipping.")
                        error_files += 1
                        continue
                    
                    emotion_code = int(parts[2])
                    
                    # For binary classification, only use neutral (1) and happy (3)
                    if emotion_code not in [1, 3]:
                        continue
                        
                    gender = 1 if int(actor_dir[-2:]) % 2 == 1 else 0  # Male if odd-numbered actor
                    path = os.path.join(actor_path, file)

                    # Load and extract features
                    y, sr = librosa.load(path, sr=sample_rate)
                    
                    mfcc = librosa.feature.mfcc(
                        y=y, 
                        sr=sr, 
                        n_mfcc=n_mfcc,
                        n_fft=2048,           # Increased FFT window size
                        hop_length=512,       # Hop length for frame shifting
                        fmin=20,              # Minimum frequency
                        fmax=8000,            # Maximum frequency for analysis 
                        htk=True              # Use HTK formula for mel scale
                    )

                    # Add delta and delta-delta features for better representation
                    mfcc_delta = librosa.feature.delta(mfcc)
                    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
                    
                    # Combine features and take statistics
                    combined_features = np.concatenate([
                        np.mean(mfcc.T, axis=0),
                        np.std(mfcc.T, axis=0),
                        np.mean(mfcc_delta.T, axis=0),
                        np.mean(mfcc_delta2.T, axis=0)
                    ])
                    
                    features.append(combined_features)
                    labels.append(1 if emotion_code == 3 else 0)  # Happy=1, Neutral=0
                    genders.append(gender)
                    processed_files += 1
                    
                    # Print progress every 50 files
                    if processed_files % 50 == 0:
                        print(f"Processed {processed_files} files...")
                    
                except Exception as e:
                    print(f"Error processing file {file}: {str(e)}")
                    error_files += 1
    
    print(f"Feature extraction complete. Processed {processed_files} files with {error_files} errors.")
    
    if not features:
        raise ValueError("No valid features extracted. Check your dataset.")
    
    # Create and clean DataFrame
    df = pd.DataFrame(features)
    df['label'] = labels
    df['gender'] = genders
    df.dropna(inplace=True)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution: {df['label'].value_counts().to_dict()}")
    print(f"Gender distribution: {df['gender'].value_counts().to_dict()}")
    
    return df

def optimize_hyperparameters(X, y, gender, random_state):
    """Find optimal hyperparameters for the classifier."""
    print("Optimizing hyperparameters...")
    
    # Split data for hyperparameter tuning
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define parameter grid
    param_grid = {
        'C': [0.1, 1.0, 10.0],
        'class_weight': [None, 'balanced'],
        'solver': ['liblinear', 'lbfgs']
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=random_state),
        param_grid=param_grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    # Evaluate best model on test set
    best_model = grid_search.best_estimator_
    test_score = best_model.score(X_test_scaled, y_test)
    print(f"Test accuracy with best parameters: {test_score:.3f}")
    
    return grid_search.best_params_

def balanced_bias_induction(df, drop_fraction_male, drop_fraction_female, random_state):
    """
    Induce bias in a more balanced way by adjusting both male-happy and female-neutral samples.
    This helps prevent extreme disparate impact values.
    """
    print(f"Inducing balanced bias...")
    
    biased_df = df.copy()
    
    # Original distribution
    original_class_dist = df['label'].value_counts().to_dict()
    original_gender_dist = pd.crosstab(df['gender'], df['label'])
    print("Original distribution by gender and emotion:")
    print(original_gender_dist)
    
    # Get distribution of samples by gender and emotion
    male_happy = (biased_df['gender'] == 1) & (biased_df['label'] == 1)
    male_neutral = (biased_df['gender'] == 1) & (biased_df['label'] == 0)
    female_happy = (biased_df['gender'] == 0) & (biased_df['label'] == 1)
    female_neutral = (biased_df['gender'] == 0) & (biased_df['label'] == 0)
    
    # Calculate samples to drop
    male_happy_drop = int(drop_fraction_male * male_happy.sum())
    female_neutral_drop = int(drop_fraction_female * female_neutral.sum())
    
   # Drop male-happy samples (creates bias against men for positive outcome)
    if male_happy_drop > 0:
        drop_indices_male = biased_df[male_happy].sample(n=male_happy_drop, random_state=random_state).index
        biased_df = biased_df.drop(index=drop_indices_male)
    else:
        print("No male-happy samples to drop (male_happy_drop = 0)")

    # Drop female-neutral samples (balances by reducing female negative outcomes)
    if female_neutral_drop > 0:
        drop_indices_female = biased_df[female_neutral].sample(n=female_neutral_drop, random_state=random_state).index
        biased_df = biased_df.drop(index=drop_indices_female)
    else:
        print("No female-neutral samples to drop (female_neutral_drop = 0)")
    
    # Check new distribution
    new_gender_dist = pd.crosstab(biased_df['gender'], biased_df['label'])
    print("\nBiased distribution by gender and emotion:")
    print(new_gender_dist)
    
    # Calculate implied disparate impact to check if it's in target range
    female_pos_rate = new_gender_dist.loc[0, 1] / new_gender_dist.loc[0].sum()
    male_pos_rate = new_gender_dist.loc[1, 1] / new_gender_dist.loc[1].sum()
    
    implied_di = female_pos_rate / male_pos_rate
    print(f"\nImplied Disparate Impact (female/male positive rate): {implied_di:.3f}")
    print(f"Target DI range: {DI_LOWER_BOUND}-{DI_UPPER_BOUND}")
    
    return biased_df

def to_binary_label_dataset(X, y, protected):
    """Convert data to AIF360 BinaryLabelDataset format."""
    df = pd.DataFrame(X)
    df['label'] = y
    df['protected'] = protected
    return BinaryLabelDataset(df=df, label_names=['label'], protected_attribute_names=['protected'])

def plot_confusion_matrices(metrics_baseline, metrics_fair_pr, metrics_fair_rw, plot_dir):
    """Plot confusion matrices for each method."""
    print("Plotting confusion matrices...")
    
    # Compute average confusion matrices
    cm_baseline = np.mean(np.array(metrics_baseline['confusion']), axis=0)
    cm_pr = np.mean(np.array(metrics_fair_pr['confusion']), axis=0)
    cm_rw = np.mean(np.array(metrics_fair_rw['confusion']), axis=0)
    
    # Create a figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot each confusion matrix
    for i, (cm, title) in enumerate(zip(
        [cm_baseline, cm_pr, cm_rw],
        ['Baseline', 'Prejudice Remover', 'Reweighing']
    )):
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(
            cm_norm, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues', 
            ax=axes[i],
            cbar=False,
            xticklabels=['Neutral', 'Happy'],
            yticklabels=['Neutral', 'Happy']
        )
        axes[i].set_title(f'{title} Confusion Matrix', fontsize=12)
        axes[i].set_ylabel('True Label', fontsize=10)
        axes[i].set_xlabel('Predicted Label', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'confusion_matrices.png'), dpi=300)
    plt.close()

def statistical_significance_test(metrics_baseline, metrics_fair_pr, metrics_fair_rw):
    """Perform statistical significance tests between methods."""
    print("\nPerforming statistical significance tests...")
    
    # Prepare arrays
    acc_data = {
        'Baseline': metrics_baseline['acc'],
        'PrejudiceRemover': metrics_fair_pr['acc'],
        'Reweighing': metrics_fair_rw['acc']
    }
    
    di_data = {
        'Baseline': metrics_baseline['di'],
        'PrejudiceRemover': metrics_fair_pr['di'],
        'Reweighing': metrics_fair_rw['di']
    }
    
    # Perform Friedman test on accuracy (non-parametric repeated measures ANOVA)
    print("Accuracy comparison:")
    try:
        from scipy.stats import friedmanchisquare
        
        # Extract values as lists
        acc_values = list(acc_data.values())
        
        # Perform Friedman test if we have enough samples
        if len(acc_values[0]) >= 3:  # Need at least 3 samples
            friedman_stat, p_value = friedmanchisquare(*acc_values)
            print(f"  Friedman test: statistic={friedman_stat:.3f}, p-value={p_value:.3f}")
            if p_value < 0.05:
                print("  There are statistically significant differences between methods.")
            else:
                print("  No statistically significant differences between methods.")
        else:
            print("  Not enough samples for Friedman test.")
    except Exception as e:
        print(f"  Error in statistical test: {e}")
    
    # Perform pairwise Wilcoxon signed-rank tests for disparate impact
    print("\nDisparate Impact comparison (pairwise):")
    try:
        from scipy.stats import wilcoxon
        
        pairs = [
            ('Baseline', 'PrejudiceRemover'),
            ('Baseline', 'Reweighing'),
            ('PrejudiceRemover', 'Reweighing')
        ]
        
        for method1, method2 in pairs:
            stat, p = wilcoxon(di_data[method1], di_data[method2])
            print(f"  {method1} vs {method2}: p-value={p:.3f}")
            if p < 0.05:
                print(f"    Significant difference found.")
            else:
                print(f"    No significant difference.")
    except Exception as e:
        print(f"  Error in Wilcoxon test: {e}")
def run_fairness_experiment(X, y, gender, args, dirs):
    """Run the fairness experiment with cross-validation."""

    # Initialize Metrics with proper keys for all measurements
    baseline_metrics = {'acc': [], 'roc': [], 'confusion': [], 'di': []}
    metrics_fair_pr = {'acc': [], 'roc': [], 'di': [], 'eo': [], 'avg_odds': [], 'confusion': []}
    metrics_fair_rw = {'acc': [], 'roc': [], 'di': [], 'eo': [], 'avg_odds': [], 'confusion': []}

    # Scale features (this was missing)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize the stratified k-fold cross validation
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    fold = 1

    # Run k-fold cross validation experiment
    for train_idx, test_idx in skf.split(X_scaled, y):
        print(f"\nProcessing fold {fold}/{args.n_splits}...")
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train, g_test = gender[train_idx], gender[test_idx]
        
        # Create datasets for AIF360
        train_bld = to_binary_label_dataset(X_train, y_train, g_train)
        test_bld = to_binary_label_dataset(X_test, y_test, g_test)
        
        # ===== Baseline Model =====
        clf = LogisticRegression(C=args.C, max_iter=1000, random_state=args.random_state, 
                                solver='liblinear', class_weight='balanced')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        baseline_metrics['acc'].append(accuracy_score(y_test, y_pred))
        baseline_metrics['roc'].append(roc_auc_score(y_test, y_prob))
        baseline_metrics['confusion'].append(confusion_matrix(y_test, y_pred))
        
        # Create BinaryLabelDataset for baseline predictions for fairness metrics
        pred_dataset_baseline = test_bld.copy()
        pred_dataset_baseline.labels = np.reshape(y_pred, [-1, 1])
        
        # Calculate disparate impact for baseline
        dataset_metric_baseline = BinaryLabelDatasetMetric(
            pred_dataset_baseline,
            unprivileged_groups=[{'protected': 0}], 
            privileged_groups=[{'protected': 1}]
        )
        baseline_metrics['di'].append(dataset_metric_baseline.disparate_impact())
        
        # ===== Fair Model 1: Prejudice Remover =====
        # Adjust eta for better fairness-accuracy trade-off
        pr = PrejudiceRemover(sensitive_attr='protected', eta=args.eta)
        pr.fit(train_bld)
        fair_pred_pr = pr.predict(test_bld)
        
        # Get predictions
        fair_labels_pr = fair_pred_pr.labels.ravel()
        
        # Calculate metrics
        metrics_fair_pr['acc'].append(accuracy_score(y_test, fair_labels_pr))
        try:
            # Ensure predictions have both classes for ROC calculation
            if len(np.unique(fair_labels_pr)) > 1:
                metrics_fair_pr['roc'].append(roc_auc_score(y_test, fair_labels_pr))
            else:
                # If predictions are all one class, ROC AUC is undefined
                # Default to 0.5 (random classifier performance)
                print(f"Warning: All predictions in fold {fold} for Prejudice Remover are same class. Setting ROC AUC to 0.5.")
                metrics_fair_pr['roc'].append(0.5)
        except Exception as e:
            print(f"Error calculating ROC AUC for Prejudice Remover: {e}")
            metrics_fair_pr['roc'].append(0.5)
        
        metrics_fair_pr['confusion'].append(confusion_matrix(y_test, fair_labels_pr))
        
        # Calculate fairness metrics for Prejudice Remover
        dataset_metric_pr = BinaryLabelDatasetMetric(
            fair_pred_pr,
            unprivileged_groups=[{'protected': 0}], 
            privileged_groups=[{'protected': 1}]
        )
        di = dataset_metric_pr.disparate_impact()
        metrics_fair_pr['di'].append(di)
        
        # Equal Opportunity Difference
        classification_metric = ClassificationMetric(
            test_bld, fair_pred_pr, 
            unprivileged_groups=[{'protected': 0}], 
            privileged_groups=[{'protected': 1}]
        )
        eo_diff = classification_metric.equal_opportunity_difference()
        metrics_fair_pr['eo'].append(eo_diff)
        
        # Add Average Odds Difference calculation
        avg_odds_diff = classification_metric.average_odds_difference()
        metrics_fair_pr['avg_odds'].append(avg_odds_diff)

        # ===== Fair Model 2: Reweighing =====
        # Configure reweighing
        rw = Reweighing(unprivileged_groups=[{'protected': 0}], privileged_groups=[{'protected': 1}])
        train_bld_rw = rw.fit_transform(train_bld)
        
        # Get sample weights
        sample_weights = train_bld_rw.instance_weights
        
        # Train classifier with balanced weights and regularization
        clf_rw = LogisticRegression(C=args.C, max_iter=1000, random_state=args.random_state, 
                                  solver='liblinear')
        clf_rw.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Get predictions
        y_pred_rw = clf_rw.predict(X_test)
        y_prob_rw = clf_rw.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics_fair_rw['acc'].append(accuracy_score(y_test, y_pred_rw))
        metrics_fair_rw['roc'].append(roc_auc_score(y_test, y_prob_rw))
        metrics_fair_rw['confusion'].append(confusion_matrix(y_test, y_pred_rw))
        
        # Create BinaryLabelDataset for reweighing predictions
        pred_dataset_rw = test_bld.copy()
        pred_dataset_rw.labels = np.reshape(y_pred_rw, [-1, 1])
        
        # Calculate fairness metrics for Reweighing
        dataset_metric_rw = BinaryLabelDatasetMetric(
            pred_dataset_rw,
            unprivileged_groups=[{'protected': 0}], 
            privileged_groups=[{'protected': 1}]
        )
        metrics_fair_rw['di'].append(dataset_metric_rw.disparate_impact())
        
        # Equal Opportunity Difference
        classification_metric_rw = ClassificationMetric(
            test_bld, pred_dataset_rw, 
            unprivileged_groups=[{'protected': 0}], 
            privileged_groups=[{'protected': 1}]
        )
        metrics_fair_rw['eo'].append(classification_metric_rw.equal_opportunity_difference())
        
        # Add Average Odds Difference calculation
        avg_odds_diff_rw = classification_metric_rw.average_odds_difference()
        metrics_fair_rw['avg_odds'].append(avg_odds_diff_rw)
        
        # Print intermediate results for this fold
        print(f"Fold {fold} - Disparate Impact:")
        print(f"  Baseline: {baseline_metrics['di'][-1]:.3f}")
        print(f"  Prejudice Remover: {metrics_fair_pr['di'][-1]:.3f}")
        print(f"  Reweighing: {metrics_fair_rw['di'][-1]:.3f}")
        
        fold += 1

    # Save the results to CSV
    results_df = pd.DataFrame({
        'baseline_acc': [np.mean(baseline_metrics['acc'])],
        'baseline_roc': [np.mean(baseline_metrics['roc'])],
        'baseline_di': [np.mean(baseline_metrics['di'])],
        'pr_acc': [np.mean(metrics_fair_pr['acc'])],
        'pr_roc': [np.mean(metrics_fair_pr['roc'])],
        'pr_di': [np.mean(metrics_fair_pr['di'])],
        'pr_eo': [np.mean(metrics_fair_pr['eo'])],
        'pr_avg_odds': [np.mean(metrics_fair_pr['avg_odds'])],
        'rw_acc': [np.mean(metrics_fair_rw['acc'])],
        'rw_roc': [np.mean(metrics_fair_rw['roc'])],
        'rw_di': [np.mean(metrics_fair_rw['di'])],
        'rw_eo': [np.mean(metrics_fair_rw['eo'])],
        'rw_avg_odds': [np.mean(metrics_fair_rw['avg_odds'])]
    })
    
    # Use the data directory from dirs dictionary
    results_path = os.path.join(dirs['data'], 'experiment_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")
    
    # Generate plots if enabled
    if not args.no_plots:
        create_plots(baseline_metrics, metrics_fair_pr, metrics_fair_rw, dirs['plots'])
    
    return baseline_metrics, metrics_fair_pr, metrics_fair_rw

def create_plots(metrics_baseline, metrics_fair_pr, metrics_fair_rw, plot_dir):
    """Create visualizations comparing model performance and fairness metrics."""
    print("\nGenerating visualization plots...")
    
    # Set plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # ===== Accuracy Comparison =====
    plt.figure(figsize=(10, 6))
    x = np.arange(1, len(metrics_baseline['acc']) + 1)
    plt.plot(x, metrics_baseline['acc'], 'o--', color='#1f77b4', linewidth=2, label='Baseline')
    plt.plot(x, metrics_fair_pr['acc'], 'o-', color='#ff7f0e', linewidth=2, label='Prejudice Remover')
    plt.plot(x, metrics_fair_rw['acc'], 'o-.', color='#2ca02c', linewidth=2, label='Reweighing')
    plt.title('Accuracy Comparison across Methods', fontsize=14)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(x)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "accuracy_comparison.png"), dpi=300)
    plt.close()
    
    # ===== ROC AUC Comparison =====
    plt.figure(figsize=(10, 6))
    plt.plot(x, metrics_baseline['roc'], 'o--', color='#1f77b4', linewidth=2, label='Baseline')
    plt.plot(x, metrics_fair_pr['roc'], 'o-', color='#ff7f0e', linewidth=2, label='Prejudice Remover')
    plt.plot(x, metrics_fair_rw['roc'], 'o-.', color='#2ca02c', linewidth=2, label='Reweighing')
    plt.title('ROC AUC Comparison across Methods', fontsize=14)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('ROC AUC', fontsize=12)
    plt.xticks(x)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "roc_auc_comparison.png"), dpi=300)
    plt.close()
    
    # ===== Disparate Impact Comparison =====
    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.3, metrics_baseline['di'], width=0.2, color='#1f77b4', label='Baseline')
    plt.bar(x - 0.1, metrics_fair_pr['di'], width=0.2, color='#ff7f0e', label='Prejudice Remover')
    plt.bar(x + 0.1, metrics_fair_rw['di'], width=0.2, color='#2ca02c', label='Reweighing')
    plt.axhline(1.0, color='gray', linestyle='-', label='Ideal DI = 1')
    plt.axhline(DI_LOWER_BOUND, color='red', linestyle='--', label=f'DI < {DI_LOWER_BOUND} indicates bias')
    plt.axhline(DI_UPPER_BOUND, color='red', linestyle='--', label=f'DI > {DI_UPPER_BOUND} indicates bias')
    plt.title('Disparate Impact Comparison', fontsize=14)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Disparate Impact', fontsize=12)
    plt.xticks(x)
    
    # Calculate limit for y-axis to ensure all lines are visible
    max_di = max([max(metrics_baseline['di']), max(metrics_fair_pr['di']), max(metrics_fair_rw['di'])])
    plt.ylim(0, max(max_di, DI_UPPER_BOUND) * 1.2)
    
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "disparate_impact_comparison.png"), dpi=300)
    plt.close()
    
    # ===== Equal Opportunity Difference =====
    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2, metrics_fair_pr['eo'], width=0.4, color='#ff7f0e', label='Prejudice Remover')
    plt.bar(x + 0.2, metrics_fair_rw['eo'], width=0.4, color='#2ca02c', label='Reweighing')
    plt.axhline(0.0, color='gray', linestyle='--', label='Ideal EOD = 0')
    plt.title('Equal Opportunity Difference Comparison', fontsize=14)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Equal Opportunity Difference', fontsize=12)
    plt.xticks(x)
    max_abs_eo = max(
        abs(min(min(metrics_fair_pr['eo']), min(metrics_fair_rw['eo']))),
        abs(max(max(metrics_fair_pr['eo']), max(metrics_fair_rw['eo'])))
    )
    plt.ylim(-max_abs_eo * 1.2, max_abs_eo * 1.2)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "equal_opportunity_comparison.png"), dpi=300)
    plt.close()
    
    # ===== Performance vs. Fairness Trade-off =====
    plt.figure(figsize=(10, 8))
    
    # Calculate mean values for scatter plot
    baseline_acc_mean = np.mean(metrics_baseline['acc'])
    baseline_roc_mean = np.mean(metrics_baseline['roc'])
    baseline_di_mean = np.mean(metrics_baseline['di'])
    
    pr_acc_mean = np.mean(metrics_fair_pr['acc'])
    pr_roc_mean = np.mean(metrics_fair_pr['roc'])
    pr_di_mean = np.mean(metrics_fair_pr['di'])
    
    rw_acc_mean = np.mean(metrics_fair_rw['acc'])
    rw_roc_mean = np.mean(metrics_fair_rw['roc'])
    rw_di_mean = np.mean(metrics_fair_rw['di'])
    
    # Fairness vs Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.scatter([baseline_acc_mean], [baseline_di_mean], s=100, color='#1f77b4', label='Baseline')
    plt.scatter([pr_acc_mean], [pr_di_mean], s=100, color='#ff7f0e', label='Prejudice Remover')
    plt.scatter([rw_acc_mean], [rw_di_mean], s=100, color='#2ca02c', label='Reweighing')
    
    plt.axhline(1.0, color='gray', linestyle='-', alpha=0.7)
    plt.axhline(DI_LOWER_BOUND, color='red', linestyle='--', alpha=0.7)
    plt.axhline(DI_UPPER_BOUND, color='red', linestyle='--', alpha=0.7)
    plt.axhspan(DI_LOWER_BOUND, DI_UPPER_BOUND, color='green', alpha=0.1, label='Fair region')
    
    plt.title('Accuracy vs. Fairness Trade-off', fontsize=14)
    plt.xlabel('Accuracy', fontsize=12)
    plt.ylabel('Disparate Impact (0.8-1.25 is fair)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Fairness vs ROC AUC subplot
    plt.subplot(1, 2, 2)
    plt.scatter([baseline_roc_mean], [baseline_di_mean], s=100, color='#1f77b4', label='Baseline')
    plt.scatter([pr_roc_mean], [pr_di_mean], s=100, color='#ff7f0e', label='Prejudice Remover')
    plt.scatter([rw_roc_mean], [rw_di_mean], s=100, color='#2ca02c', label='Reweighing')
    
    plt.axhline(1.0, color='gray', linestyle='-', alpha=0.7)
    plt.axhline(DI_LOWER_BOUND, color='red', linestyle='--', alpha=0.7)
    plt.axhline(DI_UPPER_BOUND, color='red', linestyle='--', alpha=0.7)
    plt.axhspan(DI_LOWER_BOUND, DI_UPPER_BOUND, color='green', alpha=0.1, label='Fair region')
    
    plt.title('ROC AUC vs. Fairness Trade-off', fontsize=14)
    plt.xlabel('ROC AUC', fontsize=12)
    plt.ylabel('Disparate Impact (0.8-1.25 is fair)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "fairness_tradeoff.png"), dpi=300)
    plt.close()
    
    print(f"Saved all plots to {plot_dir}")

def evaluate_fairness(di_value):
    """Evaluate if a disparate impact value is within fair range."""
    return DI_LOWER_BOUND <= di_value <= DI_UPPER_BOUND

def print_summary(metrics_baseline, metrics_fair_pr, metrics_fair_rw):
    """Print summary statistics for each method."""
    print("\n===== EXPERIMENT SUMMARY =====")
    
    # Calculate mean metrics
    baseline_acc = np.mean(metrics_baseline['acc'])
    baseline_roc = np.mean(metrics_baseline['roc'])
    baseline_di = np.mean(metrics_baseline['di'])
    
    pr_acc = np.mean(metrics_fair_pr['acc'])
    pr_roc = np.mean(metrics_fair_pr['roc'])
    pr_di = np.mean(metrics_fair_pr['di'])
    pr_eo = np.mean(metrics_fair_pr['eo'])
    
    rw_acc = np.mean(metrics_fair_rw['acc'])
    rw_roc = np.mean(metrics_fair_rw['roc'])
    rw_di = np.mean(metrics_fair_rw['di'])
    rw_eo = np.mean(metrics_fair_rw['eo'])
    
    # Print accuracy metrics
    print("\nPerformance Metrics:")
    print(f"  Baseline:           Acc = {baseline_acc:.3f}, ROC AUC = {baseline_roc:.3f}")
    print(f"  Prejudice Remover:  Acc = {pr_acc:.3f}, ROC AUC = {pr_roc:.3f}")
    print(f"  Reweighing:         Acc = {rw_acc:.3f}, ROC AUC = {rw_roc:.3f}")
    
    # Print fairness metrics
    print("\nFairness Metrics:")
    print(f"  Baseline:")
    print(f"    - Disparate Impact = {baseline_di:.3f} ({evaluate_fairness(baseline_di)})")
    print(f"  Prejudice Remover:")
    print(f"    - Disparate Impact = {pr_di:.3f} ({evaluate_fairness(pr_di)})")
    print(f"    - Equal Opportunity Diff = {pr_eo:.3f}")
    print(f"  Reweighing:")
    print(f"    - Disparate Impact = {rw_di:.3f} ({evaluate_fairness(rw_di)})")
    print(f"    - Equal Opportunity Diff = {rw_eo:.3f}")
    
    # Compare methods
    print("\nMethod Comparison:")
    
    # Performance comparison
    best_acc = max(baseline_acc, pr_acc, rw_acc)
    best_roc = max(baseline_roc, pr_roc, rw_roc)
    
    if best_acc == baseline_acc:
        acc_winner = "Baseline"
    elif best_acc == pr_acc:
        acc_winner = "Prejudice Remover"
    else:
        acc_winner = "Reweighing"
        
    if best_roc == baseline_roc:
        roc_winner = "Baseline"
    elif best_roc == pr_roc:
        roc_winner = "Prejudice Remover"
    else:
        roc_winner = "Reweighing"
    
    print(f"  Best Accuracy: {acc_winner} ({best_acc:.3f})")
    print(f"  Best ROC AUC: {roc_winner} ({best_roc:.3f})")
    
    # Fairness comparison
    di_diff_from_one = [abs(baseline_di - 1), abs(pr_di - 1), abs(rw_di - 1)]
    best_di_idx = np.argmin(di_diff_from_one)
    best_di_methods = ["Baseline", "Prejudice Remover", "Reweighing"]
    best_di_values = [baseline_di, pr_di, rw_di]
    
    best_eo_idx = np.argmin([float('inf'), abs(pr_eo), abs(rw_eo)])  # Baseline has no EO
    best_eo_methods = ["N/A", "Prejudice Remover", "Reweighing"]
    best_eo_values = [float('inf'), abs(pr_eo), abs(rw_eo)]
    
    print(f"  Best Disparate Impact: {best_di_methods[best_di_idx]} ({best_di_values[best_di_idx]:.3f})")
    print(f"  Best Equal Opportunity: {best_eo_methods[best_eo_idx]} ({best_eo_values[best_eo_idx]:.3f})")
    
    # Overall recommendation
    fairness_winners = []
    if evaluate_fairness(baseline_di):
        fairness_winners.append("Baseline")
    if evaluate_fairness(pr_di) and abs(pr_eo) < 0.1:  # EO less than 0.1 is considered acceptable
        fairness_winners.append("Prejudice Remover")
    if evaluate_fairness(rw_di) and abs(rw_eo) < 0.1:
        fairness_winners.append("Reweighing")
    
    print("\nOverall Recommendation:")
    if not fairness_winners:
        print("  None of the methods achieved acceptable fairness levels.")
        if min(di_diff_from_one) < 0.5:  # If any method is somewhat close to fairness
            best_method_idx = np.argmin(di_diff_from_one)
            print(f"  The best option is {best_di_methods[best_method_idx]} but it needs further improvements.")
    elif len(fairness_winners) == 1:
        print(f"  {fairness_winners[0]} achieves acceptable fairness.")
    else:
        # If multiple methods are fair, choose the one with best accuracy
        fair_accs = []
        for method in fairness_winners:
            if method == "Baseline":
                fair_accs.append(baseline_acc)
            elif method == "Prejudice Remover":
                fair_accs.append(pr_acc)
            else:  # Reweighing
                fair_accs.append(rw_acc)
        
        best_fair_idx = np.argmax(fair_accs)
        best_fair_method = fairness_winners[best_fair_idx]
        print(f"  {best_fair_method} provides the best balance of fairness and accuracy.")
    
    print("\n============================")

def main():
    """Main function to run the fair speech emotion recognition pipeline."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create directories for outputs
    dirs = setup_directories(args.output_dir)
    
    try:
        # Extract features from audio files
        df = extract_features_and_labels(args.data_dir, args.sample_rate, args.n_mfcc)
        
        # Induce bias for experiment
        biased_df = balanced_bias_induction(df, args.bias_drop, args.balance_factor, args.random_state)
        
        # Prepare data for models
        X = biased_df.drop(['label', 'gender'], axis=1).values
        y = biased_df['label'].values
        gender = biased_df['gender'].values
        
        # Run fairness experiment - pass the directories dictionary instead of just output_dir
        metrics_baseline, metrics_fair_pr, metrics_fair_rw = run_fairness_experiment(X, y, gender, args, dirs)
        
        # Print summary of results
        print_summary(metrics_baseline, metrics_fair_pr, metrics_fair_rw)
        
        # Plot confusion matrices if plots are enabled
        if not args.no_plots:
            plot_confusion_matrices(metrics_baseline, metrics_fair_pr, metrics_fair_rw, dirs['plots'])
            
        # Run statistical significance tests
        statistical_significance_test(metrics_baseline, metrics_fair_pr, metrics_fair_rw)
        
    except Exception as e:
        print(f"Error running experiment: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
