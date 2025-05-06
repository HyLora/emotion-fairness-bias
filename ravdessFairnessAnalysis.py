#!/usr/bin/env python3
# Fair Speech Emotion Recognition
# Author: [Your Name]
# Date: May 2025
# Course: Logics for AI - Logic of Computation and Information 2024-2025

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
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.inprocessing import PrejudiceRemover
from aif360.algorithms.preprocessing import Reweighing
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

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
    parser.add_argument('--bias_drop', type=float, default=0.7,
                        help='Fraction of male-happy samples to drop for bias induction')
    parser.add_argument('--eta', type=float, default=0.5,
                        help='Eta parameter for PrejudiceRemover')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of cross-validation splits')
    parser.add_argument('--no_plots', action='store_true',
                        help='Disable plot generation')
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
                    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
                    features.append(np.mean(mfcc.T, axis=0))
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

def induce_bias(df, drop_fraction, random_state):
    """Induce bias by removing a portion of male-happy samples."""
    print(f"Inducing bias by dropping {drop_fraction*100:.1f}% of male-happy samples...")
    
    biased_df = df.copy()
    mask = (biased_df['gender'] == 1) & (biased_df['label'] == 1)
    drop_count = int(drop_fraction * mask.sum())
    
    original_class_dist = biased_df['label'].value_counts().to_dict()
    original_gender_dist = pd.crosstab(biased_df['gender'], biased_df['label'])
    
    drop_indices = biased_df[mask].sample(n=drop_count, random_state=random_state).index
    biased_df = biased_df.drop(index=drop_indices)
    
    new_gender_dist = pd.crosstab(biased_df['gender'], biased_df['label'])
    
    print("Original distribution by gender and emotion:")
    print(original_gender_dist)
    print("\nBiased distribution by gender and emotion:")
    print(new_gender_dist)
    
    return biased_df

def to_binary_label_dataset(X, y, protected):
    """Convert data to AIF360 BinaryLabelDataset format."""
    df = pd.DataFrame(X)
    df['label'] = y
    df['protected'] = protected
    return BinaryLabelDataset(df=df, label_names=['label'], protected_attribute_names=['protected'])

def run_fairness_experiment(X, y, gender, args, dirs):
    """Run the fairness experiment with cross-validation."""
    print("\nStarting fairness experiment with cross-validation...")
    
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    
    # Initialize metrics dictionaries
    metrics_baseline = {'acc': [], 'roc': [], 'confusion': []}
    metrics_fair_pr = {'acc': [], 'roc': [], 'di': [], 'eo': [], 'confusion': []}  # PR = Prejudice Remover
    metrics_fair_rw = {'acc': [], 'roc': [], 'di': [], 'eo': [], 'confusion': []}  # RW = Reweighing
    
    fold = 1
    for train_idx, test_idx in skf.split(X, y):
        print(f"\nProcessing fold {fold}/{args.n_splits}...")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        g_train, g_test = gender[train_idx], gender[test_idx]
        
        # Create datasets for AIF360
        train_bld = to_binary_label_dataset(X_train, y_train, g_train)
        test_bld = to_binary_label_dataset(X_test, y_test, g_test)
        
        # ===== Baseline Model =====
        clf = LogisticRegression(max_iter=1000, random_state=args.random_state)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        metrics_baseline['acc'].append(accuracy_score(y_test, y_pred))
        metrics_baseline['roc'].append(roc_auc_score(y_test, y_prob))
        metrics_baseline['confusion'].append(confusion_matrix(y_test, y_pred))
        
        # ===== Fair Model 1: Prejudice Remover =====
        pr = PrejudiceRemover(sensitive_attr='protected', eta=args.eta)
        pr.fit(train_bld)
        fair_pred_pr = pr.predict(test_bld)
        
        fair_labels_pr = fair_pred_pr.labels.ravel()
        metrics_fair_pr['acc'].append(accuracy_score(y_test, fair_labels_pr))
        try:
            metrics_fair_pr['roc'].append(roc_auc_score(y_test, fair_labels_pr))
        except:
            # Handle case where all predictions are of one class
            metrics_fair_pr['roc'].append(0.5)
        
        metrics_fair_pr['confusion'].append(confusion_matrix(y_test, fair_labels_pr))
        
        # Fairness metrics for Prejudice Remover
        classification_metric = ClassificationMetric(
            test_bld, fair_pred_pr, 
            unprivileged_groups=[{'protected': 0}], 
            privileged_groups=[{'protected': 1}]
        )
        
        # Disparate Impact (ratio of positive outcomes between unprivileged and privileged groups)
        dataset_metric_pr = BinaryLabelDatasetMetric(
            fair_pred_pr,
            unprivileged_groups=[{'protected': 0}], 
            privileged_groups=[{'protected': 1}]
        )
        di = dataset_metric_pr.disparate_impact()
        metrics_fair_pr['di'].append(di)
        
        # Equal Opportunity Difference (difference in TPR between privileged and unprivileged groups)
        eo_diff = classification_metric.equal_opportunity_difference()
        metrics_fair_pr['eo'].append(eo_diff)
        
        # ===== Fair Model 2: Reweighing =====
        rw = Reweighing(unprivileged_groups=[{'protected': 0}], privileged_groups=[{'protected': 1}])
        train_bld_rw = rw.fit_transform(train_bld)
        
        sample_weights = train_bld_rw.instance_weights
        clf_rw = LogisticRegression(max_iter=1000, random_state=args.random_state)
        clf_rw.fit(X_train, y_train, sample_weight=sample_weights)
        
        y_pred_rw = clf_rw.predict(X_test)
        y_prob_rw = clf_rw.predict_proba(X_test)[:, 1]
        
        metrics_fair_rw['acc'].append(accuracy_score(y_test, y_pred_rw))
        metrics_fair_rw['roc'].append(roc_auc_score(y_test, y_prob_rw))
        metrics_fair_rw['confusion'].append(confusion_matrix(y_test, y_pred_rw))
        
        # Create BinaryLabelDataset for reweighing predictions for fairness metrics
        pred_dataset_rw = test_bld.copy()
        pred_dataset_rw.labels = np.reshape(y_pred_rw, [-1, 1])
        
        # Fairness metrics for Reweighing
        classification_metric_rw = ClassificationMetric(
            test_bld, pred_dataset_rw, 
            unprivileged_groups=[{'protected': 0}], 
            privileged_groups=[{'protected': 1}]
        )
        
        # Calculate disparate impact for reweighing
        dataset_metric_rw = BinaryLabelDatasetMetric(
            pred_dataset_rw,
            unprivileged_groups=[{'protected': 0}], 
            privileged_groups=[{'protected': 1}]
        )
        metrics_fair_rw['di'].append(dataset_metric_rw.disparate_impact())
        
        # Equal Opportunity Difference
        metrics_fair_rw['eo'].append(classification_metric_rw.equal_opportunity_difference())
        
        fold += 1

    # Save the results to CSV
    results = {
        'baseline_acc': np.mean(metrics_baseline['acc']),
        'baseline_roc': np.mean(metrics_baseline['roc']),
        'pr_acc': np.mean(metrics_fair_pr['acc']),
        'pr_roc': np.mean(metrics_fair_pr['roc']),
        'pr_di': np.mean(metrics_fair_pr['di']),
        'pr_eo': np.mean(metrics_fair_pr['eo']),
        'rw_acc': np.mean(metrics_fair_rw['acc']),
        'rw_roc': np.mean(metrics_fair_rw['roc']),
        'rw_di': np.mean(metrics_fair_rw['di']),
        'rw_eo': np.mean(metrics_fair_rw['eo'])
    }
    
    results_df = pd.DataFrame([results])
    results_path = os.path.join(dirs['data'], 'experiment_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Saved results to {results_path}")
    
    # Generate plots if enabled
    if not args.no_plots:
        create_plots(metrics_baseline, metrics_fair_pr, metrics_fair_rw, dirs['plots'])
    
    return metrics_baseline, metrics_fair_pr, metrics_fair_rw

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
    plt.bar(x - 0.2, metrics_fair_pr['di'], width=0.4, color='#ff7f0e', label='Prejudice Remover')
    plt.bar(x + 0.2, metrics_fair_rw['di'], width=0.4, color='#2ca02c', label='Reweighing')
    plt.axhline(1.0, color='gray', linestyle='--', label='Ideal DI = 1')
    plt.axhline(0.8, color='red', linestyle='--', label='DI < 0.8 indicates bias')
    plt.title('Disparate Impact Comparison', fontsize=14)
    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Disparate Impact', fontsize=12)
    plt.xticks(x)
    plt.ylim(0, max(max(metrics_fair_pr['di']), max(metrics_fair_rw['di'])) * 1.2)
    plt.grid(True)
    plt.legend(fontsize=12)
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
    
    pr_acc_mean = np.mean(metrics_fair_pr['acc'])
    pr_roc_mean = np.mean(metrics_fair_pr['roc'])
    pr_di_mean = np.mean(metrics_fair_pr['di'])
    
    rw_acc_mean = np.mean(metrics_fair_rw['acc'])
    rw_roc_mean = np.mean(metrics_fair_rw['roc'])
    rw_di_mean = np.mean(metrics_fair_rw['di'])
    
    # Fairness vs Accuracy subplot
    plt.subplot(1, 2, 1)
    plt.scatter([baseline_acc_mean], [0.5], s=100, color='#1f77b4', label='Baseline')
    plt.scatter([pr_acc_mean], [pr_di_mean], s=100, color='#ff7f0e', label='Prejudice Remover')
    plt.scatter([rw_acc_mean], [rw_di_mean], s=100, color='#2ca02c', label='Reweighing')
    
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(0.8, color='red', linestyle='--', alpha=0.7)
    
    plt.title('Accuracy vs. Fairness Trade-off', fontsize=14)
    plt.xlabel('Accuracy', fontsize=12)
    plt.ylabel('Disparate Impact (closer to 1 is better)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    # Fairness vs ROC AUC subplot
    plt.subplot(1, 2, 2)
    plt.scatter([baseline_roc_mean], [0.5], s=100, color='#1f77b4', label='Baseline')
    plt.scatter([pr_roc_mean], [pr_di_mean], s=100, color='#ff7f0e', label='Prejudice Remover')
    plt.scatter([rw_roc_mean], [rw_di_mean], s=100, color='#2ca02c', label='Reweighing')
    
    plt.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    plt.axhline(0.8, color='red', linestyle='--', alpha=0.7)
    
    plt.title('ROC AUC vs. Fairness Trade-off', fontsize=14)
    plt.xlabel('ROC AUC', fontsize=12)
    plt.ylabel('Disparate Impact (closer to 1 is better)', fontsize=12)
    plt.grid(True)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "fairness_tradeoff.png"), dpi=300)
    plt.close()
    
    print(f"Saved all plots to {plot_dir}")

def print_summary(metrics_baseline, metrics_fair_pr, metrics_fair_rw):
    """Print summary statistics of the experiment."""
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    
    def print_model_stats(name, acc, roc, di=None, eo=None):
        print(f"\n{name} Results:")
        print(f"  Accuracy: {np.mean(acc):.3f} ± {np.std(acc):.3f}")
        print(f"  ROC AUC : {np.mean(roc):.3f} ± {np.std(roc):.3f}")
        if di is not None:
            print(f"  Disparate Impact: {np.mean(di):.3f} ± {np.std(di):.3f}")
        if eo is not None:
            print(f"  Equal Opportunity Diff: {np.mean(eo):.3f} ± {np.std(eo):.3f}")
    
    # Print statistics for each model
    print_model_stats("Baseline", 
                     metrics_baseline['acc'], 
                     metrics_baseline['roc'])
    
    print_model_stats("Prejudice Remover", 
                     metrics_fair_pr['acc'], 
                     metrics_fair_pr['roc'],
                     metrics_fair_pr['di'],
                     metrics_fair_pr['eo'])
    
    print_model_stats("Reweighing", 
                     metrics_fair_rw['acc'], 
                     metrics_fair_rw['roc'],
                     metrics_fair_rw['di'],
                     metrics_fair_rw['eo'])
    
    # Analyze fairness-accuracy tradeoff
    acc_diff_pr = np.mean(metrics_baseline['acc']) - np.mean(metrics_fair_pr['acc'])
    acc_diff_rw = np.mean(metrics_baseline['acc']) - np.mean(metrics_fair_rw['acc'])
    
    print("\nFairness-Accuracy Tradeoff:")
    print(f"  Prejudice Remover: {acc_diff_pr:.3f} accuracy loss for {np.mean(metrics_fair_pr['di']):.3f} disparate impact")
    print(f"  Reweighing: {acc_diff_rw:.3f} accuracy loss for {np.mean(metrics_fair_rw['di']):.3f} disparate impact")
    
    print("\nConclusion:")
    
    # Determine which fair model performed better
    if np.mean(metrics_fair_pr['di']) > np.mean(metrics_fair_rw['di']) and acc_diff_pr < acc_diff_rw:
        best_model = "Prejudice Remover"
    elif np.mean(metrics_fair_rw['di']) > np.mean(metrics_fair_pr['di']) and acc_diff_rw < acc_diff_pr:
        best_model = "Reweighing"
    else:
        # Compare fairness-accuracy trade-off ratio
        pr_ratio = np.mean(metrics_fair_pr['di']) / (1 + acc_diff_pr)  # Higher is better
        rw_ratio = np.mean(metrics_fair_rw['di']) / (1 + acc_diff_rw)
        best_model = "Prejudice Remover" if pr_ratio > rw_ratio else "Reweighing"
    
    print(f"  The {best_model} model provides the best fairness-accuracy trade-off for this task.")
    print("="*50)

def main():
    """Main function to run the experiment."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Create output directories
    dirs = setup_directories(args.output_dir)
    
    try:
        # Step 1: Extract features from audio files
        df = extract_features_and_labels(args.data_dir, args.sample_rate, args.n_mfcc)
        
        # Step 2: Induce bias in the dataset
        biased_df = induce_bias(df, args.bias_drop, args.random_state)
        
        # Step 3: Prepare data for modeling
        X = biased_df.drop(columns=['label', 'gender']).values
        y = biased_df['label'].values
        gender = biased_df['gender'].values
        
        # Step 4: Run fairness experiment
        metrics_baseline, metrics_fair_pr, metrics_fair_rw = run_fairness_experiment(X, y, gender, args, dirs)
        
        # Step 5: Print summary statistics
        print_summary(metrics_baseline, metrics_fair_pr, metrics_fair_rw)
        
        print(f"\nExperiment completed successfully! Results saved in {args.output_dir}")
        
    except Exception as e:
        print(f"Error during experiment execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
