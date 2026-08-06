import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate

# Set visualization aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'

def evaluate_pipelines(pipelines, X_test, y_test, output_dir="outputs"):
    """
    Evaluates trained pipelines across comprehensive metrics and saves publication-quality visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing (AUC = 0.50)')
    
    pr_fig, pr_ax = plt.subplots(figsize=(10, 8))
    
    for name, pipeline in pipelines.items():
        y_pred = pipeline.predict(X_test)
        
        # Probabilities for ROC-AUC / PR curves
        if hasattr(pipeline, "predict_proba"):
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
            avg_pr = average_precision_score(y_test, y_proba)
            
            # ROC Curve plotting
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.3f})")
            
            # PR Curve plotting
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            pr_ax.plot(rec, prec, lw=2, label=f"{name} (AP = {avg_pr:.3f})")
        else:
            roc_auc = np.nan
            avg_pr = np.nan
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Rejected (0)', 'Approved (1)'],
                    yticklabels=['Rejected (0)', 'Approved (1)'])
        plt.title(f'Confusion Matrix\n{name}', fontsize=12, fontweight='bold')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        plt.savefig(os.path.join(output_dir, f'cm_{safe_name}.png'), dpi=300)
        plt.close()
        
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'ROC-AUC': roc_auc,
            'PR-AUC': avg_pr
        })
        
    # Finalize and save multi-model ROC Curve
    plt.title('Multi-Model ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
    plt.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves_comparison.png'), dpi=300)
    plt.close()
    
    # Finalize and save PR Curve
    pr_ax.set_title('Multi-Model Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
    pr_ax.set_xlabel('Recall', fontsize=12)
    pr_ax.set_ylabel('Precision', fontsize=12)
    pr_ax.legend(loc='lower left', frameon=True)
    pr_fig.tight_layout()
    pr_fig.savefig(os.path.join(output_dir, 'precision_recall_comparison.png'), dpi=300)
    plt.close(pr_fig)
    
    results_df = pd.DataFrame(results)
    return results_df

def run_stratified_cv(pipelines, X, y, n_splits=5):
    """
    Evaluates models using 5-Fold Stratified Cross Validation across multiple metrics.
    """
    print("\n========== Running 5-Fold Stratified Cross Validation ==========")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_records = []
    
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    for name, pipeline in pipelines.items():
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        cv_records.append({
            'Model': name,
            'CV Accuracy Mean': scores['test_accuracy'].mean(),
            'CV Accuracy Std': scores['test_accuracy'].std(),
            'CV F1 Mean': scores['test_f1'].mean(),
            'CV F1 Std': scores['test_f1'].std(),
            'CV ROC-AUC Mean': scores['test_roc_auc'].mean(),
            'CV ROC-AUC Std': scores['test_roc_auc'].std()
        })
        print(f"{name:<28} | Acc: {scores['test_accuracy'].mean():.4f} (+/- {scores['test_accuracy'].std():.4f}) | ROC-AUC: {scores['test_roc_auc'].mean():.4f}")
        
    return pd.DataFrame(cv_records)

def plot_model_comparison_bar(results_df, output_dir="outputs"):
    """
    Plots multi-metric performance comparison bar charts.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_to_plot = ['Accuracy', 'F1 Score', 'ROC-AUC']
    df_melted = results_df.melt(id_vars=['Model'], value_vars=metrics_to_plot, var_name='Metric', value_name='Score')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='Score', hue='Metric', data=df_melted, palette='mako')
    plt.title('Comprehensive Model Benchmark Comparison', fontsize=14, fontweight='bold')
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=25, ha='right')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_benchmark.png'), dpi=300)
    plt.close()

def plot_pipeline_feature_importance(pipeline, output_dir="outputs"):
    """
    Extracts and visualizes feature importances from a fitted Random Forest pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if classifier has feature importances
    classifier = pipeline.named_steps.get('rf') or pipeline.named_steps.get('classifier')
    if not hasattr(classifier, 'feature_importances_'):
        print("Classifier does not have feature_importances_ attribute.")
        return
        
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Retrieve feature names from ColumnTransformer
    num_features = preprocessor.transformers_[0][2]
    cat_encoder = preprocessor.transformers_[1][1].named_steps['onehot']
    cat_features_raw = preprocessor.transformers_[1][2]
    
    try:
        cat_features_encoded = list(cat_encoder.get_feature_names_out(cat_features_raw))
    except Exception:
        cat_features_encoded = [f"cat_{i}" for i in range(len(cat_encoder.categories_))]
        
    all_feature_names = list(num_features) + cat_features_encoded
    importances = classifier.feature_importances_
    
    # Align length if necessary
    if len(all_feature_names) == len(importances):
        feat_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(12, 7))
        sns.barplot(x='Importance', y='Feature', data=feat_df.head(15), palette='viridis')
        plt.title('Top 15 Most Important Features for Loan Approval', fontsize=14, fontweight='bold')
        plt.xlabel('Gini Importance Score')
        plt.ylabel('Engineered / Original Features')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_top15.png'), dpi=300)
        plt.close()
        print("Top 15 feature importance plot saved successfully.")
