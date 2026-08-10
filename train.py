import os
import json
import joblib
import warnings
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
from src.data_preprocessing import load_and_preprocess_data, get_preprocessor, engineer_features
from src.model_training import (
    get_base_models, create_model_pipelines, create_ensemble_pipelines,
    tune_random_forest_pipeline, create_pca_pipelines
)
from src.evaluation import (
    evaluate_pipelines, run_stratified_cv, plot_model_comparison_bar,
    plot_pipeline_feature_importance
)

warnings.filterwarnings('ignore')

def main():
    dataset_path = "dataset/loan_data.csv"
    output_dir = "outputs"
    models_dir = "models"
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 60)
    print("LOAN APPROVAL PREDICTION SYSTEM - ENHANCED ML PIPELINE")
    print("=" * 60)

    # 1. Data Preprocessing & Feature Engineering
    print("\n[Step 1/8] Loading Data & Performing Domain Feature Engineering...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_path)
    preprocessor, num_features, cat_features = get_preprocessor()

    # 2. Build Base Models & Pipelines
    print("\n[Step 2/8] Constructing Base Model Pipelines...")
    base_models = get_base_models()
    base_pipelines = create_model_pipelines(preprocessor, base_models)

    # 3. Build Advanced Classical Ensembles (Voting & Stacking)
    print("\n[Step 3/8] Building Classical Ensemble Pipelines (Voting & Stacking)...")
    ensemble_pipelines = create_ensemble_pipelines(preprocessor, base_models)
    
    all_pipelines = {**base_pipelines, **ensemble_pipelines}

    # Train all pipelines
    print("\n[Step 4/8] Training All Base Models and Ensembles...")
    fitted_pipelines = {}
    for name, pipe in all_pipelines.items():
        print(f"  -> Fitting {name}...")
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

    # 4. Evaluation & Multi-Model Visualizations
    print("\n[Step 5/8] Evaluating Models & Generating ROC/PR Curves...")
    test_results_df = evaluate_pipelines(fitted_pipelines, X_test, y_test, output_dir=output_dir)
    print("\n--- Test Set Benchmark Performance ---")
    print(test_results_df.sort_values(by='ROC-AUC', ascending=False).to_string(index=False))
    
    test_results_df.to_csv(os.path.join(output_dir, 'model_comparison_test.csv'), index=False)
    plot_model_comparison_bar(test_results_df, output_dir=output_dir)

    # 5. Stratified Cross-Validation
    print("\n[Step 6/8] Running 5-Fold Stratified Cross-Validation on Full Dataset...")
    full_df = pd.read_csv(dataset_path).drop_duplicates()
    full_df['Loan_Status'] = full_df['Loan_Status'].map({'Y': 1, 'N': 0}).dropna().astype(int)
    full_df['Dependents'] = full_df['Dependents'].astype(str).str.replace('+', '', regex=False)
    full_df_engineered = engineer_features(full_df)
    
    X_full = full_df_engineered.drop(columns=['Loan_Status'])
    y_full = full_df_engineered['Loan_Status']
    
    cv_summary_df = run_stratified_cv(all_pipelines, X_full, y_full, n_splits=5)
    cv_summary_df.to_csv(os.path.join(output_dir, 'model_comparison_cv.csv'), index=False)

    # 6. PCA Dimensionality Reduction Benchmark
    print("\n[Step 7/8] Benchmarking with PCA Dimensionality Reduction...")
    pca_pipelines = create_pca_pipelines(preprocessor, base_models, n_components=0.95)
    for name, pipe in pca_pipelines.items():
        pipe.fit(X_train, y_train)
    pca_results_df = evaluate_pipelines(pca_pipelines, X_test, y_test, output_dir=os.path.join(output_dir, "pca"))
    pca_results_df.to_csv(os.path.join(output_dir, 'pca_model_comparison.csv'), index=False)

    # 7. Hyperparameter Tuning on Random Forest Pipeline
    print("\n[Step 8/8] Hyperparameter Tuning for Random Forest Pipeline...")
    best_tuned_rf = tune_random_forest_pipeline(preprocessor, X_train, y_train)
    tuned_eval_df = evaluate_pipelines({'Tuned Random Forest Pipeline': best_tuned_rf}, X_test, y_test, output_dir=output_dir)
    print("\nTuned Random Forest Test Results:\n", tuned_eval_df.to_string(index=False))

    # Feature Importance Visualization
    print("\nGenerating Top Feature Importance Plot...")
    plot_pipeline_feature_importance(best_tuned_rf, output_dir=output_dir)

    # 8. Selection & Saving of Best Model Artifact
    # We compare Tuned RF, Stacking, and Voting to pick the highest ROC-AUC model
    candidate_models = {
        'Tuned Random Forest': best_tuned_rf,
        'Voting Classifier (Soft)': fitted_pipelines['Voting Classifier (Soft)'],
        'Stacking Classifier': fitted_pipelines['Stacking Classifier'],
        'Logistic Regression': fitted_pipelines['Logistic Regression']
    }
    
    best_name = None
    best_score = -1
    best_pipeline = None
    
    for name, model in candidate_models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        score = roc_auc_score(y_test, y_prob)
        if score > best_score:
            best_score = score
            best_name = name
            best_pipeline = model
            
    print(f"\n============================================================")
    print(f"*** CHAMPION MODEL SELECTED: {best_name} (ROC-AUC: {best_score:.4f}) ***")
    print(f"============================================================")

    # Save champion pipeline
    model_save_path = os.path.join(models_dir, 'best_model.joblib')
    joblib.dump(best_pipeline, model_save_path)
    print(f"Saved complete end-to-end pipeline model to: {model_save_path}")
    
    # Save metadata
    metadata = {
        'model_name': best_name,
        'roc_auc': float(best_score),
        'accuracy': float(accuracy_score(y_test, best_pipeline.predict(X_test))),
        'numeric_features': num_features,
        'categorical_features': cat_features
    }
    with open(os.path.join(models_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print("Training pipeline finished successfully!\n")

if __name__ == "__main__":
    main()
