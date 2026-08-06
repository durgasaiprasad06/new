import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from sklearn.base import clone

def get_base_models():
    """
    Returns a dictionary of classical ML base models configured with balanced class weights.
    """
    return {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5, class_weight='balanced'),
        'Support Vector Machine': SVC(probability=True, random_state=42, class_weight='balanced'),
        'Gaussian Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
    }

def create_model_pipelines(preprocessor, models_dict):
    """
    Wraps each classifier with the preprocessing pipeline to guarantee zero data leakage.
    """
    pipelines = {}
    for name, model in models_dict.items():
        pipelines[name] = Pipeline(steps=[
            ('preprocessor', clone(preprocessor)),
            ('classifier', clone(model))
        ])
    return pipelines

def create_ensemble_pipelines(preprocessor, base_models):
    """
    Creates classical ensemble pipelines (Soft Voting and Stacking Classifier).
    """
    voting_estimators = [
        ('lr', clone(base_models['Logistic Regression'])),
        ('svm', clone(base_models['Support Vector Machine'])),
        ('gnb', clone(base_models['Gaussian Naive Bayes'])),
        ('rf', clone(base_models['Random Forest']))
    ]
    
    voting_clf = VotingClassifier(estimators=voting_estimators, voting='soft')
    
    stacking_estimators = [
        ('lr', clone(base_models['Logistic Regression'])),
        ('knn', clone(base_models['K-Nearest Neighbors'])),
        ('dt', clone(base_models['Decision Tree'])),
        ('svm', clone(base_models['Support Vector Machine'])),
        ('gnb', clone(base_models['Gaussian Naive Bayes'])),
        ('rf', clone(base_models['Random Forest']))
    ]
    
    stacking_clf = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    
    ensembles = {
        'Voting Classifier (Soft)': Pipeline(steps=[('preprocessor', clone(preprocessor)), ('classifier', voting_clf)]),
        'Stacking Classifier': Pipeline(steps=[('preprocessor', clone(preprocessor)), ('classifier', stacking_clf)])
    }
    
    return ensembles

def tune_random_forest_pipeline(preprocessor, X_train, y_train):
    """
    Tunes Random Forest within a complete Scikit-Learn Pipeline using Stratified GridSearchCV.
    """
    print("\nStarting Hyperparameter Tuning for Random Forest Pipeline...")
    
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    
    param_grid = {
        'rf__n_estimators': [50, 100, 200],
        'rf__max_depth': [4, 6, 10, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4]
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=rf_pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    print(f"Best Tuning ROC-AUC Score: {grid_search.best_score_:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_

def create_pca_pipelines(preprocessor, models_dict, n_components=0.95):
    """
    Creates complete ML pipelines that incorporate PCA dimensionality reduction.
    """
    pca_pipelines = {}
    for name, model in models_dict.items():
        pca_pipelines[name] = Pipeline(steps=[
            ('preprocessor', clone(preprocessor)),
            ('pca', PCA(n_components=n_components, random_state=42)),
            ('classifier', clone(model))
        ])
    return pca_pipelines
