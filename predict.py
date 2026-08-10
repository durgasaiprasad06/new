import os
import joblib
import pandas as pd
import numpy as np
import warnings
from src.data_preprocessing import engineer_features

warnings.filterwarnings('ignore')

def load_champion_model(model_path="models/best_model.joblib"):
    """
    Loads the trained end-to-end pipeline model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Run 'python train.py' first.")
    return joblib.load(model_path)

def analyze_risk_factors(input_data):
    """
    Identifies key positive and negative financial indicators for explainability.
    """
    risk_factors = []
    positive_factors = []
    
    # Credit History analysis
    credit_hist = float(input_data.get('Credit_History', 1.0))
    if credit_hist == 0:
        risk_factors.append("Poor Credit History (Significant risk factor)")
    else:
        positive_factors.append("Positive Credit History record")
        
    # Income to EMI ratio
    app_inc = float(input_data.get('ApplicantIncome', 0))
    coapp_inc = float(input_data.get('CoapplicantIncome', 0))
    loan_amt = float(input_data.get('LoanAmount', 100))
    term = float(input_data.get('Loan_Amount_Term', 360))
    
    total_income = app_inc + coapp_inc
    emi = (loan_amt * 1000) / (term if term > 0 else 360)
    monthly_income = total_income / 12.0
    
    if monthly_income <= 0 or emi > (monthly_income * 0.5):
        risk_factors.append(f"High Debt Burden (EMI of ${emi:.0f}/mo exceeds 50% of monthly income ${monthly_income:.0f}/mo)")
    else:
        positive_factors.append(f"Healthy Disposable Income (${monthly_income - emi:.0f}/mo after EMI)")
        
    if total_income < 2500:
        risk_factors.append("Low household total income")
    else:
        positive_factors.append("Adequate household income profile")
        
    return risk_factors, positive_factors

def predict_loan(input_dict, pipeline=None):
    """
    Takes a dictionary of applicant features and returns the prediction, probability,
    and explanatory financial indicators.
    """
    if pipeline is None:
        pipeline = load_champion_model()
        
    df_raw = pd.DataFrame([input_dict])
    
    # Format dependents consistently
    if 'Dependents' in df_raw.columns:
        df_raw['Dependents'] = df_raw['Dependents'].astype(str).str.replace('+', '', regex=False)
        
    # Apply feature engineering
    df_features = engineer_features(df_raw)
    
    # Generate inference
    prediction = pipeline.predict(df_features)[0]
    probabilities = pipeline.predict_proba(df_features)[0]
    
    # Class 1 is 'Approved', Class 0 is 'Rejected'
    prob_approved = probabilities[1]
    status = "Approved" if prediction == 1 else "Rejected"
    
    risk_factors, positive_factors = analyze_risk_factors(input_dict)
    
    return {
        'status': status,
        'probability_approved': float(prob_approved),
        'probability_rejected': float(1.0 - prob_approved),
        'risk_factors': risk_factors,
        'positive_factors': positive_factors
    }

if __name__ == "__main__":
    sample = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '0',
        'Education': 'Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 5000,
        'CoapplicantIncome': 2000,
        'LoanAmount': 150,
        'Loan_Amount_Term': 360,
        'Credit_History': 1.0,
        'Property_Area': 'Urban'
    }
    res = predict_loan(sample)
    print(f"Prediction: Loan {res['status']} ({res['probability_approved']*100:.2f}%)")
