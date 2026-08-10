import sys
from predict import load_champion_model, predict_loan

def prompt_with_default(prompt_text, default_value, val_type=str):
    user_val = input(f"{prompt_text} [Default: {default_value}]: ").strip()
    if not user_val:
        return val_type(default_value)
    try:
        return val_type(user_val)
    except ValueError:
        print(f"Invalid format! Using default: {default_value}")
        return val_type(default_value)

def main():
    print("=" * 60)
    print("     [LOANGUARD AI] LOAN APPROVAL PREDICTION SYSTEM")
    print("=" * 60)
    print("Press Enter to accept defaults, or enter custom applicant data:\n")
    
    try:
        model = load_champion_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first by executing: python train.py")
        sys.exit(1)
        
    data = {}
    data['Gender'] = prompt_with_default("Gender (Male/Female)", "Male", str)
    data['Married'] = prompt_with_default("Married Status (Yes/No)", "Yes", str)
    data['Dependents'] = prompt_with_default("Dependents (0/1/2/3+)", "0", str)
    data['Education'] = prompt_with_default("Education (Graduate/Not Graduate)", "Graduate", str)
    data['Self_Employed'] = prompt_with_default("Self Employed (Yes/No)", "No", str)
    data['ApplicantIncome'] = prompt_with_default("Applicant Monthly Income ($)", 5400, float)
    data['CoapplicantIncome'] = prompt_with_default("Co-applicant Monthly Income ($)", 1800, float)
    data['LoanAmount'] = prompt_with_default("Loan Amount in Thousands ($k)", 140, float)
    data['Loan_Amount_Term'] = prompt_with_default("Loan Term in Months", 360, float)
    data['Credit_History'] = prompt_with_default("Credit History (1.0 for Good, 0.0 for Bad)", 1.0, float)
    data['Property_Area'] = prompt_with_default("Property Area (Urban/Semiurban/Rural)", "Semiurban", str)
    
    print("\n" + "-" * 60)
    print("Evaluating application through Champion Machine Learning Pipeline...")
    print("-" * 60)
    
    result = predict_loan(data, model)
    
    print(f"\nDECISION:  >>> LOAN {result['status'].upper()} <<<")
    print(f"Approval Probability: {result['probability_approved'] * 100:.2f}%")
    print(f"Rejection Risk:       {result['probability_rejected'] * 100:.2f}%\n")
    
    if result['positive_factors']:
        print("[+] Positive Financial Factors:")
        for factor in result['positive_factors']:
            print(f"   * {factor}")
            
    if result['risk_factors']:
        print("\n[-] Identified Risk Factors:")
        for factor in result['risk_factors']:
            print(f"   * {factor}")
            
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
