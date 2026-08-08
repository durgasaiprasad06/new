import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def engineer_features(df_input):
    """
    Applies domain-specific financial feature engineering:
    - Total_Income: Applicant + Coapplicant
    - EMI: Estimated Monthly Installment
    - Balance_Income: Disposable income after EMI
    - Income_to_Loan_Ratio: Ability to repay ratio
    - Log transforms for skewed numerical features
    """
    df = df_input.copy()
    
    # 1. Total Income
    df['Total_Income'] = df['ApplicantIncome'] + df['CoapplicantIncome']
    
    # 2. EMI Calculation (LoanAmount is in thousands, Loan_Amount_Term in months)
    term = df['Loan_Amount_Term'].replace(0, 360).fillna(360)
    loan_amt = df['LoanAmount'].fillna(df['LoanAmount'].median() if not df['LoanAmount'].empty else 140)
    df['EMI'] = (loan_amt * 1000) / term
    
    # 3. Balance Income (Monthly Total Income - Monthly EMI)
    monthly_income = df['Total_Income'] / 12.0
    df['Balance_Income'] = monthly_income - df['EMI']
    
    # 4. Income to Loan Amount Ratio
    df['Income_to_Loan_Ratio'] = df['Total_Income'] / (loan_amt * 1000 + 1)
    
    # 5. Log Transformations
    df['ApplicantIncome_Log'] = np.log1p(np.maximum(0, df['ApplicantIncome']))
    df['Total_Income_Log'] = np.log1p(np.maximum(0, df['Total_Income']))
    df['LoanAmount_Log'] = np.log1p(np.maximum(0, loan_amt))
    
    return df

def get_preprocessor():
    """
    Constructs a robust Scikit-Learn ColumnTransformer for numerical and categorical features.
    """
    numeric_features = [
        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
        'Credit_History', 'Total_Income', 'EMI', 'Balance_Income',
        'Income_to_Loan_Ratio', 'ApplicantIncome_Log', 'Total_Income_Log', 'LoanAmount_Log'
    ]
    
    categorical_features = [
        'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'
    ]
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor, numeric_features, categorical_features

def load_and_preprocess_data(filepath, test_size=0.2, random_state=42):
    """
    Loads dataset, handles initial cleaning, engineers features, 
    and splits into train and test sets.
    """
    print("Loading raw dataset from:", filepath)
    df = pd.read_csv(filepath)
    print(f"Raw shape: {df.shape}")
    
    # Drop Loan_ID
    if 'Loan_ID' in df.columns:
        df = df.drop(columns=['Loan_ID'])
        
    # Drop duplicate rows
    df = df.drop_duplicates()
    
    # Clean target variable
    df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
    df = df.dropna(subset=['Loan_Status'])
    df['Loan_Status'] = df['Loan_Status'].astype(int)
    
    # Handle initial missing values for Dependents formatting
    df['Dependents'] = df['Dependents'].astype(str).str.replace('+', '', regex=False)
    
    # Apply Feature Engineering
    df_engineered = engineer_features(df)
    
    X = df_engineered.drop(columns=['Loan_Status'])
    y = df_engineered['Loan_Status']
    
    # Stratified split to maintain class balance ratio in train & test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
    print(f"Target distribution (Train): {dict(y_train.value_counts(normalize=True).round(3))}")
    print(f"Target distribution (Test):  {dict(y_test.value_counts(normalize=True).round(3))}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess_data("../dataset/loan_data.csv")
    preprocessor, num_cols, cat_cols = get_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    print("Transformed training shape:", X_train_trans.shape)
