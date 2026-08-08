import unittest
import pandas as pd
import numpy as np
from src.data_preprocessing import engineer_features, get_preprocessor
from predict import predict_loan, load_champion_model

class TestLoanMLPipeline(unittest.TestCase):
    
    def setUp(self):
        self.sample_raw_data = {
            'Gender': 'Male',
            'Married': 'Yes',
            'Dependents': '0',
            'Education': 'Graduate',
            'Self_Employed': 'No',
            'ApplicantIncome': 6000,
            'CoapplicantIncome': 2400,
            'LoanAmount': 180,
            'Loan_Amount_Term': 360,
            'Credit_History': 1.0,
            'Property_Area': 'Semiurban'
        }
        
    def test_feature_engineering(self):
        """Test domain feature calculations"""
        df = pd.DataFrame([self.sample_raw_data])
        df_eng = engineer_features(df)
        
        # Verify Total Income
        self.assertEqual(df_eng['Total_Income'].iloc[0], 8400)
        
        # Verify EMI (180 * 1000 / 360 = 500)
        self.assertAlmostEqual(df_eng['EMI'].iloc[0], 500.0, places=1)
        
        # Verify Balance Income ((8400 / 12) - 500 = 700 - 500 = 200)
        self.assertAlmostEqual(df_eng['Balance_Income'].iloc[0], 200.0, places=1)
        
        # Verify log transform exists
        self.assertTrue('ApplicantIncome_Log' in df_eng.columns)
        self.assertTrue('Total_Income_Log' in df_eng.columns)
        self.assertTrue('LoanAmount_Log' in df_eng.columns)
        
    def test_preprocessor_shape(self):
        """Test ColumnTransformer output consistency"""
        df = pd.DataFrame([self.sample_raw_data])
        df_eng = engineer_features(df)
        
        preprocessor, num_cols, cat_cols = get_preprocessor()
        transformed = preprocessor.fit_transform(df_eng)
        
        # Verify it produces a 2D numpy array with no NaNs
        self.assertEqual(len(transformed.shape), 2)
        self.assertFalse(np.isnan(transformed).any())
        
    def test_prediction_output_bounds(self):
        """Test prediction output validity and probability bounds"""
        try:
            model = load_champion_model()
        except FileNotFoundError:
            self.skipTest("Champion model not trained yet.")
            
        res = predict_loan(self.sample_raw_data, model)
        
        # Check output structure
        self.assertIn(res['status'], ['Approved', 'Rejected'])
        self.assertGreaterEqual(res['probability_approved'], 0.0)
        self.assertLessEqual(res['probability_approved'], 1.0)
        self.assertAlmostEqual(res['probability_approved'] + res['probability_rejected'], 1.0, places=4)
        self.assertIsInstance(res['risk_factors'], list)
        self.assertIsInstance(res['positive_factors'], list)

if __name__ == '__main__':
    unittest.main()
