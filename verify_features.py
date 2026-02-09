import unittest
import sys
import os

# Add current dir to path to find app.py
sys.path.append(os.getcwd())

from app import app

class TestCreditRiskApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_predict_endpoint(self):
        # Sample data matching the form fields
        data = {
            'person_age': '30',
            'person_income': '50000',
            'person_home_ownership': 'RENT',
            'person_emp_length': '5.0',
            'loan_intent': 'EDUCATION',
            'loan_grade': 'B',
            'loan_amnt': '10000',
            'loan_int_rate': '10.5',
            'loan_percent_income': '0.20',
            'cb_person_default_on_file': 'N',
            'cb_person_cred_hist_length': '3'
        }
        
        response = self.app.post('/predict', data=data)
        self.assertEqual(response.status_code, 200)
        
        html = response.get_data(as_text=True)
        
        # Verify LIME Explanations section
        # New text is "Key Factors"
        self.assertIn('Key Factors', html)
        
        # Verify Chart.js canvas elements
        self.assertIn('id="riskGauge"', html)
        self.assertIn('id="incomeChart"', html)
        
        # Verify Data Injection
        # 50000 income should be present
        self.assertIn('50000', html)
        
        print("Verification Successful: New Frontend Elements found.")

if __name__ == '__main__':
    unittest.main()
