from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os
import lime
import lime.lime_tabular
from sklearn.preprocessing import LabelEncoder
import traceback
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# --- Global State ---
model = None
explainer = None
encoders = {}
feature_names = []
categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
stats_metadata = {}

# --- Initialization ---
def initialize_app():
    global model, explainer, encoders, feature_names, stats_metadata
    
    # 1. Load Model
    try:
        model = joblib.load('models/model.pkl')
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("CRITICAL ERROR: models/model.pkl not found. Run backend training first.")
        return

    # 2. Load Data for LIME & Stats
    # Try different paths to be robust
    possible_paths = [
        "data/credit_risk_dataset.csv",
        "../data/credit_risk_dataset.csv"
    ]
    
    df = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading dataset from: {path}")
            df = pd.read_csv(path)
            break
            
    if df is None:
        print("CRITICAL WARNING: Dataset not found. Explainability and Dashboard stats will fail.")
        return

    # 3. Calculate Stats for Dashboard (Comparison)
    stats_metadata['avg_income'] = df['person_income'].mean()
    stats_metadata['median_income'] = df['person_income'].median()
    stats_metadata['income_percentiles'] = np.percentile(df['person_income'], [25, 50, 75])
    
    # 4. Initialize LIME Explainer
    print("Initializing LIME Explainer...")
    X = df.drop('loan_status', axis=1)
    feature_names = X.columns.tolist()
    
    categorical_names = {}
    for feature in categorical_features:
        le = LabelEncoder()
        # Handle missing values for LIME init string conversion
        X[feature] = le.fit_transform(X[feature].astype(str))
        encoders[feature] = le
        categorical_names[feature_names.index(feature)] = le.classes_
    
    # Fill numeric NaNs for LIME initialization
    X = X.fillna(X.median(numeric_only=True))
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X.values,
        feature_names=feature_names,
        class_names=['Low Risk', 'High Risk'],
        categorical_features=[feature_names.index(c) for c in categorical_features],
        categorical_names=categorical_names,
        mode='classification',
        discretize_continuous=True 
    )
    print("LIME Initialized.")

# Run Init
initialize_app()

# --- LIME Prediction Wrapper ---
def predict_fn_lime(data_numpy):
    """
    LIME passes a numpy array of encoded integers/floats.
    We must convert this back to a DataFrame with proper types/strings 
    for the main model pipeline (which has OneHotEncoder).
    """
    df_temp = pd.DataFrame(data_numpy, columns=feature_names)
    
    for feature in categorical_features:
        # Decode integer to original string
        # Safely handle float conversions from LIME
        df_temp[feature] = df_temp[feature].map(
            lambda x: encoders[feature].inverse_transform([int(round(x))])[0]
        )
    
    return model.predict_proba(df_temp)

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "System not ready. Model not loaded.", 500
        
    try:
        # 1. Parse Input
        def parse_num(val, func):
            if isinstance(val, str):
                return func(val.replace(',', '').replace('$', ''))
            return func(val)

        form_data = {
            'person_age': int(request.form['person_age']),
            'person_income': parse_num(request.form['person_income'], int),
            'person_home_ownership': request.form['person_home_ownership'],
            'person_emp_length': float(request.form['person_emp_length']),
            'loan_intent': request.form['loan_intent'],
            'loan_grade': request.form['loan_grade'],
            'loan_amnt': parse_num(request.form['loan_amnt'], int),
            'loan_int_rate': float(request.form['loan_int_rate']),
            'loan_percent_income': float(request.form['loan_percent_income']),
            'cb_person_default_on_file': request.form['cb_person_default_on_file'],
            'cb_person_cred_hist_length': int(request.form['cb_person_cred_hist_length'])
        }
        
        # DataFrame for Model
        df_input = pd.DataFrame([form_data])
        
        # 2. Predict
        prediction = model.predict(df_input)[0]
        probability = model.predict_proba(df_input)[0][1] # Probability of '1' (Default/Risk)
        result_text = "High" if prediction == 1 else "Low"
        
        # 3. Generate Explanation (LIME)
        top_reasons = []
        if explainer:
            try:
                # Encode input for LIME
                lime_input_row = df_input.copy()
                for feature in categorical_features:
                    lime_input_row[feature] = encoders[feature].transform(lime_input_row[feature].astype(str))
                
                # Get LIME explanation
                exp = explainer.explain_instance(
                    lime_input_row.values[0],
                    predict_fn_lime,
                    num_features=3
                )
                
                # Feature Mapping for nice display
                feature_map = {
                    'person_income': 'Annual Income',
                    'person_age': 'Age',
                    'person_emp_length': 'Employment Length',
                    'loan_intent': 'Loan Purpose',
                    'loan_grade': 'Loan Grade',
                    'loan_amnt': 'Loan Amount',
                    'loan_int_rate': 'Interest Rate',
                    'loan_percent_income': 'Loan-to-Income Ratio',
                    'cb_person_default_on_file': 'Default History',
                    'cb_person_cred_hist_length': 'Credit History',
                    'person_home_ownership': 'Home Ownership'
                }

                import re

                # --- AI Recommendation Engine ---
                risk_analysis = []
                
                for feature_cond, weight in exp.as_list():
                    is_risk = weight > 0
                    impact_type = "Negative" if is_risk else "Positive"
                    
                    # 1. Feature Identification
                    raw_feature = None
                    human_name = feature_cond
                    
                    for key, val in feature_map.items():
                        if key in feature_cond:
                            human_name = feature_map[key]
                            raw_feature = key
                            break
                    
                    # 2. Natural Language Translation
                    # Default fallback
                    readable_cond = feature_cond.replace(raw_feature, human_name) if raw_feature else feature_cond
                    
                    # Custom Logic for specific features
                    if raw_feature == 'person_home_ownership':
                        if 'RENT' in feature_cond: readable_cond = "Applicant currently rents their home"
                        elif 'OWN' in feature_cond: readable_cond = "Applicant owns their home"
                        elif 'MORTGAGE' in feature_cond: readable_cond = "Applicant has an active mortgage"
                        else: readable_cond = f"Home ownership status: {feature_cond.split('=')[-1]}"

                    elif raw_feature == 'loan_grade':
                        grade = feature_cond.split('=')[-1].strip()
                        readable_cond = f"Assigned Loan Grade is {grade}"
                        
                    elif raw_feature == 'loan_intent':
                        intent = feature_cond.split('=')[-1].strip().lower().title()
                        readable_cond = f"Loan purpose is for {intent}"

                    elif raw_feature == 'cb_person_default_on_file':
                        if 'Y' in feature_cond: readable_cond = "Applicant has a history of default"
                        elif 'N' in feature_cond: readable_cond = "No prior defaults on file"

                    elif raw_feature in ['person_income', 'loan_amnt']:
                        # Add formatting to numbers
                        readable_cond = re.sub(r'(\d+\.?\d*)', r'$\1', readable_cond)
                        # Replace symbols with words for better flow
                        readable_cond = readable_cond.replace('>', 'is above').replace('<', 'is below').replace('=', 'is')
                        
                    else:
                         # General cleanup for other fields
                        readable_cond = readable_cond.replace('>', 'is above ').replace('<', 'is below ').replace('=', 'is ')
                        readable_cond = re.sub(r'(\d+\.?\d*)', r'\1', readable_cond) # clean up floats if needed

                    # 3. Generate Smart Recommendation (Keep existing logic but refine context)
                    recommendation = ""
                    action_icon = ""

                    if is_risk:
                        # --- Negative Factors (Actionable Advice) ---
                        action_icon = "fa-triangle-exclamation"
                        if raw_feature == 'loan_amnt':
                            recommendation = "Consider lowering the loan amount slightly to improve approval chances."
                        elif raw_feature == 'person_income':
                            recommendation = "Income level is a constraint. A co-signer or proof of additional income may help."
                        elif raw_feature == 'loan_int_rate':
                            recommendation = "High interest rate reflects perceived risk. improving credit score is key."
                        elif raw_feature == 'loan_percent_income':
                            recommendation = "Debt-to-Income ratio is high. Paying off smaller existing debts could help."
                        elif raw_feature == 'cb_person_default_on_file':
                            recommendation = "Past defaults are a major flag. Consistent on-time payments are needed to rebuild trust."
                        elif raw_feature == 'person_home_ownership':
                            recommendation = "Lack of property assets reduces collateral options."
                        elif raw_feature == 'loan_grade':
                            recommendation = "The assigned grade suggests high risk. A secured loan might be a better option."
                        else:
                            recommendation = "This factor negatively impacts the risk profile."
                    else:
                        # --- Positive Factors (Strengths) ---
                        action_icon = "fa-check-double"
                        if raw_feature == 'person_income':
                            recommendation = "Strong income level supports this loan application."
                        elif raw_feature == 'person_emp_length':
                            recommendation = "Employment stability is a significant strength."
                        elif raw_feature == 'cb_person_default_on_file':
                            recommendation = "Clean credit history is a strong positive indicator."
                        elif raw_feature == 'loan_percent_income':
                            recommendation = "Healthy debt-to-income ratio shows good financial management."
                        elif raw_feature == 'loan_grade':
                            recommendation = "Excellent loan grade reflects a strong borrower profile."
                        else:
                            recommendation = "This factor strengthens the application."

                    risk_analysis.append({
                        'factor': human_name,
                        'condition': readable_cond,
                        'impact': impact_type,
                        'recommendation': recommendation,
                        'icon': action_icon,
                        'raw_feature': raw_feature
                    })

                # Pass to template
                top_reasons = risk_analysis
                    
            except Exception as e:
                print(f"LIME Error: {e}")
                traceback.print_exc()
                top_reasons = ["Explanation data unavailable."]

        # 4. Prepare Comparison Data
        income = form_data['person_income']
        avg_inc = stats_metadata.get('avg_income', 0)
        
        # Determine Rank
        rank = ""
        if 'income_percentiles' in stats_metadata:
            p25, p50, p75 = stats_metadata['income_percentiles']
            if income < p25: rank = "Bottom 25%"
            elif income > p75: rank = "Top 25%"
            else: rank = "Average"
        else:
            rank = "N/A"

        comparison = {
            'user_income': income,
            'avg_income': round(avg_inc, 2),
            'income_rank': rank
        }
        
        return render_template(
            'result.html', 
            result=result_text,
            probability=round(probability * 100, 2),
            top_reasons=top_reasons,
            comparison=comparison
        )
        
    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}", 400

if __name__ == '__main__':
    app.run(debug=True)
