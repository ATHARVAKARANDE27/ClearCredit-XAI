from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os
import traceback
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# --- Global State ---
model = None
feature_names = []
categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
stats_metadata = {}
training_means = {}   # Per-feature means from training data (for local importance)
rf_importances = {}   # Global feature importances from the Random Forest

# --- Initialization ---
def initialize_app():
    global model, feature_names, stats_metadata, training_means, rf_importances
    
    # 1. Load Model
    try:
        model = joblib.load('models/model.pkl')
        print("ClearCredit XAI: Model loaded successfully.")
    except FileNotFoundError:
        print("CRITICAL ERROR: models/model.pkl not found. Run backend training first.")
        return

    # 2. Load Data for SHAP & Stats
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
    if df is not None:
        approved_df = df[df['loan_status'] == 0]
        stats_metadata['avg_income'] = float(df['person_income'].mean())
        stats_metadata['median_income'] = float(df['person_income'].median())
        stats_metadata['income_percentiles'] = [float(x) for x in np.percentile(df['person_income'].dropna(), [25, 50, 75])]
        
        # Database-derived thresholds for the UI
        if not approved_df.empty:
            stats_metadata['database_stats'] = {
                'avg_lti_approved': float(approved_df['loan_percent_income'].mean()),
                'p75_lti_approved': float(np.percentile(approved_df['loan_percent_income'].dropna(), 75)),
                'avg_income_approved': float(approved_df['person_income'].mean())
            }
        else:
            stats_metadata['database_stats'] = {
                'avg_lti_approved': 0.15,
                'p75_lti_approved': 0.20,
                'avg_income_approved': 50000.0
            }
    else:
        stats_metadata['avg_income'] = 0
        stats_metadata['database_stats'] = {}

    # 4. Build dependency-free local explainer
    print("Building feature importance explainer...")
    X = df.drop('loan_status', axis=1)
    feature_names = X.columns.tolist()

    # Compute per-feature means / modes for deviation scoring
    for feat in feature_names:
        if feat in categorical_features:
            training_means[feat] = X[feat].mode()[0]   # most common category
        else:
            training_means[feat] = float(X[feat].median())

    # Extract global feature importances from the trained RF classifier
    inner_clf = model.named_steps['classifier']
    # Get transformed feature names so we can map back to originals
    preprocessor = model.named_steps['preprocessor']
    try:
        tf_names = list(preprocessor.get_feature_names_out())
    except AttributeError:
        tf_names = []

    importances = inner_clf.feature_importances_
    # Aggregate importance back to original feature names
    for feat in feature_names:
        rf_importances[feat] = 0.0

    if tf_names and len(tf_names) == len(importances):
        for idx, tf_name in enumerate(tf_names):
            for orig in feature_names:
                if orig in tf_name:
                    rf_importances[orig] += importances[idx]
                    break
    else:
        # Fallback: spread evenly
        for i, feat in enumerate(feature_names):
            rf_importances[feat] = importances[i] if i < len(importances) else 0.0

    print("Explainer ready (dependency-free RF importance mode).")

# Run Init
initialize_app()


# --- Local Explainer: counterfactual-based per-prediction importance ---
def compute_local_importance(form_data, base_proba):
    """
    For each feature, replace it with its training median/mode and re-run
    the model. The change in probability tells us the TRUE direction:

      impact = base_proba - counterfactual_proba
      impact > 0  →  real value REDUCES risk vs neutral  →  Positive (good)
      impact < 0  →  real value INCREASES risk vs neutral →  Negative (bad)

    Score = rf_importance × impact  (weighted by global importance)
    """
    scores = {}
    for feat in feature_names:
        global_imp = rf_importances.get(feat, 0.0)
        if global_imp < 1e-6:
            scores[feat] = 0.0
            continue

        # Build counterfactual: swap just this feature for its neutral value
        cf_data = dict(form_data)
        cf_data[feat] = training_means[feat]

        try:
            cf_proba = model.predict_proba(pd.DataFrame([cf_data]))[0][1]
            # Positive impact → current value lowers probability → GOOD (reduces risk)
            # Negative impact → current value raises probability → BAD (increases risk)
            impact = base_proba - cf_proba
            scores[feat] = global_imp * impact
        except Exception:
            scores[feat] = 0.0

    # Sort by absolute magnitude (most influential features first)
    ranked = sorted(scores.items(), key=lambda x: abs(x[1]), reverse=True)
    return ranked

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
        
        # 3. Generate Explanation (RF feature importance × deviation)
        top_reasons = []
        try:
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

            # Get ranked features (sorted by absolute local importance)
            ranked = compute_local_importance(form_data, probability)
            top_features = ranked[:3]

            risk_analysis = []
            for raw_feature, score in top_features:
                # score > 0 → feature REDUCES risk vs neutral → Positive (good for applicant)
                # score < 0 → feature INCREASES risk vs neutral → Negative (bad for applicant)
                is_risk = score < 0
                impact_type = "Negative" if is_risk else "Positive"
                human_name = feature_map.get(raw_feature, raw_feature)
                actual_value = form_data.get(raw_feature, '')

                # Natural language condition based on actual value
                if raw_feature == 'person_home_ownership':
                    val = str(actual_value).upper()
                    if 'RENT' in val: readable_cond = "Applicant currently rents their home"
                    elif 'OWN' in val: readable_cond = "Applicant owns their home"
                    elif 'MORTGAGE' in val: readable_cond = "Applicant has an active mortgage"
                    else: readable_cond = f"Home ownership: {actual_value}"

                elif raw_feature == 'loan_grade':
                    readable_cond = f"Assigned Loan Grade is {actual_value}"

                elif raw_feature == 'loan_intent':
                    readable_cond = f"Loan purpose is for {str(actual_value).lower().title()}"

                elif raw_feature == 'cb_person_default_on_file':
                    if str(actual_value).upper() == 'Y':
                        readable_cond = "Applicant has a history of default"
                    else:
                        readable_cond = "No prior defaults on file"

                elif raw_feature in ['person_income', 'loan_amnt']:
                    readable_cond = f"{human_name} is ${actual_value:,}"

                elif raw_feature == 'loan_percent_income':
                    readable_cond = f"Loan-to-Income ratio is {actual_value:.0%}"

                elif raw_feature == 'loan_int_rate':
                    readable_cond = f"Interest rate is {actual_value}%"

                elif raw_feature == 'person_age':
                    readable_cond = f"Applicant age is {actual_value} years"

                elif raw_feature == 'person_emp_length':
                    readable_cond = f"Employment length is {actual_value} years"

                elif raw_feature == 'cb_person_cred_hist_length':
                    readable_cond = f"Credit history is {actual_value} years long"

                else:
                    readable_cond = f"{human_name}: {actual_value}"

                # Smart Recommendation
                recommendation = ""
                action_icon = ""

                if is_risk:
                    action_icon = "fa-triangle-exclamation"
                    if raw_feature == 'loan_amnt':
                        recommendation = "Consider lowering the loan amount slightly to improve approval chances."
                    elif raw_feature == 'person_income':
                        recommendation = "Income level is a constraint. A co-signer or proof of additional income may help."
                    elif raw_feature == 'loan_int_rate':
                        recommendation = "High interest rate reflects perceived risk. Improving credit score is key."
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

            top_reasons = risk_analysis

        except Exception as e:
            print(f"Explainer Error: {e}")
            traceback.print_exc()
            top_reasons = []

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
            comparison=comparison,
            original_data=form_data,
            db_stats=stats_metadata.get('database_stats', {})
        )


        
    except Exception as e:
        traceback.print_exc()
        return f"Error: {str(e)}", 400

@app.route('/predict_api', methods=['POST'])
def predict_api():
    """API endpoint for 'What-If' simulation (returns JSON)"""
    if model is None:
        return {"error": "Model not loaded"}, 500
    try:
        data = request.get_json()
        
        # Parse inputs (re-using form logic but for JSON)
        form_data = {
            'person_age': int(data['person_age']),
            'person_income': int(data['person_income']),
            'person_home_ownership': data['person_home_ownership'],
            'person_emp_length': float(data['person_emp_length']),
            'loan_intent': data['loan_intent'],
            'loan_grade': data['loan_grade'],
            'loan_amnt': int(data['loan_amnt']),
            'loan_int_rate': float(data['loan_int_rate']),
            'loan_percent_income': float(data['loan_percent_income']),
            'cb_person_default_on_file': data['cb_person_default_on_file'],
            'cb_person_cred_hist_length': int(data['cb_person_cred_hist_length'])
        }
        
        df_input = pd.DataFrame([form_data])
        probability = model.predict_proba(df_input)[0][1]
        prediction = model.predict(df_input)[0]
        result_text = "High" if prediction == 1 else "Low"
        
        return {
            "probability": round(probability * 100, 2),
            "result": result_text
        }
    except Exception as e:
        return {"error": str(e)}, 400

if __name__ == '__main__':
    app.run(debug=True)

