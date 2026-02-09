import app
import pandas as pd
import numpy as np

# Access via app module
app.init_lime_and_stats()
feature_names = app.feature_names
encoders = app.encoders
explainer = app.explainer
predict_fn = app.predict_fn


print("Feature names:", feature_names)

# Sample input (similar to what app receives)
sample_data = {
    'person_age': 30,
    'person_income': 50000,
    'person_home_ownership': 'RENT',
    'person_emp_length': 5.0,
    'loan_intent': 'EDUCATION',
    'loan_grade': 'B',
    'loan_amnt': 10000,
    'loan_int_rate': 10.5,
    'loan_percent_income': 0.20,
    'cb_person_default_on_file': 'N',
    'cb_person_cred_hist_length': 3
}

df_input = pd.DataFrame([sample_data])
print("Input Header:", df_input.columns.tolist())

# Prepare for LIME (encode)
categorical_features = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
lime_input = df_input.copy()
for feature in categorical_features:
    lime_input[feature] = encoders[feature].transform(lime_input[feature].astype(str))

print("Encoded Input Shape:", lime_input.values.shape)
print("Encoded Input:", lime_input.values)

# Test predict_fn directly
try:
    probs = predict_fn(lime_input.values)
    print("Direct predict_fn result:", probs)
except Exception as e:
    print("Direct predict_fn FAILED:")
    print(e)
    import traceback
    traceback.print_exc()

# Test explain_instance
try:
    print("Starting explanation...")
    exp = explainer.explain_instance(
        lime_input.values[0], 
        predict_fn, 
        num_features=3
    )
    print("Explanation:", exp.as_list())
except Exception as e:
    print("Explanation FAILED:")
    print(e)
    import traceback
    traceback.print_exc()
