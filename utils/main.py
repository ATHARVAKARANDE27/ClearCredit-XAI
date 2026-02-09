
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print("Data loaded successfully.")
        print(f"Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print("Error: File not found.")
        return None

def preprocess_data(df):
    print("Preprocessing data...")
    
    # Identify target and features
    if 'loan_status' not in df.columns:
        raise ValueError("Target column 'loan_status' not found in dataset")
        
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    
    print(f"Numerical features: {list(numeric_features)}")
    print(f"Categorical features: {list(categorical_features)}")

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor

def train_and_evaluate(X, y, preprocessor):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    results = {}
    
    print("\n--- Model Evaluation ---")
    
    best_model_name = None
    best_f1 = -1
    best_pipeline = None
    
    for name, model in models.items():
        # Create full pipeline
        clf = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', model)])
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results[name] = {'Accuracy': acc, 'F1-Score': f1}
        
        print(f"\n{name}:")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(classification_report(y_test, y_pred))
        
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_pipeline = clf

    print("\n--- Final Recommendation ---")
    print(f"The best performing model based on F1-Score is: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['Accuracy']:.4f}")
    print(f"F1-Score: {results[best_model_name]['F1-Score']:.4f}")
    
    return best_pipeline

def save_metadata(df):
    metadata = {
        'median_income': df['person_income'].median(),
        'median_loan_amnt': df['loan_amnt'].median(),
        'income_percentiles': np.percentile(df['person_income'].dropna(), [25, 50, 75]).tolist(),
        'loan_percentiles': np.percentile(df['loan_amnt'].dropna(), [25, 50, 75]).tolist()
    }
    joblib.dump(metadata, 'metadata.pkl')
    print("Metadata saved to metadata.pkl")

if __name__ == "__main__":
    # Assuming the script is run from the project root or back/ directory locally
    # Adjust path as needed based on execution context
    import os
    
    # Try to find the dataset
    dataset_path = "data/credit_risk_dataset.csv"
    if not os.path.exists(dataset_path):
        dataset_path = "../data/credit_risk_dataset.csv"
    
    df = load_data(dataset_path)
    
    if df is not None:
        X, y, preprocessor = preprocess_data(df)
        best_model = train_and_evaluate(X, y, preprocessor)
        
        # Save the model
        model_filename = 'models/model.pkl' if os.path.exists('models') else '../models/model.pkl'
        joblib.dump(best_model, model_filename)
        print(f"Best model saved to {model_filename}")
        
        # Save metadata
        metadata_filename = 'models/metadata.pkl' if os.path.exists('models') else '../models/metadata.pkl'
        joblib.dump(df['person_income'].median(), metadata_filename) # Simplified illustration
        # Re-using the save_metadata logic
        metadata = {
            'median_income': df['person_income'].median(),
            'median_loan_amnt': df['loan_amnt'].median(),
            'income_percentiles': np.percentile(df['person_income'].dropna(), [25, 50, 75]).tolist(),
            'loan_percentiles': np.percentile(df['loan_amnt'].dropna(), [25, 50, 75]).tolist()
        }
        joblib.dump(metadata, metadata_filename)
        print(f"Metadata saved to {metadata_filename}")

