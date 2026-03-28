# ClearCredit XAI

## Project Overview
This project is an end-to-end credit risk prediction system that classifies loan applicants as High Risk or Low Risk, estimates probability of default, and provides explainable insights for each decision.

## Business Objective
Financial institutions require fast and reliable credit decisions while ensuring transparency and regulatory compliance.
This project addresses that by combining predictive modeling with explainability techniques to support informed and auditable decision-making.

## Key Features
- Predicts applicant risk category and probability of default
- Provides clear explanation of individual predictions using Dependency-Free Counterfactual Explainability
- Calculates financial indicators such as Debt-to-Income (DTI) ratio
- Interactive web interface for real-time prediction and analysis

## Tech Stack
- **Backend:** Python, Flask
- **Machine Learning:** Scikit-learn, Random Forest Feature Importances
- **Frontend:** HTML, CSS, JavaScript
- **Dataset:** Kaggle Credit Risk Dataset

## Model Development
- Performed data cleaning and preprocessing on historical credit data
- Engineered features such as Debt-to-Income ratio to capture financial risk
- Trained and compared models: Logistic Regression, Decision Tree, Random Forest
- Selected Random Forest based on superior F1-score performance
- Evaluated model using Precision, Recall, and F1-score

## Explainability
Counterfactual Explanations and local feature deviations are used alongside Random Forest's native feature importances to explain individual predictions by identifying the most influential features (e.g., income, loan amount, credit history).
This improves transparency and supports regulatory compliance in credit decision-making without relying on heavy external dependencies.

## Deployment
The model is deployed using Flask, allowing users to input applicant details through a web interface and receive:
- Risk classification
- Probability of default
- Explanation of key contributing factors

## Limitations & Future Scope
- Uses static historical data (no real-time data integration)
- Does not account for behavioral or time-based changes
- Future improvements: model monitoring, fairness analysis, and drift detection

---

## Getting Started

Follow these steps to run the project locally.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ATHARVAKARANDE27/ClearCredit-XAI.git
   cd ClearCredit-XAI
   ```

2. **Set up Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Data Setup:**
   - Create a folder named "data" in the root directory.
   - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset).
   - Place "credit_risk_dataset.csv" inside the "data" folder.

4. **Train and Run:**
   - Train the model: "python utils/main.py"
   - Start the server: "python app.py"
   - Access via browser: http://127.0.0.1:5000
