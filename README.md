# ClearCredit XAI | Explainable ML Application

## Problem Statement
Financial institutions must assess loan applicants quickly while maintaining transparency and regulatory explainability. Traditional models often fail to clearly justify why a customer is approved or rejected.

## Project Overview
This project is an end-to-end ClearCredit XAI web application that predicts whether a loan applicant is High Risk or Low Risk, estimates the probability of default, and provides explainable reasons behind each decision using LIME (Local Interpretable Model-agnostic Explanations).

## Business Value
- Enables faster, consistent credit decisions: Automating the initial screening process.
- Supports regulator-friendly explainability: Provides clear justification for loan decisions.
- Reduces manual underwriting effort: Allows experts to focus on complex cases.
- Automated Financial Calculations: Real-time LTI and DTI estimation grounded in historical data.
- Helps risk teams prioritize high-risk applicants: Visual dashboards highlighting critical risk factors.


## Tech Stack
- Backend: Python, Flask
- Machine Learning: Scikit-learn, LIME
- Frontend: HTML, CSS, JavaScript (Vanilla)
- Data: [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (Historical data)

## Model Development
- Models evaluated: Logistic Regression, Decision Tree, Random Forest.
- Metric used: F1-Score, prioritizing the balance between detecting risky borrowers and minimizing false positives.
- Final model: Random Forest.

## Explainability
For applicants flagged as High Risk, LIME identifies the most influential features (e.g., income level, loan amount, credit history) that contributed to the decision, enabling transparent and interpretable outcomes.

## Limitations and Future Enhancements
- Uses static historical data (no real-time bureau integration).
- Does not account for behavioral changes over time.
- Future Scope: Includes model monitoring, fairness checks, and drift detection.

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
