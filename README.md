# 💳 Credit Risk Assessment & Explainable ML Application

## 📌 Problem Statement
Financial institutions must assess loan applicants quickly while maintaining transparency and regulatory explainability. Traditional models often fail to clearly justify why a customer is approved or rejected.

## 🚀 Project Overview
This project is an end-to-end **Credit Risk Assessment** web application that predicts whether a loan applicant is **High Risk** or **Low Risk**, estimates the probability of default, and provides explainable reasons behind each decision using **LIME** (Local Interpretable Model-agnostic Explanations).

## 💼 Business Value
*   **Enables faster, consistent credit decisions**: Automating the initial screening process.
*   **Supports regulator-friendly explainability**: Provides clear justification for loan decisions.
*   **Reduces manual underwriting effort**: Allows experts to focus on complex cases.
*   **Helps risk teams prioritize high-risk applicants**: Visual dashboards highlighting critical risk factors.

## 🛠️ Tech Stack
*   **Backend**: Python, Flask
*   **Machine Learning**: Scikit-learn, LIME
*   **Frontend**: HTML, CSS, JavaScript (Vanilla)
*   **Data**: Historical credit risk dataset

## 🧠 Model Development
*   **Models evaluated**: Logistic Regression, Decision Tree, Random Forest.
*   **Metric used**: **F1-Score**, prioritizing the balance between detecting risky borrowers and minimizing false positives.
*   **Final model**: Random Forest.

## 🔍 Explainability
For applicants flagged as **High Risk**, LIME identifies the most influential features (e.g., income level, loan amount, credit history) that contributed to the decision, enabling transparent and interpretable outcomes.

## ⚠️ Limitations & Future Enhancements
*   Uses static historical data (no real-time bureau integration).
*   Does not account for behavioral changes over time.
*   **Future Scope**: Includes model monitoring, fairness checks, and drift detection.

---

### 🛠️ Getting Started
*(Since data and models are not tracked in this repository, you must train the model first)*

1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Train the model**: `python utils/main.py`
4. **Run the application**: `python app.py`
