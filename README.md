## Credit Scoring Model is a predictive analytics solution that assesses an individual's likelihood of repaying debts based on historical financial data. Financial institutions use these models to make informed decisions about loan approvals, credit limits, and interest rates. By analyzing factors like income, debt levels, and payment history, the model categorizes individuals as high or low credit risks.

## 🎯 Objective
The objective of this project is to build a machine learning model that predicts an individual's **creditworthiness** (Good Credit / Bad Credit) using classification algorithms. The model will help in automating credit risk assessment, reducing manual intervention, and improving decision accuracy for financial services.

## 🛠️ Features
- 📊 Feature Engineering from financial history (Income, Debts, Payment History, Credit Utilization).
- 🤖 Classification Models:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
- 🧪 Model Evaluation Metrics:
  - Precision, Recall, F1-Score, ROC-AUC Score

## 📂 Dataset
The dataset includes the following features:
- Income
- Debt
- Payment_History
- Credit_Utilization
- Default_Status (Target)

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies:
    pip install -r requirements.txt
3. Run EDA Notebook: (Not yet implemented)
4. Train Models:
    python src/model_training.py

## 📈 Output
- Classification Reports for each model (Precision, Recall, F1-Score)
- ROC-AUC Scores
- Outputs are printed in the console.

## 🔗 References
- Scikit-learn Documentation: https://scikit-learn.org/
- Dataset: Synthetic Example
- ML Metrics: Precision, Recall, F1-Score, ROC-AUC
