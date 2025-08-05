# 🏦 Credit Scoring Model

A **Credit Scoring Model** is a predictive analytics solution that assesses an individual's likelihood of repaying debts based on historical financial data. Financial institutions use these models to make informed decisions about loan approvals, credit limits, and interest rates. By analyzing factors like income, debt levels, and payment history, the model categorizes individuals as high or low credit risks.

---

## 🎯 Objective
The objective of this project is to build a machine learning model that predicts an individual's **creditworthiness** (Good Credit / Bad Credit) using classification algorithms. The model will help in automating credit risk assessment, reducing manual intervention, and improving decision accuracy for financial services.

---
## 🛠️ Features
- 📊 **Feature Engineering** from financial history (Income, Debts, Payment History, Credit Utilization).
- 🤖 **Classification Models**:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
- 🧪 **Model Evaluation Metrics**:
  - Precision, Recall, F1-Score
  - ROC-AUC Score

---

## 📂 Dataset
The dataset includes the following features:
- **Income** — Annual income of the individual.
- **Debt** — Total existing debts.
- **Payment_History** — Categorical (Good / Average / Poor).
- **Credit_Utilization** — Percentage of credit used.
- **Default_Status** — Target variable (0: Good Credit, 1: Bad Credit).

📁 Location: `data/credit_data.csv`

---

## 🚀 How to Run

### 1️⃣ Clone the Repository:
```bash
git clone https://github.com/your-username/Credit-Scoring-Model.git
cd Credit-Scoring-Model
2️⃣ Install Dependencies:
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Run Exploratory Data Analysis (EDA):
Open Jupyter Notebook and run:

bash
Copy
Edit
jupyter notebook notebooks/EDA_FeatureEngineering.ipynb
4️⃣ Train Models and Evaluate:
bash
Copy
Edit
python src/model_training.py
📈 Output
After running model_training.py, you will get:

Classification Report for each model (Precision, Recall, F1-Score).

ROC-AUC Score.

Results will be displayed in the console and saved to:

outputs/model_performance_report.txt

🔗 References
Scikit-learn Documentation: https://scikit-learn.org/

Dataset Structure: Synthetic dataset created for demo purposes.

Machine Learning Concepts: Precision, Recall, F1-Score, ROC-AUC.

📝 Author
Shiva Teja

