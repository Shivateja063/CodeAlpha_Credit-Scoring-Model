# CodeAlpha_Credit-Scoring-Model
Structure
css
Copy
Edit
Credit-Scoring-Model/
├── README.md
├── data/
│   └── credit_data.csv
├── notebooks/
│   └── EDA_FeatureEngineering.ipynb
├── src/
│   └── model_training.py
│   └── utils.py
├── outputs/
│   └── model_performance_report.txt
├── requirements.txt
└── .gitignore
📄 README.md
markdown
Copy
Edit
# Credit Scoring Model

## Objective
Predict an individual's creditworthiness using historical financial data. This project leverages classification algorithms to classify individuals as "Good Credit" or "Bad Credit".

## Approach
- **Data Preprocessing & Feature Engineering** from financial history.
- Model building using:
  - Logistic Regression
  - Decision Trees
  - Random Forest
- Model evaluation using metrics like:
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC Curve

## Dataset
Sample dataset includes:
- Income
- Debts
- Payment History
- Credit Utilization
- Default Status (Target)

## Folder Structure
- `data/`: Contains the dataset (`credit_data.csv`).
- `notebooks/`: Jupyter notebooks for EDA & Feature Engineering.
- `src/`: Python scripts for model training and utility functions.
- `outputs/`: Model reports and metrics.
- `requirements.txt`: List of Python dependencies.

## How to Run
1. Clone this repository.
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run EDA Notebook:
    - Open `notebooks/EDA_FeatureEngineering.ipynb` and execute.
4. Train Model:
    ```bash
    python src/model_training.py
    ```

## Metrics Evaluated
- Accuracy
- Precision & Recall
- F1-Score
- ROC-AUC

## Future Work
- Hyperparameter Tuning
- Deploy as API using Flask/FastAPI
📄 requirements.txt
nginx
Copy
Edit
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
📁 data/credit_data.csv (Sample Data)
csv
Copy
Edit
Income,Debt,Payment_History,Credit_Utilization,Default_Status
55000,15000,Good,30,0
48000,12000,Poor,60,1
62000,5000,Good,20,0
75000,20000,Average,50,0
39000,10000,Poor,70,1
...
📄 notebooks/EDA_FeatureEngineering.ipynb
Main EDA + Feature Engineering notebook where you:

Analyze data distributions.

Encode categorical variables.

Create new features if needed.

Scale/normalize features.

📄 src/utils.py
python
Copy
Edit
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    le = LabelEncoder()
    df['Payment_History'] = le.fit_transform(df['Payment_History'])
    X = df.drop('Default_Status', axis=1)
    y = df['Default_Status']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y
📄 src/model_training.py
python
Copy
Edit
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from utils import load_data, preprocess_data

# Load and preprocess data
df = load_data('data/credit_data.csv')
X, y = preprocess_data(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier()
}

# Train & Evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred)}")
📄 outputs/model_performance_report.txt
(This will be generated after model training execution)

📄 .gitignore
markdown
Copy
Edit
__pycache__/
*.pyc
.ipynb_checkpoints/
outputs/
