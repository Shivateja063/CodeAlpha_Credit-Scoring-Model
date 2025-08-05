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