import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

# Get the current directory where the script is located
current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

# Read the CSV
df = pd.read_csv(csv_path)

# Fix column names with spaces
df.columns = df.columns.str.strip()
print(df.columns)

# Clean and preprocess
df.drop("customerID", axis=1, inplace=True)
df.replace(" ", np.nan, inplace=True)
df.dropna(inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"])
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

# Encode categorical features
categorical_cols = df.select_dtypes(include=["object"]).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and labels
X = df.drop("Churn", axis=1)
y = df["Churn"]
feature_names = X.columns.tolist()

# Scale numeric data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the preprocessed dataset to a CSV file
preprocessed_data_path = os.path.join(current_dir, "preprocessed_customer_data.csv")
pd.DataFrame(X_scaled, columns=feature_names).to_csv(preprocessed_data_path, index=False)
print(f"✅ Preprocessed data saved to {preprocessed_data_path}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and preprocessing tools
joblib.dump(model, "churn_model_rf.pkl")
joblib.dump(scaler, "churn_scaler.pkl")
joblib.dump(feature_names, "feature_names.pkl")

print("✅ Model training complete and files saved.")
