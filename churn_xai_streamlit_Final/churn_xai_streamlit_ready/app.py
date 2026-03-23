import streamlit as st
import pandas as pd
import joblib
import numpy as np
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Load required artifacts
model = joblib.load("churn_model_rf.pkl")
scaler = joblib.load("churn_scaler.pkl")
features = joblib.load("feature_names.pkl")

st.title("📉 Customer Churn Prediction App")
st.markdown("""
This project uses a machine learning model to predict whether a customer is likely to churn based on their details.
Upload a CSV file with customer data to get predictions.
""")

uploaded_file = st.file_uploader("📂 Upload customer data (.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Uploaded Data", df.head())

    # Check if required columns are present
    if all(f in df.columns for f in features):

        # Step 1: Encode categorical columns (e.g., Gender, Geography)
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Step 2: Ensure all expected features from training are present
        for col in features:
            if col not in df.columns:
                df[col] = 0  # Add missing columns with default value

        # Step 3: Reorder columns to match training
        df = df[features]

        # Step 4: Scale the data
        scaled_data = scaler.transform(df)

        # Step 5: Predict
        predictions = model.predict(scaled_data)
        probs = model.predict_proba(scaled_data)[:, 1]

        df["Churn Prediction"] = np.where(predictions == 1, "Yes", "No")
        df["Churn Probability"] = np.round(probs, 2)

        st.success("✅ Prediction complete!")
        st.write(df[["Churn Prediction", "Churn Probability"]])

        # SHAP Explanation
        st.subheader("SHAP Values")

        explainer_shap = shap.Explainer(model, scaled_data)
        shap_values = explainer_shap(scaled_data)

        # Check SHAP shape and use only class 1 explanations if multi-class
        if len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]

        shap.initjs()

        # Visualize for the first instance
        st.write("### SHAP Explanation for Instance #0")
        instance_index = 0
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[instance_index], show=False)
        st.pyplot(fig)

        # LIME Explanation
        st.subheader("LIME Explanation")
        explainer_lime = lime.lime_tabular.LimeTabularExplainer(
            training_data=np.array(scaled_data),
            feature_names=features,
            class_names=['No Churn', 'Churn'],
            mode='classification'
        )

        # Generate LIME explanation for the first instance
        lime_exp = explainer_lime.explain_instance(
            data_row=scaled_data[0],
            predict_fn=model.predict_proba
        )

        lime_fig = lime_exp.as_pyplot_figure()
        plt.title("LIME Feature Importance")
        st.pyplot(lime_fig)

    else:
        st.error("Uploaded CSV is missing required features. Required columns: " + ", ".join(features))
