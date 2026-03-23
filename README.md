# Customer Churn Prediction with Explainable AI (XAI)

## Project Overview

This project predicts customer churn using the Telco Customer Churn dataset from Kaggle.

The system uses Machine Learning to classify whether a customer will leave the telecom service and provides Explainable AI (XAI) to understand model predictions.

The project also includes a Streamlit web app for interactive prediction.

---

## Dataset

Dataset: Telco Customer Churn Dataset
Source: Kaggle

Features:

* Gender
* SeniorCitizen
* Tenure
* MonthlyCharges
* TotalCharges
* Contract
* PaymentMethod
* InternetService
* Churn

---

## Technologies Used

Language:

* Python

Libraries:

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* SHAP
* Streamlit
* Pickle

Tools:

* Jupyter Notebook
* VS Code
* GitHub

---

## Machine Learning Model

* Random Forest Classifier
* Feature Scaling using StandardScaler
* Model saved using Pickle

Files:

* churn_model_rf.pkl
* churn_scaler.pkl
* feature_names.pkl

---

## Project Structure

CUSTOMER_CHURN_PREDICTION_XAI/

│
├── churn_xai_streamlit_ready/
│   ├── app.py
│   ├── training_model.py
│   ├── churn_model_rf.pkl
│   ├── churn_scaler.pkl
│   ├── feature_names.pkl
│   ├── requirements.txt
│   ├── dataset.csv
│
├── README.md
├── .gitignore

---

## How to Run

Install dependencies

pip install -r requirements.txt

Run Streamlit app

streamlit run app.py

---

## Features

* Customer churn prediction
* Explainable AI (XAI)
* Streamlit UI
* Model saving using pickle
* Kaggle dataset

---

## Future Improvements

* Deploy on cloud
* Add Flask API
* Add Docker
* Add MLflow
* Add CI/CD

