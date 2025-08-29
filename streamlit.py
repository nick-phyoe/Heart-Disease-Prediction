import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")
st.title("Heart Disease Prediction — Choose Model & Predict")

# --- Paths 
MODEL_DIR = '/content/drive/MyDrive/Saved Models'
LR_FILE = os.path.join(MODEL_DIR, 'logistic_regression.h5')
SVM_FILE = os.path.join(MODEL_DIR, 'svm.h5')
RF_FILE = os.path.join(MODEL_DIR, 'random_forest.h5')

DEFAULT_DATA_CSV = '/content/drive/MyDrive/Heart Disease Prediction.csv'

#st.markdown(f"Models directory (default): `{MODEL_DIR}`")

def try_load(path):
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        return None

if not all([lr, svm, rf]):
    st.warning("One or more models failed to load. Place them in the path above (joblib .h5 files).")
    st.stop()
st.success("Models loaded successfully.")

# Feature order expected
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Model selection
model_choice = st.selectbox("SELECT MODEL", ["LOGISTIC REGRESSION", "SVM", "RANDOM FOREST"])
model_map = {'LOGISTIC REGRESSION': lr, 'SVM': svm, 'RANDOM FOREST': rf}
model = model_map[model_choice]

st.markdown("### Predict for a single patient")

with st.form("patient_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("age", 50)
        sex = st.selectbox("sex (1=male,0=female)", [1, 0], index=0)
        cp = st.number_input("cp (0-3)", 0, 3, 1)
        trestbps = st.number_input("trestbps", 0, 300, 130)
        chol = st.number_input("chol", 0, 1000, 250)
    with c2:
        fbs = st.selectbox("fbs (0/1)", [0, 1], index=0)
        restecg = st.number_input("restecg (0-2)", 0, 2, 1)
        thalach = st.number_input("thalach (max heart rate)", 0, 300, 150)
        exang = st.selectbox("exang (0/1)", [0, 1], index=0)
    with c3:
        oldpeak = st.number_input("oldpeak", 0.0, 10.0, 1.0, step=0.1, format="%.1f")
        slope = st.number_input("slope (0-2)", 0, 2, 2)
        ca = st.number_input("ca (0-3)", 0, 3, 0)
        thal = st.number_input("thal (1-3)", 1, 3, 2)

    submit = st.form_submit_button("Predict")

if submit:
    input_df = pd.DataFrame([{
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }])
    try:
        pred = model.predict(Xs)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    st.write("Predicted target:", int(pred))
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(Xs)[0]
        st.write(f"Probability target=1: {proba[1]:.4f}")
    elif hasattr(model, "decision_function"):
        score = model.decision_function(Xs)[0]
        st.write(f"Decision score: {score:.4f}")
    else:
        st.write("Probability/score not available for this model.")

