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

st.markdown(f"Models directory (default): `{MODEL_DIR}`")

def try_load(path):
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        return None

with st.expander("Load models from path"):
    st.write("Expected files:")
    st.write(f"- {LR_FILE}")
    st.write(f"- {SVM_FILE}")
    st.write(f"- {RF_FILE}")

lr = try_load(LR_FILE)
svm = try_load(SVM_FILE)
rf = try_load(RF_FILE)

if not all([lr, svm, rf]):
    st.warning("One or more models failed to load. Place them in the path above (joblib .h5 files).")
    st.stop()
st.success("Models loaded successfully.")

# Feature order expected
FEATURES = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

# Optional: load dataset to compute evaluation scores
# st.markdown("### Evaluation dataset (optional)")
# uploaded = st.file_uploader("Upload CSV with same columns (target column 'target')", type=['csv'])
# df = None
# if uploaded is not None:
#    df = pd.read_csv(uploaded)
# else:
#    if os.path.exists(DEFAULT_DATA_CSV):
#       try:
#            df = pd.read_csv(DEFAULT_DATA_CSV)
#            st.info(f"Loaded default dataset from {DEFAULT_DATA_CSV}")
#        except Exception as e:
#            st.warning(f"Could not load default CSV: {e}")

# scores = {}
# if df is not None:
#    if 'target' not in df.columns:
#        st.error("Dataset must include a 'target' column.")
#    else:
#        missing = [c for c in FEATURES if c not in df.columns]
#        if missing:
#            st.error(f"Dataset is missing required columns: {missing}")
#        else:
#            X = df[FEATURES]
#            y = df['target']
#            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique())>1 else None)
#            X_test_scaled = scaler.transform(X_test)

#            def metrics(model, Xs, ys):
#                yp = model.predict(Xs)
#                return {
#                    'accuracy': accuracy_score(ys, yp),
#                    'precision': precision_score(ys, yp, zero_division=0),
#                    'recall': recall_score(ys, yp, zero_division=0),
#                    'f1': f1_score(ys, yp, zero_division=0)
                }

#            scores['Logistic Regression'] = metrics(lr, X_test_scaled, y_test)
#            scores['SVM'] = metrics(svm, X_test_scaled, y_test)
#            scores['Random Forest'] = metrics(rf, X_test_scaled, y_test)

# Model selection
model_choice = st.selectbox("Select model", ["Logistic Regression", "SVM", "Random Forest"])
model_map = {'Logistic Regression': lr, 'SVM': svm, 'Random Forest': rf}
model = model_map[model_choice]

# if scores:
#    st.markdown("#### Evaluation scores (on test split)")
#    sc = scores.get(model_choice)
#    if sc:
#        st.write(f"Accuracy: {sc['accuracy']:.4f}")
#        st.write(f"Precision: {sc['precision']:.4f}")
#        st.write(f"Recall: {sc['recall']:.4f}")
#        st.write(f"F1-score: {sc['f1']:.4f}")

st.markdown("### Predict for a single patient")

with st.form("patient_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input(int("age"))
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

st.markdown("---")
st.write("Notes:")
st.write("- This app expects models saved with joblib in the folder: /content/drive/MyDrive/Saved Models")
st.write("- If you trained SVC and want probabilities, train with probability=True before saving.")
