import os
import streamlit as st
import joblib
import pandas as pd

st.set_page_config(page_title="HEART DISEASE PREDICTION")
st.title("HEART DISEASE PREDICTION")

DRIVE_IDS = {
    "LOGISTIC REGRESSION": "1ktl6rV0aW9iRx2vjTSXJ7BwKvk0LnPQu",
    "SVM": "1tyIX057hxBbljSKWZP9tH9sb120R16o9",
    "RANDOM FOREST": "1KZU765YHFMp5quHlw5clqtCCJtvgclZn",
}
LOCAL = {"LOGISTIC REGRESSION": "lr.h5", "SVM": "svm.h5", "RANDOM FOREST": "rf.h5"}
FEATURES = ['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']

# require gdown
try:
    import gdown
except Exception:
    st.error("Install gdown: pip install gdown")
    st.stop()

os.makedirs("models", exist_ok=True)

@st.cache_resource
def download_and_load(fid, localname):
    url = f"https://drive.google.com/uc?id={fid}"
    path = os.path.join("models", localname)
    if not os.path.exists(path) or os.path.getsize(path) < 1024:
        gdown.download(url, path, quiet=True)
    try:
        return joblib.load(path)
    except Exception:
        return None

# load models
models = {}
for name, fid in DRIVE_IDS.items():
    m = download_and_load(fid, LOCAL[name])
    if m:
        models[name] = m

if not models:
    st.error("No models loaded. Check Drive IDs and that files are sklearn joblib dumps.")
    st.stop()

model_name = st.selectbox("Choose model", list(models.keys()))
model = models[model_name]

with st.form("f"):
    c1,c2,c3 = st.columns(3)
    with c1:
        age = st.number_input("age", 0, 120, 50)
        sex = st.selectbox("sex (1=male,0=female)", [1,0])
        cp = st.number_input("cp (0-3)", 0, 3, 1)
        trestbps = st.number_input("trestbps", 0, 400, 130)
        chol = st.number_input("chol", 0, 2000, 250)
    with c2:
        fbs = st.selectbox("fbs (0/1)", [0,1])
        restecg = st.number_input("restecg (0-2)", 0, 2, 1)
        thalach = st.number_input("thalach(max heart rate)", 0, 400, 150)
        exang = st.selectbox("exang (0/1)", [0,1])
    with c3:
        oldpeak = st.number_input("oldpeak", 0.0, 20.0, 1.0, step=0.1, format="%.1f")
        slope = st.number_input("slope (0-2)", 0, 2, 2)
        ca = st.number_input("ca (0-3)", 0, 3, 0)
        thal = st.number_input("thal (1-3)", 1, 3, 2)

    submit = st.form_submit_button("Predict")

if submit:
    row = {'age':age,'sex':sex,'cp':cp,'trestbps':trestbps,'chol':chol,'fbs':fbs,'restecg':restecg,
           'thalach':thalach,'exang':exang,'oldpeak':oldpeak,'slope':slope,'ca':ca,'thal':thal}
    X = pd.DataFrame([row], columns=FEATURES).apply(pd.to_numeric, errors='coerce')
    try:
        p = model.predict(X)[0]
        st.write("Predicted target:", int(p))
    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.write("Note:") 
st.write("Target 0 - Lower chance of heart attack.") 
st.write("Target 1 - Higher chance of heart attack.")



