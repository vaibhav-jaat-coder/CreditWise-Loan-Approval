import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from train_model import train

st.set_page_config(page_title="CreditWise | Loan Approval", page_icon="💳", layout="wide")
ARTIFACT_PATH = os.path.join("artifacts", "loan_model.joblib")

@st.cache_resource
def load_artifact():
    if not os.path.exists(ARTIFACT_PATH):
        train()
    return joblib.load(ARTIFACT_PATH)

try:
    artifact = load_artifact()
except Exception as exc:
    st.error(f"Unable to load/train the model: {exc}")
    st.info("Keep loan_approval_data.csv in the same folder as app.py, then run: python train_model.py")
    st.stop()

st.markdown("""
<style>
.stApp {background:#f5f8fc;}
.block-container {max-width:1150px;padding-top:2rem;}
.hero {background:linear-gradient(115deg,#102a43,#2563a6);padding:26px 30px;border-radius:18px;color:white;margin-bottom:20px;}
.hero h1 {color:white;margin:0;} .hero p {color:#e5efff;margin:6px 0 0;}
div[data-testid="stForm"] {background:white;padding:20px;border:1px solid #e5eaf1;border-radius:16px;}
</style>
<div class="hero"><h1>💳 CreditWise</h1><p>Loan Approval Prediction System · ML-powered estimate</p></div>
""", unsafe_allow_html=True)

metrics = artifact["metrics"]
a,b,c,d = st.columns(4)
a.metric("Test accuracy", f"{metrics['accuracy']:.1%}")
b.metric("Test precision", f"{metrics['precision']:.1%}")
c.metric("Test recall", f"{metrics['recall']:.1%}")
d.metric("Test F1 score", f"{metrics['f1']:.1%}")
st.caption(f"Evaluation on {metrics['rows']:,} records using an 80/20 train-test split. These are dataset test metrics, not a guarantee of real-world performance.")

opts = artifact["category_options"]
with st.form("loan_form"):
    st.subheader("Applicant details")
    c1,c2,c3 = st.columns(3)
    with c1:
        employment = st.selectbox("Employment status", opts["Employment_Status"])
        age = st.number_input("Age (years)", 18, 100, 30)
        marital = st.selectbox("Marital status", opts["Marital_Status"])
        dependents = st.number_input("Dependents", 0, 20, 0)
        gender = st.selectbox("Gender", opts["Gender"])
    with c2:
        income = st.number_input("Applicant income", min_value=0.0, value=30000.0, step=1000.0)
        co_income = st.number_input("Co-applicant income", min_value=0.0, value=0.0, step=1000.0)
        education = st.selectbox("Education level", opts["Education_Level"])
        employer = st.selectbox("Employer category", opts["Employer_Category"])
        savings = st.number_input("Savings", min_value=0.0, value=20000.0, step=1000.0)
    with c3:
        credit = st.number_input("Credit score", min_value=0.0, max_value=1000.0, value=700.0)
        loans = st.number_input("Existing loans", 0, 30, 0)
        dti = st.number_input("DTI ratio", min_value=0.0, max_value=1.0, value=0.30, step=0.01,
                              help="Enter as a decimal: 0.30 means 30%.")
        collateral = st.number_input("Collateral value", min_value=0.0, value=0.0, step=1000.0)
        area = st.selectbox("Property area", opts["Property_Area"])
    st.divider()
    st.subheader("Loan details")
    x,y,z = st.columns(3)
    with x: loan_amount = st.number_input("Loan amount", min_value=0.0, value=100000.0, step=5000.0)
    with y: term = st.number_input("Loan term (months)", 1, 600, 120)
    with z: purpose = st.selectbox("Loan purpose", opts["Loan_Purpose"])
    submitted = st.form_submit_button("Predict loan approval", type="primary", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        "Applicant_ID": 0, "Applicant_Income": income, "Coapplicant_Income": co_income,
        "Employment_Status": employment, "Age": age, "Marital_Status": marital,
        "Dependents": dependents, "Credit_Score": credit, "Existing_Loans": loans,
        "DTI_Ratio": dti, "Savings": savings, "Collateral_Value": collateral,
        "Loan_Amount": loan_amount, "Loan_Term": term, "Loan_Purpose": purpose,
        "Property_Area": area, "Education_Level": education, "Gender": gender,
        "Employer_Category": employer
    }])

    row[artifact["numeric_cols"]] = artifact["numeric_imputer"].transform(row[artifact["numeric_cols"]])
    row[artifact["cat_feature_cols"]] = artifact["cat_imputer"].transform(row[artifact["cat_feature_cols"]])
    row["Education_Level"] = artifact["education_encoder"].transform(row["Education_Level"].astype(str))
    encoded = artifact["onehot"].transform(row[artifact["ohe_cols"]].astype(str))
    encoded_df = pd.DataFrame(encoded, columns=artifact["onehot"].get_feature_names_out(artifact["ohe_cols"]), index=row.index)
    prepared = pd.concat([row.drop(columns=artifact["ohe_cols"]), encoded_df], axis=1)
    prepared["DTI_Ratio_sq"] = prepared["DTI_Ratio"] ** 2
    prepared["Credit_Score_sq"] = prepared["Credit_Score"] ** 2
    prepared["Coapplicant_Income_log"] = np.log1p(prepared["Coapplicant_Income"])
    prepared = prepared.drop(columns=["Applicant_ID", "DTI_Ratio", "Credit_Score", "Coapplicant_Income"])
    prepared = prepared.reindex(columns=artifact["feature_columns"], fill_value=0)
    scaled = artifact["scaler"].transform(prepared)
    prediction = int(artifact["model"].predict(scaled)[0])
    label = str(artifact["target_encoder"].inverse_transform([prediction])[0])

    st.divider()
    st.subheader("Prediction result")
    if label.lower() == "yes":
        st.success("Loan status predicted: APPROVED")
    else:
        st.warning("Loan status predicted: NOT APPROVED")
    st.write(f"Model output: **{label}**")
    if hasattr(artifact["model"], "predict_proba"):
        probs = artifact["model"].predict_proba(scaled)[0]
        prob_data = {str(k): float(v) for k,v in zip(artifact["target_encoder"].classes_, probs)}
        st.write("Model class probabilities")
        st.write({k: f"{v:.1%}" for k,v in prob_data.items()})
    st.info("Academic demonstration only. This prediction is not a bank decision, credit assessment, or loan offer. Do not use it as the sole basis for a financial decision.")

with st.expander("About CreditWise"):
    st.write("This app uses Gaussian Naive Bayes, as in the final model section of the supplied notebook. It applies mean/mode imputation, label encoding for Education_Level and Loan_Approved, one-hot encoding, the notebook's feature engineering, and StandardScaler.")
