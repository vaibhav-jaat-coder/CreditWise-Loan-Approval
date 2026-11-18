import os, json, joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_PATH = os.getenv("LOAN_DATA_PATH", "loan_approval_data.csv")
ARTIFACT_DIR = "artifacts"

OHE_COLS = ["Employment_Status", "Marital_Status", "Loan_Purpose",
            "Property_Area", "Gender", "Employer_Category"]
NUMERIC_COLS = ["Applicant_ID", "Applicant_Income", "Coapplicant_Income", "Age",
                "Dependents", "Credit_Score", "Existing_Loans", "DTI_Ratio",
                "Savings", "Collateral_Value", "Loan_Amount", "Loan_Term"]
CAT_FEATURE_COLS = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                    "Property_Area", "Education_Level", "Gender", "Employer_Category"]
REQUIRED = NUMERIC_COLS + CAT_FEATURE_COLS + ["Loan_Approved"]

def train():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}. Keep loan_approval_data.csv beside app.py.")
    df = pd.read_csv(DATA_PATH)
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError("CSV is missing columns: " + ", ".join(missing))
    df = df[REQUIRED].copy()

    # Same approach as the notebook: mean for numeric values, mode for categories.
    numeric_imputer = SimpleImputer(strategy="mean")
    df[NUMERIC_COLS] = numeric_imputer.fit_transform(df[NUMERIC_COLS])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[CAT_FEATURE_COLS] = cat_imputer.fit_transform(df[CAT_FEATURE_COLS])
    target_imputer = SimpleImputer(strategy="most_frequent")
    df[["Loan_Approved"]] = target_imputer.fit_transform(df[["Loan_Approved"]])

    education_encoder = LabelEncoder()
    df["Education_Level"] = education_encoder.fit_transform(df["Education_Level"].astype(str))
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(df["Loan_Approved"].astype(str))

    onehot = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded = onehot.fit_transform(df[OHE_COLS].astype(str))
    encoded_df = pd.DataFrame(encoded, columns=onehot.get_feature_names_out(OHE_COLS), index=df.index)
    model_df = pd.concat([df.drop(columns=OHE_COLS + ["Loan_Approved"]), encoded_df], axis=1)

    # Exact feature engineering and feature drops used in the notebook.
    model_df["DTI_Ratio_sq"] = model_df["DTI_Ratio"] ** 2
    model_df["Credit_Score_sq"] = model_df["Credit_Score"] ** 2
    model_df["Coapplicant_Income_log"] = np.log1p(model_df["Coapplicant_Income"])
    X = model_df.drop(columns=["Applicant_ID", "DTI_Ratio", "Credit_Score", "Coapplicant_Income"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = GaussianNB()
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "rows": int(len(df))
    }

    artifact = {
        "numeric_imputer": numeric_imputer, "cat_imputer": cat_imputer,
        "target_imputer": target_imputer, "education_encoder": education_encoder,
        "target_encoder": target_encoder, "onehot": onehot, "scaler": scaler,
        "model": model, "feature_columns": list(X.columns),
        "numeric_cols": NUMERIC_COLS, "cat_feature_cols": CAT_FEATURE_COLS,
        "ohe_cols": OHE_COLS, "metrics": metrics,
        "category_options": {c: sorted(df[c].astype(str).unique().tolist()) for c in CAT_FEATURE_COLS}
    }
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(artifact, os.path.join(ARTIFACT_DIR, "loan_model.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("Model saved. Test metrics:", metrics)

if __name__ == "__main__":
    train()
