# CreditWise — Loan Approval Prediction

## Files
- `app.py`: Streamlit interactive user interface
- `train_model.py`: training/preprocessing pipeline
- `loan_approval_data.csv`: dataset supplied for this project
- `requirements.txt`: Python dependencies

## Run locally
Open Anaconda Prompt/terminal in this folder and run:

```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

Open the local URL shown in the terminal (usually http://localhost:8501).

The training script uses the same dataset columns and final Gaussian Naive Bayes feature workflow as the supplied notebook. It creates the `artifacts/` folder automatically.

## Deploy
Push the project to GitHub and deploy `app.py` with Streamlit Community Cloud. For a public repository, use only synthetic/anonymized data; do not publish real applicants' personal or financial records.

## Important
This is an educational project. Model predictions and test metrics are not validated for real-world credit decisions.
