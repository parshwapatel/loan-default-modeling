import gradio as gr
import pandas as pd
import joblib
import os

# 1. Load the model
model_path = os.path.join("exported_model", "model.pkl")
model = joblib.load(model_path)

# 2. Prediction function
def predict(
    loan_purpose, loan_type,
    loan_tenure_months,
    number_of_open_accounts,
    enquiry_count,
    credit_utilization_ratio,
    loan_to_income,
    delinquency_ratio,
    avg_dpd_per_delinquency
):
    # Build feature dict
    row = {
        "loan_tenure_months": loan_tenure_months,
        "number_of_open_accounts": number_of_open_accounts,
        "enquiry_count": enquiry_count,
        "credit_utilization_ratio": credit_utilization_ratio,
        "loan_to_income": loan_to_income,
        "delinquency_ratio": delinquency_ratio,
        "avg_dpd_per_delinquency": avg_dpd_per_delinquency,
        # One-hot encode
        "loan_purpose_Home": 1 if loan_purpose=="Home" else 0,
        "loan_purpose_Education": 1 if loan_purpose=="Education" else 0,
        "loan_purpose_Personal": 1 if loan_purpose=="Personal" else 0,
        "loan_type_Unsecured": 1 if loan_type=="Unsecured" else 0,
    }
    df = pd.DataFrame([row])
    pred = model.predict(df)[0]
    return "Approved" if pred == 0 else "Default"

# 3. Build Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Radio(["Home","Education","Personal","Auto"], label="Loan Purpose"),
        gr.Radio(["Unsecured","Secured"], label="Loan Type"),
        gr.Number(label="Loan Tenure (months)", value=12),
        gr.Number(label="Number of Open Accounts", value=2),
        gr.Number(label="Enquiry Count", value=1),
        gr.Number(label="Credit Utilization Ratio", value=0.3),
        gr.Number(label="Loan to Income Ratio", value=1.0),
        gr.Number(label="Delinquency Ratio", value=0.1),
        gr.Number(label="Avg DPD per Delinquency", value=5.0),
    ],
    outputs="text",
    title="Loan Approval Predictor",
    description="Enter the borrower’s features, then click Predict."
)

if __name__ == "__main__":
    iface.launch()
