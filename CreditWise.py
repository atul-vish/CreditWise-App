import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="CreditWise Loan System", page_icon="💳", layout="centered"
)

st.title("💳 CreditWise Loan Approval System")
st.write("Robust & production-safe ML loan approval app")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("clean_dataset.csv")


df = load_data()

# --------------------------------------------------
# BASIC CLEANING
# --------------------------------------------------
df = df.drop(columns=["Applicant_ID"], errors="ignore")
df = df[df["Loan_Approved"].isin(["Yes", "No"])]

# --------------------------------------------------
# SPLIT FEATURES & TARGET
# --------------------------------------------------
y = df["Loan_Approved"].map({"Yes": 1, "No": 0})
X = df.drop(columns=["Loan_Approved"])

num_cols = X.select_dtypes(include="number").columns
cat_cols = X.select_dtypes(include="object").columns

# --------------------------------------------------
# PREPROCESSING (TRAINING)
# --------------------------------------------------
num_imputer = SimpleImputer(strategy="median")
cat_imputer = SimpleImputer(strategy="most_frequent")

X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
X_encoded = ohe.fit_transform(X[cat_cols])

X_encoded_df = pd.DataFrame(
    X_encoded, columns=ohe.get_feature_names_out(cat_cols), index=X.index
)

X_final = pd.concat([X.drop(columns=cat_cols), X_encoded_df], axis=1)

# --------------------------------------------------
# SCALE & TRAIN
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_final)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# Save column order (VERY IMPORTANT)
TRAIN_COLUMNS = X_final.columns

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.header("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Dataset Preview", "Loan Prediction"])

# --------------------------------------------------
# DATASET PAGE
# --------------------------------------------------
if page == "Dataset Preview":
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    st.write("Model trained on", len(TRAIN_COLUMNS), "features")

# --------------------------------------------------
# LOAN PREDICTION PAGE
# --------------------------------------------------
if page == "Loan Prediction":
    st.subheader("📝 Enter Applicant Details")

    with st.form("loan_form"):
        Applicant_Income = st.number_input("Applicant Income", min_value=0)
        Coapplicant_Income = st.number_input("Coapplicant Income", min_value=0)
        Loan_Amount = st.number_input("Loan Amount", min_value=0)
        Loan_Term = st.number_input("Loan Term (months)", min_value=1)

        Gender = st.selectbox("Gender", ["Male", "Female"])
        Marital_Status = st.selectbox("Marital Status", ["Single", "Married"])
        Education_Level = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
        Employment_Status = st.selectbox(
            "Employment Status", ["Salaried", "Self-Employed"]
        )

        submit = st.form_submit_button("Predict Loan Status")

    if submit:
        # Start with EMPTY row having ALL columns
        input_df = pd.DataFrame(np.zeros((1, X.shape[1])), columns=X.columns)

        # Fill only known inputs
        input_df.loc[0, "Applicant_Income"] = Applicant_Income
        input_df.loc[0, "Coapplicant_Income"] = Coapplicant_Income
        input_df.loc[0, "Loan_Amount"] = Loan_Amount
        input_df.loc[0, "Loan_Term"] = Loan_Term
        input_df.loc[0, "Gender"] = Gender
        input_df.loc[0, "Marital_Status"] = Marital_Status
        input_df.loc[0, "Education_Level"] = Education_Level
        input_df.loc[0, "Employment_Status"] = Employment_Status

        # Apply preprocessing
        input_df[num_cols] = num_imputer.transform(input_df[num_cols])
        input_df[cat_cols] = cat_imputer.transform(input_df[cat_cols])

        input_encoded = ohe.transform(input_df[cat_cols])

        input_encoded_df = pd.DataFrame(
            input_encoded,
            columns=ohe.get_feature_names_out(cat_cols),
            index=input_df.index,
        )

        input_final = pd.concat(
            [input_df.drop(columns=cat_cols), input_encoded_df], axis=1
        )

        # Align column order
        input_final = input_final.reindex(columns=TRAIN_COLUMNS, fill_value=0)

        input_scaled = scaler.transform(input_final)
        prediction = model.predict(input_scaled)[0]

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")
