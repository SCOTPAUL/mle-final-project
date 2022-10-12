from operator import mod
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st. set_page_config(layout="wide") 

st.write("""
Columns:\n\n

\ncredit.policy: 1 if the customer meets the credit underwriting criteria of LendingClub.com, and 0 otherwise.
\npurpose: The purpose of the loan (takes values "creditcard", "debtconsolidation", "educational", "majorpurchase", "smallbusiness", and "all_other").
\nint.rate: The interest rate of the loan, as a proportion (a rate of 11% would be stored as 0.11). Borrowers judged by LendingClub.com to be more risky are assigned higher interest rates.
\ninstallment: The monthly installments owed by the borrower if the loan is funded.
\nlog.annual.inc: The natural log of the self-reported annual income of the borrower.
\ndti: The debt-to-income ratio of the borrower (amount of debt divided by annual income).
\nfico: The FICO credit score of the borrower.
\ndays.with.cr.line: The number of days the borrower has had a credit line.
\nrevol.bal: The borrower's revolving balance (amount unpaid at the end of the credit card billing cycle).
\nrevol.util: The borrower's revolving line utilization rate (the amount of the credit line used relative to total credit available).
\ninq.last.6mths: The borrower's number of inquiries by creditors in the last 6 months.
\ndelinq.2yrs: The number of times the borrower had been 30+ days past due on a payment in the past 2 years.
\npub.rec: The borrower's number of derogatory public records (bankruptcy filings, tax liens, or judgments).""")

df = pd.read_csv("loan_data.csv")

df["annual.inc"] = np.exp(df["log.annual.inc"])
df = df.drop(columns=["log.annual.inc"])

df.columns = df.columns.str.lower().str.replace(' ', '_') # replacing any potential spaces in a column header
df.columns = df.columns.str.lower().str.replace('.', '_') # replacing period to avoid confusion

#df

df_without_target = df.drop(columns="not_fully_paid")
df_with_dummies = pd.get_dummies(df_without_target)

# credit_policy
# purpose
# int_rate
# installment
# dti
# fico
# days_with_cr_line
# revol_bal
# revol_util
# inq_last_6mths
# delinq_2yrs
# pub_rec
# annual_inc
# not_fully_paid

cols = st.columns(7)

credit_policy = cols[0].number_input("credit_policy", max_value=1, min_value=0, step=1, key="credit_policy", value=1)
purpose = cols[1].selectbox("purpose", options=["all_other", "credit_card", "debt_consolidation", "educational", "home_improvement", "major_purchase", "small_business"], index=2, key="purpose")
int_rate = cols[2].number_input("int_rate", max_value=1., min_value=0., key="int_rate", value=0.1496)
installment = cols[3].number_input("installment", min_value=0., key="installment", value=194.0200)
dti = cols[4].number_input("dti", min_value=0., key="dti", value=4.0000)
fico = cols[5].number_input("fico", min_value=0, key="fico", value=667)
days_with_cr_line = cols[6].number_input("days_with_cr_line", min_value=0., key="days_with_cr_line", value=3180.0417)

cols2 = st.columns(6)
revol_bal = cols2[0].number_input("revol_bal", min_value=0, key="revol_bal", value=3839)
revol_util = cols2[1].number_input("revol_util", min_value=0., key="revol_util", value=76.8000)
inq_last_6mths = cols2[2].number_input("inq_last_6mths", min_value=0, key="inq_last_6mths", value=0)
delinq_2yrs = cols2[3].number_input("delinq_2yrs", min_value=0, key="delinq_2yrs", value=0)
pub_rec = cols2[4].number_input("pub_rec", min_value=0, key="pub_rec", value=1)
annual_inc = cols2[5].number_input("annual_inc", min_value=0., key="annual_inc", value=45000.0001)

# for i, (col, col_name) in enumerate(zip(cols, col_names)):

#     col.text_input(f"{col_name}", key=f"var_inp_{i}")

#st.session_state

#1	debt_consolidation	0.1496	194.0200	4.0000	667	3,180.0417	3839	76.8000	0	0	1	1	45,000.0001

input_df = pd.DataFrame([[
    st.session_state.credit_policy,
    st.session_state.purpose,
    st.session_state.int_rate,
    st.session_state.installment,
    st.session_state.dti,
    st.session_state.fico, 
    st.session_state.days_with_cr_line,
    st.session_state.revol_bal,
    st.session_state.revol_util,
    st.session_state.inq_last_6mths,
    st.session_state.delinq_2yrs,
    st.session_state.pub_rec,
    st.session_state.annual_inc]], columns=df_without_target.columns)

input_df = pd.get_dummies(input_df)

missing_cols = list(set(df_with_dummies) - set(input_df))

for missing_col in missing_cols:
    input_df[missing_col] = 0

#input_df


with open("xgb.bin", "rb") as model_file:
    clf = pickle.load(model_file)
    f"Prediction: {clf.predict_proba(input_df)[0][1] * 100:.2f}% chance of default"
