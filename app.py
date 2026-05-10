import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(
    page_title="LoanGuard — Default Predictor",
    page_icon="🏦",
    layout="wide"
)

# ── FINANCE THEME CSS ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #111827 100%);
}

#MainMenu, footer { visibility: hidden; }

.hero {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f5a623, #e74c3c, #c0392b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: #94a3b8;
    font-size: 0.95rem;
    letter-spacing: 0.08em;
}
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
}
.result-safe {
    background: linear-gradient(135deg, #0d4f2f, #1e8449);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    color: white;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 1rem 0;
}
.result-risk {
    background: linear-gradient(135deg, #5c0a0a, #c0392b);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    color: white;
    font-size: 1.5rem;
    font-weight: 700;
    margin: 1rem 0;
}
.section-label {
    color: #f5a623;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.8rem;
}
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] { color: #e2e8f0 !important; }

[data-testid="stSlider"] label,
.stSelectbox label,
.stNumberInput label { color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ── AUTO TRAIN ────────────────────────────────────────────
if not os.path.exists('model.pkl'):
    st.info("⏳ Training model for first time — please wait...")
    import subprocess
    subprocess.run(['python','train.py'])
    st.rerun()

# ── LOAD MODEL ────────────────────────────────────────────
@st.cache_resource
def load_all():
    with open('model.pkl','rb') as f:
        model = pickle.load(f)
    with open('label_encoder.pkl','rb') as f:
        le = pickle.load(f)
    with open('feature_names.pkl','rb') as f:
        features = pickle.load(f)
    with open('model_name.pkl','rb') as f:
        model_name = pickle.load(f)
    return model, le, features, model_name

model, le, feature_names, model_name = load_all()

# ── HEADER ────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🏦 LoanGuard</div>
    <div class="hero-sub">AI-POWERED LOAN DEFAULT PREDICTION · FINANCE INTELLIGENCE</div>
</div>
""", unsafe_allow_html=True)

c1,c2,c3,c4 = st.columns(4)
c1.metric("🤖 Algorithm",    model_name)
c2.metric("🎯 Output",       "Default / Repay")
c3.metric("📊 Training data","600 borrowers")
c4.metric("🏦 Industry",     "Finance / Banking")
st.divider()

# ── TWO COLUMN LAYOUT ─────────────────────────────────────
left, right = st.columns([1,1], gap="large")

with left:
    st.markdown('<div class="section-label">Borrower Profile</div>',
                unsafe_allow_html=True)

    age   = st.slider("👤 Age", 22, 65, 35)
    income = st.number_input("💰 Annual Income (₹)",
                              min_value=15000, max_value=120000,
                              value=50000, step=1000)
    loan_amount = st.number_input("🏦 Loan Amount (₹)",
                                   min_value=5000, max_value=80000,
                                   value=20000, step=1000)
    credit_score = st.slider("📊 Credit Score", 300, 850, 650)

    st.markdown("---")
    employment_years   = st.slider("💼 Years Employed", 0, 30, 5)
    num_existing_loans = st.slider("📋 Existing Loans",  0,  5, 1)
    loan_purpose = st.selectbox("🎯 Loan Purpose",
        ['Education','Medical','Business','Personal','Home'])

    predict_btn = st.button("🔍 Predict Default Risk",
                             type="primary",
                             use_container_width=True)

with right:
    st.markdown('<div class="section-label">Risk Analysis</div>',
                unsafe_allow_html=True)

    if not predict_btn:
        st.markdown("""
        <div class="card">
            <div style="text-align:center;padding:1.5rem;">
                <div style="font-size:3.5rem;">🏦</div>
                <div style="color:#94a3b8;margin-top:0.8rem;font-size:0.9rem;">
                    Fill in the borrower profile<br>
                    and click <b style="color:#f5a623;">Predict Default Risk</b>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="section-label">How it works</div>
            <div style="color:#cbd5e1;font-size:0.9rem;line-height:1.7;">
                This model was trained on <b>600 borrower profiles</b> using
                <b>XGBoost</b> — the most popular algorithm in Finance AI.<br><br>
                Key factors:<br>
                🔴 Low credit score = high risk<br>
                🔴 High loan vs income = high risk<br>
                🔴 Many existing loans = high risk<br>
                🟢 High income + good credit = safe
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── BUILD INPUT ───────────────────────────────────
        loan_purpose_encoded = int(le.transform([loan_purpose])[0])

        # Feature engineering — SAME as train.py
        debt_to_income = loan_amount / income
        loan_to_credit = loan_amount / credit_score
        high_risk = int(credit_score < 550 and num_existing_loans >= 3)

        input_dict = {
            'age'               : age,
            'income'            : income,
            'loan_amount'       : loan_amount,
            'credit_score'      : credit_score,
            'employment_years'  : employment_years,
            'num_existing_loans': num_existing_loans,
            'loan_purpose'      : loan_purpose_encoded,
            'debt_to_income'    : debt_to_income,
            'loan_to_credit'    : loan_to_credit,
            'high_risk'         : high_risk,
        }

        input_df = pd.DataFrame([input_dict])[feature_names]

        prediction  = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        default_pct = round(float(probability[1]) * 100, 1)
        repay_pct   = round(float(probability[0]) * 100, 1)

        # ── RESULT ────────────────────────────────────────
        if prediction == 0:
            st.markdown(
                '<div class="result-safe">✅ LIKELY TO REPAY</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="result-risk">⚠️ HIGH DEFAULT RISK</div>',
                unsafe_allow_html=True)

        # Metrics
        r1,r2 = st.columns(2)
        r1.metric("✅ Repay Probability",   f"{repay_pct}%")
        r2.metric("⚠️ Default Probability", f"{default_pct}%")

        # Risk bar
        bar_color = "#27ae60" if prediction==0 else "#e74c3c"
        fill_pct  = repay_pct if prediction==0 else default_pct
        st.markdown(f"""
        <div style="background:rgba(255,255,255,0.1);border-radius:8px;
                    height:14px;margin:8px 0 16px;">
            <div style="background:{bar_color};width:{fill_pct}%;
                        height:14px;border-radius:8px;"></div>
        </div>""", unsafe_allow_html=True)

        st.divider()

        # ── FEATURE VALUES ────────────────────────────────
        st.markdown('<div class="section-label">Profile Breakdown</div>',
                    unsafe_allow_html=True)

        dti = round(debt_to_income, 3)
        dti_color = "#27ae60" if dti < 0.4 else ("#f5a623" if dti < 0.8 else "#e74c3c")
        cs_color  = "#27ae60" if credit_score >= 700 else ("#f5a623" if credit_score >= 550 else "#e74c3c")

        items = [
            ("💰 Annual Income",    f"₹{income:,.0f}",      "#27ae60"),
            ("🏦 Loan Amount",      f"₹{loan_amount:,.0f}", "#e2e8f0"),
            ("📊 Credit Score",     str(int(credit_score)), cs_color),
            ("📈 Debt-to-Income",   str(dti),               dti_color),
            ("💼 Employment Years", str(employment_years),  "#e2e8f0"),
            ("📋 Existing Loans",   str(num_existing_loans),"#e2e8f0"),
        ]
        for label, val, col in items:
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;
                        color:#cbd5e1;font-size:0.88rem;
                        padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.05);">
                <span>{label}</span>
                <span style="color:{col};font-weight:600;">{val}</span>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # ── TIPS ──────────────────────────────────────────
        st.markdown('<div class="section-label">Recommendations</div>',
                    unsafe_allow_html=True)
        tips = []
        if credit_score < 600:
            tips.append(("🔴", "Credit score below 600 is the biggest red flag. Pay existing debts on time to improve it."))
        if debt_to_income > 0.5:
            tips.append(("🔴", f"Debt-to-income ratio is {dti} — loan is more than 50% of annual income. Very risky."))
        if num_existing_loans >= 3:
            tips.append(("🟡", "3 or more existing loans increases default risk significantly."))
        if employment_years < 2:
            tips.append(("🟡", "Less than 2 years of employment suggests income instability."))
        if income < 25000:
            tips.append(("🔴", "Very low income relative to the loan amount requested."))

        if tips:
            for icon, tip in tips:
                st.markdown(f"""
                <div style="background:rgba(231,76,60,0.1);border-left:3px solid #e74c3c;
                            border-radius:6px;padding:8px 12px;margin-bottom:6px;
                            color:#e2e8f0;font-size:0.88rem;">
                    {icon} {tip}
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(39,174,96,0.1);border-left:3px solid #27ae60;
                        border-radius:6px;padding:8px 12px;color:#e2e8f0;font-size:0.88rem;">
                🟢 Strong borrower profile. Low default risk indicators.
            </div>""", unsafe_allow_html=True)

# ── FEATURE IMPORTANCE ────────────────────────────────────
st.divider()
st.markdown('<div class="section-label" style="text-align:center;">What Affects Default Risk Most?</div>',
            unsafe_allow_html=True)
imp_df = pd.DataFrame({
    'Factor'    : feature_names,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
st.bar_chart(imp_df.set_index('Factor'), color="#e74c3c")
st.caption(f"Model: {model_name} · 600 borrowers · Binary classification · Finance AI")