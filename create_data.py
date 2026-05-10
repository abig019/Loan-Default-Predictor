import pandas as pd
import numpy as np

np.random.seed(42)
n = 600

age = np.random.randint(22, 65, n)

income = np.random.randint(15000, 120000, n).astype(float)

loan_amount = np.random.randint(5000, 80000, n).astype(float)

credit_score = np.random.randint(300, 850, n).astype(float)

employment_years = np.random.randint(0, 30, n).astype(float)

num_existing_loans = np.random.randint(0, 6, n)

loan_purpose = np.random.choice(
    ['Education', 'Medical', 'Business', 'Personal', 'Home'], n)

missing_income_idx = np.random.choice(n, size=int(n*0.08), replace=False)
income[missing_income_idx] = np.nan

# 6% of credit_score values will be missing
missing_credit_idx = np.random.choice(n, size=int(n*0.06), replace=False)
credit_score[missing_credit_idx] = np.nan

# 5% of employment_years will be missing
missing_emp_idx = np.random.choice(n, size=int(n*0.05), replace=False)
employment_years[missing_emp_idx] = np.nan

default = []
for i in range(n):
    risk = 0

    inc = income[i] if not np.isnan(income[i]) else 40000

    if inc < 30000:
        risk += 3
    elif inc < 60000:
        risk += 1

    # High loan amount = higher risk
    if loan_amount[i] > 50000:
        risk += 2
    elif loan_amount[i] > 30000:
        risk += 1

    # Low credit score = higher risk
    cs = credit_score[i] if not np.isnan(credit_score[i]) else 600
    if cs < 500:
        risk += 3
    elif cs < 650:
        risk += 2

    # Short employment = higher risk
    ey = employment_years[i] if not np.isnan(employment_years[i]) else 5
    if ey < 2:
        risk += 2
    elif ey < 5:
        risk += 1

     # Many existing loans = higher risk
    if num_existing_loans[i] >= 4:
        risk += 2
    elif num_existing_loans[i] >= 2:
        risk += 1

    # Add randomness to make it realistic
    risk += np.random.randint(-1, 2)

    # If total risk score is high — they will default
    default.append(1 if risk >= 5 else 0)
    # 1 = Default (will not repay)
    # 0 = Repay (will repay)

default = np.array(default)

# ── BUILD TABLE ───────────────────────────────────────────
df = pd.DataFrame({
    'age'               : age,
    'income'            : income,
    'loan_amount'       : loan_amount,
    'credit_score'      : credit_score,
    'employment_years'  : employment_years,
    'num_existing_loans': num_existing_loans,
    'loan_purpose'      : loan_purpose,
    'default'           : default
})

df.to_csv('loan_data.csv', index=False)

print("✅ Dataset created!")
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print()
print("Default distribution:")
print(df['default'].value_counts())
print()
print("Missing values:")
print(df.isnull().sum())
print()
print(df.head())