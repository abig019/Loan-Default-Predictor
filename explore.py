import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('loan_data.csv')

print("── Shape ───────────────────────────")
print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print()

# ── CHECK MISSING VALUES (NEW SKILL!) ─────────────────────
print("── Missing Values ──────────────────")
missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
missing_report = pd.DataFrame({
    'Missing Count' : missing,
    'Missing %'     : missing_pct
})
print(missing_report[missing_report['Missing Count'] > 0])
# WHY check percentage? 5% missing is fine to fill
# 50% missing means that column is useless — drop it
print()

print("── Default distribution ────────────")
print(df['default'].value_counts())
print()

print("── Numeric summary ─────────────────")
print(df.describe().round(2))
print()

# Chart 1 — Default distribution
plt.figure(figsize=(6,4))
df['default'].value_counts().plot(kind='bar',
    color=['#27ae60','#e74c3c'], edgecolor='white')
plt.xticks([0,1], ['Will Repay','Will Default'], rotation=0)
plt.title('Loan Default Distribution')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('default_distribution.png')
plt.show()
print("✅ Chart 1 saved")

# Chart 2 — Credit score by default
plt.figure(figsize=(7,5))
df_clean = df.dropna(subset=['credit_score'])
# WHY dropna here? seaborn cannot plot NaN values
# dropna(subset=['credit_score']) drops only rows
# where credit_score is missing — not all missing rows
sns.boxplot(data=df_clean, x='default', y='credit_score',
            hue='default',
            palette={0:'#27ae60', 1:'#e74c3c'},
            legend=False)
plt.xticks([0,1], ['Will Repay','Will Default'])
plt.title('Credit Score by Default Status')
plt.tight_layout()
plt.savefig('credit_score_boxplot.png')
plt.show()
print("✅ Chart 2 saved")

# Chart 3 — Income vs loan amount scatter
plt.figure(figsize=(7,5))
df_clean2 = df.dropna(subset=['income'])
colors = df_clean2['default'].map({0:'#27ae60', 1:'#e74c3c'})
plt.scatter(df_clean2['income'], df_clean2['loan_amount'],
            c=colors, alpha=0.5)
plt.xlabel('Income')
plt.ylabel('Loan Amount')
plt.title('Income vs Loan Amount (Green=Repay, Red=Default)')
plt.tight_layout()
plt.savefig('income_vs_loan.png')
plt.show()
print("✅ Chart 3 saved")

# Chart 4 — Loan purpose distribution
plt.figure(figsize=(7,4))
df.groupby(['loan_purpose','default']).size().unstack().plot(
    kind='bar', color=['#27ae60','#e74c3c'], edgecolor='white')
plt.title('Default Rate by Loan Purpose')
plt.xlabel('Loan Purpose')
plt.xticks(rotation=0)
plt.legend(['Will Repay','Will Default'])
plt.tight_layout()
plt.savefig('loan_purpose.png')
plt.show()
print("✅ Chart 4 saved")