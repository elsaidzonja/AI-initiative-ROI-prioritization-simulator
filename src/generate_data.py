"""
Generate a synthetic customer base for *Kaffeeliebe*, a fictional DACH-region
specialty-coffee subscription retailer.

Design goals
------------
1. Realistic, *noisy* churn signal so a model earns ~0.80-0.85 ROC-AUC
   (believable), never a fake 0.99. Churn is a logistic function of several
   drivers WITH an interaction term and gaussian noise.
2. A per-customer value figure (annual_margin) so the Phase-3 retention ROI
   is grounded in euros.
3. A little realistic missingness (satisfaction, email engagement) to justify
   a light preprocessing step -- deliberately lighter than the SQL project,
   whose focus was cleaning; here the focus is modelling + business value.

One row per customer.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
N = 8000

def z(x):
    x = np.asarray(x, dtype=float)
    return (x - np.nanmean(x)) / np.nanstd(x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# --- Customer attributes -------------------------------------------------
region = rng.choice(["DE", "AT", "CH", "Other"], size=N, p=[0.62, 0.14, 0.16, 0.08])
channel = rng.choice(["Organic", "Paid Search", "Social", "Referral"],
                     size=N, p=[0.34, 0.30, 0.24, 0.12])
tier = rng.choice(["Basic", "Plus", "Premium"], size=N, p=[0.50, 0.35, 0.15])

tenure_months = np.clip(rng.exponential(scale=18, size=N) + 1, 1, 72).round().astype(int)

# Frequency (orders in last 12m): higher tier & longer tenure -> more orders
tier_freq_boost = np.select([tier == "Basic", tier == "Plus", tier == "Premium"],
                            [0.0, 1.5, 3.0])
freq_lambda = np.clip(2.5 + 0.05 * tenure_months + tier_freq_boost, 1, None)
num_orders_12m = rng.poisson(freq_lambda).clip(0, 40)

# Recency: inversely related to frequency, plus noise
recency = np.clip(
    220 - 8 * num_orders_12m + rng.normal(0, 35, N), 1, 365
).round().astype(int)

# Average order value: driven by tier
tier_aov = np.select([tier == "Basic", tier == "Plus", tier == "Premium"],
                     [28, 42, 65])
avg_order_value = np.clip(rng.normal(tier_aov, 8), 8, None).round(2)

# Engagement
email_engagement = np.clip(rng.beta(2.2, 3.0, N), 0, 1)          # open/click rate 0-1
app_logins_30d = rng.poisson(np.clip(4 * email_engagement, 0.2, None)).clip(0, 40)

# Support experience (mostly low, a heavy tail of unhappy customers)
support_tickets_12m = rng.poisson(0.7, N).clip(0, 12)

# Discount dependence: share of orders placed with a discount
discount_rate = np.clip(rng.beta(2.0, 4.0, N), 0, 1)

# Satisfaction (1-5), pulled down by support tickets
satisfaction = np.clip(
    4.4 - 0.35 * support_tickets_12m + rng.normal(0, 0.5, N), 1, 5
).round(1)

# --- Value: expected annual margin (the "value at risk") -----------------
GROSS_MARGIN = 0.38
expected_annual_revenue = avg_order_value * np.clip(num_orders_12m, 1, None)
annual_margin = (expected_annual_revenue * GROSS_MARGIN).round(2)

# --- Churn: logistic model of the drivers (+ interaction + noise) --------
low_eng = (email_engagement < np.median(email_engagement)).astype(float)
tier_churn = np.select([tier == "Basic", tier == "Plus", tier == "Premium"],
                       [0.15, 0.0, -0.35])

logit = (
    -1.75
    + 0.95 * z(recency)
    - 0.75 * z(num_orders_12m)
    - 0.55 * z(email_engagement)
    - 0.35 * z(app_logins_30d)
    + 0.60 * z(support_tickets_12m)
    - 0.65 * z(tenure_months)
    + 0.45 * z(discount_rate)
    + tier_churn
    + 0.30 * z(recency) * low_eng          # recent-inactive AND disengaged = worst
    + rng.normal(0, 0.5, N)                 # irreducible noise
)
p_churn = sigmoid(logit)
churned = rng.binomial(1, p_churn)

df = pd.DataFrame({
    "customer_id": [f"KL{100000 + i}" for i in range(N)],
    "region": region,
    "acquisition_channel": channel,
    "plan_tier": tier,
    "tenure_months": tenure_months,
    "num_orders_12m": num_orders_12m,
    "days_since_last_order": recency,
    "avg_order_value": avg_order_value,
    "email_engagement_rate": email_engagement.round(3),
    "app_logins_30d": app_logins_30d,
    "support_tickets_12m": support_tickets_12m,
    "discount_rate": discount_rate.round(3),
    "satisfaction_score": satisfaction,
    "annual_margin": annual_margin,
    "churned": churned,
})

# --- Inject light, realistic missingness (MCAR) --------------------------
for col, frac in [("satisfaction_score", 0.04), ("email_engagement_rate", 0.03)]:
    idx = rng.choice(N, size=int(frac * N), replace=False)
    df.loc[idx, col] = np.nan

df.to_csv("/mnt/user-data/outputs/customer-retention/data/customers.csv", index=False)

print(f"rows: {len(df):,}")
print(f"churn base rate: {df['churned'].mean():.1%}")
print(f"missing satisfaction: {df['satisfaction_score'].isna().mean():.1%}, "
      f"missing email_eng: {df['email_engagement_rate'].isna().mean():.1%}")
print(f"total annual margin at stake: EUR {df['annual_margin'].sum():,.0f}")
