"""
Phase 1 -- EDA & RFM segmentation for the Kaffeeliebe customer base.

RFM = Recency, Frequency, Monetary. Each customer is scored 1-5 on each
dimension (quintiles), then mapped to a named behavioural segment. The point
is to see where *value* and *churn risk* concentrate BEFORE modelling --
segments like "At Risk" and "Can't Lose Them" are exactly who a retention
budget should defend.

Outputs
-------
data/customer_segments.csv     customer_id + R/F/M scores + segment
reports/figures/*.png          segment value & churn charts
(prints a segment summary table to stdout)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = "/mnt/user-data/outputs/customer-retention"
df = pd.read_csv(f"{BASE}/data/customers.csv")

# ----------------------------------------------------------------- EDA -----
print("=" * 60)
print("EDA SUMMARY")
print("=" * 60)
print(f"customers: {len(df):,}   |   churn rate: {df['churned'].mean():.1%}")
print("\nchurn rate by plan tier:")
print(df.groupby("plan_tier")["churned"].mean().round(3).to_string())
print("\nchurn rate by acquisition channel:")
print(df.groupby("acquisition_channel")["churned"].mean().round(3).sort_values(ascending=False).to_string())

# --------------------------------------------------------- RFM scoring -----
# Recency: fewer days since last order is better -> reverse the quintile.
df["R"] = pd.qcut(df["days_since_last_order"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
# Frequency & Monetary: higher is better. rank() breaks ties so qcut won't fail.
df["F"] = pd.qcut(df["num_orders_12m"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
df["M"] = pd.qcut(df["annual_margin"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

# Map the (R,F) grid to named segments -- the widely used RFM segmentation map.
seg_map = {
    r"[1-2][1-2]": "Hibernating",
    r"[1-2][3-4]": "At Risk",
    r"[1-2]5":     "Can't Lose Them",
    r"3[1-2]":     "About to Sleep",
    r"33":         "Needs Attention",
    r"[3-4][4-5]": "Loyal Customers",
    r"41":         "Promising",
    r"51":         "New Customers",
    r"[4-5][2-3]": "Potential Loyalists",
    r"5[4-5]":     "Champions",
}
rf = df["R"].astype(str) + df["F"].astype(str)
df["segment"] = rf.replace(seg_map, regex=True)

# ------------------------------------------------- segment summary ---------
summary = (df.groupby("segment")
             .agg(customers=("customer_id", "size"),
                  avg_recency=("days_since_last_order", "mean"),
                  avg_frequency=("num_orders_12m", "mean"),
                  avg_margin=("annual_margin", "mean"),
                  total_margin=("annual_margin", "sum"),
                  churn_rate=("churned", "mean"))
             .sort_values("total_margin", ascending=False)
             .round(2))
summary["pct_base"] = (summary["customers"] / len(df) * 100).round(1)

print("\n" + "=" * 60)
print("SEGMENT SUMMARY (sorted by total annual margin)")
print("=" * 60)
print(summary[["customers", "pct_base", "avg_frequency", "avg_margin",
               "total_margin", "churn_rate"]].to_string())

# margin sitting in high-churn segments = the retention opportunity
at_risk_segs = ["At Risk", "Can't Lose Them", "About to Sleep", "Needs Attention"]
opp = summary.loc[summary.index.isin(at_risk_segs), "total_margin"].sum()
print(f"\nAnnual margin in high-risk segments: EUR {opp:,.0f} "
      f"({opp / summary['total_margin'].sum():.0%} of total)")

df.to_csv(f"{BASE}/data/customer_segments.csv", index=False)

# ------------------------------------------------------- figures -----------
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# 1) segment value vs churn -- the "where to defend" chart
fig, ax = plt.subplots(figsize=(9, 5))
s = summary.sort_values("total_margin")
colors = plt.cm.RdYlGn_r(s["churn_rate"] / s["churn_rate"].max())
ax.barh(s.index, s["total_margin"], color=colors)
ax.set_xlabel("Total annual margin (EUR)")
ax.set_title("Where value & churn risk concentrate\n(bar length = margin, redder = higher churn)")
sm = plt.cm.ScalarMappable(cmap="RdYlGn_r",
                           norm=plt.Normalize(s["churn_rate"].min(), s["churn_rate"].max()))
fig.colorbar(sm, ax=ax, label="churn rate")
plt.tight_layout()
plt.savefig(f"{BASE}/reports/figures/segment_value_vs_churn.png", bbox_inches="tight")

# 2) churn rate by segment
fig, ax = plt.subplots(figsize=(9, 5))
c = summary.sort_values("churn_rate")
ax.barh(c.index, c["churn_rate"], color="#c0504d")
ax.set_xlabel("Churn rate")
ax.set_title("Churn rate by RFM segment")
for i, v in enumerate(c["churn_rate"]):
    ax.text(v + 0.005, i, f"{v:.0%}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{BASE}/reports/figures/churn_by_segment.png", bbox_inches="tight")

print("\nSaved: data/customer_segments.csv and reports/figures/*.png")
