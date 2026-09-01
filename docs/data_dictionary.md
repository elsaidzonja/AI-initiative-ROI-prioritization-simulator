# Data Dictionary — Kaffeeliebe Customer Retention

**Scenario.** *Kaffeeliebe* is a fictional DACH-region specialty-coffee
subscription retailer. Each row is one customer, observed over a trailing
12-month window. The task: predict churn and decide where to spend a
retention budget. Data is synthetic but built with realistic, noisy signal
(churn is a logistic function of the drivers below, plus an interaction and
gaussian noise), so a model earns a believable ~0.87 ROC-AUC.

## `customers.csv`

| Column | Type | Description |
|---|---|---|
| `customer_id` | string | Unique customer key (`KL######`). |
| `region` | category | DE / AT / CH / Other. |
| `acquisition_channel` | category | How the customer was acquired. No churn signal by design. |
| `plan_tier` | category | Basic / Plus / Premium. Premium is stickier. |
| `tenure_months` | int | Months since signup (1–72). Longer tenure → lower churn. |
| `num_orders_12m` | int | Orders in the last 12 months (Frequency). |
| `days_since_last_order` | int | Days since most recent order (Recency). Strongest churn driver. |
| `avg_order_value` | float | Mean order value in EUR, driven by tier. |
| `email_engagement_rate` | float | Email open/click rate 0–1. ~3% missing (MCAR). |
| `app_logins_30d` | int | App logins in the last 30 days. |
| `support_tickets_12m` | int | Support tickets raised; a heavy tail of unhappy customers. |
| `discount_rate` | float | Share of orders placed with a discount. Higher → more churn. |
| `satisfaction_score` | float | 1–5, pulled down by support tickets. ~4% missing (MCAR). |
| `annual_margin` | float | **Value at risk**: expected annual gross margin (EUR). Basis for ROI. |
| `churned` | int | **Target**. 1 = churned in the window, 0 = retained. Base rate ~27.5%. |

## Derived (produced by `01_segmentation.py`, saved to `customer_segments.csv`)

| Column | Description |
|---|---|
| `R`, `F`, `M` | Recency / Frequency / Monetary quintile scores, 1–5 (5 = best). |
| `segment` | Named RFM segment (Champions, Loyal, At Risk, Can't Lose Them, Hibernating, …). |

## Repo layout

```
customer-retention/
├── data/            customers.csv, customer_segments.csv
├── src/             generate_data.py, 01_segmentation.py
├── reports/figures/ segmentation charts
└── docs/            this dictionary
```
