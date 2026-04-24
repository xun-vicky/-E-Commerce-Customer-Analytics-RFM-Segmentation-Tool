# 🛒 E-Commerce Customer Analytics & RFM Segmentation Tool

An interactive Streamlit application that analyzes e-commerce transaction data using RFM (Recency, Frequency, Monetary) analysis and K-Means clustering to segment customers and uncover actionable business insights.

## Problem & Users

**Research Question:** How can we segment e-commerce customers to identify high-value groups, at-risk customers, and opportunities for targeted marketing?

**Target Users:** Marketing managers, CRM analysts, and business strategists who need data-driven customer segmentation to design retention campaigns, optimize promotional spend, and maximize customer lifetime value.

## Data

- **Source:** Synthetic e-commerce transaction dataset modeling a multi-category online retailer
- **Generation Date:** Programmatically generated with realistic purchase patterns
- **Scope:** 800 customers, ~7,000 transactions, 6 product categories, 10 countries, 2 years (2023-2024)
- **Key Fields:**
  - `order_id`, `order_date` — Transaction metadata
  - `customer_id`, `country` — Customer identifiers
  - `product_name`, `category` — Product taxonomy
  - `quantity`, `unit_price`, `total_amount` — Purchase metrics
  - `payment_method` — Payment channel

## Method (Python Core Steps)

1. **Data Generation** — Synthetic data with realistic customer behavior profiles (champions, loyals, at-risk, lost), including deliberate quality issues
2. **Data Cleaning** — Duplicate removal, negative quantity filtering, missing value handling, feature engineering (order month, day of week)
3. **RFM Analysis** — Recency/Frequency/Monetary computation, quintile scoring (1-5), rule-based customer segmentation into 8 segments
4. **K-Means Clustering** — StandardScaler normalization, Elbow Method for optimal K, unsupervised clustering on RFM features
5. **Interactive Visualization** — Sales dashboards, segment treemaps, 3D cluster plots, customer lookup, product heatmaps

## Main Findings

1. **Pareto Effect:** Champions (~8% of customers) generate ~35% of total revenue, confirming the 80/20 principle
2. **At-Risk Opportunity:** ~20% of customers are "At Risk" — previously active but recently absent, representing high-value retention targets
3. **Category-Value Link:** Electronics drives highest per-order revenue; Clothing has the highest frequency — different strategies needed per category
4. **K-Means Validates RFM:** Data-driven clusters align with rule-based segments, strengthening confidence in the segmentation approach
5. **Repeat Rate Gap:** Repeat purchase rate below industry benchmarks suggests untapped loyalty program potential

## How to Run

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Structure

```
Project2_Ecommerce_RFM/
├── app.py                  # Streamlit interactive tool
├── notebook/
│   └── ecommerce_rfm_analysis.ipynb  # Complete analysis notebook
├── data/                   # Generated data files
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── reflection.md          # 500-800 word reflection with AI disclosure
└── demo_script.md         # 1.5-2 min demo script
```

## Product / Demo Link

> **GitHub repository:** [xun-vicky/ecommerce-rfm-segmentation](https://github.com/xun-vicky/ecommerce-rfm-segmentation)  
> **How to use:** Clone the repository, install dependencies, and run `streamlit run app.py` locally.  
> **Demo video:** Mediasite link — to be added after recording.

## Limitations & Future Improvements

- **Synthetic Data:** Customer behavior patterns are simulated; real e-commerce data would contain richer signals like browsing history, cart abandonment, and session data.
- **Static Segmentation:** RFM is a snapshot analysis. Implementing rolling windows or cohort analysis would capture customer lifecycle dynamics.
- **Limited Features:** Adding demographic data (age, gender), product ratings, and marketing channel attribution would improve segmentation quality.
- **Scalability:** For large-scale production use, the clustering pipeline should be modularized and scheduled with tools like Airflow or Prefect.
- **Recommendation Engine:** Integrating collaborative filtering or association rules could extend the tool from segmentation to personalized product recommendations.
