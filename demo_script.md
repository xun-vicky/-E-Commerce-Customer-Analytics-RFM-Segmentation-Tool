# Demo Script — E-Commerce RFM Tool (1.5-2 min)

## Opening (10s)
This is my second ACC102 data product — an E-Commerce Customer Analytics tool using RFM segmentation and K-Means clustering, built with Python and Streamlit.

## Problem & Data (15s)
The question is: how do we segment customers to find our best buyers and the ones about to leave? I created a synthetic dataset with 800 customers and about 7,000 transactions across six product categories.

## KPI Overview (10s)
At the top, five key metrics: total revenue, orders, unique customers, average order value, and the repeat purchase rate. The sidebar lets you filter by date, category, and country.

## Sales Overview (15s)
The first tab shows monthly revenue and order trends, revenue breakdown by category, and payment method distribution. Electronics leads in revenue, while Clothing has the highest volume.

## RFM Segmentation (20s)
This is the core of the tool. RFM scores each customer on Recency, Frequency, and Monetary value. The bar chart shows customer counts per segment, and the treemap shows revenue contribution. Notice how Champions are only about 8% of customers but drive over 30% of revenue — a classic Pareto pattern.

## K-Means Clustering (15s)
Here I validate the RFM segments with machine learning. The 3D scatter plot shows four K-Means clusters. The Elbow Method on the right confirms K=4 as optimal. The box plots below compare each cluster's RFM distributions.

## Customer Lookup (10s)
Pick any customer ID and instantly see their segment, RFM scores, purchase history, and order details. This is what makes it a tool, not just a dashboard.

## Closing (10s)
Built with pandas, scikit-learn, and plotly. The complete analysis notebook and reflection are included in the repo. Thanks for watching!
