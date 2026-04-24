# Reflection — E-Commerce Customer Analytics & RFM Segmentation Tool

## Problem, audience, and product aim

**Analytical question:** Who are the highest-value customers today, who is slipping in engagement, and how do **RFM (Recency, Frequency, Monetary)** scores and **K-Means** clustering compare as segmentation views?

**Target users:** **Marketing and CRM-oriented stakeholders** (e.g., small e-commerce teams, interns, or students simulating a retention workshop) who need an interactive tool—not only a static notebook—to explore segments and drill into individual customers.

**Product aim:** Publish a Streamlit application that mirrors a practical workflow: sales overview → RFM scoring → cluster validation → product/customer drill-down, with transparency via a raw data tab.

## Dataset and why it was chosen

The project uses a **synthetic transactional dataset** (orders, customers, timestamps, monetary values) generated for educational use. It was selected to demonstrate **RFM engineering**, **quintile scoring**, and **clustering** with realistic column semantics while avoiding privacy and licensing issues with real retailer data. Generation/access timing is stated in the notebook and README.

**Trade-off:** Synthetic data is **clean by design**; production systems need identity resolution, returns, promotions, and seasonality controls.

## Python workflow (substantive steps)

- **Data preparation** with **pandas** (filters, aggregates, customer-level rollups, sanity checks).
- **RFM construction:** recency windows, frequency counts, monetary totals; **quintile scoring** with tie handling via **ranking** (`method='first'`) to avoid `qcut` bin-edge failures.
- **Machine learning:** **K-Means** on **standardized** RFM features (**StandardScaler**) to prevent scale dominance by monetary magnitude.
- **Visualization and product layer:** **Plotly** (including 3D scatter for intuition), **Streamlit** tabs ordered from executive overview to operational detail, plus **customer lookup**.

## Main insights (high level)

- A small share of customers typically contributes a disproportionate share of revenue; RFM labels make that **actionable** (e.g., who to reward vs reactivate).
- **Preprocessing order matters:** clustering before scaling produced misleading clusters; scaling aligned the algorithm with the intended geometry of RFM space.
- **Segment rules are business-specific:** textbook thresholds illustrate the method; a real firm would tune cutoffs using margins, churn costs, and campaign constraints.

## Limitations, reliability, and improvements

- **Synthetic transactions** miss messy retail realities (refunds, multi-channel identity, coupon distortions).
- **RFM is a snapshot:** cohort and survival models would better predict **future** churn.
- **Identity assumption:** one `customer_id` equals one person—often false in practice.
- **3D plots** aid demos but are not always the best operational chart type.

**Improvements:** calibrate segments using **profit** not only revenue; add **time-based cross-validation**; incorporate **product affinity** and **seasonality**; export segments for A/B test design documentation.

## Personal contribution and what I learned

I scoped the business question, implemented the RFM pipeline (including the **tie-aware quintile** workaround), built the Streamlit navigation model, validated cluster behavior after scaling, and verified key headline metrics (e.g., concentration of revenue in top segments) manually in the notebook. I also decided tab order and which views prioritize **manager usability** over “maximum charts.”

The hardest lesson was that **conceptual knowledge of RFM does not guarantee smooth implementation**—edge cases like duplicate quantile edges are where coursework meets real engineering.

## AI use disclosure (required format)

| Tool | Model / version (if known) | Access date (2026) | Specific purpose |
|------|----------------------------|--------------------|------------------|
| Claude | Not recorded | Apr 2026 | Suggested `.rank(method='first')` workaround for `pd.qcut` non-unique bin edges |
| ChatGPT / Claude | Not recorded | Apr 2026 | Suggested tab ordering from overview → detail for an analytics dashboard |
| ChatGPT / Claude | Not recorded | Apr 2026 | Provided a **template** for Plotly `scatter_3d`; I adapted parameters to my dataframe |
| ChatGPT / Claude | Not recorded | Apr 2026 | Confirmed scaling should apply to features used in K-Means; I validated results in-app |

Business interpretations, segment definitions, metric checks, and overall accountability for the analysis remain **my own**.

---

*Approximate word count: ~760 words*
