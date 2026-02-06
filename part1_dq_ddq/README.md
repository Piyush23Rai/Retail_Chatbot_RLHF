# Part 1 – Data Quality (DQ) & Distributional Data Quality (DDQ)

## Objective
Assess and improve data quality for an ecommerce chatbot using synthetic retail conversation data.

## Components
- Synthetic data generation with realistic noise
- Data Quality (DQ) metrics
- Distributional Data Quality (DDQ) using KL-Divergence
- Automated JSON reporting

## Files
- dq_ddq.py : End-to-end pipeline
- quality_report.json : Metrics output

## Metrics Used
### Data Quality
- Completeness ≥ 95%
- Duplicate Rate ≥ 98%
- Format Validity ≥ 99%

### Distributional Data Quality
- KL-Divergence < 0.1 for:
  - Product Category
  - Sentiment

## Execution
```bash
python dq_ddq.py
