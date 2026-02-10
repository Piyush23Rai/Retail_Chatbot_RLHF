# Part 1 – Data Quality (DQ) & Distributional Data Quality (DDQ)

## Objective
Evaluate data readiness for an ecommerce chatbot by assessing
intrinsic data quality and distributional alignment with production.

## Dataset
10,000 synthetic retail chat conversations with:
- customer_id
- customer_segment (VIP, regular, new)
- product_category
- sentiment
- timestamp
- outcome

## Data Quality (DQ) Metrics
| Metric | Threshold |
|-----|-----|
Completeness | ≥ 95%
Duplicate Rate | ≥ 98%
Format Validity | ≥ 99%

## Distributional Data Quality (DDQ)
KL divergence computed for:
- Product Category Balance
- Sentiment Distribution
- Customer Segment Representation
- Temporal Consistency (pre/post COVID)

Acceptance: KL < 0.1

## Files
- dq_ddq.py : End-to-end pipeline
- text_templates.py: Conversation templates for synthetic data generator
- quality_report.json : Metrics output

## Execution
```bash
python -m part1_dq_ddq.dq_ddq
