# Credit Scoring Model for Bati Bank's Buy-Now-Pay-Later Service

## Overview

This project develops a credit scoring system for Bati Bank's buy-now-pay-later partnership with an eCommerce platform. The model analyzes customer transaction patterns to predict creditworthiness and determine optimal loan terms, following Basel II regulatory guidelines.

## Business Context

Bati Bank requires a robust credit scoring solution that:

- Predicts default probability for loan applicants
- Assigns credit scores for decision-making
- Recommends appropriate loan amounts and durations
- Complies with Basel II Capital Accord requirements
- Uses interpretable models for regulatory approval

## Key Features

The system implements five core components:

1. **Default Proxy Definition** - Creates a binary risk indicator from behavioral data
2. **Feature Engineering** - Derives predictive features using RFMS methodology (Recency, Frequency, Monetary, Standard Deviation)
3. **Risk Probability Model** - Predicts probability of default (PD) for applicants
4. **Credit Score Model** - Converts risk probabilities to interpretable credit scores (300-850 range)
5. **Loan Optimization** - Recommends optimal loan amount and duration based on risk profile

## Methodology

### RFMS Feature Engineering

The model extracts four key behavioral signals from transaction data:

- **Recency**: Time since last purchase (lower = more engaged)
- **Frequency**: Number of transactions (higher = more active)
- **Monetary**: Average transaction value (indicates spending capacity)
- **Standard Deviation**: Spending volatility (measures financial stability)

### Model Approach

Following Basel II requirements for transparency, the project prioritizes:

- **Logistic Regression**: Primary model for interpretability and regulatory compliance
- **Weight of Evidence (WoE)**: Feature transformation for monotonic relationships
- **Gradient Boosting** (optional): Enhanced performance with SHAP explanations

## Data Requirements

### Input Data

- Transaction records from eCommerce platform
- Required fields: customer_id, transaction_date, transaction_amount, product_category
- Format: CSV or Parquet files

### Expected Schema

```
customer_id: string
transaction_date: datetime
transaction_amount: float
product_category: string (optional)
```

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/batibank/credit-scoring.git
cd credit-scoring

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Required Packages

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
statsmodels>=0.14.0
matplotlib>=3.6.0
seaborn>=0.12.0
shap>=0.41.0 (optional for model interpretation)
```

## Project Structure

```
credit-risk-model/
├── .github/
│   └── workflows/
│       └── ci.yml           # CI/CD pipeline configuration
├── data/                    # ⚠️ Added to .gitignore - not tracked
│   ├── raw/                 # Raw transaction data from eCommerce
│   └── processed/           # Processed RFMS features for training
├── notebooks/
│   └── eda.ipynb           # Exploratory data analysis and visualization
├── src/
│   ├── __init__.py         # Package initialization
│   ├── data_processing.py  # Feature engineering (RFMS calculation)
│   ├── train.py            # Model training pipeline
│   ├── predict.py          # Inference and prediction logic
│   └── api/
│       ├── main.py         # FastAPI application entry point
│       └── pydantic_models.py # Request/response schemas
├── tests/
│   └── test_data_processing.py # Unit tests for feature engineering
├── Dockerfile              # Container configuration
├── docker-compose.yml      # Multi-container orchestration
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore rules (includes data/)
└── README.md              # This file
```

## EDA Insights

### Dataset Overview

The exploratory data analysis (EDA) reveals a comprehensive transaction dataset with 95,662 records across 16 columns, covering customer interactions in Uganda (CountryCode: 256). All transactions are denominated in UGX (Ugandan Shilling). Memory usage is approximately 66.48 MB, with high-cardinality ID fields (e.g., 95,662 unique TransactionIds, 3,633 unique AccountIds) indicating a diverse and granular customer base.

### Key Statistics

Numerical features show significant variability, highlighting opportunities for RFMS-based feature engineering:

| Feature         | Count  | Mean     | Std        | Min        | 25% | 50%   | 75%   | Max       |
| --------------- | ------ | -------- | ---------- | ---------- | --- | ----- | ----- | --------- |
| CountryCode     | 95,662 | 256.00   | 0.00       | 256        | 256 | 256   | 256   | 256       |
| Amount          | 95,662 | 6,717.85 | 123,306.80 | -1,000,000 | -50 | 1,000 | 2,800 | 9,880,000 |
| Value           | 95,662 | 9,900.58 | 123,122.10 | 2          | 275 | 1,000 | 5,000 | 9,880,000 |
| PricingStrategy | 95,662 | 2.26     | 0.73       | 0          | 2   | 2     | 2     | 4         |
| FraudResult     | 95,662 | 0.002    | 0.045      | 0          | 0   | 0     | 0     | 1         |

- **Imbalance in Target**: FraudResult (used as a default proxy) is highly imbalanced, with only ~0.2% positive cases (193 frauds). This underscores the need for techniques like SMOTE or class weighting in modeling to address rarity of defaults.
- **Transaction Variability**: Amounts include negatives (e.g., refunds), with extreme outliers (max ~9.88M UGX). Value often exceeds Amount, suggesting inclusion of fees or adjustments—critical for Monetary feature derivation.
- **Categorical Distribution**: ProductCategory features include airtime (high frequency, low value), financial_services (mixed signs), and utility_bill (higher values). ChannelId and ProviderId show batching patterns, useful for Frequency and Recency signals.

### Time-Based Patterns

Transactions span from November 2018 onward (exact range: min 2018-11-15T02:18:49Z). Key temporal insights:

- **Daily Volume**: Steady growth with peaks on weekends, indicating consumer-driven spikes.
- **Hourly Peaks**: Highest activity between 12-18h (afternoon/evening), aligning with post-work shopping—ideal for Recency thresholding.
- **Day-of-Week**: Fridays and Saturdays dominate (~15-20% higher than weekdays), while Sundays are lowest.
- **Monthly Trends**: Even distribution, but Q4 shows slight upticks, possibly seasonal eCommerce effects.

These patterns inform RFMS features (e.g., recency <7 days flags active users) and reveal stability for credit risk proxying, though high volatility (Std in Amount) signals need for Standard Deviation monitoring to detect unstable profiles.

Visualizations (daily trends, hourly bars, DoW/monthly distributions) in `notebooks/eda.ipynb` support model validation and stakeholder communication.

## Model Performance Metrics

The model is evaluated using:

- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Gini Coefficient**: Measure of discriminatory power
- **KS Statistic**: Kolmogorov-Smirnov test for separation
- **Precision/Recall**: At various probability thresholds
- **Confusion Matrix**: Classification performance breakdown

Target performance benchmarks:

- AUC-ROC > 0.75 (regulatory minimum)
- Gini > 0.50
- KS Statistic > 0.30

## Regulatory Compliance

### Basel II Alignment

The model adheres to Basel II Capital Accord requirements:

- **Pillar 1**: Estimates PD for minimum capital requirements
- **Pillar 2**: Documented methodology for supervisory review
- **Pillar 3**: Transparent model for market discipline

### Model Interpretability

All predictions include:

- Feature importance rankings
- Weight of Evidence transformations
- Decision thresholds and rationale
- SHAP values (for complex models)

## Risk Considerations

### Proxy Variable Limitations

- Default proxy may not capture all credit risk dimensions
- Requires periodic validation against actual loan performance
- May introduce bias in underrepresented customer segments

### Model Monitoring

- Monthly performance tracking recommended
- Quarterly recalibration if population drift detected
- Annual full model validation required

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## References

- Huang et al. (2018). RFMS Method for Credit Scoring Based on Bank Card Transaction Data. _Statistica Sinica_.
- Basel Committee on Banking Supervision (2004). _International Convergence of Capital Measurement and Capital Standards_.
- ICCR/World Bank (2019). _Credit Scoring Approaches Guidelines_.

## Version History

- **v1.0.0** (Current): Initial release with logistic regression baseline
- Planned: v1.1.0 - Gradient boosting implementation with SHAP
- Planned: v1.2.0 - Real-time scoring API integration
