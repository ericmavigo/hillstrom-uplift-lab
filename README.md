# Hillstrom Uplift Lab

An end-to-end portfolio project built as **three reproducible Python notebooks** plus a **Streamlit executive review**. The notebooks open directly in Google Colab and clone this repository to retrieve their code and data.

## Start here

| Step | Notebook | Purpose |
|---|---|---|
| 1 | [00 · Data audit](https://colab.research.google.com/github/ericmavigo/hillstrom-uplift-lab/blob/main/notebooks/00_data_audit.ipynb) | Inspect tables, schema, missingness, repeated rows, group sizes, and pre-treatment balance. |
| 2 | [01 · A/B experiment](https://colab.research.google.com/github/ericmavigo/hillstrom-uplift-lab/blob/main/notebooks/01_ab_experiment.ipynb) | Estimate campaign effects on purchases, visits, and spend, with confidence intervals and corrected tests. |
| 3 | [02 · Uplift models](https://colab.research.google.com/github/ericmavigo/hillstrom-uplift-lab/blob/main/notebooks/02_uplift_models.ipynb) | Train and compare five uplift strategies on a held-out randomized sample. |

Every notebook contains an **Open in Colab** badge and can be run independently. No Kaggle login or uploaded credential is required.

## Executive Streamlit review

The dashboard brings the findings together: a business recommendation, treatment-versus-control purchase lift, experiment uncertainty, a data-quality snapshot, and the five-model evaluation. The app fetches the public CSV automatically if it is not present locally.

Run it locally:

```bash
python -m pip install -r requirements.txt
python download_data.py
streamlit run app.py
```

To publish with [Streamlit Community Cloud](https://share.streamlit.io/), connect this public GitHub repository and select branch `main` and entrypoint `app.py`.

## Business question

Did sending a marketing email increase purchase conversion compared with sending no email? After measuring the average campaign effect, can machine learning rank customers by their estimated incremental response?

The treatment in this historical randomized experiment is **receiving an email**, not receiving a price discount. The arms are `Mens E-Mail`, `Womens E-Mail`, and `No E-Mail`. Purchase conversion is the preselected primary outcome; visits and spend are supporting outcomes.

## Five uplift strategies

The dashboard and modeling notebook share the implementation in `src/uplift.py`:

- **S-learner:** one outcome model uses features and treatment assignment.
- **T-learner:** separate outcome models for treatment and control.
- **X-learner:** estimates effects within each arm and combines them by treatment propensity.
- **Class transformation:** converts randomized treatment-effect estimation into a classification task.
- **Transformed outcome:** regresses a treatment-weighted pseudo-outcome.

All methods use the same selected base learner, pre-treatment customer features, and held-out split. Evaluation reports Qini area above random ranking and observed lift among the top 30% of scored customers. A model score is not proof of future profit; a proposed targeting policy should be confirmed in a new randomized test.

## Data checks and limitations

The source file contains 64,000 customers and no customer ID. Exact repeated rows are reported but retained because they may represent different people with identical recorded values. Identical rows are kept together when creating the model evaluation split.

The dataset does not include campaign delivery costs, product margins, or discount amounts. It supports measuring the effect of email campaigns, but cannot establish that sending them is profitable or estimate the effect of a price reduction.

## Data source

- [Hillstrom dataset on Kaggle](https://www.kaggle.com/datasets/bofulee/kevin-hillstrom-minethatdata-e-mailanalytics)
- [MineThatData email analytics challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
- [Experiment description](https://stochasticsolutions.com/pdf/HillstromChallenge.pdf)

