# Hillstrom Uplift Lab

An English-language Streamlit app for exploring the Hillstrom randomized email experiment, auditing its data, estimating campaign impact, and comparing five uplift-learning approaches.

## Project question

Did an email campaign increase visits or purchases? After measuring its average effect against a randomized no-email control, can machine learning identify customer profiles with larger incremental responses?

The treatment in this dataset is receiving a marketing email, **not receiving a discount**. The three randomized groups are `Mens E-Mail`, `Womens E-Mail`, and `No E-Mail`.

## Run locally

```bash
python -m pip install -r requirements.txt
python download_data.py
streamlit run app.py
```

The app includes a data overview and dictionary, data quality checks, A/B experiment results, and an interactive model comparison. The source data is downloaded at runtime and is excluded from Git.

## Five uplift approaches

- **S-learner:** a single outcome model receives both customer features and treatment assignment.
- **T-learner:** separate outcome models are fit for treatment and control.
- **X-learner:** imputed treatment effects from both groups are combined using the treatment propensity.
- **Class transformation:** transforms the binary outcome so treatment-effect estimation becomes a classification task.
- **Transformed outcome:** regresses a treatment-weighted outcome that is unbiased under randomized assignment.

The app lets you compare the methods with the same base learner and a held-out evaluation set. Features are limited to information available before treatment. Exact repeated rows are kept together when splitting to reduce leakage.

## Evaluation and interpretation

The experiment view reports treatment and control means, an effect interval, a p-value, and pre-treatment balance checks. The model view reports Qini area above random ranking and observed lift among the top 30% of scored customers. These measures assess whether a model prioritizes incremental responders; they do not replace validation in a new randomized campaign before deploying a targeting policy.

Binary outcomes use a two-proportion z-test and Newcombe-Wilson confidence interval. Spend uses Welch's test and a bootstrap confidence interval.

## Data and references

- [Hillstrom dataset on Kaggle](https://www.kaggle.com/datasets/bofulee/kevin-hillstrom-minethatdata-e-mailanalytics)
- [MineThatData email analytics challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)
- [Experiment description](https://stochasticsolutions.com/pdf/HillstromChallenge.pdf)

This is a public randomized marketing experiment. Customer and feature names are historical and reflect the source dataset. The dataset contains no customer identifier, so exact repeated rows cannot be confidently classified as duplicate records.

