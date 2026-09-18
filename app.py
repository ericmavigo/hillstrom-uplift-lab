from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.proportion import proportion_confint, proportions_ztest


st.set_page_config(page_title="Hillstrom Uplift Lab", page_icon="📬", layout="wide")
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "raw" / "hillstrom.csv"
ARMS = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
OUTCOMES = {"Visit": "visit", "Purchase": "conversion", "Spend": "spend"}
PRE_TREATMENT = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel"]
UPLIFT_MODELS = ["S-learner", "T-learner", "X-learner", "Class transformation", "Transformed outcome"]


@st.cache_data(show_spinner=False)
def load_data(path: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path)


def wilson_difference(treated: np.ndarray, control: np.ndarray) -> tuple[float, float]:
    nt, nc = len(treated), len(control)
    pt, pc = treated.mean(), control.mean()
    lt, ut = proportion_confint(int(treated.sum()), nt, method="wilson")
    lc, uc = proportion_confint(int(control.sum()), nc, method="wilson")
    diff = pt - pc
    lo = diff - np.sqrt((pt - lt) ** 2 + (uc - pc) ** 2)
    hi = diff + np.sqrt((ut - pt) ** 2 + (pc - lc) ** 2)
    return float(lo), float(hi)


def bootstrap_mean_difference(treated: np.ndarray, control: np.ndarray, seed: int = 42, reps: int = 2500) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    differences = np.empty(reps)
    for i in range(reps):
        differences[i] = (rng.choice(treated, len(treated), replace=True).mean()
                          - rng.choice(control, len(control), replace=True).mean())
    return tuple(np.quantile(differences, [0.025, 0.975]).astype(float))


def experiment_result(df: pd.DataFrame, treatment: str, control: str, outcome: str) -> dict:
    y_t = df.loc[df.segment == treatment, outcome].to_numpy()
    y_c = df.loc[df.segment == control, outcome].to_numpy()
    diff = float(y_t.mean() - y_c.mean())
    if outcome in ("visit", "conversion"):
        p_value = float(proportions_ztest([y_t.sum(), y_c.sum()], [len(y_t), len(y_c)])[1])
        lo, hi = wilson_difference(y_t, y_c)
    else:
        p_value = float(stats.ttest_ind(y_t, y_c, equal_var=False).pvalue)
        lo, hi = bootstrap_mean_difference(y_t, y_c)
    return {"treated": float(y_t.mean()), "control": float(y_c.mean()), "diff": diff,
            "lo": lo, "hi": hi, "p": p_value, "n_t": len(y_t), "n_c": len(y_c)}


def smd(a: pd.Series, b: pd.Series) -> float:
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled_sd) if pooled_sd else 0.0


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["history_segment", "zip_code", "channel"]),
        ("numeric", StandardScaler(), ["recency", "history", "mens", "womens", "newbie"]),
    ], remainder="passthrough")


def make_classifier(family: str):
    # Keep probabilities aligned with the natural outcome rate for effect estimation.
    estimator = (LogisticRegression(max_iter=1500) if family == "Logistic regression"
                 else RandomForestClassifier(n_estimators=180, min_samples_leaf=25, max_features=0.8,
                                             n_jobs=-1, random_state=42))
    return make_pipeline(make_preprocessor(), estimator)


def make_regressor(family: str):
    estimator = (Ridge(alpha=2.0) if family == "Logistic regression"
                 else RandomForestRegressor(n_estimators=180, min_samples_leaf=25, max_features=0.8,
                                            n_jobs=-1, random_state=42))
    return make_pipeline(make_preprocessor(), estimator)


def with_treatment(x: pd.DataFrame, treatment: np.ndarray) -> pd.DataFrame:
    out = x.copy()
    out["__treatment"] = treatment.astype(int)
    return out


def train_uplift_models(train: pd.DataFrame, test: pd.DataFrame, outcome: str, family: str) -> dict[str, np.ndarray]:
    x_train = train[PRE_TREATMENT].reset_index(drop=True)
    x_test = test[PRE_TREATMENT].reset_index(drop=True)
    t_train = train["__treatment"].to_numpy().astype(int)
    y_train = train[outcome].to_numpy().astype(float)
    p_t = float(t_train.mean())
    models: dict[str, np.ndarray] = {}

    # S-learner: one outcome model sees both the covariates and treatment flag.
    x_with_t = with_treatment(x_train, t_train)
    s_model = make_classifier(family).fit(x_with_t, y_train)
    p1 = s_model.predict_proba(with_treatment(x_test, np.ones(len(x_test))))[:, 1]
    p0 = s_model.predict_proba(with_treatment(x_test, np.zeros(len(x_test))))[:, 1]
    models["S-learner"] = p1 - p0

    # T-learner: separate outcome models for treatment and control.
    m1 = make_classifier(family).fit(x_train.loc[t_train == 1], y_train[t_train == 1])
    m0 = make_classifier(family).fit(x_train.loc[t_train == 0], y_train[t_train == 0])
    mu1 = m1.predict_proba(x_test)[:, 1]
    mu0 = m0.predict_proba(x_test)[:, 1]
    models["T-learner"] = mu1 - mu0

    # X-learner: impute individual effects in both arms and combine by propensity.
    treated, control = t_train == 1, t_train == 0
    tau_treated = y_train[treated] - m0.predict_proba(x_train.loc[treated])[:, 1]
    tau_control = m1.predict_proba(x_train.loc[control])[:, 1] - y_train[control]
    tau1 = make_regressor(family).fit(x_train.loc[treated], tau_treated).predict(x_test)
    tau0 = make_regressor(family).fit(x_train.loc[control], tau_control).predict(x_test)
    models["X-learner"] = (1 - p_t) * tau1 + p_t * tau0

    # Class transformation: the transformed binary label encodes which outcome was observed.
    z = np.where(t_train == 1, y_train, 1 - y_train)
    class_model = make_classifier(family).fit(x_train, z)
    models["Class transformation"] = 2 * class_model.predict_proba(x_test)[:, 1] - 1

    # Transformed outcome: an unbiased pseudo-outcome under randomized assignment.
    pseudo_y = y_train * (t_train - p_t) / (p_t * (1 - p_t))
    transformed = make_regressor(family).fit(x_train, pseudo_y)
    models["Transformed outcome"] = transformed.predict(x_test)
    return models


def uplift_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score)
    y, t = y[order], t[order]
    n_t = np.cumsum(t)
    n_c = np.cumsum(1 - t)
    sum_t = np.cumsum(y * t)
    sum_c = np.cumsum(y * (1 - t))
    safe_c = np.maximum(n_c, 1)
    gain = sum_t - sum_c * n_t / safe_c
    fraction = np.arange(1, len(y) + 1) / len(y)
    return fraction, gain


def evaluate_uplift(test: pd.DataFrame, outcome: str, predictions: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = test[outcome].to_numpy().astype(float)
    t = test["__treatment"].to_numpy().astype(int)
    fraction = np.arange(1, len(y) + 1) / len(y)
    rows, curves = [], []
    for name, score in predictions.items():
        x, gain = uplift_curve(y, t, score)
        # Center Qini area against random ranking; higher values mean better prioritization.
        qini = float(np.trapz(gain - x * gain[-1], x))
        top = np.argsort(-score)[:max(1, int(0.30 * len(score)))]
        top_t, top_c = t[top] == 1, t[top] == 0
        top_lift = (float(y[top][top_t].mean() - y[top][top_c].mean())
                    if top_t.any() and top_c.any() else np.nan)
        rows.append({"Model": name, "Qini area above random": qini,
                     "Top 30% observed lift": top_lift, "Customers targeted": len(top)})
        curves.append(pd.DataFrame({"Targeted share": x, "Incremental outcome gain": gain, "Model": name}))
    global_effect = float(y[t == 1].mean() - y[t == 0].mean())
    curves.append(pd.DataFrame({"Targeted share": fraction, "Incremental outcome gain": fraction * global_effect,
                                "Model": "Random targeting baseline"}))
    return pd.DataFrame(rows).sort_values("Qini area above random", ascending=False), pd.concat(curves, ignore_index=True)


@st.cache_data(show_spinner="Training and evaluating uplift models...")
def cached_model_run(df: pd.DataFrame, treatment: str, control: str, outcome: str,
                     family: str, selected_models: tuple[str, ...], test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    experiment = df[df.segment.isin([treatment, control])].copy().reset_index(drop=True)
    experiment["__treatment"] = (experiment.segment == treatment).astype(int)
    # Group exact duplicate rows together so identical observations cannot cross the split.
    group_ids = pd.util.hash_pandas_object(experiment.drop(columns=["__treatment"]), index=False).astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_idx, test_idx = next(splitter.split(experiment, groups=group_ids))
    train, test = experiment.iloc[train_idx].copy(), experiment.iloc[test_idx].copy()
    all_predictions = train_uplift_models(train, test, outcome, family)
    predictions = {name: all_predictions[name] for name in selected_models}
    scores, curves = evaluate_uplift(test, outcome, predictions)
    split = pd.DataFrame([{"Training rows": len(train), "Evaluation rows": len(test),
                           "Treatment share in train": train.__treatment.mean(),
                           "Treatment share in evaluation": test.__treatment.mean()}])
    return scores, curves, split


st.title("Hillstrom Uplift Lab")
st.caption("Explore a randomized email experiment, measure its average impact, and test five machine learning approaches for customer targeting.")

if not DATA_PATH.exists():
    st.error("The dataset file is missing. Download it using the instructions in the README, then restart the app.")
    st.stop()

df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime)
data_tab, quality_tab, experiment_tab, models_tab, method_tab = st.tabs([
    "1 · Data overview", "2 · Data quality", "3 · Experiment results", "4 · Uplift models", "Method and sources"
])

with data_tab:
    st.subheader("Dataset overview")
    st.write("The MineThatData email challenge randomly assigned retail customers to a men's merchandise email, a women's merchandise email, or no email. The treatment is receiving an email, **not receiving a discount**. Outcomes were recorded after the campaign.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{df.shape[1]}")
    c3.metric("Experiment groups", f"{df.segment.nunique()}")
    st.markdown("**Pre-treatment features** describe customers before assignment. `segment` is the randomized assignment. `visit`, `conversion`, and `spend` are post-treatment outcomes and must not be used as predictors.")
    st.markdown("**Sample rows**")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    dictionary = pd.DataFrame([
        ("recency", "Months since the most recent purchase", "Pre-treatment"),
        ("history_segment", "Band of spending in the prior year", "Pre-treatment"),
        ("history", "Spending in the prior year", "Pre-treatment"),
        ("mens / womens", "Past purchases by merchandise category", "Pre-treatment"),
        ("zip_code", "Customer area type", "Pre-treatment"),
        ("newbie", "Whether the customer is new", "Pre-treatment"),
        ("channel", "Primary shopping channel", "Pre-treatment"),
        ("segment", "Randomized email assignment", "Assignment"),
        ("visit", "Visited after the campaign", "Post-treatment outcome"),
        ("conversion", "Purchased after the campaign", "Post-treatment outcome"),
        ("spend", "Spending after the campaign", "Post-treatment outcome"),
    ], columns=["Variable", "Meaning", "Timing"])
    st.markdown("**Data dictionary**")
    st.dataframe(dictionary, use_container_width=True, hide_index=True)

with quality_tab:
    st.subheader("Data quality and table structure")
    schema = pd.DataFrame({"Data type": df.dtypes.astype(str), "Unique values": df.nunique(),
                           "Missing values": df.isna().sum()})
    left, right = st.columns(2)
    with left:
        st.markdown("**Column schema**")
        st.dataframe(schema, use_container_width=True)
        st.metric("Rows with missing values", f"{int(df.isna().any(axis=1).sum()):,}")
        st.metric("Exact repeated rows", f"{int(df.duplicated().sum()):,}")
    with right:
        counts = df.segment.value_counts().reindex(ARMS).rename_axis("Group").reset_index(name="Customers")
        st.plotly_chart(px.bar(counts, x="Group", y="Customers", color="Group", title="Randomized group sizes"), use_container_width=True)
        checks = {
            "Binary outcome outside 0/1": int((~df.visit.isin([0, 1])).sum() + (~df.conversion.isin([0, 1])).sum()),
            "Negative spend": int((df.spend < 0).sum()),
            "Purchase without visit": int(((df.conversion == 1) & (df.visit == 0)).sum()),
            "Positive spend without purchase": int(((df.spend > 0) & (df.conversion == 0)).sum()),
        }
        st.markdown("**Consistency checks**")
        st.dataframe(pd.DataFrame({"Check": checks.keys(), "Violations": checks.values()}), use_container_width=True, hide_index=True)
    st.info(f"There are {int(df.duplicated().sum()):,} rows identical across every column. The source has no customer ID, so we cannot tell whether these are duplicate records or different customers with the same values. We keep them and ensure identical rows stay in the same train/evaluation split. Zero outcomes are expected for a campaign with low purchase rates.")

with experiment_tab:
    st.subheader("Did the email work on average?")
    st.write("Compare one email treatment with the no-email control. Random assignment supports a causal interpretation of this average difference, subject to the experiment and data being intact.")
    control = st.selectbox("Control group", ["No E-Mail"])
    treatment = st.selectbox("Email treatment", ["Mens E-Mail", "Womens E-Mail"])
    outcome_label = st.selectbox("Primary outcome", list(OUTCOMES), index=1)
    outcome = OUTCOMES[outcome_label]
    result = experiment_result(df, treatment, control, outcome)
    rel = result["diff"] / result["control"] * 100 if result["control"] else np.nan
    m1, m2, m3, m4 = st.columns(4)
    fmt = (lambda x: f"{x:.2%}") if outcome != "spend" else (lambda x: f"${x:.2f}")
    m1.metric(f"{outcome_label}: treatment", fmt(result["treated"]))
    m2.metric(f"{outcome_label}: control", fmt(result["control"]))
    m3.metric("Absolute difference", f"{result['diff']:+.2%}" if outcome != "spend" else f"${result['diff']:+.2f}")
    m4.metric("Relative difference", f"{rel:+.1f}%")
    st.write(f"95% confidence interval for the difference: **[{result['lo']:+.4f}, {result['hi']:+.4f}]** · p-value = **{result['p']:.4g}**")
    if result["lo"] > 0 or result["hi"] < 0:
        st.success("The difference is statistically distinguishable from zero at the 5% level. Consider the effect size and business objective before deciding whether to roll out the campaign.")
    else:
        st.warning("The data do not show a clear difference at the 5% level. This does not prove that the effect is exactly zero.")
    summary = (df.groupby("segment", observed=True)
               .agg(Visit=("visit", "mean"), Purchase=("conversion", "mean"), Spend=("spend", "mean"))
               .reset_index().melt(id_vars="segment", var_name="Outcome", value_name="Mean"))
    st.plotly_chart(px.bar(summary, x="segment", y="Mean", color="Outcome", barmode="group", title="Observed average outcomes by group"), use_container_width=True)
    st.markdown("**Pre-treatment balance check**")
    a, b = df[df.segment == treatment], df[df.segment == control]
    balance = [{"Feature": col, "Standardized mean difference": smd(a[col], b[col])}
               for col in ["recency", "history", "mens", "womens", "newbie"]]
    st.dataframe(pd.DataFrame(balance), use_container_width=True, hide_index=True)
    st.caption("Values near zero indicate similar groups on observed pre-treatment features. Post-treatment outcomes are excluded from this check.")

with models_tab:
    st.subheader("Who benefits from receiving the email?")
    st.write("A purchase-propensity model finds likely buyers, including people who would buy without the campaign. Uplift modeling instead estimates how much the email changes each customer's outcome and can prioritize customers who may respond because of the email.")
    treatment = st.selectbox("Email treatment to model", ["Mens E-Mail", "Womens E-Mail"], key="model_treatment")
    control = "No E-Mail"
    outcome_label = st.selectbox("Outcome to optimize", ["Visit", "Purchase"], index=1, key="model_outcome")
    outcome = OUTCOMES[outcome_label]
    family = st.selectbox("Shared base learner", ["Random forest", "Logistic regression"], help="Using the same base learner keeps the comparison focused on the uplift strategy.")
    selected = st.multiselect("Five uplift approaches", UPLIFT_MODELS, default=UPLIFT_MODELS)
    test_size = st.slider("Evaluation share", min_value=0.15, max_value=0.35, value=0.25, step=0.05)
    st.markdown("**Why these five?** S- and T-learners provide simple baselines; X-learner combines imputed effects from both groups; class transformation recasts treatment-effect learning as classification; transformed outcome creates an unbiased effect target under randomized assignment.")
    st.caption("All approaches use only pre-treatment customer features. Exact repeated rows are grouped so identical records cannot occur in both training and evaluation sets.")
    run = st.button("Train and compare selected models", type="primary", disabled=not selected)
    if run:
        scores, curves, split = cached_model_run(df, treatment, control, outcome, family, tuple(selected), test_size)
        st.markdown("**Held-out evaluation split**")
        st.dataframe(split, use_container_width=True, hide_index=True)
        st.markdown("**Model comparison**")
        st.dataframe(scores, use_container_width=True, hide_index=True)
        st.plotly_chart(px.line(curves, x="Targeted share", y="Incremental outcome gain", color="Model",
                                title="Qini curves on held-out customers"), use_container_width=True)
        st.caption("Qini area above random measures how well a model ranks incremental responders. Top 30% observed lift compares treatment and control outcomes among customers ranked highest by that model. A positive score is not guaranteed; if models fail to beat random, the experiment may have little predictable heterogeneity.")
    else:
        st.info("Choose the outcome and base learner, then train the five uplift approaches on the same held-out evaluation split.")

with method_tab:
    st.subheader("Method, interpretation, and data source")
    st.markdown("- [Hillstrom dataset on Kaggle](https://www.kaggle.com/datasets/bofulee/kevin-hillstrom-minethatdata-e-mailanalytics)\n- [MineThatData email analytics challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)\n- [Experiment description](https://stochasticsolutions.com/pdf/HillstromChallenge.pdf)")
    st.write("This is a randomized email marketing experiment. It estimates the effect of receiving a marketing email, not the effect of a price discount. The overall A/B comparison answers whether the campaign changed outcomes on average; uplift models explore whether that change varies across customer profiles.")
    st.write("Binary outcomes use a two-proportion z-test and Newcombe-Wilson confidence interval. Spend uses Welch's test and a bootstrap interval. Models are compared on a held-out set using Qini area above a random-ranking baseline and observed lift in the top 30% of ranked customers.")
    st.warning("Predictive heterogeneity is not the same as a confirmed business policy. Use the randomized group for causal evaluation, and validate any targeting policy in a new experiment before deployment.")

