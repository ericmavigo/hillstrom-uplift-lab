from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.model_selection import GroupShuffleSplit
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint, proportions_ztest
from src.uplift import evaluate_uplift, train_uplift_models


st.set_page_config(page_title="Hillstrom | Executive Campaign Review", page_icon="📬", layout="wide")
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "raw" / "hillstrom.csv"
DATA_URL = "http://www.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
ARMS = ["Mens E-Mail", "Womens E-Mail", "No E-Mail"]
OUTCOMES = {"Visit": "visit", "Purchase": "conversion", "Spend per customer": "spend"}


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Use a local CSV when available; otherwise fetch the public source for Cloud deployment."""
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return pd.read_csv(DATA_URL)


def wilson_difference(treated: np.ndarray, control: np.ndarray) -> tuple[float, float]:
    pt, pc = treated.mean(), control.mean()
    lt, ut = proportion_confint(int(treated.sum()), len(treated), method="wilson")
    lc, uc = proportion_confint(int(control.sum()), len(control), method="wilson")
    diff = pt - pc
    lo = diff - np.sqrt((pt - lt) ** 2 + (uc - pc) ** 2)
    hi = diff + np.sqrt((ut - pt) ** 2 + (pc - lc) ** 2)
    return float(lo), float(hi)


def bootstrap_mean_difference(treated: np.ndarray, control: np.ndarray, seed: int = 42, reps: int = 2500) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    differences = np.empty(reps)
    for start in range(0, reps, 64):
        stop = min(start + 64, reps)
        treated_means = rng.choice(treated, size=(stop - start, len(treated)), replace=True).mean(axis=1)
        control_means = rng.choice(control, size=(stop - start, len(control)), replace=True).mean(axis=1)
        differences[start:stop] = treated_means - control_means
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


@st.cache_data(show_spinner=False)
def all_experiment_results(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm in ["Mens E-Mail", "Womens E-Mail"]:
        for label, outcome in OUTCOMES.items():
            result = experiment_result(df, arm, "No E-Mail", outcome)
            rows.append({"Campaign": arm, "Outcome": label, **result})
    results = pd.DataFrame(rows)
    conversion_rows = results["Outcome"] == "Purchase"
    results["p_adjusted"] = np.nan
    results.loc[conversion_rows, "p_adjusted"] = multipletests(results.loc[conversion_rows, "p"], method="holm")[1]
    return results


def smd(a: pd.Series, b: pd.Series) -> float:
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / pooled_sd) if pooled_sd else 0.0


@st.cache_data(show_spinner="Training and evaluating the models...")
def cached_model_run(df: pd.DataFrame, treatment: str, outcome: str,
                     family: str, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    experiment = df[df.segment.isin([treatment, "No E-Mail"])].copy().reset_index(drop=True)
    experiment["__treatment"] = (experiment.segment == treatment).astype(int)
    row_groups = pd.util.hash_pandas_object(experiment.drop(columns=["__treatment"]), index=False).astype(str)
    split = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    train_i, test_i = next(split.split(experiment, groups=row_groups))
    train, test = experiment.iloc[train_i].copy(), experiment.iloc[test_i].copy()
    scores, curves = evaluate_uplift(test, outcome, train_uplift_models(train, test, outcome, family))
    split_info = pd.DataFrame([{"Training customers": len(train), "Evaluation customers": len(test),
                                "Treatment share (train)": train.__treatment.mean(),
                                "Treatment share (evaluation)": test.__treatment.mean()}])
    return scores, curves, split_info


try:
    df = load_data()
except Exception as exc:
    st.error("The Hillstrom data could not be loaded. Run `python download_data.py` locally or check the public source connection.")
    st.exception(exc)
    st.stop()

def queue_model_run() -> None:
    st.session_state["run_model_comparison"] = True


if st.session_state.pop("run_model_comparison", False):
    treatment_choice = st.session_state.get("model_treatment", "Mens E-Mail")
    outcome_choice = st.session_state.get("model_outcome", "Purchase")
    family_choice = st.session_state.get("model_family", "Logistic regression")
    test_size_choice = st.session_state.get("model_test_size", 0.25)
    scores, curves, split_info = cached_model_run(
        df, treatment_choice, OUTCOMES[outcome_choice], family_choice, test_size_choice
    )
    st.session_state["model_scores"] = scores
    st.session_state["model_curves"] = curves
    st.session_state["model_split_info"] = split_info
    st.session_state["model_config"] = f"{treatment_choice} vs no email · {outcome_choice.lower()} · {family_choice} · {1-test_size_choice:.0%} training / {test_size_choice:.0%} evaluation"

results = all_experiment_results(df)
st.title("Executive Campaign Review")
st.caption("Hillstrom randomized email experiment · 64,000 retail customers · Primary business outcome: purchase conversion")

summary_tab, model_tab, data_tab, method_tab = st.tabs([
    "Executive summary", "Customer targeting models", "Data audit", "Method and limitations"
])

with summary_tab:
    primary = results[results.Outcome == "Purchase"].copy()
    primary["Significant after Holm correction"] = primary.p_adjusted < 0.05
    best = primary.sort_values("diff", ascending=False).iloc[0]
    positives = primary[(primary["diff"] > 0) & (primary["p_adjusted"] < 0.05)]
    st.subheader("Recommendation")
    if len(positives) == 2:
        st.success(f"Both email campaigns increased purchase conversion versus no email. The stronger observed candidate is **{best.Campaign}**; validate its economics and customer targeting in a follow-up experiment before broad rollout.")
    elif len(positives) == 1:
        st.success(f"{positives.iloc[0].Campaign} increased purchase conversion versus no email after correcting for the two campaign comparisons. Confirm campaign costs and margin before rollout.")
    else:
        st.warning("Neither email campaign showed a statistically clear purchase-conversion lift after correcting for the two campaign comparisons. Do not select a winner from these data alone.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Customers in experiment", f"{len(df):,}")
    m2.metric("Primary outcome", "14-day purchase conversion")
    m3.metric("Randomized arms", "2 email + control")
    m4.metric("Rows missing any field", f"{int(df.isna().any(axis=1).sum()):,}")

    st.markdown("#### Campaign effect on the primary outcome")
    display = primary[["Campaign", "n_t", "treated", "control", "diff", "lo", "hi", "p_adjusted"]].rename(columns={
        "n_t": "Customers", "treated": "Campaign conversion", "control": "Control conversion",
        "diff": "Absolute lift", "lo": "95% CI lower", "hi": "95% CI upper", "p_adjusted": "Holm-adjusted p-value"})
    st.dataframe(display.style.format({"Campaign conversion": "{:.2%}", "Control conversion": "{:.2%}",
                                       "Absolute lift": "{:+.2%}", "95% CI lower": "{:+.2%}",
                                       "95% CI upper": "{:+.2%}", "Holm-adjusted p-value": "{:.3g}"}),
                 use_container_width=True, hide_index=True)
    chart = go.Figure()
    for _, row in primary.iterrows():
        chart.add_trace(go.Bar(name=row.Campaign, x=[row.Campaign], y=[row.treated],
                               error_y={"type": "data", "symmetric": False,
                                        "array": [max(0, row.control + row.hi - row.treated)],
                                        "arrayminus": [max(0, row.treated - row.control - row.lo)]}))
    chart.add_hline(y=float(df.loc[df.segment == "No E-Mail", "conversion"].mean()), line_dash="dash",
                    annotation_text="No-email control")
    chart.update_layout(title="Purchase conversion by email treatment (95% confidence interval)",
                        yaxis_title="Purchase conversion", yaxis_tickformat=".1%", showlegend=False)
    st.plotly_chart(chart, use_container_width=True)

    model_result = st.session_state.get("model_scores")
    st.markdown("#### Customer targeting evidence")
    if model_result is None:
        st.info("The five uplift models are ready to run in the **Customer targeting models** tab. Their held-out ranking results will appear here after training.")
    else:
        st.caption(f"Last run: {st.session_state.get('model_config', 'configuration unavailable')}. Results are held-out estimates, not a guaranteed profit forecast.")
        st.dataframe(model_result, use_container_width=True, hide_index=True)

    st.markdown("#### Decision guardrails")
    st.write("The dataset measures visits, purchases, and customer spend after an email. It does not include campaign delivery cost, product margin, or discount amounts, so the observed revenue lift is not profit. The treatment is an email, not a price discount.")
    st.caption("Visit and spend results are available in the experiment analysis notebook. Purchase conversion is the preselected primary outcome; Holm correction is applied to the two email-versus-control comparisons.")

with model_tab:
    st.subheader("Which customers respond because of the email?")
    st.write("Uplift models estimate the incremental change associated with treatment. This is different from predicting who is likely to buy regardless of the email.")
    treatment = st.selectbox("Email campaign", ["Mens E-Mail", "Womens E-Mail"], key="model_treatment")
    outcome_label = st.selectbox("Customer outcome", ["Purchase", "Visit"], index=0, key="model_outcome")
    outcome = OUTCOMES[outcome_label]
    base = st.selectbox("Shared base learner", ["Logistic regression", "Random forest"], key="model_family")
    test_size = st.slider("Held-out evaluation share", 0.15, 0.35, 0.25, 0.05, key="model_test_size")
    st.write("The comparison includes S-learner, T-learner, X-learner, class transformation, and transformed outcome. All use pre-treatment features and the same train/evaluation split. Exact repeated rows are kept together to limit leakage.")
    st.button("Train five models", type="primary", on_click=queue_model_run)
    if "model_scores" in st.session_state:
        if "model_split_info" in st.session_state:
            st.markdown("**Evaluation split**")
            st.dataframe(st.session_state["model_split_info"], use_container_width=True, hide_index=True)
        st.markdown("**Held-out results**")
        st.dataframe(st.session_state["model_scores"], use_container_width=True, hide_index=True)
        if "model_curves" in st.session_state:
            st.plotly_chart(px.line(st.session_state["model_curves"], x="Targeted share", y="Incremental outcome gain", color="Model",
                                    title="Qini curves versus random targeting"), use_container_width=True)
        st.caption("Qini area above random measures how well a model ranks incremental responders. Observed lift in the top 30% compares treatment and control outcomes among the highest-ranked customers. Use the randomized experiment to confirm any proposed targeting policy.")
    else:
        st.info("Choose the outcome and base learner, then run the five-model comparison. Results will also appear in the executive summary.")

with data_tab:
    st.subheader("Data audit")
    a, b, c, d = st.columns(4)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", f"{df.shape[1]}")
    c.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
    d.metric("Exact repeated rows", f"{int(df.duplicated().sum()):,}")
    st.write("Exact repeated rows cannot be confidently classified as duplicate customer records because the source contains no customer identifier. We retain them and keep identical rows on the same side of the model evaluation split.")
    schema = pd.DataFrame({"Data type": df.dtypes.astype(str), "Unique values": df.nunique(), "Missing values": df.isna().sum()})
    st.markdown("**Column profile**")
    st.dataframe(schema, use_container_width=True)
    checks = {
        "Binary outcome outside 0/1": int((~df.visit.isin([0, 1])).sum() + (~df.conversion.isin([0, 1])).sum()),
        "Negative spend": int((df.spend < 0).sum()),
        "Purchase without visit": int(((df.conversion == 1) & (df.visit == 0)).sum()),
        "Positive spend without purchase": int(((df.spend > 0) & (df.conversion == 0)).sum()),
    }
    left, right = st.columns(2)
    with left:
        st.markdown("**Randomized group sizes**")
        counts = df.segment.value_counts().reindex(ARMS).rename_axis("Group").reset_index(name="Customers")
        st.plotly_chart(px.bar(counts, x="Group", y="Customers", color="Group"), use_container_width=True)
    with right:
        st.markdown("**Consistency checks**")
        st.dataframe(pd.DataFrame({"Check": checks.keys(), "Violations": checks.values()}), use_container_width=True, hide_index=True)
    st.markdown("**Observed outcomes by group**")
    group_summary = df.groupby("segment", observed=True).agg(
        Customers=("segment", "size"), Visit_rate=("visit", "mean"), Purchase_rate=("conversion", "mean"),
        Spend_per_customer=("spend", "mean")).reset_index()
    st.dataframe(group_summary.style.format({"Visit_rate": "{:.2%}", "Purchase_rate": "{:.2%}", "Spend_per_customer": "${:.2f}"}),
                 use_container_width=True, hide_index=True)

with method_tab:
    st.subheader("How to read this review")
    st.markdown("1. **Data audit:** check source fields, missingness, repeated rows, randomization sizes, and outcome consistency.\n2. **Experiment:** compare each email arm with the no-email control. Purchase conversion is the primary outcome. The two campaign comparisons use Holm correction.\n3. **Uplift modeling:** compare five approaches on a held-out sample; report Qini area and lift in the top 30%.\n4. **Decision:** check effect size and confidence interval, then account for costs and margins that are not present in this dataset.")
    st.markdown("### Statistical choices")
    st.write("Binary outcomes use a two-proportion z-test with Newcombe-Wilson confidence intervals. Spend uses Welch's test and a bootstrap confidence interval. These support different data types and make uncertainty visible alongside the estimated effect.")
    st.markdown("### Limitations")
    st.write("The data come from a historical randomized email experiment. The treatment is email exposure, not a discount. Purchase rates are low, and exact duplicated rows have no customer ID to resolve their provenance. An estimated individual uplift is not directly observable; model ranking must be validated against held-out randomized outcomes and any final targeting rule should be tested prospectively.")
    st.markdown("### Sources")
    st.markdown("- [Hillstrom dataset on Kaggle](https://www.kaggle.com/datasets/bofulee/kevin-hillstrom-minethatdata-e-mailanalytics)\n- [MineThatData email analytics challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html)\n- [Experiment description](https://stochasticsolutions.com/pdf/HillstromChallenge.pdf)")

