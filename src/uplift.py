from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PRE_TREATMENT = ["recency", "history_segment", "history", "mens", "womens", "zip_code", "newbie", "channel"]
UPLIFT_MODELS = ["S-learner", "T-learner", "X-learner", "Class transformation", "Transformed outcome"]


def make_preprocessor() -> ColumnTransformer:
    """Encode categories and scale numeric fields; pass through the S-learner treatment flag."""
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), ["history_segment", "zip_code", "channel"]),
        ("numeric", StandardScaler(), ["recency", "history", "mens", "womens", "newbie"]),
    ], remainder="passthrough")


def make_classifier(family: str):
    # Unweighted probabilities preserve the observed outcome prevalence for treatment-effect estimation.
    estimator = (LogisticRegression(max_iter=1500) if family == "Logistic regression"
                 else RandomForestClassifier(n_estimators=140, min_samples_leaf=25, max_features=0.8,
                                             n_jobs=-1, random_state=42))
    return make_pipeline(make_preprocessor(), estimator)


def make_regressor(family: str):
    estimator = (Ridge(alpha=2.0) if family == "Logistic regression"
                 else RandomForestRegressor(n_estimators=140, min_samples_leaf=25, max_features=0.8,
                                            n_jobs=-1, random_state=42))
    return make_pipeline(make_preprocessor(), estimator)


def with_treatment(x: pd.DataFrame, treatment: np.ndarray) -> pd.DataFrame:
    out = x.copy()
    out["__treatment"] = treatment.astype(int)
    return out


def train_uplift_models(train: pd.DataFrame, test: pd.DataFrame, outcome: str,
                        family: str) -> dict[str, np.ndarray]:
    """Fit five uplift strategies and score the evaluation customers."""
    x_train = train[PRE_TREATMENT].reset_index(drop=True)
    x_test = test[PRE_TREATMENT].reset_index(drop=True)
    t_train = train["__treatment"].to_numpy().astype(int)
    y_train = train[outcome].to_numpy().astype(float)
    p_t = float(t_train.mean())
    models: dict[str, np.ndarray] = {}

    # S-learner: one outcome model receives both features and treatment assignment.
    x_with_t = with_treatment(x_train, t_train)
    s_model = make_classifier(family).fit(x_with_t, y_train)
    p1 = s_model.predict_proba(with_treatment(x_test, np.ones(len(x_test))))[:, 1]
    p0 = s_model.predict_proba(with_treatment(x_test, np.zeros(len(x_test))))[:, 1]
    models["S-learner"] = p1 - p0

    # T-learner: separate outcome models for treatment and control.
    m1 = make_classifier(family).fit(x_train.loc[t_train == 1], y_train[t_train == 1])
    m0 = make_classifier(family).fit(x_train.loc[t_train == 0], y_train[t_train == 0])
    mu1, mu0 = m1.predict_proba(x_test)[:, 1], m0.predict_proba(x_test)[:, 1]
    models["T-learner"] = mu1 - mu0

    # X-learner: impute effects within each arm, then combine by treatment propensity.
    treated, control = t_train == 1, t_train == 0
    tau_treated = y_train[treated] - m0.predict_proba(x_train.loc[treated])[:, 1]
    tau_control = m1.predict_proba(x_train.loc[control])[:, 1] - y_train[control]
    tau1 = make_regressor(family).fit(x_train.loc[treated], tau_treated).predict(x_test)
    tau0 = make_regressor(family).fit(x_train.loc[control], tau_control).predict(x_test)
    models["X-learner"] = (1 - p_t) * tau1 + p_t * tau0

    # Class transformation assumes near-balanced randomized treatment assignment.
    z = np.where(t_train == 1, y_train, 1 - y_train)
    class_model = make_classifier(family).fit(x_train, z)
    models["Class transformation"] = 2 * class_model.predict_proba(x_test)[:, 1] - 1

    # Transformed outcome is unbiased under randomized assignment.
    pseudo_y = y_train * (t_train - p_t) / (p_t * (1 - p_t))
    transformed = make_regressor(family).fit(x_train, pseudo_y)
    models["Transformed outcome"] = transformed.predict(x_test)
    return models


def uplift_curve(y: np.ndarray, t: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-score)
    y, t = y[order], t[order]
    n_t, n_c = np.cumsum(t), np.cumsum(1 - t)
    sum_t, sum_c = np.cumsum(y * t), np.cumsum(y * (1 - t))
    gain = sum_t - sum_c * n_t / np.maximum(n_c, 1)
    return np.arange(1, len(y) + 1) / len(y), gain


def evaluate_uplift(test: pd.DataFrame, outcome: str,
                    predictions: dict[str, np.ndarray]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = test[outcome].to_numpy().astype(float)
    t = test["__treatment"].to_numpy().astype(int)
    fraction = np.arange(1, len(y) + 1) / len(y)
    rows, curves = [], []
    for name, score in predictions.items():
        x, gain = uplift_curve(y, t, score)
        qini = float(trapezoid(gain - x * gain[-1], x))
        top = np.argsort(-score)[:max(1, int(0.30 * len(score)))]
        top_t, top_c = t[top] == 1, t[top] == 0
        top_lift = float(y[top][top_t].mean() - y[top][top_c].mean()) if top_t.any() and top_c.any() else np.nan
        rows.append({"Model": name, "Qini area above random": qini,
                     "Top 30% observed lift": top_lift, "Customers targeted": len(top)})
        curves.append(pd.DataFrame({"Targeted share": x, "Incremental outcome gain": gain, "Model": name}))
    overall = float(y[t == 1].mean() - y[t == 0].mean())
    curves.append(pd.DataFrame({"Targeted share": fraction, "Incremental outcome gain": fraction * overall,
                                "Model": "Random targeting baseline"}))
    return pd.DataFrame(rows).sort_values("Qini area above random", ascending=False), pd.concat(curves, ignore_index=True)

