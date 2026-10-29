"""Customer Churn Predictor - Streamlit app.

Run:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import business, explain, theme
from src.preprocess import FEATURES, NUMERIC_FEATURES, add_features, load_model_frame

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.markdown(
    """
    <style>
      .hero {font-size: 3.2rem; font-weight: 700; line-height: 1.05; margin: 0;}
      .hero-sub {color: #898781; font-size: 0.9rem; margin-top: 2px;}
      .band {display:inline-block; padding: 4px 12px; border-radius: 999px;
             font-weight: 600; font-size: 0.95rem; margin-top: 8px; border: 1.5px solid;}
      .band-High {color:#d03b3b; border-color:#d03b3b;}
      .band-Medium {color:#b97d00; border-color:#b97d00;}
      .band-Low {color:#0a8a0a; border-color:#0a8a0a;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------ loading
@st.cache_resource(show_spinner="Training the model for the first time (about 20 seconds)...")
def load_artifacts():
    if not (MODELS_DIR / "churn_model.joblib").exists():
        import train  # first run on a fresh deployment: build the model
        train.main()
    model = joblib.load(MODELS_DIR / "churn_model.joblib")
    metrics = json.loads((MODELS_DIR / "metrics.json").read_text())
    return model, metrics


@st.cache_data
def load_data() -> pd.DataFrame:
    return load_model_frame()


@st.cache_resource
def get_explainer(_model):
    prep, clf = _model.named_steps["prep"], _model.named_steps["clf"]
    background = prep.transform(load_data()[FEATURES].sample(200, random_state=42))
    return explain.build_explainer(clf, background)


model, metrics = load_artifacts()
data = load_data()
THRESHOLD = metrics["threshold"]


def style_fig(fig, height=330):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=theme.MUTED), title_font=dict(size=15),
        legend=dict(orientation="h", y=1.12, x=0),
    )
    fig.update_xaxes(showgrid=False, linecolor=theme.MUTED)
    fig.update_yaxes(gridcolor=theme.GRID, zeroline=False)
    return fig


# ------------------------------------------------------------------ helpers
def risk_band(score: float) -> str:
    if score >= 0.5:
        return "High"
    return "Medium" if score >= THRESHOLD else "Low"


BAND_TEXT = {
    "High": ("▲ High risk", "Priority customer: act now."),
    "Medium": ("◆ Medium risk", "Above the profit-maximising threshold: a retention offer pays off."),
    "Low": ("▼ Low risk", "Below the threshold: no offer needed."),
}


def recommend(r: dict) -> list:
    actions = []
    has_net = r["InternetService"] != "No"
    if r["Contract"] == "Month-to-month":
        actions.append("Offer a discounted 1-year or 2-year contract (month-to-month is the strongest churn driver).")
    if r["InternetService"] == "Fiber optic" and r["MonthlyCharges"] >= 80:
        actions.append("Review pricing: loyalty discount or bundle for high-bill fibre customers.")
    if has_net and r["TechSupport"] == "No":
        actions.append("Give 3 months of free Tech Support.")
    if has_net and r["OnlineSecurity"] == "No":
        actions.append("Offer a free Online Security trial.")
    if r["PaymentMethod"] == "Electronic check":
        actions.append("Offer a small bill credit for switching to automatic card or bank payment.")
    if r["tenure"] <= 12:
        actions.append("Schedule an onboarding check-in call within the first year.")
    return actions[:4] or ["Send a thank-you / loyalty message and keep monitoring."]


def explain_customer(row_df: pd.DataFrame, row: dict, top=7) -> pd.DataFrame:
    prep = model.named_steps["prep"]
    Xt = prep.transform(row_df[FEATURES])
    names = [explain.readable(n) for n in prep.get_feature_names_out()]
    values = explain.positive_class_values(get_explainer(model), Xt)[0]
    numeric_idx = {names.index(c) for c in NUMERIC_FEATURES if c in names}
    keep = [i for i in range(len(names)) if i in numeric_idx or Xt[0][i] != 0]
    keep = sorted(keep, key=lambda i: -abs(values[i]))[:top]
    labels = [f"{names[i]} = {row_df.iloc[0][names[i]]:.0f}" if i in numeric_idx else names[i] for i in keep]
    return pd.DataFrame({"factor": labels, "impact": values[keep]})


# ------------------------------------------------------------------ layout
st.title("Customer Churn Predictor")
st.caption(
    f"Telco customer dataset ({metrics['n_customers']:,} customers, "
    f"{metrics['churn_rate']:.1%} churned) · model: {metrics['best_model']}"
)

tab_predict, tab_insights, tab_model, tab_business = st.tabs(
    ["Predict a customer", "Insights", "Model performance", "Business impact"]
)

# ============================================================ 1. PREDICT
with tab_predict:
    left, right = st.columns([1.1, 1])

    with left:
        st.subheader("Customer details")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Account**")
            tenure = st.slider("Tenure (months)", 0, 72, 5)
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            payment = st.selectbox(
                "Payment method",
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 85.0, step=0.5)
        with c2:
            st.markdown("**Services**")
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            phone = st.selectbox("Phone service", ["Yes", "No"])
            lines = st.selectbox("Multiple lines", ["No", "Yes"] if phone == "Yes" else ["No phone service"])
            opts = ["No internet service"] if internet == "No" else ["No", "Yes"]
            security = st.selectbox("Online security", opts)
            backup = st.selectbox("Online backup", opts)
        with c3:
            st.markdown("**More**")
            protection = st.selectbox("Device protection", opts)
            support = st.selectbox("Tech support", opts)
            tv = st.selectbox("Streaming TV", opts)
            movies = st.selectbox("Streaming movies", opts)
            st.markdown("**Profile**")
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Has partner", ["No", "Yes"])
            dependents = st.selectbox("Has dependents", ["No", "Yes"])

    row = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": lines, "InternetService": internet,
        "OnlineSecurity": security, "OnlineBackup": backup, "DeviceProtection": protection,
        "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies, "Contract": contract,
        "PaperlessBilling": paperless, "PaymentMethod": payment, "MonthlyCharges": monthly,
        "TotalCharges": monthly * tenure,
    }
    row_df = add_features(pd.DataFrame([row]))
    score = float(model.predict_proba(row_df[FEATURES])[0, 1])
    band = risk_band(score)

    with right:
        st.subheader("Prediction")
        label, note = BAND_TEXT[band]
        st.markdown(
            f'<p class="hero">{score:.0%}</p><p class="hero-sub">churn risk score</p>'
            f'<span class="band band-{band}">{label}</span>',
            unsafe_allow_html=True,
        )
        st.write(note)

        st.markdown("**Why this score**")
        why = explain_customer(row_df, row)
        why_plot = why.iloc[::-1]
        fig = go.Figure(go.Bar(
            x=why_plot["impact"], y=why_plot["factor"], orientation="h",
            marker_color=[theme.CHURN if v > 0 else theme.STAY for v in why_plot["impact"]],
            hovertemplate="%{y}<br>impact %{x:+.2f}<extra></extra>",
        ))
        fig.update_layout(xaxis_title="← lowers risk      raises risk →")
        st.plotly_chart(style_fig(fig, 300), use_container_width=True)
        with st.expander("Show as table"):
            st.dataframe(why.round(3), use_container_width=True, hide_index=True)

        st.markdown("**Suggested retention actions**")
        if band == "Low":
            st.write("No retention offer needed for this customer.")
        else:
            for action in recommend(row):
                st.markdown(f"- {action}")

    st.caption(
        "The score comes from a model trained with SMOTE oversampling, so it ranks customers well "
        "but is not a calibrated probability. Total charges are estimated as tenure × monthly charges."
    )

# ============================================================ 2. INSIGHTS
with tab_insights:
    churned = data[data["Churn"] == 1]
    kept = data[data["Churn"] == 0]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customers", f"{len(data):,}")
    k2.metric("Churn rate", f"{data['Churn'].mean():.1%}")
    k3.metric("Avg monthly bill: churned", f"${churned['MonthlyCharges'].mean():.0f}",
              delta=f"{churned['MonthlyCharges'].mean() - kept['MonthlyCharges'].mean():+.0f} vs stayed",
              delta_color="off")
    k4.metric("Avg tenure: churned", f"{churned['tenure'].mean():.0f} months",
              delta=f"{churned['tenure'].mean() - kept['tenure'].mean():+.0f} vs stayed", delta_color="off")

    def churn_rate_chart(col, title, order=None):
        g = data.groupby(col)["Churn"].agg(["mean", "size"]).reset_index()
        if order:
            g[col] = pd.Categorical(g[col], order, ordered=True)
            g = g.sort_values(col)
        fig = go.Figure(go.Bar(
            x=g[col].astype(str), y=g["mean"], marker_color=theme.CHURN,
            text=[f"{v:.0%}" for v in g["mean"]], textposition="outside", cliponaxis=False,
            customdata=g["size"],
            hovertemplate="%{x}<br>churn rate %{y:.1%}<br>%{customdata:,} customers<extra></extra>",
        ))
        fig.update_layout(title=title, yaxis_tickformat=".0%", yaxis_range=[0, g["mean"].max() * 1.2])
        return style_fig(fig)

    r1, r2 = st.columns(2)
    r1.plotly_chart(churn_rate_chart("Contract", "Churn rate by contract type",
                                     ["Month-to-month", "One year", "Two year"]), use_container_width=True)
    r2.plotly_chart(churn_rate_chart("TenureGroup", "Churn rate by tenure (months)",
                                     ["0-12", "13-24", "25-48", "49-72"]), use_container_width=True)
    r3, r4 = st.columns(2)
    r3.plotly_chart(churn_rate_chart("InternetService", "Churn rate by internet service"), use_container_width=True)
    r4.plotly_chart(churn_rate_chart("PaymentMethod", "Churn rate by payment method"), use_container_width=True)

    fig = go.Figure()
    for label, frame, color in (("Stayed", kept, theme.STAY), ("Churned", churned, theme.CHURN)):
        fig.add_trace(go.Histogram(
            x=frame["MonthlyCharges"], name=label, marker_color=color, opacity=0.75,
            histnorm="percent", xbins=dict(size=5)))
    fig.update_layout(barmode="overlay", title="Monthly charges: churned customers skew to higher bills",
                      xaxis_title="Monthly charges ($)", yaxis_title="% of group")
    st.plotly_chart(style_fig(fig, 340), use_container_width=True)

# ============================================================ 3. MODEL
with tab_model:
    st.subheader("Model comparison")
    st.caption("Three models, each trained with SMOTE inside the cross-validation loop. "
               "The best one by cross-validated ROC-AUC is used by the app.")
    st.dataframe(pd.read_csv(MODELS_DIR / "model_comparison.csv"), use_container_width=True, hide_index=True)

    at = metrics["at_threshold"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Chosen model", metrics["best_model"])
    m2.metric(f"Recall @ {THRESHOLD:.2f}", f"{at['recall']:.0%}",
              help="Share of real churners the model flags for an offer.")
    m3.metric(f"Precision @ {THRESHOLD:.2f}", f"{at['precision']:.0%}",
              help="Share of flagged customers who really would have churned.")
    m4.metric("Customers flagged", f"{at['contacted']:,} of {metrics['test_size']:,}")

    a, b = st.columns(2)
    a.image(str(REPORTS_DIR / "roc_curves.png"), use_container_width=True)
    b.image(str(REPORTS_DIR / "confusion_matrix.png"), use_container_width=True)
    c, d = st.columns(2)
    c.image(str(REPORTS_DIR / "feature_importance.png"), use_container_width=True)
    d.image(str(REPORTS_DIR / "net_benefit.png"), use_container_width=True)
    if (REPORTS_DIR / "shap_beeswarm.png").exists():
        st.markdown("**SHAP detail: each dot is a customer** (red = high feature value, blue = low)")
        st.image(str(REPORTS_DIR / "shap_beeswarm.png"), width=700)

# ============================================================ 4. BUSINESS
with tab_business:
    st.subheader("What is the model worth?")
    st.caption("Change the assumptions to match your business. Results use the held-out test customers.")
    s1, s2, s3, s4 = st.columns(4)
    offer_cost = s1.number_input("Cost of one retention offer ($)", 1.0, 500.0,
                                 float(metrics["business"]["offer_cost"]), step=5.0)
    cust_value = s2.number_input("Value of a saved customer ($)", 50.0, 5000.0,
                                 float(metrics["business"]["customer_value"]), step=50.0)
    success = s3.slider("Offer success rate", 0.05, 0.9, float(metrics["business"]["success_rate"]), step=0.05)
    thr = s4.slider("Risk threshold", 0.05, 0.95, float(THRESHOLD), step=0.01)

    preds = pd.read_csv(MODELS_DIR / "test_predictions.csv")
    y_true, proba = preds["churned"].to_numpy(), preds["probability"].to_numpy()
    n = len(preds)

    kw = dict(offer_cost=offer_cost, value=cust_value, success=success)
    with_model = business.net_benefit(y_true, proba, thr, **kw)
    offer_all = business.contact_everyone(y_true, offer_cost=offer_cost, value=cust_value, success=success)
    contacted = int((proba >= thr).sum())
    caught = int(((proba >= thr) & (y_true == 1)).sum())

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Customers contacted", f"{contacted:,} of {n:,}")
    b2.metric("Churners reached", f"{caught:,} of {int(y_true.sum()):,}")
    b3.metric("Net benefit with model", f"${with_model:,.0f}")
    b4.metric("vs. offer to everyone", f"${with_model - offer_all:+,.0f}",
              help=f"Contacting all {n:,} customers without a model would net ${offer_all:,.0f}.")

    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    curve = business.benefit_curve(y_true, proba, grid, **kw) / n * 1000
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grid, y=curve, mode="lines", line=dict(color=theme.STAY, width=2.5),
                             name="Net benefit",
                             hovertemplate="threshold %{x:.2f}<br>$%{y:,.0f} per 1,000 customers<extra></extra>"))
    fig.add_trace(go.Scatter(x=[thr], y=[with_model / n * 1000], mode="markers",
                             marker=dict(color=theme.CHURN, size=11, line=dict(color="white", width=2)),
                             name="Selected threshold", hoverinfo="skip"))
    fig.update_layout(title="Net benefit per 1,000 customers vs. threshold",
                      xaxis_title="Risk threshold", yaxis_title="$ per 1,000 customers")
    st.plotly_chart(style_fig(fig, 360), use_container_width=True)

    base = st.number_input("Your customer base size", 1000, 10_000_000, 100_000, step=5000)
    per_customer = with_model / n
    st.info(f"At these settings the model is worth about **${per_customer * base:,.0f}** for a base of "
            f"{base:,} customers (one retention campaign). Treat it as an estimate: the offer success "
            f"rate and customer value are assumptions, not measured numbers.")