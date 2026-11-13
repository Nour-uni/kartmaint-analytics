"""KartMaint Analytics dashboard (Streamlit).

Run:  streamlit run dashboard/app.py
Needs the pipeline to have run first (data/predictions, data/model, data/clean).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kartmaint.config import CLEAN_DIR, MODEL_DIR, PARTS, PRED_DIR, SERVICE_INTERVAL_HOURS  # noqa: E402

st.set_page_config(page_title="KartMaint Analytics", page_icon="🏎️", layout="wide")


@st.cache_data
def load_all():
    scores = pd.read_parquet(PRED_DIR / "latest_scores.parquet")
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    importance = pd.read_csv(MODEL_DIR / "feature_importance.csv")
    sessions = pd.read_parquet(CLEAN_DIR / "sessions")
    events = pd.read_parquet(CLEAN_DIR / "events")
    sessions["session_date"] = pd.to_datetime(sessions["session_date"])
    events["event_date"] = pd.to_datetime(events["event_date"])
    return scores, metrics, importance, sessions, events


if not (PRED_DIR / "latest_scores.parquet").exists():
    st.error("No predictions found. Run the pipeline first (see README: generate -> etl -> features -> train -> score).")
    st.stop()

scores, metrics, importance, sessions, events = load_all()
BAND_ICON = {"high": "🔴 high", "medium": "🟠 medium", "low": "🟢 low"}

st.title("🏎️ KartMaint Analytics: breakdown risk for the next 14 days")
st.caption(
    f"Data through {scores['data_through'].iloc[0]} · model: {scores['model_name'].iloc[0]} · "
    "SYNTHETIC data (simulated karting fleet), not real client records."
)

best = metrics["selected_model"]
test_m = metrics[best]["test"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Karts scored", len(scores))
c2.metric("High-risk karts", int((scores["risk_band"] == "high").sum()))
c3.metric("Test ROC-AUC", f"{test_m['roc_auc']:.2f}")
c4.metric("Precision in top 10% (test)", f"{test_m['precision_at_top10pct']:.0%}",
          help=f"Base rate in the test period was {test_m['positive_rate']:.0%}")

tab_fleet, tab_kart, tab_model = st.tabs(["Fleet ranking", "Kart detail", "Model & honesty"])

with tab_fleet:
    band = st.multiselect("Risk band", ["high", "medium", "low"], default=["high", "medium"])
    view = scores[scores["risk_band"].isin(band)].copy()
    view["band"] = view["risk_band"].map(BAND_ICON)
    st.dataframe(
        view[["risk_rank", "kart_id", "model", "band", "risk_score", "most_overdue_part", "most_overdue_ratio", "hours_30d"]],
        use_container_width=True,
        hide_index=True,
        column_config={
            "risk_score": st.column_config.ProgressColumn("P(breakdown in 14 days)", min_value=0.0, max_value=1.0, format="%.2f"),
            "most_overdue_ratio": st.column_config.NumberColumn("overdue ratio", help="hours since service / recommended interval"),
            "hours_30d": st.column_config.NumberColumn("hours (30d)", format="%.1f"),
        },
    )

with tab_kart:
    kart_id = st.selectbox("Kart", scores["kart_id"].tolist(), format_func=lambda k: f"Kart {k}")
    row = scores[scores["kart_id"] == kart_id].iloc[0]
    a, b, c = st.columns(3)
    a.metric("Risk (14 days)", f"{row['risk_score']:.0%}")
    b.metric("Fleet rank", f"{int(row['risk_rank'])} / {len(scores)}")
    c.metric("Band", BAND_ICON[row["risk_band"]])

    left, right = st.columns(2)
    with left:
        st.subheader("Wear vs service interval")
        wear = pd.DataFrame({"ratio": [row[f"hours_since_{p}"] / SERVICE_INTERVAL_HOURS[p] for p in PARTS]}, index=PARTS)
        st.bar_chart(wear)
        st.caption("1.0 = the part is exactly at its recommended service interval; above 1.0 = overdue.")
    with right:
        st.subheader("Weekly usage (last 26 weeks)")
        s = sessions[sessions["kart_id"] == kart_id].set_index("session_date")["hours_used"]
        weekly = s.resample("W").sum().tail(26)
        st.line_chart(weekly)

    st.subheader("Recent maintenance events")
    ev = events[events["kart_id"] == kart_id].sort_values("event_date", ascending=False).head(15)
    st.dataframe(ev[["event_date", "event_type", "part", "cost", "downtime_hours", "is_breakdown"]],
                 hide_index=True, use_container_width=True)

with tab_model:
    st.subheader("Does the model beat simple alternatives? (test period)")
    rows = []
    for key, label in [("baseline_rule_days_since_preventive", "Rule: days since last preventive service"),
                       ("logistic_regression", "Logistic regression"),
                       ("gradient_boosting", "Gradient boosting")]:
        m = metrics[key]["test"]
        rows.append({"model": label, "ROC-AUC": round(m["roc_auc"], 3), "PR-AUC": round(m["pr_auc"], 3),
                     "precision@top10%": round(m["precision_at_top10pct"], 3),
                     "recall@top10%": round(m["recall_at_top10pct"], 3), "base rate": round(m["positive_rate"], 3)})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.subheader("What drives the predictions (permutation importance)")
    st.bar_chart(importance.set_index("feature")["importance"].head(10))
    st.info(
        "These numbers come from a SIMULATED fleet whose breakdown risk was built to grow with hours since "
        "last service. They show that the pipeline works end to end, not how well it would perform on a real "
        "karting centre. Validation is time-based (train < validation < test), with a 14-day gap."
    )
