import os
import json
import math
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime

from profiler import ProfilerPipeline
from explainability import RiskExplainer

# -------------------------------------------------------------------------
# PAGE CONFIGURATION & STYLING
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="SOC Behavioral Anomaly Engine",
    page_icon="SOC",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Apple-inspired Light Theme & Liquid Glass (Ultra Minimal) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Inter", sans-serif !important;
    }

    /* Force Light Mode Background & Base Colors */
    .stApp {
        background-color: #F5F5F7 !important;
    }
    
    /* Strict Text Coloring for Light Theme */
    h1, h2, h3, h4, h5, h6, p, label, li, span, .stMarkdown, .stText {
        color: #1D1D1F !important;
    }
    
    /* Exception: Keep metric deltas colored properly */
    [data-testid="stMetricDelta"] * {
        color: inherit !important;
    }

    /* Headings refined typography */
    h1 { font-weight: 700 !important; letter-spacing: -0.04em !important; font-size: 2.2rem !important; }
    h2 { font-weight: 600 !important; letter-spacing: -0.02em !important; }
    h3 { font-weight: 600 !important; letter-spacing: -0.01em !important; }

    /* Glassmorphism for Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(245, 245, 247, 0.65) !important;
        backdrop-filter: blur(30px) !important;
        -webkit-backdrop-filter: blur(30px) !important;
        border-right: 1px solid rgba(0, 0, 0, 0.05) !important;
    }
    
    /* Fix Sidebar close button & SVG icons visibility */
    section[data-testid="stSidebar"] button svg {
        stroke: #1D1D1F !important;
        fill: #1D1D1F !important;
        color: #1D1D1F !important;
    }

    /* Metrics Liquid Glass Cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: saturate(180%) blur(20px);
        -webkit-backdrop-filter: saturate(180%) blur(20px);
        padding: 20px;
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.6);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
    }
    [data-testid="stMetric"]:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
        background: rgba(255, 255, 255, 0.9);
    }
    [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        font-size: 2rem !important;
        letter-spacing: -0.02em !important;
    }

    /* Button styling - Apple iOS style */
    .stButton>button {
        width: 100%;
        border-radius: 12px !important;
        font-weight: 600 !important;
        background-color: rgba(255, 255, 255, 0.8) !important;
        border: 1px solid rgba(0, 0, 0, 0.08) !important;
        color: #007AFF !important;
        backdrop-filter: blur(10px);
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.02) !important;
        padding: 10px 16px !important;
    }
    .stButton>button:hover {
        background-color: #007AFF !important;
        border: 1px solid #007AFF !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3) !important;
    }

    /* Inputs and Selectboxes */
    div[data-baseweb="select"] > div {
        border-radius: 12px !important;
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Segmented Control Tabs (iOS style) */
    .stTabs [data-baseweb="tab-list"] {
        background-color: rgba(118, 118, 128, 0.12) !important;
        padding: 3px !important;
        border-radius: 9px !important;
        gap: 0px !important;
        border: none !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 7px !important;
        padding: 6px 16px !important;
        font-weight: 500 !important;
        color: #1D1D1F !important;
        border: none !important;
        margin: 0 !important;
        background-color: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.12), 0 3px 1px rgba(0, 0, 0, 0.04) !important;
        border-bottom: none !important;
    }
    
    /* Clean up dataframes (Excel sheet look) */
    [data-testid="stDataFrame"] {
        background: #FFFFFF !important;
        border-radius: 16px !important;
        padding: 0 !important;
        border: 1px solid rgba(0, 0, 0, 0.06) !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04) !important;
        overflow: hidden !important;
    }
    
    /* Alert Metadata JSON Block */
    .stJson {
        background: #FFFFFF !important;
        border-radius: 16px !important;
        padding: 16px !important;
        border: 1px solid rgba(0, 0, 0, 0.06) !important;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.04) !important;
    }
    </style>
""", unsafe_allow_html=True)

FEEDBACK_FILE = "data/analyst_feedback.json"
DEFAULT_ALERT_THRESHOLD = 0.50  # fallback only if no trained threshold is found

# -------------------------------------------------------------------------
# HELPER FUNCTIONS & DATA LOADING
# -------------------------------------------------------------------------
@st.cache_resource
def load_ml_pipeline():
    """Loads saved models, encoders, and SHAP explainer."""
    if not os.path.exists('models/detector.json') or not os.path.exists('models/encoders.pkl'):
        return None, None, DEFAULT_ALERT_THRESHOLD

    profiler = ProfilerPipeline()
    profiler.encoders = joblib.load('models/encoders.pkl')
    profiler.global_model = joblib.load('models/global_profiler.pkl')
    profiler.entity_models = joblib.load('models/entity_profiles.pkl')
    if os.path.exists('models/entity_known_fingerprints.pkl'):
        profiler.entity_known_fingerprints = joblib.load('models/entity_known_fingerprints.pkl')
    if os.path.exists('models/entity_history.pkl'):
        profiler.entity_history = joblib.load('models/entity_history.pkl')

    explainer = RiskExplainer()

    # Real top-1%-alert-budget threshold computed during training (see
    # detector.py's optimize_precision_at_k). Falls back to 0.50 only if an
    # older model directory doesn't have it yet.
    alert_threshold = DEFAULT_ALERT_THRESHOLD
    if os.path.exists('models/alert_threshold.pkl'):
        alert_threshold = joblib.load('models/alert_threshold.pkl')

    return profiler, explainer, alert_threshold


@st.cache_data
def load_raw_logs(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_feedback(feedback_data):
    os.makedirs("data", exist_ok=True)
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback_data, f, indent=2)


@st.cache_data
def run_inference(df_raw: pd.DataFrame, _profiler, _explainer):
    """Processes raw logs through profiler and detector to predict anomaly
    types and scores. Cached on df_raw's content: the original version of
    this function ran, uncached, on every single widget interaction
    (slider drag, tab switch, button click) because Streamlit reruns the
    whole script top-to-bottom on every interaction. Combined with the
    (since-fixed) slow row-by-row scoring loop in profiler.py, that meant
    every click could take on the order of minutes on a full-size dataset.
    Leading underscores on _profiler/_explainer tell Streamlit's cache not
    to try to hash those (they're already de-duplicated via
    @st.cache_resource in load_ml_pipeline)."""
    df_proc = _profiler.transform_features(df_raw.copy())
    df_proc['baseline_anomaly_score'] = _profiler.compute_anomaly_scores(df_raw)

    feature_cols = _explainer.feature_cols
    X = df_proc[feature_cols]

    probs = _explainer.model.predict_proba(X)
    classes = _explainer.label_encoder.classes_

    normal_idx = list(classes).index('normal') if 'normal' in classes else 0

    pred_idx = np.argmax(probs, axis=1)
    pred_class = classes[pred_idx]
    prob_score = probs[np.arange(len(probs)), pred_idx]
    risk_score = 1.0 - probs[:, normal_idx]

    df_res = pd.DataFrame({
        "predicted_label": pred_class,
        "risk_score": np.round(risk_score, 4),
        "confidence": np.round(prob_score, 4),
    })

    df_combined = pd.concat([df_raw.reset_index(drop=True), df_res], axis=1)
    df_proc = df_proc.reset_index(drop=True)
    return df_combined, df_proc


def maybe_refresh_entity_profile(profiler, feedback, df_results, entity_id):
    """Concept-drift hook: once an entity has a few analyst-rejected
    (false-positive) alerts on record, re-fit that entity's baseline
    excluding those rows, so a legitimate new pattern stops being
    permanently re-flagged. This is what actually consumes the analyst
    feedback loop -- in the original submission, feedback was written to
    disk for display purposes only and never fed back into any model."""
    rejected = [
        int(idx) for idx, v in feedback.items()
        if v.get("status") == "FALSE_POSITIVE"
        and idx.isdigit()
        and int(idx) in df_results.index
        and df_results.loc[int(idx), 'entity_id'] == entity_id
    ]
    if len(rejected) >= 3:
        return profiler.update_profile(entity_id, rejected_indices=rejected)
    return False


# -------------------------------------------------------------------------
# MAIN DASHBOARD INTERFACE
# -------------------------------------------------------------------------
def main():
    st.title("AI-Powered Behavioral Anomaly Detection SOC Dashboard")
    st.caption("Real-Time Autonomous Intrusion Profiling & SHAP Explainability Engine")

    profiler, explainer, trained_alert_threshold = load_ml_pipeline()
    if not profiler or not explainer:
        st.error("Model artifacts not found in `models/`. Please execute `data_generator.py` and `detector.py` first.")
        st.stop()

    stream_path = "data/streaming_logs.csv"
    if not os.path.exists(stream_path):
        stream_path = "data/train_logs.csv"

    df_raw = load_raw_logs(stream_path)
    df_results, df_processed = run_inference(df_raw, profiler, explainer)

    feedback = load_feedback()

    # ---------------------------------------------------------------------
    # ALERT BUDGET: exactly top-K by rank, not a probability threshold
    # ---------------------------------------------------------------------
    # A percentile/threshold cutoff (what this used to do) breaks the moment
    # MORE than 1% of events tie at the same top score -- `>= threshold`
    # then lets ALL of them through, not just 1%. Confirmed on your real
    # model: it's confident enough that ~4% of events tie at the exact max
    # score, so a value-based cutoff can't hit a 1% budget no matter what
    # value it uses. A budget is a COUNT, so it needs a rank-based cap:
    # always take exactly the top K rows (ties broken by recency), which
    # holds regardless of how large the tied cluster at the top is.
    total_events = len(df_results)
    budget_k = max(1, math.ceil(0.01 * total_events))
    ranked_all = df_results.sort_values(['risk_score', 'timestamp'], ascending=[False, False], kind='stable')
    budget_df = ranked_all.head(budget_k)
    budget_min_exact = float(budget_df['risk_score'].min()) if len(budget_df) else 0.0
    slider_default = round(budget_min_exact, 2)  # widget-safe value on the 0.01 step grid
    tied_at_boundary = int((df_results['risk_score'] == budget_min_exact).sum())

    # ---------------------------------------------------------------------
    # SIDEBAR CONTROLS
    # ---------------------------------------------------------------------
    st.sidebar.header("Triaging Controls")

    st.sidebar.caption(
        f"Alert budget: top **{budget_k}** events (~1% of {total_events:,}). "
        f"(Reference only -- trained validation-set threshold was {trained_alert_threshold:.3f}.)"
    )
    if tied_at_boundary > budget_k:
        st.sidebar.caption(
            f"Notice: {tied_at_boundary} events currently tie at the boundary score "
            f"({budget_min_exact:.3f}) -- the model is very confident on this synthetic "
            f"data, so more events qualify than the strict budget. Showing the most "
            f"recent {budget_k}; use the slider below to see the rest."
        )

    min_risk = st.sidebar.slider(
        "Minimum Risk Score (narrows within budget; drag left to see more)",
        0.0, 1.0, slider_default, step=0.01
    )
    with st.sidebar.expander("Why a budget instead of just a slider?"):
        st.write(
            "\"Top 1% alert budget\" is a count, not a probability value -- a SOC team "
            "can review N alerts a day, not \"whatever crosses probability X\". This dashboard "
            "always ranks every event and takes the top K by rank first (K = 1% of total "
            "events), then the slider lets you widen or narrow further from there. This stays "
            "correct even when many events tie at the same top confidence score, which a "
            "fixed probability cutoff alone cannot guarantee."
        )

    selected_entities = st.sidebar.multiselect(
        "Filter Entity Type",
        options=df_results['entity_type'].unique(),
        default=df_results['entity_type'].unique()
    )

    attack_categories = [c for c in df_results['predicted_label'].unique() if c != 'normal']
    selected_attacks = st.sidebar.multiselect(
        "Filter Attack Taxonomy",
        options=attack_categories,
        default=attack_categories
    )

    # min_risk defaults to slider_default (i.e. "show the whole budget", using
    # budget_df exactly rather than re-filtering by the rounded value, so
    # rounding to the widget's step grid can never accidentally exclude rows
    # that genuinely belong in the budget). Moving the slider left of the
    # default widens beyond the strict top-K budget into the full ranked
    # list; moving it right narrows within the budget.
    if min_risk >= slider_default:
        base_df = budget_df[budget_df['risk_score'] >= min_risk] if min_risk > slider_default else budget_df
    else:
        base_df = ranked_all[ranked_all['risk_score'] >= min_risk]

    filtered_df = base_df[
        (base_df['entity_type'].isin(selected_entities)) &
        ((base_df['predicted_label'].isin(selected_attacks)) | (base_df['predicted_label'] == 'normal'))
    ]

    # ---------------------------------------------------------------------
    # TOP KPI METRICS BAR
    # ---------------------------------------------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)

    total_alerts = len(budget_df)  # the headline number always reflects the actual budget size
    critical_alerts = len(df_results[df_results['risk_score'] >= 0.85])
    # "Cold-start" = entity_id never appeared in TRAINING at all (uses the
    # full training-time entity roster, not just entities dense enough to
    # get their own IsolationForest -- the original KPI conflated "brand
    # new" with "seen, but with sparse history").
    known_entities = set(getattr(profiler, 'entity_history', {}).keys()) or set(profiler.entity_models.keys())
    cold_starts = len(df_results[~df_results['entity_id'].isin(known_entities)])
    triage_count = len(feedback)

    col1.metric("Total Events Telemetry", f"{total_events:,}")
    col2.metric("Alert Budget (Top ~1%)", f"{total_alerts:,}", delta=f"{round(total_alerts/max(1,total_events)*100, 1)}%")
    col3.metric("Critical Risk Alerts", f"{critical_alerts:,}", delta_color="inverse")
    col4.metric("Cold-Start Entities", f"{cold_starts:,}")
    col5.metric("Analyst Triaged", f"{triage_count}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Ranked Alert Queue", "Alert Deep Dive", "Entity Behavioral History"])

    # ---------------------------------------------------------------------
    # TAB 1: RANKED ALERT QUEUE
    # ---------------------------------------------------------------------
    with tab1:
        st.subheader(f"High-Priority Alert Queue -- {len(filtered_df):,} of {total_events:,} events (Top ~1% Budget)")

        if filtered_df.empty:
            st.info("No alerts match the selected threshold criteria.")
        else:
            display_df = filtered_df.copy()
            display_df['Analyst_Status'] = display_df.index.astype(str).map(lambda idx: feedback.get(idx, {}).get("status", "UNREVIEWED"))

            cols = ['risk_score', 'predicted_label', 'entity_id', 'entity_type', 'timestamp',
                    'geo_location', 'resource_accessed', 'Analyst_Status', 'source_ip']

            st.dataframe(
                display_df[cols].style.background_gradient(cmap="Reds", subset=['risk_score']),
                use_container_width=True,
                height=400
            )

    # ---------------------------------------------------------------------
    # TAB 2: DEEP DIVE & SHAP EXPLAINABILITY
    # ---------------------------------------------------------------------
    with tab2:
        st.subheader("Alert Feature Attribution & Triage")

        alerts_df = filtered_df[filtered_df['predicted_label'] != 'normal']

        if alerts_df.empty:
            st.success("No anomalous alerts selected for deep dive inspection.")
        else:
            alert_options = {f"Alert #{idx} | {row['predicted_label'].upper()} | Entity: {row['entity_id']} | Risk: {row['risk_score']}": idx
                             for idx, row in alerts_df.iterrows()}

            selected_key = st.selectbox("Select Alert for Deep Investigation:", list(alert_options.keys()))
            alert_idx = alert_options[selected_key]
            alert_row = df_results.loc[alert_idx]
            proc_row = df_processed.loc[[alert_idx]][explainer.feature_cols]

            d_col1, d_col2 = st.columns([1, 1])

            with d_col1:
                st.markdown("### Event Metadata")
                st.json({
                    "Entity ID": alert_row['entity_id'],
                    "Entity Type": alert_row['entity_type'],
                    "Timestamp": str(alert_row['timestamp']),
                    "Source IP": alert_row['source_ip'],
                    "Geo Location": alert_row['geo_location'],
                    "Resource": alert_row['resource_accessed'],
                    "Auth Method": alert_row['auth_method'],
                    "Device Fingerprint": alert_row['device_fingerprint'],
                    "Command Sequence": alert_row['command_sequence']
                })

            with d_col2:
                st.markdown("### AI Assessment")
                # insider_drift is explicitly an "edge case ... used for
                # false-positive tuning" per the problem statement, not a
                # confirmed threat -- so it gets an amber warning banner
                # instead of the same red "error" treatment as a confirmed
                # attack pattern like brute force or credential stuffing.
                if alert_row['predicted_label'] == 'insider_drift':
                    st.warning(f"Ambiguous Pattern (edge case): {alert_row['predicted_label'].upper()} -- gradually expanding access. May be a legitimate role change; recommend manual review rather than auto-escalation.")
                else:
                    st.error(f"Threat Classification: {alert_row['predicted_label'].upper()}")
                st.metric("Risk Score Confidence", f"{round(alert_row['risk_score'] * 100, 2)}%")

                is_cold = alert_row['entity_id'] not in known_entities
                if is_cold:
                    st.warning("Cold-Start Entity: First time entity seen. Assessed via Global Profiler Model.")

                explanation = explainer.explain_alert(proc_row)
                st.info(f"SOC Summary: {explanation['human_readable']}")

            st.markdown("---")
            st.markdown("### SHAP Feature Attribution (Why was this flagged?)")

            if explanation['top_factors']:
                shap_df = pd.DataFrame(explanation['top_factors'])

                fig = px.bar(
                    shap_df,
                    x='impact',
                    y='feature',
                    orientation='h',
                    title="Top Feature Contributions to Anomaly Detection",
                    color='impact',
                    color_continuous_scale=[(0, "#FFC6C4"), (1, "#FF3B30")] # Apple Red gradient
                )
                
                # Apply Apple-style minimal layout to the Plotly chart
                fig.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#1D1D1F", family="-apple-system, BlinkMacSystemFont, 'SF Pro Display', Arial, sans-serif"),
                    yaxis={'categoryorder': 'total ascending', 'showgrid': False, 'zeroline': False},
                    xaxis={'showgrid': True, 'gridcolor': "rgba(0,0,0,0.04)", 'zeroline': False},
                    margin=dict(l=20, r=20, t=50, b=20)
                )
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            st.markdown("### Analyst Feedback & Model Tuning")
            f_col1, f_col2, f_col3 = st.columns([1, 1, 2])

            str_idx = str(alert_idx)
            current_status = feedback.get(str_idx, {}).get("status", "UNREVIEWED")
            st.write(f"Current Analyst Verdict: **{current_status}**")

            if f_col1.button("Confirm True Positive (Escalate)"):
                feedback[str_idx] = {"status": "CONFIRMED_TRUE_POSITIVE", "timestamp": str(datetime.now())}
                save_feedback(feedback)
                st.success("Incident Confirmed & Saved.")
                st.rerun()

            if f_col2.button("Flag as False Positive (Tune)"):
                feedback[str_idx] = {"status": "FALSE_POSITIVE", "timestamp": str(datetime.now())}
                save_feedback(feedback)
                updated = maybe_refresh_entity_profile(profiler, feedback, df_results, alert_row['entity_id'])
                if updated:
                    st.warning(f"Flagged as False Positive. Enough rejections accumulated for `{alert_row['entity_id']}` -- baseline profile refreshed to reduce repeat false alarms.")
                else:
                    st.warning("Flagged as False Positive for retraining feedback.")
                st.rerun()

    # ---------------------------------------------------------------------
    # TAB 3: ENTITY BEHAVIORAL HISTORY & TIMELINE
    # ---------------------------------------------------------------------
    with tab3:
        st.subheader("Entity Context & Timeline Reconstruction")

        selected_entity = st.selectbox("Select Entity ID to Inspect History:", df_results['entity_id'].unique())

        entity_df = df_results[df_results['entity_id'] == selected_entity].sort_values("timestamp")

        st.markdown(f"**Historical Telemetry Events for `{selected_entity}` ({len(entity_df)} records)**")

        fig_timeline = px.scatter(
            entity_df,
            x="timestamp",
            y="risk_score",
            color="predicted_label",
            size="session_duration",
            hover_data=["resource_accessed", "geo_location", "source_ip"],
            title=f"Behavioral Anomaly Timeline: {selected_entity}",
            color_discrete_map={"normal": "#34C759"} # Apple Green for normal
        )
        
        # Apply Apple-style minimal layout to timeline chart
        fig_timeline.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#1D1D1F", family="-apple-system, BlinkMacSystemFont, 'SF Pro Display', Arial, sans-serif"),
            margin=dict(l=20, r=20, t=50, b=20),
            xaxis={'showgrid': False, 'zeroline': False},
            yaxis={'showgrid': True, 'gridcolor': "rgba(0,0,0,0.04)", 'zeroline': False}
        )
        
        fig_timeline.add_hline(y=min_risk, line_dash="dash", line_color="#FF3B30", # Apple Red
                               annotation_text=f"Alert Threshold ({min_risk:.3f})")
        st.plotly_chart(fig_timeline, use_container_width=True)

        st.dataframe(entity_df[['timestamp', 'predicted_label', 'risk_score', 'resource_accessed', 'geo_location', 'session_duration', 'command_sequence']], use_container_width=True)

if __name__ == "__main__":
    main()