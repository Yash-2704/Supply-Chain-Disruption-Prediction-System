"""Streamlit dashboard entry point — run with: streamlit run dashboard_ui/app.py

Read-only. Data comes ONLY from dashboard_data.queries; all formatting via
dashboard_ui.formatters. No SQL, no invented scoring, no hardcoded resource slugs.
"""
import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dashboard_data.queries import (  # noqa: E402
    get_all_resources, get_alert_feed, get_leaderboard, get_resource_detail,
)
from dashboard_ui import formatters as fmt  # noqa: E402


# --------------------------------------------------------------------------- #
# shared render helpers (thin wrappers over pure formatters)
# --------------------------------------------------------------------------- #

def _render_data_mode_badge(is_real: bool):
    """A validated model gets a success badge; synthetic gets a persistent warning."""
    badge = fmt.format_data_mode_badge(is_real)
    (st.success if is_real else st.warning)(badge)


def _render_prediction(pred, compact: bool = False):
    """Render a PredictionSummary or the 'no prediction' state — visibly distinct."""
    if pred is None:
        st.info(fmt.format_missing_prediction())
        return
    prob_str = fmt.format_probability(pred.calibrated_probability)
    # Synthetic/not-validated: plain text (never a big 'final answer' metric) + warning badge.
    if pred.is_real_data_model:
        st.metric(label="Calibrated probability", value=prob_str)
    else:
        st.write(f"**Calibrated probability:** {prob_str}")
    _render_data_mode_badge(pred.is_real_data_model)
    # Neutral progress bar proportional to the raw number (no invented risk-tier colors).
    if pred.calibrated_probability is not None:
        st.progress(min(max(pred.calibrated_probability, 0.0), 1.0))
    if not compact:
        st.caption(f"horizon: {pred.horizon or '—'} · run_id: {pred.run_id} · "
                   f"data_mode: {pred.data_mode} · generated: {pred.predicted_at or '—'}")
        if pred.top_shap_features:
            st.caption("Top SHAP features:")
            st.json(pred.top_shap_features)


def _render_forecast_status_map(status_by_series: dict):
    if not status_by_series:
        st.caption("No forecasts available for this resource.")
        return
    for series_type, status in sorted(status_by_series.items()):
        st.write(f"- **{series_type}**: {fmt.format_forecast_status(status)}")


# --------------------------------------------------------------------------- #
# view: leaderboard
# --------------------------------------------------------------------------- #

def view_leaderboard():
    st.header("Resource leaderboard")
    st.caption("All resources at a glance. Sparse/synthetic data is shown honestly, "
               "not padded or inflated.")
    board = get_leaderboard()
    for summary in board:
        with st.container(border=True):
            st.subheader(summary.name or summary.resource_id)
            cols = st.columns(3)
            with cols[0]:
                st.markdown("**Prediction**")
                _render_prediction(summary.latest_prediction, compact=True)
            with cols[1]:
                st.markdown("**Forecast status**")
                _render_forecast_status_map(summary.latest_forecast_status_by_series)
            with cols[2]:
                st.markdown("**Explanation**")
                st.write(fmt.format_explanation_status(summary.latest_explanation_status))


# --------------------------------------------------------------------------- #
# view: resource detail
# --------------------------------------------------------------------------- #

def _render_price_chart(price_history):
    if not price_history:
        st.info("No price history available for this resource.")
        return
    dates = [p.date for p in price_history]
    prices = [p.price_usd for p in price_history]
    unit = next((p.unit for p in price_history if p.unit), "value")
    fig = go.Figure(go.Scatter(x=dates, y=prices, mode="lines", name=unit))
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320,
                      xaxis_title="date", yaxis_title=unit)
    st.plotly_chart(fig, width="stretch")
    st.caption("Producer stock-proxy price — a weak market-sentiment proxy, NOT the "
               "resource's own commodity price.")


def _render_forecast_breakdown(rows):
    if not rows:
        st.info("No forecast rows available for this resource.")
        return
    table = [{
        "series_type": r.series_type, "horizon": r.horizon,
        "status": fmt.format_forecast_status(r.status),
        "point_value": r.point_value, "lower_bound": r.lower_bound,
        "upper_bound": r.upper_bound, "disagreement_pct": r.disagreement_pct,
    } for r in rows]
    st.dataframe(table, width="stretch", hide_index=True)


def _render_explanation(expl):
    if expl is None:
        # No row exists at all -> the 'not_attempted' sentinel message.
        st.info(fmt.format_explanation_status("not_attempted"))
        return
    st.write(fmt.format_explanation_status(expl.status))
    _render_data_mode_badge(expl.is_real_data_model)
    st.caption(f"evidence_count: {expl.evidence_count} · generated: {expl.generated_at or '—'}")
    if expl.narrative is not None:
        # Render VERBATIM, including the built-in disclaimer. Never edited.
        st.markdown("**Narrative:**")
        st.write(expl.narrative)
    if expl.citations:
        st.markdown("**Citations:**")
        for c in expl.citations:
            st.write(f"- signal_id `{c.get('signal_id')}`: “{c.get('excerpt', '')}”")


def _render_historical_context(episodes):
    st.markdown("### Historical / reference context")
    st.caption("Known past episodes for background only — this is NOT a current live signal.")
    if not episodes:
        st.info("No historical context available for this resource.")
        return
    for ep in episodes:
        with st.container(border=True):
            st.write(f"**{ep.episode_name or '(unnamed episode)'}**")
            rng = f"{ep.start_date or '?'} → {ep.end_date or '?'}"
            st.caption(f"labeled range: {rng}")
            if ep.justification:
                st.write(ep.justification)


def view_detail(resources):
    st.header("Resource detail")
    if not resources:
        st.info("No resources available.")
        return
    # Dynamic selector — never hardcoded slugs.
    options = [r["id"] for r in resources]
    name_by_id = {r["id"]: r["name"] for r in resources}
    selected = st.selectbox("Select a resource", options,
                            format_func=lambda rid: name_by_id.get(rid, rid))
    detail = get_resource_detail(selected)

    st.subheader(detail.name or detail.resource_id)
    st.markdown("#### Latest prediction")
    _render_prediction(detail.latest_prediction)

    st.markdown("#### Forecast status (summary)")
    _render_forecast_status_map(detail.latest_forecast_status_by_series)

    st.markdown("#### Price history")
    _render_price_chart(detail.price_history)

    st.markdown("#### Forecast breakdown (all series / horizons)")
    _render_forecast_breakdown(detail.forecast_breakdown)

    st.markdown("#### Explanation")
    _render_explanation(detail.explanation)

    _render_historical_context(detail.historical_context)


# --------------------------------------------------------------------------- #
# view: alert feed
# --------------------------------------------------------------------------- #

def view_alert_feed():
    st.header("Recent activity")
    feed = get_alert_feed(limit=20)
    if not feed:
        st.info("No recent activity.")
        return
    for item in feed:
        with st.container(border=True):
            title = f"{item.item_type.upper()} · {item.resource_name or item.resource_id}"
            st.write(f"**{title}**")
            st.caption(f"{item.timestamp or '—'}")
            if item.item_type == "explanation":
                st.write(fmt.format_explanation_status(item.status) if item.status
                         else "—")
                if isinstance(item.short_summary, str) and item.short_summary:
                    st.caption(item.short_summary)
            else:  # prediction
                st.write(f"Calibrated probability: "
                         f"{fmt.format_probability(item.short_summary)}")
            if not item.is_real_data_model:
                _render_data_mode_badge(False)


# --------------------------------------------------------------------------- #
# app shell
# --------------------------------------------------------------------------- #

def main():
    st.set_page_config(page_title="Supply Chain Disruption Dashboard", layout="wide")
    st.title("Supply Chain Disruption Prediction — Dashboard")
    st.caption("Read-only view of the pipeline's honest output. Synthetic / "
               "not-yet-validated results are labeled as such throughout.")

    resources = get_all_resources()
    view = st.sidebar.radio("View", ["Leaderboard", "Resource detail", "Alert feed"])
    if view == "Leaderboard":
        view_leaderboard()
    elif view == "Resource detail":
        view_detail(resources)
    else:
        view_alert_feed()


if __name__ == "__main__":
    main()
else:
    # Streamlit executes this module as a script; run on import under `streamlit run`.
    main()
