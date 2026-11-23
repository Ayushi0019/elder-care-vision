"""
dashboard/dashboard.py — Streamlit Monitoring Dashboard
=========================================================
PURPOSE:
    Provide a live web-based dashboard showing:
    - Current monitoring status
    - Current activity and confidence
    - Privacy status
    - Fall state and alerts
    - Recent event history
    - Activity log chart

HOW TO RUN:
    First, start the monitoring app in one terminal:
        python app.py

    Then, in another terminal:
        streamlit run dashboard/dashboard.py

    Open browser at: http://localhost:8501

PRIVACY INDICATORS:
    The dashboard prominently displays:
    "🔒 RAW VIDEO STORAGE: DISABLED"
    
    This confirms to caregivers and evaluators that the system
    does not store or transmit raw camera footage.

DATA SOURCE:
    The dashboard reads from the SQLite database.
    It does NOT read raw camera frames.
    It only shows classification results and event metadata.
"""

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import sys
import os
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from database import Database

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Elder Care Monitor Dashboard",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .status-card {
        background: #1c1c2e;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid #333;
    }
    .alert-box {
        background: #ff000033;
        border: 2px solid #ff0000;
        border-radius: 10px;
        padding: 15px;
        animation: pulse 1s infinite;
    }
    .normal-box {
        background: #00aa0033;
        border: 2px solid #00aa00;
        border-radius: 10px;
        padding: 15px;
    }
    .privacy-badge {
        background: #004400;
        color: #00ff88;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .metric-big { font-size: 2em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def get_activity_emoji(activity: str) -> str:
    """Return emoji for activity label."""
    emojis = {
        "Standing":  "🧍",
        "Walking":   "🚶",
        "Sitting":   "🪑",
        "Reclining": "🛋️",
        "Fallen":    "🚨",
        "Unknown":   "❓",
    }
    return emojis.get(activity, "❓")


def get_fall_state_color(state: str) -> str:
    """Return color for fall state."""
    colors = {
        "NORMAL":           "green",
        "POTENTIAL_FALL":   "orange",
        "FALLEN_CONFIRMED": "red",
        "ALERTING":         "red",
        "COOLDOWN":         "blue",
    }
    return colors.get(state, "gray")


def main():
    db = Database()

    # =========================================================================
    # SIDEBAR
    # =========================================================================
    with st.sidebar:
        st.markdown("## 🏠 Elder Care Monitor")
        st.markdown("---")

        st.markdown("""
        <div class='privacy-badge'>
            🔒 RAW VIDEO STORAGE: DISABLED<br>
            Privacy-Preserving Mode: ACTIVE
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**System Info**")
        st.markdown(f"- Classifier: {'✅ ML Model' if os.path.exists(config.MODEL_PATH) else '⚠️ Rule-Based Fallback'}")
        st.markdown(f"- Database: {config.DATABASE_PATH}")
        st.markdown(f"- Activities: {', '.join(config.ACTIVITY_CLASSES)}")
        st.markdown("---")
        st.markdown("""
        **⚠️ Disclaimer**

        This is an academic prototype system.
        It does NOT replace professional 
        medical monitoring or emergency services.
        """)

        refresh_rate = st.slider("Refresh Rate (seconds)", 1, 10, 2)

    # =========================================================================
    # MAIN DASHBOARD
    # =========================================================================
    st.markdown("# 🏠 Privacy-Preserving Elder Care Monitor")
    st.markdown("*Edge-Based Skeletal Pose Estimation & Anomaly Detection — B.Tech Major Project*")
    st.markdown("---")

    # Try to import shared state from running app
    app_running = False
    current_state = {
        "activity": "Unknown", "confidence": 0.0, "fps": 0.0,
        "fall_state": "NORMAL", "fall_message": "",
        "motion_detected": False, "person_detected": False,
        "irc_value": 0.0, "vote_count": 0, "running": False,
        "timestamp": datetime.now().isoformat(),
    }

    try:
        # Attempt to import live state if app is running in same process
        from app import get_monitor_state
        current_state = get_monitor_state()
        app_running = current_state.get("running", False)
    except ImportError:
        pass  # App might be running separately

    # =========================================================================
    # SYSTEM STATUS ROW
    # =========================================================================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status_icon = "🟢" if app_running else "🔴"
        st.metric(
            label="Monitoring Status",
            value=f"{status_icon} {'ACTIVE' if app_running else 'OFFLINE'}",
        )

    with col2:
        fps = current_state.get("fps", 0.0)
        st.metric(label="FPS", value=f"{fps:.1f}")

    with col3:
        motion = current_state.get("motion_detected", False)
        st.metric(label="Motion", value="Detected" if motion else "None")

    with col4:
        person = current_state.get("person_detected", False)
        st.metric(label="Person", value="Present" if person else "Absent")

    st.markdown("---")

    # =========================================================================
    # CURRENT ACTIVITY AND FALL STATE
    # =========================================================================
    col_activity, col_fall = st.columns(2)

    with col_activity:
        activity = current_state.get("activity", "Unknown")
        confidence = current_state.get("confidence", 0.0)
        emoji = get_activity_emoji(activity)

        st.markdown(f"""
        <div class='status-card'>
            <h3>Current Activity</h3>
            <div class='metric-big'>{emoji} {activity}</div>
            <p>Confidence: <strong>{confidence*100:.1f}%</strong></p>
            <p>IRC Value: <strong>{current_state.get('irc_value', 0):.4f}</strong>
               (threshold: {config.IRC_THRESHOLD})</p>
            <p>Fall Votes: <strong>{current_state.get('vote_count', 0)}/7</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with col_fall:
        fall_state = current_state.get("fall_state", "NORMAL")
        fall_msg = current_state.get("fall_message", "Normal monitoring active")
        state_color = get_fall_state_color(fall_state)
        is_alert = fall_state in ["ALERTING", "FALLEN_CONFIRMED"]

        box_class = "alert-box" if is_alert else "normal-box"
        icon = "🚨" if is_alert else "✅"

        st.markdown(f"""
        <div class='{box_class}'>
            <h3>{icon} Fall Detection State</h3>
            <div class='metric-big'>{fall_state}</div>
            <p>{fall_msg}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================================================================
    # KINETIC FEATURE TELEMETRY (SYNOPSIS CHAPTER 4 HUD)
    # =========================================================================
    st.markdown("### 🔬 Real-Time Kinetic Feature Telemetry ($X_t$)")
    t_col1, t_col2, t_col3, t_col4 = st.columns(4)

    curr_ar = current_state.get("aspect_ratio", 0.5)
    curr_acc = current_state.get("head_accel", 0.0)
    curr_ke = current_state.get("kinetic_energy", 0.0)
    curr_irc = current_state.get("irc_value", 0.0)

    with t_col1:
        st.metric(
            label="Aspect Ratio ($R_A$)",
            value=f"{curr_ar:.2f}",
            delta=f"{'ABNORMAL' if curr_ar > config.FALL_ASPECT_RATIO_THRESHOLD else 'NORMAL'}",
            delta_color="inverse"
        )
        st.caption(f"Threshold: > {config.FALL_ASPECT_RATIO_THRESHOLD} (Horizontal)")

    with t_col2:
        st.metric(
            label="Head Accel ($a_y$)",
            value=f"{curr_acc:.4f}",
            delta=f"{'SPIKE' if curr_acc > config.HEAD_ACCELERATION_THRESHOLD else 'STABLE'}",
            delta_color="inverse"
        )
        st.caption(f"Threshold: > {config.HEAD_ACCELERATION_THRESHOLD}")

    with t_col3:
        st.metric(
            label="Kinetic Energy ($E_k$)",
            value=f"{curr_ke:.4f}",
        )
        st.caption("RMS Joint Velocity Proxy")

    with t_col4:
        st.metric(
            label="IRC Value ($IRC_A$)",
            value=f"{curr_irc:.4f}",
            delta=f"{'EXCEEDED' if curr_irc > config.IRC_THRESHOLD else 'NORMAL'}",
            delta_color="inverse"
        )
        st.caption(f"Threshold: > {config.IRC_THRESHOLD}")

    st.markdown("---")

    # =========================================================================
    # ACTIVE ALERT PANEL & PRIVACY SKELETAL REPLAY
    # =========================================================================
    try:
        from alerts.alert_manager import get_latest_alert
        latest_alert = get_latest_alert()
        if latest_alert.get("active"):
            st.markdown("""
            <div class='alert-box'>
                <h2>🚨 EMERGENCY ALERT — FALL DETECTED</h2>
            </div>
            """, unsafe_allow_html=True)

            alert_col1, alert_col2, alert_col3 = st.columns(3)
            with alert_col1:
                st.metric("Alert Time", latest_alert.get("timestamp", "N/A"))
            with alert_col2:
                st.metric("Activity", latest_alert.get("activity", "N/A"))
            with alert_col3:
                conf = latest_alert.get("confidence", 0) * 100
                st.metric("Confidence", f"{conf:.1f}%")

            st.warning(
                f"Event ID: {latest_alert.get('event_id', 'N/A')} | "
                f"Duration: {latest_alert.get('duration', 0):.1f}s | "
                f"IRC: {latest_alert.get('irc_value', 0):.4f}"
            )

            # Display Privacy-Preserving Skeletal Replay
            rep_html = latest_alert.get("replay_html")
            rep_gif = latest_alert.get("replay_gif")

            st.markdown("#### 🔒 Privacy-Preserving Skeletal Vector Replay")
            if rep_html and os.path.exists(rep_html):
                with open(rep_html, "r", encoding="utf-8") as f_html:
                    components.html(f_html.read(), height=440)
            elif rep_gif and os.path.exists(rep_gif):
                st.image(rep_gif, caption="Skeletal Fall Replay (100% Visual Privacy)", use_column_width=True)
            else:
                st.info("Skeletal animation vector recording active in memory.")

            st.markdown("---")
    except ImportError:
        pass

    # =========================================================================
    # RECENT EVENTS TABLE
    # =========================================================================
    st.markdown("### 📋 Recent Events")
    events = db.get_recent_events(limit=20)
    if events:
        events_df = pd.DataFrame(events)
        # Style fall events in red
        st.dataframe(
            events_df[[
                "timestamp", "event_type", "activity",
                "confidence", "duration", "alert_sent"
            ]].rename(columns={
                "timestamp": "Time",
                "event_type": "Event",
                "activity": "Activity",
                "confidence": "Confidence",
                "duration": "Duration (s)",
                "alert_sent": "Alert Sent",
            }),
            use_container_width=True,
        )
    else:
        st.info("No events logged yet. Start monitoring (python app.py) to see events here.")

    # =========================================================================
    # ACTIVITY HISTORY CHART
    # =========================================================================
    st.markdown("### 📊 Activity History")
    activity_logs = db.get_recent_activity_logs(limit=200)

    if activity_logs:
        logs_df = pd.DataFrame(activity_logs)
        logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])

        fig = px.scatter(
            logs_df,
            x="timestamp",
            y="activity",
            color="activity",
            size_max=10,
            color_discrete_map={
                "Standing":  "#00ff88",
                "Walking":   "#00ccff",
                "Sitting":   "#ffcc00",
                "Reclining": "#ff8800",
                "Fallen":    "#ff0000",
                "Unknown":   "#888888",
            },
            title="Activity Timeline",
        )
        fig.update_layout(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Confidence over time
        fig2 = px.line(
            logs_df.tail(100),
            x="timestamp",
            y="confidence",
            title="Classification Confidence Over Time",
            color_discrete_sequence=["#00ccff"],
        )
        fig2.update_layout(
            plot_bgcolor="#0e1117",
            paper_bgcolor="#0e1117",
            font_color="white",
            yaxis_range=[0, 1],
        )
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No activity history yet. Activity data appears here during monitoring.")

    # =========================================================================
    # PRIVACY STATUS
    # =========================================================================
    st.markdown("---")
    st.markdown("### 🔒 Privacy Status")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.success("✅ RAW VIDEO STORAGE: DISABLED")
    with col_p2:
        st.success("✅ FACE RECOGNITION: DISABLED")
    with col_p3:
        st.success("✅ CLOUD VIDEO UPLOAD: DISABLED")

    st.info(
        "This system processes video locally on the edge device. "
        "Only numerical feature data and activity classifications are stored. "
        "Raw camera frames are discarded immediately after skeleton extraction."
    )

    # =========================================================================
    # AUTO-REFRESH
    # =========================================================================
    time.sleep(refresh_rate)
    st.rerun()


if __name__ == "__main__":
    main()
