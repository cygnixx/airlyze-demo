#!/usr/bin/env python
# AirLyze — Full Rewritten Streamlit App (with enhancements)
# Features:
# - Passwordless login (SQLite)
# - Upload raw timestamp+spo2 CSVs
# - Auto-generate breathing_rate if missing
# - Synchronized / zoomable plots with shared time-range controls
# - Multi-axis combined plot (SpO2 + BR)
# - Automatic downsampling for long recordings
# - Desaturation detection and export
# - Option to load previously generated all_patients_waveform.csv if present

import streamlit as st
import sqlite3
import os
from pathlib import Path
import secrets
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import socket
from typing import Optional, Dict, Any, Tuple
import zipfile
import io

# ---------------------------
# Config
# ---------------------------
APP_TITLE = "AirLyze — Enhanced Demo"
DB_FILE = "airlyze_ultra.db"
USER_DATA_DIR = Path("user_data")
USER_DATA_DIR.mkdir(exist_ok=True)
DEFAULT_PORT = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
SESSION_TIMEOUT_SECONDS = 30 * 60  # 30 minutes
MAX_POINTS_PLOT = 5000  # threshold for downsampling

# ---------------------------
# DB helpers
# ---------------------------
def init_db(db_file: str = DB_FILE) -> sqlite3.Connection:
    conn = sqlite3.connect(db_file, check_same_thread=False)
    cur = conn.cursor()
    cur.execute(\"\"\"
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT
        )
    \"\"\")
    conn.commit()
    return conn

_db = init_db()

def add_user_db(username: str, role: str='user') -> bool:
    cur = _db.cursor()
    created = datetime.now().isoformat()
    try:
        cur.execute("INSERT INTO users (username, role, created_at) VALUES (?, ?, ?)",
                    (username, role, created))
        _db.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def get_user_db(username: str) -> Optional[Dict[str, Any]]:
    cur = _db.cursor()
    cur.execute("SELECT username, role, created_at FROM users WHERE username = ?", (username,))
    r = cur.fetchone()
    if not r:
        return None
    return {"username": r[0], "role": r[1], "created_at": r[2]}

if get_user_db("admin") is None:
    add_user_db("admin", role="admin")

# ---------------------------
# Networking helper
# ---------------------------
def get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        try:
            return socket.gethostbyname(socket.gethostname())
        except:
            return "127.0.0.1"

def build_urls(port: str = DEFAULT_PORT) -> Tuple[str, str]:
    local = f"http://localhost:{port}"
    network = f"http://{get_local_ip()}:{port}"
    return local, network

# ---------------------------
# Simulation & detection
# ---------------------------
def simulate_sensor_data(duration_minutes:int=180, freq_seconds:int=30, seed:int=None, desaturation_prob:float=0.002) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)
    n = max(1, int(duration_minutes * 60 / freq_seconds))
    base_time = datetime.now() - pd.to_timedelta(n * freq_seconds, unit='s')
    times = [base_time + pd.to_timedelta(i * freq_seconds, unit='s') for i in range(n)]
    spo2 = 96 + 1.5 * np.sin(np.linspace(0, 8 * np.pi, n)) + np.random.normal(0, 0.6, n)
    for i in range(n):
        if np.random.rand() < desaturation_prob:
            length = np.random.randint(2, min(12, n - i))
            depth = np.random.uniform(4, 12)
            spo2[i:i+length] -= depth
    spo2 = np.clip(spo2, 70, 100).round(1)
    br = 15 + 1.5 * np.sin(np.linspace(0, 4 * np.pi, n)) + np.random.normal(0, 0.7, n)
    br = np.clip(br, 6, 40).round(1)
    sleep_flag = [1 if (t.hour >= 22 or t.hour < 7) else 0 for t in times]
    return pd.DataFrame({"timestamp": times, "spo2": spo2, "breathing_rate": br, "sleep_flag": sleep_flag})

def detect_desaturation_events(df: pd.DataFrame, threshold: int=92) -> list:
    if df is None or len(df) == 0:
        return []
    df = df.sort_values("timestamp").reset_index(drop=True)
    events = []
    in_event = False
    start_idx = None
    for i, v in df["spo2"].items():
        if pd.isna(v):
            continue
        if v < threshold and not in_event:
            in_event = True
            start_idx = i
        elif v >= threshold and in_event:
            seg = df.loc[start_idx:i-1]
            if len(seg) > 0:
                events.append({
                    "start": seg["timestamp"].iloc[0].isoformat(),
                    "end": seg["timestamp"].iloc[-1].isoformat(),
                    "nadir": float(seg["spo2"].min()),
                    "duration_s": float((seg["timestamp"].iloc[-1] - seg["timestamp"].iloc[0]).total_seconds())
                })
            in_event = False
    if in_event:
        seg = df.loc[start_idx:]
        if len(seg) > 0:
            events.append({
                "start": seg["timestamp"].iloc[0].isoformat(),
                "end": seg["timestamp"].iloc[-1].isoformat(),
                "nadir": float(seg["spo2"].min()),
                "duration_s": float((seg["timestamp"].iloc[-1] - seg["timestamp"].iloc[0]).total_seconds())
            })
    return events

# ---------------------------
# Utilities
# ---------------------------
def ensure_breathing_rate(df: pd.DataFrame) -> pd.DataFrame:
    if "breathing_rate" not in df.columns or df["breathing_rate"].isna().all():
        n = len(df)
        t = np.linspace(0, 4*np.pi, n)
        br = 15 + 2*np.sin(t) + np.random.normal(0, 0.3, n)
        br = np.clip(br, 6, 35).round(1)
        df["breathing_rate"] = br
    return df

def downsample_for_plot(df: pd.DataFrame, max_points:int=MAX_POINTS_PLOT) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    # uniform subsample
    idx = np.linspace(0, len(df)-1, max_points, dtype=int)
    return df.iloc[idx].reset_index(drop=True)

def make_combined_figure(df: pd.DataFrame, show_secondary_y:bool=False, title:str="SpO₂ and Breathing Rate"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["spo2"], mode="lines", name="SpO₂ (%)", yaxis="y1"))
    if show_secondary_y:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["breathing_rate"], mode="lines", name="Breathing Rate (brpm)", yaxis="y2"))
        fig.update_layout(
            yaxis=dict(title="SpO₂ (%)", side="left"),
            yaxis2=dict(title="Breathing Rate (brpm)", overlaying="y", side="right")
        )
    else:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["breathing_rate"], mode="lines", name="Breathing Rate (brpm)", yaxis="y1"))
        fig.update_layout(yaxis=dict(title="Value"))
    fig.update_layout(title=title, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    fig.update_xaxes(nticks=10, tickformat="%Y-%m-%d\n%H:%M:%S", tickangle=-45)
    return fig

# ---------------------------
# Streamlit app
# ---------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Educational demo — not a medical device.")

if "auth" not in st.session_state:
    st.session_state.auth = {"logged_in": False, "user": None, "token": None, "last_active": time.time()}

def create_session_token(username: str) -> str:
    token = secrets.token_urlsafe(32)
    st.session_state.auth = {"logged_in": True, "user": username, "token": token, "last_active": time.time()}
    return token

def clear_session_token():
    st.session_state.auth = {"logged_in": False, "user": None, "token": None, "last_active": time.time()}

def is_session_active() -> bool:
    info = st.session_state.auth
    if not info.get("logged_in"):
        return False
    if time.time() - info.get("last_active", 0) > SESSION_TIMEOUT_SECONDS:
        return False
    return True

def touch_session():
    st.session_state.auth["last_active"] = time.time()

if st.session_state.auth.get("logged_in") and not is_session_active():
    st.warning("Session expired due to inactivity. Please log in again.")
    clear_session_token()
    st.rerun()

# Auth UI
def auth_register_ui():
    st.subheader("Create an account")
    with st.form("register_form"):
        username = st.text_input("Username", key="reg_user")
        role = st.selectbox("Role", ["user", "admin"], index=0)
        submitted = st.form_submit_button("Register")
        if submitted:
            if not username:
                st.error("Enter a username.")
            elif get_user_db(username) is not None:
                st.error("Username already exists.")
            else:
                add_user_db(username, role=role)
                st.success(f"Account created. You can log in as {username}.")

def auth_login_ui():
    st.subheader("Login")
    with st.form("login_form"):
        username = st.text_input("Username", key="login_user")
        submitted = st.form_submit_button("Login")
        if submitted:
            u = get_user_db(username)
            if u:
                create_session_token(username)
                st.success(f"Logged in as {username}")
                st.rerun()
            else:
                st.error("User not found. Please register first.")

if not st.session_state.auth.get("logged_in", False):
    st.write("## Welcome — please log in or register")
    left, right = st.columns(2)
    with left:
        auth_login_ui()
    with right:
        auth_register_ui()
    st.stop()

touch_session()
username = st.session_state.auth["user"]
user_obj = get_user_db(username)
role = user_obj["role"] if user_obj else "user"
st.sidebar.success(f"{username} ({role})")
if st.sidebar.button("Logout"):
    clear_session_token()
    st.rerun()

local_url, network_url = build_urls(DEFAULT_PORT)
st.sidebar.markdown("### Access URLs")
st.sidebar.code(local_url)
st.sidebar.code(network_url)

# Patient info
if "patient_info" not in st.session_state:
    st.session_state.patient_info = {"name":"Test Subject", "age":25, "weight":60.0, "height":165.0}
pname = st.sidebar.text_input("Name", value=st.session_state.patient_info.get("name",""))
page = st.sidebar.number_input("Age", 0,120, value=st.session_state.patient_info.get("age",25))
pweight = st.sidebar.number_input("Weight (kg)",20.0,300.0,value=float(st.session_state.patient_info.get("weight",60.0)))
pheight = st.sidebar.number_input("Height (cm)",100.0,250.0,value=float(st.session_state.patient_info.get("height",165.0)))
st.session_state.patient_info.update({"name":pname,"age":page,"weight":pweight,"height":pheight})
bmi = round(pweight / ((pheight/100)**2), 1)
st.sidebar.write(f"**BMI:** {bmi}")

# Navigation
page_sel = st.sidebar.radio("Navigate", ["Dashboard", "Upload Data", "Account", "Admin" if role=='admin' else "About"])

# Dashboard
if page_sel == "Dashboard":
    st.header("Dashboard — Visualizer & Analysis")
    col1, col2 = st.columns([2,1])
    with col1:
        use_uploaded = st.checkbox("Use uploaded data (if available)", value=True)
        mode = st.radio("Mode", ["Simulate", "Live"], index=0)
    with col2:
        simulate_duration = st.number_input("Simulation duration (minutes)", 1, 1440, 180)
        sample_interval = st.selectbox("Sample interval (s)", [1,5,10,30,60], index=1)

    threshold = st.slider("SpO₂ event threshold", 80, 98, 92)
    show_combined = st.checkbox("Show combined SpO₂ + BR plot", value=True)
    sync_range = st.checkbox("Enable shared time-range controls (synchronizes plots)", value=True)
    downsample_limit = st.number_input("Max points to plot before downsampling", min_value=1000, max_value=20000, value=MAX_POINTS_PLOT, step=500)

    user_file = USER_DATA_DIR / f"{username}_uploaded.csv"
    df = None

    if use_uploaded and user_file.exists():
        try:
            df = pd.read_csv(user_file)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            st.success("Loaded uploaded data.")
        except Exception as e:
            st.error(f"Failed to load uploaded data: {e}")

    if df is None:
        df = simulate_sensor_data(simulate_duration, sample_interval)

    df = df.sort_values("timestamp").reset_index(drop=True)
    df = ensure_breathing_rate(df)

    # automatic downsampling for plotting responsiveness
    if len(df) > downsample_limit:
        st.warning(f"Large dataset ({len(df)} samples) — downsampling to {downsample_limit} points for plotting.")
        plot_df = downsample_for_plot(df, max_points=downsample_limit)
    else:
        plot_df = df.copy()

    # Shared time-range controls
    min_t = plot_df["timestamp"].min()
    max_t = plot_df["timestamp"].max()
    if sync_range:
        start_dt, end_dt = st.slider("Display time range", min_value=min_t, max_value=max_t, value=(min_t, max_t), format="YYYY-MM-DD HH:mm:ss")
        mask = (plot_df["timestamp"] >= start_dt) & (plot_df["timestamp"] <= end_dt)
        plot_view = plot_df.loc[mask]
    else:
        plot_view = plot_df

    # Individual figures
    spo2_fig = px.line(plot_view, x="timestamp", y="spo2", labels={"timestamp":"Time", "spo2":"SpO₂ (%)"}, title="SpO₂ Timeline")
    spo2_fig.add_hline(y=threshold, line_dash="dot", annotation_text=f"Threshold {threshold}%")
    spo2_fig.update_xaxes(nticks=10, tickformat="%Y-%m-%d\n%H:%M:%S", tickangle=-45)

    br_fig = px.line(plot_view, x="timestamp", y="breathing_rate", labels={"timestamp":"Time", "breathing_rate":"BR (brpm)"}, title="Breathing Rate Timeline")
    br_fig.update_xaxes(nticks=10, tickformat="%Y-%m-%d\n%H:%M:%S", tickangle=-45)

    if show_combined:
        combined_fig = make_combined_figure(plot_view, show_secondary_y=True, title="Combined SpO₂ & Breathing Rate")
        st.plotly_chart(combined_fig, use_container_width=True)
    else:
        st.plotly_chart(spo2_fig, use_container_width=True)
        st.plotly_chart(br_fig, use_container_width=True)

    # Events table & download
    events = detect_desaturation_events(df, threshold)
    st.subheader("Detected Desaturation Events")
    if events:
        evdf = pd.DataFrame(events)
        st.dataframe(evdf)
        st.download_button("Download events CSV", data=evdf.to_csv(index=False).encode("utf-8"), file_name="desaturation_events.csv", mime="text/csv")
    else:
        st.info("No events detected.")

    # Option to export current plotted window to CSV
    if st.button("Download current view as CSV"):
        out_df = plot_view.copy()
        out_df["timestamp"] = out_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.download_button("Download CSV", data=out_df.to_csv(index=False).encode("utf-8"), file_name=f"{username}_view.csv", mime="text/csv")

# Upload Data page
elif page_sel == "Upload Data":
    st.header("Upload CSV (timestamp, spo2[, breathing_rate])")
    st.write("Upload a CSV containing at least 'timestamp' and 'spo2' columns.")

    uploaded_file = st.file_uploader("Choose CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            # try utf-8, fallback to utf-8-sig, latin1
            try:
                df_user = pd.read_csv(uploaded_file)
            except Exception:
                uploaded_file.seek(0)
                df_user = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.stop()

        # normalize column names
        df_user.columns = df_user.columns.str.strip().str.lower()
        df_user.rename(columns={"time": "timestamp", "time_stamp":"timestamp", "ts":"timestamp", "spo₂":"spo2", "o2":"spo2"}, inplace=True)

        # event-style file detection
        if set(df_user.columns) >= {"duration_s", "nadir", "patient"} and not {"timestamp","spo2"} <= set(df_user.columns):
            st.warning("This appears to be an event table (duration_s, nadir, patient). Use the Dashboard 'Generate Synthetic' option or convert externally.")
            st.dataframe(df_user.head())
            st.stop()

        required = {"timestamp","spo2"}
        missing = required - set(df_user.columns)
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
            st.stop()

        try:
            df_user["timestamp"] = pd.to_datetime(df_user["timestamp"])
        except:
            st.error("Cannot parse timestamp column.")
            st.stop()

        df_user = ensure_breathing_rate(df_user)
        out_path = USER_DATA_DIR / f"{username}_uploaded.csv"
        df_user.to_csv(out_path, index=False)
        st.success(f"Saved uploaded file as {out_path.name}")
        st.dataframe(df_user.head())

# Account
elif page_sel == "Account":
    st.header("Account")
    st.write(f"Username: **{username}**")
    st.write(f"Role: **{role}**")
    st.write(f"Created: **{user_obj.get('created_at','-')}**")

# Admin
elif page_sel == "Admin" and role == "admin":
    st.header("Admin")
    cur = _db.cursor()
    cur.execute("SELECT username, role, created_at FROM users ORDER BY created_at DESC")
    rows = cur.fetchall()
    st.dataframe(pd.DataFrame(rows, columns=["username","role","created_at"]))

    st.markdown("---")
    st.subheader("Delete user")
    del_user = st.text_input("Username to delete")
    if st.button("Delete user"):
        if del_user and del_user != username and get_user_db(del_user):
            cur.execute("DELETE FROM users WHERE username = ?", (del_user,))
            _db.commit()
            f = USER_DATA_DIR / f"{del_user}_uploaded.csv"
            if f.exists():
                f.unlink()
            st.success(f"Deleted {del_user}")
        else:
            st.error("Invalid username or attempt to delete self.")

# About
else:
    st.header("About")
    st.markdown(\"\"\"
    AirLyze — Educational demo for respiratory monitoring.
    - Upload timestamped SpO₂ CSVs (timestamp, spo2[, breathing_rate])
    - Visualize and export detected desaturation events
    - Not a medical device.
    \"\"\")

