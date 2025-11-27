#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# AirLyze Ultra Edition - Single File Streamlit App (Passwordless Login)
# Features:
#  - SQLite user store with username + role only
#  - Registration + login (username only, no password)
#  - Roles (admin / user) with Admin panel
#  - Session tokens + expiry (auto logout after inactivity)
#  - Per-user data upload (saved to user_data/<username>_uploaded.csv)
#  - Simulated sensor streaming (live) or static simulation
#  - Desaturation detection and CSV export
#  - LAN shareable URL hint in sidebar
#  - Uses only standard libs + streamlit, pandas, numpy, plotly
# IMPORTANT: Demo app — not a medical device.

import streamlit as st
import sqlite3
import os
from pathlib import Path
import secrets
import time
from datetime import datetime
import pandas as pd
import numpy as np
import plotly.express as px
import socket
from typing import Optional, Dict, Any, Tuple

# ---------------------------
# Configuration
# ---------------------------

APP_TITLE = "AirLyze — Demo Edition"
DB_FILE = "airlyze_ultra.db"
USER_DATA_DIR = Path("user_data")
USER_DATA_DIR.mkdir(exist_ok=True)
DEFAULT_PORT = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
SESSION_TIMEOUT_SECONDS = 30 * 60  # 30 minutes

# ---------------------------
# Database
# ---------------------------

def init_db(db_file: str = DB_FILE) -> sqlite3.Connection:
    conn = sqlite3.connect(db_file, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TEXT
        )
    """)
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

# Ensure demo admin exists
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
# Sensor simulation & detection
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
                    "duration_s": (seg["timestamp"].iloc[-1] - seg["timestamp"].iloc[0]).total_seconds()
                })
            in_event = False
    if in_event:
        seg = df.loc[start_idx:]
        if len(seg) > 0:
            events.append({
                "start": seg["timestamp"].iloc[0].isoformat(),
                "end": seg["timestamp"].iloc[-1].isoformat(),
                "nadir": float(seg["spo2"].min()),
                "duration_s": (seg["timestamp"].iloc[-1] - seg["timestamp"].iloc[0]).total_seconds()
            })
    return events

# ---------------------------
# Streamlit App & Session
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

# Auto logout
if st.session_state.auth.get("logged_in") and not is_session_active():
    st.warning("Session expired due to inactivity. Please log in again.")
    clear_session_token()
    st.rerun()

# ---------------------------
# Passwordless Authentication UI
# ---------------------------

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

# Show login/register if not logged in
if not st.session_state.auth.get("logged_in", False):
    st.write("## Welcome — please log in or register")
    left, right = st.columns(2)
    with left:
        auth_login_ui()
    with right:
        auth_register_ui()
    st.stop()

# ---------------------------
# Main App continues
# ---------------------------

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
st.sidebar.caption("Share the network URL with people on the same LAN.")

# ---------------------------
# Per-user patient info
# ---------------------------

if "patient_info" not in st.session_state:
    st.session_state.patient_info = {"name":"Test Subject", "age":25, "weight":60.0, "height":165.0}
pname = st.sidebar.text_input("Name", value=st.session_state.patient_info.get("name", ""))
page = st.sidebar.number_input("Age", min_value=0, max_value=120, value=st.session_state.patient_info.get("age",25))
pweight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=float(st.session_state.patient_info.get("weight",60.0)))
pheight = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=float(st.session_state.patient_info.get("height",165.0)))
st.session_state.patient_info.update({"name":pname,"age":page,"weight":pweight,"height":pheight})
bmi = round(pweight / ((pheight/100)**2), 1)
st.sidebar.write(f"**BMI:** {bmi}")

# ---------------------------
# App navigation
# ---------------------------

page_sel = st.sidebar.radio("Navigate", ["Dashboard", "Upload Data", "Account", "Admin" if role=="admin" else "About"])

# ---------------------------
# Page: Dashboard
# ---------------------------

if page_sel == "Dashboard":
    st.header("Dashboard — Live & Simulated Data")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_uploaded = st.checkbox("Use uploaded data (if available)", value=False)
    with col2:
        simulate_duration = st.number_input("Simulation duration (minutes)", min_value=1, max_value=1440, value=180)
    with col3:
        sample_interval = st.selectbox("Sampling interval (s)", [5,10,15,30,60], index=3)

    threshold = st.slider("SpO₂ event threshold", 80, 98, 92)
    manual_hr = st.number_input("Manual Heart Rate (bpm)", min_value=30, max_value=200, value=75)

    mode = st.radio("Mode", ["Static Simulation", "Live Streaming"], index=0)

    user_file = USER_DATA_DIR / f"{username}_uploaded.csv"
    df = None
    if use_uploaded and user_file.exists():
        try:
            df = pd.read_csv(user_file, parse_dates=["timestamp"])
            st.success("Loaded uploaded data.")
        except Exception as e:
            st.error("Failed to load uploaded data: " + str(e))

    if df is None:
        df = simulate_sensor_data(simulate_duration, sample_interval)

    try:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
    except Exception:
        pass

    events = detect_desaturation_events(df, threshold)

    st.subheader("Latest Vitals")
    latest = df.iloc[-1]
    m1,m2,m3 = st.columns(3)
    m1.metric("SpO₂", f"{latest['spo2']}%")
    m2.metric("Heart Rate", f"{manual_hr} bpm")
    m3.metric("Breathing Rate", f"{latest['breathing_rate']} brpm")

    st.subheader("SpO₂ Trend")
    fig = px.line(df, x="timestamp", y="spo2", labels={"timestamp":"Time","spo2":"SpO₂ (%)"})
    fig.add_hline(y=threshold, line_dash="dot", annotation_text=f"Threshold {threshold}%")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Breathing Rate")
    fig2 = px.line(df, x="timestamp", y="breathing_rate", labels={"timestamp":"Time","breathing_rate":"BR (brpm)"})
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Detected Desaturation Events")
    if events:
        evdf = pd.DataFrame(events)
        st.dataframe(evdf)
        st.download_button("Download events CSV", data=evdf.to_csv(index=False).encode("utf-8"), file_name="desaturation_events.csv", mime="text/csv")
    else:
        st.info("No events detected.")

    if mode == "Live Streaming":
        st.markdown("---")
        st.info("Live simulation: streaming data updates in real-time. Press Stop to halt.")
        live_col1, live_col2 = st.columns([1,1])
        running_key = f"live_running_{username}"
        if running_key not in st.session_state:
            st.session_state[running_key] = False
        if st.button("Start Live", key="start_live"):
            st.session_state[running_key] = True
        if st.button("Stop Live", key="stop_live"):
            st.session_state[running_key] = False
        placeholder = st.empty()
        stream_df = simulate_sensor_data(duration_minutes=10, freq_seconds=5, seed=None, desaturation_prob=0.01)
        idx = 0
        while st.session_state[running_key]:
            if idx >= len(stream_df):
                more = simulate_sensor_data(duration_minutes=5, freq_seconds=5, seed=None, desaturation_prob=0.01)
                stream_df = pd.concat([stream_df, more], ignore_index=True)
            window = stream_df.iloc[max(0, idx-100):idx+1].copy().reset_index(drop=True)
            try:
                window["timestamp"] = pd.to_datetime(window["timestamp"])
            except:
                pass
            with placeholder.container():
                st.write("Live window (most recent samples)")
                st.dataframe(window.tail(10))
                fig_live = px.line(window, x="timestamp", y="spo2", labels={"timestamp":"Time","spo2":"SpO₂ (%)"})
                fig_live.add_hline(y=threshold, line_dash="dot")
                st.plotly_chart(fig_live, use_container_width=True)
            idx += 1
            touch_session()
            time.sleep(0.75)
            if not st.session_state.get(running_key):
                break
        placeholder.empty()

# ---------------------------
# Page: Upload Data
# ---------------------------

# ---------------------------
# Sidebar: Download BIDMC Demo Data
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Download BIDMC Demo CSV Files")
import requests

bidmc_urls = []
base = "https://physionet.org/files/bidmc/1.0.0/bidmc_csv/"
for i in range(1, 54):
    subj = f"{i:02d}"
    bidmc_urls.append(base + f"bidmc_{subj}_Numerics.csv")
    bidmc_urls.append(base + f"bidmc_{subj}_Signals.csv")
    bidmc_urls.append(base + f"bidmc_{subj}_Breaths.csv")
    bidmc_urls.append(base + f"bidmc_{subj}_Fix.txt")

if st.sidebar.button("Download All 212 Files"):
    progress_bar = st.sidebar.progress(0)
    for idx, url in enumerate(bidmc_urls):
        fname = USER_DATA_DIR / url.split('/')[-1]
        try:
            r = requests.get(url)
            r.raise_for_status()
            with open(fname, "wb") as f:
                f.write(r.content)
        except Exception as e:
            st.sidebar.warning(f"Failed: {url.split('/')[-1]} ({e})")
        progress_bar.progress((idx + 1) / len(bidmc_urls))
    st.sidebar.success(f"Downloaded all files to {USER_DATA_DIR}")


# ---------------------------
# Page: Account
# ---------------------------

elif page_sel == "Account":
    st.header("Account Info")
    st.write(f"Username: **{username}**")
    st.write(f"Role: **{role}**")
    st.write(f"Created: **{user_obj.get('created_at','-')}**")

# ---------------------------
# Page: Admin
# ---------------------------

elif page_sel == "Admin" and role == "admin":
    st.header("Admin Dashboard")
    cur = _db.cursor()
    cur.execute("SELECT username, role, created_at FROM users ORDER BY created_at DESC")
    rows = cur.fetchall()
    users_df = pd.DataFrame(rows, columns=["username","role","created_at"])
    st.dataframe(users_df)
    st.markdown("---")
    st.subheader("Delete a user (and their data)")
    del_user = st.text_input("Username to delete")
    if st.button("Delete user"):
        if del_user and del_user != username and get_user_db(del_user):
            cur.execute("DELETE FROM users WHERE username = ?", (del_user,))
            _db.commit()
            f = USER_DATA_DIR / f"{del_user}_uploaded.csv"
            if f.exists():
                f.unlink()
            st.success(f"Deleted {del_user} and their data (if present).")
        else:
            st.error("Invalid username or attempt to delete self.")

# ---------------------------
# Page: About
# ---------------------------

else:
    st.header("About AirLyze Demo")
    st.markdown("""
    **AirLyze** is an educational demonstration of respiratory monitoring and desaturation event detection.  

    - Simulated or uploaded SpO₂ & breathing rate data  
    - Detects low SpO₂ events  
    - Demo purpose only, **not a medical device**  

    Built with Python, Streamlit, Pandas, NumPy, Plotly.
    """)


upload
