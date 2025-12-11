#!/usr/bin/env python
# coding: utf-8

# AirLyze Ultra Edition — Clean Rewrite (NO BIDMC, NO demo CSVs, NO io.StringIO)
# Fully stable version — passwordless login, uploads, simulation, desat detection.

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

# ============================================================
# CONFIG
# ============================================================

APP_TITLE = "AirLyze — Demo Edition"
DB_FILE = "airlyze_ultra.db"
USER_DATA_DIR = Path("user_data")
USER_DATA_DIR.mkdir(exist_ok=True)
DEFAULT_PORT = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
SESSION_TIMEOUT_SECONDS = 30 * 60  # 30 minutes

# ============================================================
# DATABASE
# ============================================================

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

# Ensure default admin exists
if get_user_db("admin") is None:
    add_user_db("admin", role="admin")

# ============================================================
# NETWORK HELPERS
# ============================================================

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

# ============================================================
# SENSOR SIMULATION + EVENT DETECTION
# ============================================================

def simulate_sensor_data(duration_minutes:int=180, freq_seconds:int=30,
                         seed:int=None, desaturation_prob:float=0.002) -> pd.DataFrame:
    if seed is not None:
        np.random.seed(seed)

    n = max(1, int(duration_minutes * 60 / freq_seconds))
    base_time = datetime.now() - pd.to_timedelta(n * freq_seconds, unit='s')
    times = [base_time + pd.to_timedelta(i * freq_seconds, unit='s') for i in range(n)]

    spo2 = 96 + 1.5 * np.sin(np.linspace(0, 8*np.pi, n)) + np.random.normal(0, 0.6, n)

    for i in range(n):
        if np.random.rand() < desaturation_prob:
            length = np.random.randint(2, min(12, n - i))
            depth = np.random.uniform(4, 12)
            spo2[i:i+length] -= depth

    spo2 = np.clip(spo2, 70, 100).round(1)

    br = 15 + 1.5 * np.sin(np.linspace(0, 4*np.pi, n)) + np.random.normal(0, 0.7, n)
    br = np.clip(br, 6, 40).round(1)

    sleep_flag = [1 if (t.hour >= 22 or t.hour < 7) else 0 for t in times]

    return pd.DataFrame({
        "timestamp": times,
        "spo2": spo2,
        "breathing_rate": br,
        "sleep_flag": sleep_flag
    })

def detect_desaturation_events(df: pd.DataFrame, threshold:int=92) -> list:
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

# ============================================================
# STREAMLIT — AUTH + SESSION
# ============================================================

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Educational demo — not a medical device.")

if "auth" not in st.session_state:
    st.session_state.auth = {
        "logged_in": False,
        "user": None,
        "token": None,
        "last_active": time.time()
    }

def create_session_token(username: str) -> str:
    token = secrets.token_urlsafe(32)
    st.session_state.auth = {
        "logged_in": True,
        "user": username,
        "token": token,
        "last_active": time.time()
    }
    return token

def clear_session_token():
    st.session_state.auth = {
        "logged_in": False,
        "user": None,
        "token": None,
        "last_active": time.time()
    }

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

# ============================================================
# LOGIN + REGISTER
# ============================================================

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
                st.success(f"Account created. You can now log in as {username}.")

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

# ============================================================
# MAIN APP AFTER LOGIN
# ============================================================

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

# ============================================================
# PATIENT INFO PANEL
# ============================================================

if "patient_info" not in st.session_state:
    st.session_state.patient_info = {
        "name":"Test Subject",
        "age":25,
        "weight":60.0,
        "height":165.0
    }

pname = st.sidebar.text_input("Name", value=st.session_state.patient_info.get("name",""))
page = st.sidebar.number_input("Age", 0,120, value=st.session_state.patient_info.get("age",25))
pweight = st.sidebar.number_input("Weight (kg)",20.0,300.0,value=float(st.session_state.patient_info.get("weight",60.0)))
pheight = st.sidebar.number_input("Height (cm)",100.0,250.0,value=float(st.session_state.patient_info.get("height",165.0)))

st.session_state.patient_info.update({"name":pname,"age":page,"weight":pweight,"height":pheight})

bmi = round(pweight / ((pheight/100)**2), 1)
st.sidebar.write(f"**BMI:** {bmi}")

# ============================================================
# NAVIGATION
# ============================================================

page_sel = st.sidebar.radio("Navigate", ["Dashboard", "Upload Data", "Account", "Admin" if role=="admin" else "About"])

# ============================================================
# DASHBOARD PAGE
# ============================================================

if page_sel == "Dashboard":
    st.header("Dashboard — Live & Simulated Data")

    col1, col2, col3 = st.columns(3)
    with col1:
        use_uploaded = st.checkbox("Use uploaded data (if available)", value=False)
    with col2:
        simulate_duration = st.number_input("Simulation duration (minutes)", 1, 1440, 180)
    with col3:
        sample_interval = st.selectbox("Sampling interval (s)", [5,10,15,30,60], index=3)

    threshold = st.slider("SpO₂ event threshold", 80, 98, 92)
    manual_hr = st.number_input("Manual Heart Rate (bpm)", 30, 200, 75)

    mode = st.radio("Mode", ["Static Simulation", "Live Streaming"], index=0)

    # ============================================================
    # STATIC MODE
    # ============================================================
    if mode == "Static Simulation":
        user_file = USER_DATA_DIR / f"{username}_uploaded.csv"
        df = None

        if use_uploaded and user_file.exists():
            try:
                df = pd.read_csv(user_file)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                st.success("Loaded uploaded data.")
            except:
                st.error("Failed to load uploaded file.")

        if df is None:
            df = simulate_sensor_data(simulate_duration, sample_interval)

        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

        events = detect_desaturation_events(df, threshold)

        st.subheader("Latest Vitals")
        latest = df.iloc[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("SpO₂", f"{latest['spo2']}%")
        c2.metric("Heart Rate", f"{manual_hr} bpm")
        c3.metric("Breathing Rate", f"{latest['breathing_rate']} brpm")

        st.subheader("SpO₂ Trend")
        fig = px.line(df, x="timestamp", y="spo2")
        fig.add_hline(y=threshold, line_dash="dot", annotation_text=f"{threshold}%")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Breathing Rate")
        fig2 = px.line(df, x="timestamp", y="breathing_rate")
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Detected Desaturation Events")
        if events:
            evdf = pd.DataFrame(events)
            st.dataframe(evdf)
            st.download_button(
                "Download events CSV",
                evdf.to_csv(index=False).encode("utf-8"),
                file_name="desaturation_events.csv",
                mime="text/csv"
            )
        else:
            st.info("No events detected.")


      def generate_live_sample():
    now = datetime.now()

    spo2 = float(96 + 1.5 * np.sin(time.time() / 20) + np.random.normal(0, 0.3))
    spo2 = round(np.clip(spo2, 70, 100), 1)

    br = float(15 + 1.2 * np.sin(time.time() / 35) + np.random.normal(0, 0.4))
    br = round(np.clip(br, 6, 40), 1)

    sleep_flag = 1 if (now.hour >= 22 or now.hour < 7) else 0

    return {
        "timestamp": now,
        "spo2": spo2,
        "breathing_rate": br,
        "sleep_flag": sleep_flag
    }


       # ============================================================
    # LIVE STREAMING MODE — TRUE 1-SECOND UPDATES
    # ============================================================
    else:
        st.success("Live streaming is active. Updates every 1 second.")

        # Initialize buffer
        if "live_df" not in st.session_state:
            st.session_state.live_df = pd.DataFrame(
                columns=["timestamp", "spo2", "breathing_rate", "sleep_flag"]
            )

        # Start button persistent state
        if "streaming_active" not in st.session_state:
            st.session_state.streaming_active = True

        # Stop button
        if st.button("STOP Live Stream"):
            st.session_state.streaming_active = False
            st.info("Live streaming stopped.")
            st.stop()

        # Only run if still active
        if st.session_state.streaming_active:

            # Add a new live data sample
            new_row = generate_live_sample()
            st.session_state.live_df = pd.concat(
                [st.session_state.live_df, pd.DataFrame([new_row])],
                ignore_index=True
            )

            df_live = st.session_state.live_df.copy()
            events_live = detect_desaturation_events(df_live, threshold)

            # VITALS
            st.subheader("Latest Vitals")
            latest = df_live.iloc[-1]

            c1, c2, c3 = st.columns(3)
            c1.metric("SpO₂", f"{latest['spo2']}%")
            c2.metric("Heart Rate", f"{manual_hr} bpm")
            c3.metric("Breathing Rate", f"{latest['breathing_rate']} brpm")

            # PLOTS
            st.subheader("Live SpO₂ Trend")
            fig1 = px.line(df_live, x="timestamp", y="spo2")
            fig1.add_hline(y=threshold, line_dash="dot")
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("Live Breathing Rate")
            fig2 = px.line(df_live, x="timestamp", y="breathing_rate")
            st.plotly_chart(fig2, use_container_width=True)

            st.subheader("Recent Samples")
            st.dataframe(df_live.tail(20))

            st.subheader("Detected Desaturation Events")
            if events_live:
                st.dataframe(pd.DataFrame(events_live))
            else:
                st.info("No events detected yet.")

            # Re-run after 1 second
            time.sleep(1)
            st.rerun()



# ============================================================
# UPLOAD DATA PAGE
# ============================================================

elif page_sel == "Upload Data":
    st.header("Upload Patient Data")
    st.write("Your CSV must contain **timestamp** and **spo2** columns. Optional: breathing_rate.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df_user = None

        try:
            df_user = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"❌ Failed to load CSV: {e}")
            st.stop()

        required = {"timestamp","spo2"}
        missing = required - set(df_user.columns)
        if missing:
            st.error("❌ Missing columns: " + ", ".join(missing))
            st.stop()

        try:
            df_user["timestamp"] = pd.to_datetime(df_user["timestamp"])
        except:
            st.error("❌ Cannot parse timestamp column.")
            st.stop()

        if "breathing_rate" not in df_user.columns:
            df_user["breathing_rate"] = np.nan

        user_file = USER_DATA_DIR / f"{username}_uploaded.csv"
        df_user.to_csv(user_file, index=False)

        st.success("File uploaded successfully.")
        st.dataframe(df_user.head())

        st.subheader("Detected Desaturation Events")
        events = detect_desaturation_events(df_user, threshold=92)

        if events:
            evdf = pd.DataFrame(events)
            st.dataframe(evdf)
            st.download_button(
                "Download Events CSV",
                evdf.to_csv(index=False).encode("utf-8"),
                "desaturation_events.csv",
                "text/csv"
            )
        else:
            st.info("No detected events.")

        st.subheader("SpO₂ Timeline")
        fig = px.line(df_user, x="timestamp", y="spo2")
        fig.add_hline(y=92, line_dash="dot")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Breathing Rate Timeline")
        fig2 = px.line(df_user, x="timestamp", y="breathing_rate")
        st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# ACCOUNT PAGE
# ============================================================

elif page_sel == "Account":
    st.header("Account Info")
    st.write(f"Username: **{username}**")
    st.write(f"Role: **{role}**")
    st.write(f"Created: **{user_obj.get('created_at','-')}**")

# ============================================================
# ADMIN PAGE
# ============================================================

elif page_sel == "Admin" and role == "admin":
    st.header("Admin Dashboard")

    cur = _db.cursor()
    cur.execute("SELECT username, role, created_at FROM users ORDER BY created_at DESC")
    rows = cur.fetchall()

    st.dataframe(pd.DataFrame(rows, columns=["username","role","created_at"]))

    st.markdown("---")
    st.subheader("Delete a user")

    del_user = st.text_input("Username to delete")
    if st.button("Delete user"):
        if del_user and del_user != username and get_user_db(del_user):
            cur.execute("DELETE FROM users WHERE username = ?", (del_user,))
            _db.commit()

            f = USER_DATA_DIR / f"{del_user}_uploaded.csv"
            if f.exists():
                f.unlink()

            st.success(f"Deleted user {del_user}.")
        else:
            st.error("Invalid username or attempting self-delete.")

# ============================================================
# ABOUT PAGE
# ============================================================

else:
    st.header("About AirLyze Demo")
    st.markdown("""
    **AirLyze** is an educational demonstration showing how SpO₂ and breathing rate
    can be monitored, analyzed, and used to detect desaturation events.

    - Upload real or synthetic CSV files  
    - Detect low oxygen saturation events  
    - Simulate multi-hour data  
    - Plot trends and export events  

    Not a medical device.
    """)

