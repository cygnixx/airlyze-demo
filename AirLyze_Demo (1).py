#!/usr/bin/env python
# coding: utf-8

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

page_sel = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Upload Data", "Account", "Admin" if role=="admin" else "About", "Download ECG+SpO2"]
)

# ---------------------------
# Page: Download ECG + SpO2 (with simulated SpO2)
# ---------------------------

if page_sel == "Download ECG+SpO2":
    st.header("Download ECG + Simulated SpO₂ Data from PhysioNet")

    start_btn = st.button("Start Download")
    if start_btn:
        import wfdb
        output_folder = "apnea_ecg_with_spo2_csv"
        merged_csv_file = "apnea_ecg_with_spo2_merged.csv"
        os.makedirs(output_folder, exist_ok=True)

        db_name = "apnea-ecg"
        records = wfdb.get_record_list(db_name)
        merged_frames = []

        st.write(f"Found {len(records)} records to process.")
        placeholder = st.empty()
        progress_bar = st.progress(0)

        def simulate_spo2(n_points: int, start_time: datetime):
            times = [start_time + pd.Timedelta(seconds=i*1) for i in range(n_points)]
            spo2 = 96 + 1.5 * np.sin(np.linspace(0, 8*np.pi, n_points)) + np.random.normal(0,0.6,n_points)
            for i in range(n_points):
                if np.random.rand() < 0.002:
                    length = np.random.randint(2, min(12, n_points-i))
                    depth = np.random.uniform(4,12)
                    spo2[i:i+length] -= depth
            spo2 = np.clip(spo2,70,100).round(1)
            return spo2, times

        for i, record in enumerate(records, start=1):
            placeholder.text(f"Processing {record} ({i}/{len(records)}) ...")
            try:
                wfdb.dl_record(record, db_name=db_name, pn_dir=db_name)
                data = wfdb.rdrecord(record, pn_dir=db_name)
                df = pd.DataFrame(data.p_signal, columns=data.sig_name)

                # Simulate SpO2
                spo2, times = simulate_spo2(len(df), start_time=datetime(2000,1,1))
                df["spo2"] = spo2
                df["time"] = times
                df["subject_id"] = record

                outpath = os.path.join(output_folder, f"{record}.csv")
                df.to_csv(outpath, index=False)
                merged_frames.append(df)
                placeholder.text(f"Saved {record} → {outpath}")
            except Exception as e:
                placeholder.text(f"Error processing {record}: {e}")
            progress_bar.progress(i / len(records))

        if merged_frames:
            full_df = pd.concat(merged_frames, ignore_index=True)
            full_df.to_csv(merged_csv_file, index=False)
            st.success(f"Merged CSV saved as: {merged_csv_file}")
        else:
            st.info("No records processed.")

        placeholder.empty()
        progress_bar.empty()

# ---------------------------
# Rest of your existing pages:
# Dashboard, Upload Data, Account, Admin, About
# (keep your previous code unchanged)
# ---------------------------
