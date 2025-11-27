#!/usr/bin/env python
# coding: utf-8

"""
AirLyze Ultra Edition - Single File Streamlit App (Passwordless Login)
Clean, fixed, drop-in rewrite of your app with robust CSV handling.
Not a medical device — demo only.
"""

# ---------------------------
# Imports (io must be first to avoid NameError)
# ---------------------------
import io
import zipfile
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
import socket
from typing import Optional, Dict, Any, Tuple

# ---------------------------
# Configuration
# ---------------------------
APP_TITLE = "AirLyze — Demo Edition"
DB_FILE = "airlyze_ultra.db"
USER_DATA_DIR = Path("user_data")
USER_DATA_DIR.mkdir(exist_ok=True)
SYNTHETIC_DIR = Path("synthetic_patient_data")
SYNTHETIC_DIR.mkdir(exist_ok=True)
DEFAULT_PORT = os.environ.get("STREAMLIT_SERVER_PORT", "8501")
SESSION_TIMEOUT_SECONDS = 30 * 60  # 30 minutes

# ---------------------------
# Database helpers
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
# Network helper
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
    """Simulate a timeseries of spo2 & breathing_rate."""
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
    """Detect contiguous segements where spo2 < threshold."""
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
# Authentication UI
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

# If not logged in, show auth UI
if not st.session_state.auth.get("logged_in", False):
    st.write("## Welcome — please log in or register")
    left, right = st.columns(2)
    with left:
        auth_login_ui()
    with right:
        auth_register_ui()
    st.stop()

# ---------------------------
# Main App Layout / Sidebar
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

# Per-user patient info
if "patient_info" not in st.session_state:
    st.session_state.patient_info = {"name":"Test Subject", "age":25, "weight":60.0, "height":165.0}
pname = st.sidebar.text_input("Name", value=st.session_state.patient_info.get("name", ""))
page = st.sidebar.number_input("Age", min_value=0, max_value=120, value=st.session_state.patient_info.get("age",25))
pweight = st.sidebar.number_input("Weight (kg)", min_value=20.0, max_value=300.0, value=float(st.session_state.patient_info.get("weight",60.0)))
pheight = st.sidebar.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=float(st.session_state.patient_info.get("height",165.0)))
st.session_state.patient_info.update({"name":pname,"age":page,"weight":pweight,"height":pheight})
bmi = round(pweight / ((pheight/100)**2), 1)
st.sidebar.write(f"**BMI:** {bmi}")

# Navigation
page_sel = st.sidebar.radio("Navigate", ["Dashboard", "Upload Data", "Account", "Admin" if role=="admin" else "About"])

# ---------------------------
# Dashboard page
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
    fig2 = px.line(df, x="timestamp", y="breathing_rate", labels={"timestamp":"Time","breathing_rate":"BR (brpm)"} )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Detected Desaturation Events")
    if events:
        evdf = pd.DataFrame(events)
        st.dataframe(evdf)
        st.download_button("Download events CSV", data=evdf.to_csv(index=False).encode("utf-8"), file_name="desaturation_events.csv", mime="text/csv")
    else:
        st.info("No events detected.")

    # Live streaming simulation (simple)
    if mode == "Live Streaming":
        st.markdown("---")
        st.info("Live simulation: streaming data updates in real-time. Press Stop to halt.")
        running_key = f"live_running_{username}"
        if running_key not in st.session_state:
            st.session_state[running_key] = False
        col_start, col_stop = st.columns(2)
        if col_start.button("Start Live"):
            st.session_state[running_key] = True
        if col_stop.button("Stop Live"):
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
# Upload Data (Full Rewritten Section)
# ---------------------------
elif page_sel == "Upload Data":
    st.header("Upload Patient Data (clean_data.csv)")
    st.write("Your CSV must contain **'timestamp'** and **'spo2'** columns. Optional: *breathing_rate*.")

    uploaded_file = st.file_uploader("Drag and drop clean_data.csv here (Max 200 MB)", type=["csv"])

    df_user = None

    if uploaded_file is not None:
        # File size check
        try:
            uploaded_file.seek(0, os.SEEK_END)
            file_size = uploaded_file.tell()
            uploaded_file.seek(0)
        except Exception:
            file_size = None

        if file_size is not None and file_size > 200 * 1024 * 1024:
            st.error("❌ File too large — limit is 200MB.")
            st.stop()

        # Try common encodings + delimiter sniff
        raw = uploaded_file.read()
        uploaded_file.seek(0)
        if isinstance(raw, bytes):
            # try decode
            try:
                sample = raw.decode('utf-8')
                encoding = 'utf-8'
            except UnicodeDecodeError:
                try:
                    sample = raw.decode('utf-8-sig')
                    encoding = 'utf-8-sig'
                except Exception:
                    sample = raw.decode('latin1')
                    encoding = 'latin1'
        else:
            sample = str(raw)
            encoding = 'utf-8'

        # auto-detect delimiter using csv.Sniffer
        import csv as _csv
        delimiter = ','
        try:
            sniff = _csv.Sniffer().sniff(sample[:2048])
            delimiter = sniff.delimiter
        except Exception:
            delimiter = ','

        # Read into dataframe
        try:
            uploaded_file.seek(0)
            df_user = pd.read_csv(uploaded_file, delimiter=delimiter, encoding=encoding)
        except Exception as e:
            st.error(f"❌ Could not read CSV: {e}")
            st.stop()

        # Normalize columns (strip, lower)
        df_user.columns = df_user.columns.str.strip().str.lower()
        df_user.rename(columns={
            "time": "timestamp",
            "time_stamp": "timestamp",
            "ts": "timestamp",
            "spo₂": "spo2",
            "o2": "spo2",
            "oxy": "spo2",
            "spo2 (%)": "spo2"
        }, inplace=True)

        # If event-style CSV (duration_s, nadir, patient) - offer to convert
        if set(df_user.columns) >= {"duration_s", "nadir", "patient"} and not {"timestamp","spo2"} <= set(df_user.columns):
            st.warning("This looks like an event-only CSV (duration_s, nadir, patient). The app expects raw timestamp+spo2 data.")
            if st.button("Auto-convert events → waveform (1 Hz)"):
                # Convert events to waveform (basic back-to-back concatenation per patient)
                # We'll create patient files and then load the selected one
                def gen_waveform_from_events(events_df, start_time=None, hz=1):
                    if start_time is None:
                        start_time = datetime(2025,1,1,0,0,0)
                    rows = []
                    current = start_time
                    for _, r in events_df.iterrows():
                        dur = int(r.get("duration_s", 30))
                        nad = float(r.get("nadir", 88))
                        # simple triangular dip
                        t = np.arange(dur)
                        d1 = max(1, dur//3)
                        d2 = max(1, dur//3)
                        d3 = dur - d1 - d2
                        spo = np.ones(dur)*97.0
                        if d1>0:
                            spo[:d1] = 97 - ( (np.sin(np.linspace(0,np.pi/2,d1))) * (97-nad) )
                        if d2>0:
                            spo[d1:d1+d2] = nad + np.random.normal(0,0.3,d2)
                        if d3>0:
                            spo[d1+d2:] = nad + ( (np.sin(np.linspace(0,np.pi/2,d3))) * (97-nad) )
                        spo += np.random.normal(0,0.2,dur)
                        spo = np.clip(spo,70,100)
                        for i, val in enumerate(spo):
                            rows.append({"timestamp": current + timedelta(seconds=i), "spo2": round(float(val),1)})
                        current = current + timedelta(seconds=dur+1)
                    return pd.DataFrame(rows)
                # Ask which patient to convert or all
                patients = sorted(df_user['patient'].unique())
                choice = st.selectbox("Select patient to convert (or choose All)", ["All"] + list(patients))
                if st.button("Generate waveform CSV(s)"):
                    if choice == "All":
                        # generate a combined file that contains all patients, labeled
                        combined_rows = []
                        for p in patients:
                            sub = df_user[df_user['patient']==p]
                            wf = gen_waveform_from_events(sub, start_time=datetime(2025,1,1,0,0,0))
                            wf['patient'] = p
                            combined_rows.append(wf)
                        combined = pd.concat(combined_rows, ignore_index=True)
                        out_path = USER_DATA_DIR / f"{username}_converted_events_all.csv"
                        combined.to_csv(out_path, index=False)
                        st.success(f"Generated waveform CSV: {out_path.name}")
                        st.download_button("Download converted CSV", data=out_path.read_bytes(), file_name=out_path.name, mime="text/csv")
                    else:
                        sub = df_user[df_user['patient']==choice]
                        wf = gen_waveform_from_events(sub, start_time=datetime(2025,1,1,0,0,0))
                        out_path = USER_DATA_DIR / f"{username}_converted_{choice}.csv"
                        wf.to_csv(out_path, index=False)
                        st.success(f"Generated waveform CSV: {out_path.name}")
                        st.download_button("Download converted CSV", data=out_path.read_bytes(), file_name=out_path.name, mime="text/csv")
            st.stop()

        # Now check required columns
        required = {"timestamp", "spo2"}
        if not required <= set(df_user.columns):
            missing = required - set(df_user.columns)
            st.error("❌ Missing required columns: " + ", ".join(sorted(missing)))
            st.stop()

        # Parse timestamp
        try:
            df_user["timestamp"] = pd.to_datetime(df_user["timestamp"])
        except Exception:
            st.error("❌ 'timestamp' column cannot be parsed as datetime.")
            st.stop()

        # Ensure breathing_rate exists
        if "breathing_rate" not in df_user.columns:
            df_user["breathing_rate"] = np.nan

        # Save uploaded file
        user_file = USER_DATA_DIR / f"{username}_uploaded.csv"
        df_user.to_csv(user_file, index=False)
        st.success(f"✅ File successfully uploaded and saved as {user_file.name}")
        st.dataframe(df_user.head())

        # Event detection and UI
        st.subheader("Detected Desaturation Events")
        events = detect_desaturation_events(df_user, threshold=92)
        if events:
            evdf = pd.DataFrame(events)
            st.dataframe(evdf)
            st.download_button("Download events CSV", data=evdf.to_csv(index=False).encode("utf-8"), file_name="desaturation_events.csv", mime="text/csv")
        else:
            st.info("No desaturation events detected in this file.")

        # Plotly charts
        st.subheader("SpO₂ Timeline")
        fig_spo2 = px.line(df_user, x="timestamp", y="spo2", labels={"timestamp":"Time","spo2":"SpO₂ (%)"})
        fig_spo2.add_hline(y=92, line_dash="dot", annotation_text="Threshold 92%")
        st.plotly_chart(fig_spo2, use_container_width=True)

        st.subheader("Breathing Rate")
        if "breathing_rate" in df_user.columns and not df_user['breathing_rate'].isnull().all():
            fig_br = px.line(df_user, x="timestamp", y="breathing_rate", labels={"timestamp":"Time","breathing_rate":"BR (brpm)"})
            st.plotly_chart(fig_br, use_container_width=True)

    for fname, content in demo_files.items():
        st.download_button(label=f"Download {fname}", data=content, file_name=fname, mime="text/csv")

        # Safe parse and event generation
        try:
            df_demo = pd.read_csv(io.StringIO(content))
            if "timestamp" in df_demo.columns:
                df_demo["timestamp"] = pd.to_datetime(df_demo["timestamp"])
            else:
                continue
            if "spo2" not in df_demo.columns:
                continue
            ev_demo = detect_desaturation_events(df_demo, threshold=92)
            if ev_demo:
                evdf_demo = pd.DataFrame(ev_demo)
                ev_name = fname.replace(".csv", "_events.csv")
                st.download_button(label=f"Download {ev_name}", data=evdf_demo.to_csv(index=False).encode("utf-8"), file_name=ev_name, mime="text/csv")
        except Exception:
            # don't raise; demo should not crash the app
            continue

    # Synthetic patient files generator + zip download
    st.markdown("---")
    st.subheader("Generate Demo Patient Files")
    synth_minutes = st.number_input("Minutes per file", min_value=1, max_value=1440, value=180)
    synth_interval = st.selectbox("Sampling interval (seconds)", [1,5,10,30,60], index=1)
    if st.button("Generate & Download All 10 Demo Patients (ZIP)"):
        # create synthetic files in memory
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for pid in range(1, 11):
                df_syn = simulate_sensor_data(duration_minutes=synth_minutes, freq_seconds=synth_interval, seed=pid)
                # ensure timestamp is isoformat string
                df_syn["timestamp"] = pd.to_datetime(df_syn["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                fname = f"patient_{pid:02d}.csv"
                csv_bytes = df_syn.to_csv(index=False).encode("utf-8")
                zf.writestr(fname, csv_bytes)
        buf.seek(0)
        st.download_button("Download synthetic_patient_data.zip", data=buf, file_name="synthetic_patient_data.zip", mime="application/zip")

# ---------------------------
# Account page
# ---------------------------
elif page_sel == "Account":
    st.header("Account Info")
    st.write(f"Username: **{username}**")
    st.write(f"Role: **{role}**")
    st.write(f"Account created at: {user_obj.get('created_at','-')}")

# ---------------------------
# Admin page
# ---------------------------
elif page_sel == "Admin" and role == "admin":
    st.header("Admin Panel")
    st.subheader("Registered Users")
    cur = _db.cursor()
    cur.execute("SELECT username, role, created_at FROM users ORDER BY created_at DESC")
    users = cur.fetchall()
    st.dataframe(pd.DataFrame(users, columns=["Username","Role","Created At"]))

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
# About
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
