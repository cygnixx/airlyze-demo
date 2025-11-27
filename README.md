# AirLyze Demo

This is a demo Streamlit app for **AirLyze** — an educational respiratory monitoring tool.

## Features

- SQLite user store with username + role
- Registration + login (username only, no password)
- Roles (admin / user) with Admin panel
- Session tokens + expiry (auto logout after inactivity)
- Per-user data upload (saved to `user_data/<username>_uploaded.csv`)
- Simulated sensor streaming (live) or static simulation
- Desaturation detection and CSV export
- LAN shareable URL hint in sidebar
- Download ECG + simulated SpO₂ data from PhysioNet Apnea-ECG records

> ⚠️ Demo app — not a medical device

## Installation

1. Clone the repository:

```bash
git clone https://github.com/cygnixx/airlyze-demo.git
cd airlyze-demo
