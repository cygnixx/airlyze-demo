--- README.md
+++ README.md
@@
+# AirLyze Demo
+
+This is a demo Streamlit app for AirLyze.
+
+## Added functionality
+
+- Download ECG + SpO₂ data from PhysioNet (Apnea-ECG subset) as CSV
+
+### How to use
+
+```bash
+python download_ecg_spo2.py
+```
+
+This will create individual CSVs in `apnea_ecg_spo2_csv/` and a merged CSV `apnea_ecg_spo2_merged.csv`.
+
--- requirements.txt
+++ requirements.txt
@@
 streamlit
+w fdb
+ pandas
+
--- /dev/null
+++ b/download_ecg_spo2.py
@@
+import wfdb
+import pandas as pd
+import os
+
+# -----------------------------------------------------------
+# CONFIGURATION
+# -----------------------------------------------------------
+
+output_folder = "apnea_ecg_spo2_csv"
+merged_csv_file = "apnea_ecg_spo2_merged.csv"
+
+os.makedirs(output_folder, exist_ok=True)
+
+db_name = "apnea-ecg"
+
+# -----------------------------------------------------------
+# STEP 1: Get record list
+# -----------------------------------------------------------
+
+print("Fetching record list from PhysioNet...")
+records = wfdb.get_record_list(db_name)
+print(f"Found {len(records)} records.")
+
+# -----------------------------------------------------------
+# STEP 2: Download and convert records with SpO2
+# -----------------------------------------------------------
+
+merged_frames = []
+
+for record in records:
+    print(f"\nProcessing {record} ...")
+    wfdb.dl_record(record, db_name=db_name, pn_dir=db_name)
+    data = wfdb.rdrecord(record, pn_dir=db_name)
+    # Only proceed if SpO2 channel exists
+    if "SpO2" not in data.sig_name:
+        print(f"  → No SpO2 channel found in {record}, skipping.")
+        continue
+
+    df = pd.DataFrame(data.p_signal, columns=data.sig_name)
+
+    # Timestamp column
+    df["time"] = pd.date_range(
+        start="2000-01-01",
+        periods=len(df),
+        freq=f"{int(1000/data.fs)}ms"
+    )
+
+    df["subject_id"] = record
+
+    outpath = os.path.join(output_folder, f"{record}.csv")
+    df.to_csv(outpath, index=False)
+    print(f"  → Saved: {outpath}")
+
+    merged_frames.append(df)
+
+# -----------------------------------------------------------
+# STEP 3: Merge all into one CSV
+# -----------------------------------------------------------
+
+if merged_frames:
+    full_df = pd.concat(merged_frames, ignore_index=True)
+    full_df.to_csv(merged_csv_file, index=False)
+    print(f"\nMerged CSV saved as: {merged_csv_file}")
+
+print("\n✅ Done.")
