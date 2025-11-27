# ---------------------------
# Page: Download ECG + SpO2 (async-friendly)
# ---------------------------

if page_sel == "Download ECG+SpO2":
    st.header("Download ECG + SpO₂ Data from PhysioNet")
    st.write("This will download all Apnea-ECG records containing SpO₂ and save CSVs locally.")

    start_btn = st.button("Start Download")
    if start_btn:
        import wfdb
        output_folder = "apnea_ecg_spo2_csv"
        merged_csv_file = "apnea_ecg_spo2_merged.csv"
        os.makedirs(output_folder, exist_ok=True)

        db_name = "apnea-ecg"
        records = wfdb.get_record_list(db_name)
        merged_frames = []

        st.write(f"Found {len(records)} records to process.")
        placeholder = st.empty()
        progress_bar = st.progress(0)

        for i, record in enumerate(records, start=1):
            placeholder.text(f"Processing {record} ({i}/{len(records)}) ...")
            try:
                wfdb.dl_record(record, db_name=db_name, pn_dir=db_name)
                data = wfdb.rdrecord(record, pn_dir=db_name)
                if "SpO2" not in data.sig_name:
                    placeholder.text(f"Skipping {record}: No SpO₂ channel found.")
                    progress_bar.progress(i/len(records))
                    continue

                df = pd.DataFrame(data.p_signal, columns=data.sig_name)
                df["time"] = pd.date_range(
                    start="2000-01-01", periods=len(df), freq=f"{int(1000/data.fs)}ms"
                )
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
            st.info("No records with SpO₂ were found.")

        placeholder.empty()
        progress_bar.empty()
