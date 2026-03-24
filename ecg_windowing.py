import os
from typing import Dict, Iterable, List, Optional

import neurokit2 as nk
import numpy as np
import pandas as pd


DEFAULT_FEATURE_COLUMNS = [
    "mean_hr",
    "rmssd",
    "sdnn",
    "pnn50",
    "sd1",
    "sd2",
    "sd1_sd2",
]


def _sliding_window_starts(
    signal_length: int,
    window_size: int,
    step_size: int,
) -> Iterable[int]:
    if signal_length < window_size:
        return []
    return range(0, signal_length - window_size + 1, step_size)


def extract_window_features(
    ecg_raw: pd.Series,
    sampling_rate: int = 256,
    window_seconds: int = 30,
    step_seconds: int = 15,
    min_rpeaks: int = 5,
) -> pd.DataFrame:
    window_size = window_seconds * sampling_rate
    step_size = step_seconds * sampling_rate
    rows: List[Dict[str, float]] = []

    for start in _sliding_window_starts(len(ecg_raw), window_size, step_size):
        stop = start + window_size
        window = pd.Series(ecg_raw.iloc[start:stop]).reset_index(drop=True)

        try:
            ecg_clean = nk.ecg_clean(window, sampling_rate=sampling_rate)
            ecg_signals, ecg_info = nk.ecg_process(ecg_clean, sampling_rate=sampling_rate)

            rpeaks = ecg_info.get("ECG_R_Peaks", [])
            if len(rpeaks) < min_rpeaks:
                continue

            hrv_time = nk.hrv_time(ecg_info, sampling_rate=sampling_rate, show=False)
            hrv_nonlinear = nk.hrv_nonlinear(ecg_info, sampling_rate=sampling_rate, show=False)

            rows.append(
                {
                    "window_start_sample": start,
                    "window_end_sample": stop,
                    "window_start_sec": start / sampling_rate,
                    "window_end_sec": stop / sampling_rate,
                    "mean_hr": float(ecg_signals["ECG_Rate"].mean()),
                    "rmssd": float(hrv_time["HRV_RMSSD"].iloc[0]),
                    "sdnn": float(hrv_time["HRV_SDNN"].iloc[0]),
                    "pnn50": float(hrv_time["HRV_pNN50"].iloc[0]),
                    "sd1": float(hrv_nonlinear["HRV_SD1"].iloc[0]),
                    "sd2": float(hrv_nonlinear["HRV_SD2"].iloc[0]),
                    "sd1_sd2": float(hrv_nonlinear["HRV_SD1SD2"].iloc[0]),
                }
            )
        except Exception:
            continue

    return pd.DataFrame(rows)


def build_windowed_dataset(
    data_path: str,
    label_map: Dict[str, str],
    participants: Optional[Iterable[int]] = None,
    sampling_rate: int = 256,
    window_seconds: int = 30,
    step_seconds: int = 15,
) -> pd.DataFrame:
    base_path = os.path.join(data_path, "Participants")
    block_details_path = os.path.join(data_path, "Block_details")
    all_rows: List[Dict[str, object]] = []

    if participants is None:
        participants = range(1, 61)

    for participant_id in participants:
        participant_name = f"Part{participant_id}"
        participant_folder = os.path.join(base_path, participant_name, "by_block")
        block_file = os.path.join(block_details_path, f"{participant_name}_Block_Details.csv")

        if not os.path.exists(block_file):
            continue

        participant_blocks = pd.read_csv(block_file)
        participant_blocks.columns = participant_blocks.columns.str.strip()

        participant_blocks = participant_blocks[
            participant_blocks["Block Type"].isin(label_map.keys())
            & (participant_blocks["ECG Quality"] == 2.0)
        ].copy()
        participant_blocks["Label"] = participant_blocks["Block Type"].map(label_map)

        for _, row in participant_blocks.iterrows():
            ecg_file = row["ECG File"].replace("_ecg.csv", "_ecg_.csv")
            ecg_path = os.path.join(participant_folder, ecg_file)

            if not os.path.exists(ecg_path):
                print(f"Missing block file for {participant_name}")

                continue

            ecg_signal = pd.read_csv(ecg_path)
            if "ecg2" not in ecg_signal.columns:
                continue

            window_df = extract_window_features(
                ecg_raw=ecg_signal["ecg2"],
                sampling_rate=sampling_rate,
                window_seconds=window_seconds,
                step_seconds=step_seconds,
            )

            if window_df.empty:
                continue

            window_df["Participant"] = participant_name
            window_df["Block"] = row["Block"]
            window_df["Block Type"] = row["Block Type"]
            window_df["Label"] = row["Label"]
            window_df["ECG File"] = ecg_file
            all_rows.extend(window_df.to_dict("records"))

    features_df = pd.DataFrame(all_rows)

    if features_df.empty:
        return features_df

    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.dropna(subset=DEFAULT_FEATURE_COLUMNS)
    features_df = features_df[
        features_df["mean_hr"].between(35, 220)
        & features_df["rmssd"].between(1, 500)
        & features_df["sdnn"].between(1, 500)
        & features_df["pnn50"].between(0, 100)
        & features_df["sd1"].between(1, 500)
        & features_df["sd2"].between(1, 500)
        & features_df["sd1_sd2"].between(0, 10)
    ].copy()

    return features_df.reset_index(drop=True)
