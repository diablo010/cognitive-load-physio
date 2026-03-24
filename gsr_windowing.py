import os
from typing import Dict, Iterable, List, Optional

import neurokit2 as nk
import numpy as np
import pandas as pd


DEFAULT_GSR_FEATURE_COLUMNS = [
    "mean_scl",
    "std_scl",
    "mean_phasic",
    "std_phasic",
    "scr_peaks",
    "scr_rate_per_min",
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
    gsr_raw: pd.Series,
    sampling_rate: int = 256,
    window_seconds: int = 60,
    step_seconds: int = 30,
) -> pd.DataFrame:
    window_size = window_seconds * sampling_rate
    step_size = step_seconds * sampling_rate
    rows: List[Dict[str, float]] = []

    for start in _sliding_window_starts(len(gsr_raw), window_size, step_size):
        stop = start + window_size
        window = pd.Series(gsr_raw.iloc[start:stop]).reset_index(drop=True)

        try:
            eda_clean = nk.eda_clean(window, sampling_rate=sampling_rate)
            eda_signals, eda_info = nk.eda_process(eda_clean, sampling_rate=sampling_rate)

            scr_peaks = len(eda_info.get("SCR_Peaks", []))
            rows.append(
                {
                    "window_start_sample": start,
                    "window_end_sample": stop,
                    "window_start_sec": start / sampling_rate,
                    "window_end_sec": stop / sampling_rate,
                    "mean_scl": float(eda_signals["EDA_Tonic"].mean()),
                    "std_scl": float(eda_signals["EDA_Tonic"].std()),
                    "mean_phasic": float(eda_signals["EDA_Phasic"].mean()),
                    "std_phasic": float(eda_signals["EDA_Phasic"].std()),
                    "scr_peaks": float(scr_peaks),
                    "scr_rate_per_min": float(scr_peaks * 60.0 / window_seconds),
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
    window_seconds: int = 60,
    step_seconds: int = 30,
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
            & (participant_blocks["EDA Quality"] == 2.0)
        ].copy()
        participant_blocks["Label"] = participant_blocks["Block Type"].map(label_map)

        for _, row in participant_blocks.iterrows():
            gsr_file = row["EDA&PPG File"].replace("_gsr_ppg.csv", "_gsr_ppg_.csv")
            gsr_path = os.path.join(participant_folder, gsr_file)

            if not os.path.exists(gsr_path):
                continue

            gsr_signal = pd.read_csv(gsr_path)
            if "gsr" not in gsr_signal.columns:
                continue

            window_df = extract_window_features(
                gsr_raw=gsr_signal["gsr"],
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
            window_df["GSR File"] = gsr_file
            all_rows.extend(window_df.to_dict("records"))

    features_df = pd.DataFrame(all_rows)

    if features_df.empty:
        return features_df

    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.dropna(subset=DEFAULT_GSR_FEATURE_COLUMNS)
    features_df = features_df[
        features_df["mean_scl"].between(-1e6, 1e6)
        & features_df["std_scl"].between(0, 1e6)
        & features_df["mean_phasic"].between(-1e6, 1e6)
        & features_df["std_phasic"].between(0, 1e6)
        & features_df["scr_peaks"].between(0, 500)
        & features_df["scr_rate_per_min"].between(0, 500)
    ].copy()

    return features_df.reset_index(drop=True)
