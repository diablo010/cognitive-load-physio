import pandas as pd

from ecg_windowing import DEFAULT_FEATURE_COLUMNS as ECG_FEATURE_COLUMNS
from ecg_windowing import build_windowed_dataset as build_ecg_windowed_dataset
from gsr_windowing import DEFAULT_GSR_FEATURE_COLUMNS
from gsr_windowing import build_windowed_dataset as build_gsr_windowed_dataset


DEFAULT_MULTIMODAL_FEATURE_COLUMNS = ECG_FEATURE_COLUMNS + DEFAULT_GSR_FEATURE_COLUMNS


def build_multimodal_windowed_dataset(
    data_path: str,
    label_map: dict,
    participants=None,
    sampling_rate: int = 256,
    window_seconds: int = 60,
    step_seconds: int = 30,
) -> pd.DataFrame:
    ecg_df = build_ecg_windowed_dataset(
        data_path=data_path,
        label_map=label_map,
        participants=participants,
        sampling_rate=sampling_rate,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )
    gsr_df = build_gsr_windowed_dataset(
        data_path=data_path,
        label_map=label_map,
        participants=participants,
        sampling_rate=sampling_rate,
        window_seconds=window_seconds,
        step_seconds=step_seconds,
    )

    if ecg_df.empty or gsr_df.empty:
        return pd.DataFrame()

    merge_keys = [
        "Participant",
        "Block",
        "Label",
        "window_start_sample",
        "window_end_sample",
        "window_start_sec",
        "window_end_sec",
    ]

    merged_df = ecg_df.merge(
        gsr_df[
            merge_keys
            + ["Block Type", "GSR File"]
            + DEFAULT_GSR_FEATURE_COLUMNS
        ],
        on=merge_keys,
        how="inner",
        suffixes=("_ecg", "_gsr"),
    )

    if "Block Type_ecg" in merged_df.columns and "Block Type_gsr" in merged_df.columns:
        merged_df["Block Type"] = merged_df["Block Type_ecg"]
        merged_df = merged_df.drop(columns=["Block Type_ecg", "Block Type_gsr"])
    elif "Block Type" not in merged_df.columns and "Block Type" in ecg_df.columns:
        merged_df["Block Type"] = ecg_df["Block Type"]

    return merged_df.reset_index(drop=True)
