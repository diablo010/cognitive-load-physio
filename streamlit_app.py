from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from ecg_windowing import DEFAULT_FEATURE_COLUMNS, extract_window_features


APP_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = APP_DIR / "cognitive_load_logistic_model.pkl"


@st.cache_resource
def load_model(model_path: str):
    return joblib.load(model_path)


def predict_windows(model, features_df: pd.DataFrame) -> pd.DataFrame:
    X = features_df[DEFAULT_FEATURE_COLUMNS]
    predictions = model.predict(X)

    prediction_df = features_df.copy()
    prediction_df["predicted_label"] = predictions

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = model.named_steps["clf"].classes_
        for idx, cls in enumerate(classes):
            prediction_df[f"prob_{cls}"] = proba[:, idx]

    return prediction_df


def aggregate_prediction(prediction_df: pd.DataFrame) -> pd.DataFrame:
    prob_cols = [col for col in prediction_df.columns if col.startswith("prob_")]

    if prob_cols:
        summary = prediction_df[prob_cols].mean().reset_index()
        summary.columns = ["class", "mean_probability"]
        summary["class"] = summary["class"].str.replace("prob_", "", regex=False)
        return summary.sort_values("mean_probability", ascending=False).reset_index(drop=True)

    return (
        prediction_df["predicted_label"]
        .value_counts(normalize=True)
        .rename_axis("class")
        .reset_index(name="mean_probability")
    )


def main():
    st.set_page_config(page_title="ECG Cognitive Load Estimator", layout="wide")
    st.title("ECG Cognitive Load Estimator")
    st.caption("Upload ECG CSV data and predict cognitive load with the ECG-only logistic model.")

    with st.sidebar:
        st.header("Settings")
        model_path = st.text_input("Model path", str(DEFAULT_MODEL_PATH))
        sampling_rate = st.number_input("Sampling rate (Hz)", min_value=1, value=256, step=1)
        window_seconds = st.slider("Window length (seconds)", min_value=30, max_value=120, value=60, step=15)
        step_seconds = st.slider("Step size (seconds)", min_value=5, max_value=60, value=30, step=5)
        st.markdown("Recommended settings: `256 Hz`, `60 s` window, `30 s` step.")

    uploaded_file = st.file_uploader("Upload ECG CSV", type=["csv"])
    if uploaded_file is None:
        st.info("Upload a CSV file containing an ECG signal column such as `ecg2`.")
        return

    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    st.subheader("Uploaded Data")
    st.write(f"Rows: {len(raw_df):,}")
    st.dataframe(raw_df.head(10), use_container_width=True)

    numeric_cols = raw_df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        st.error("No numeric columns were found in the uploaded CSV.")
        return

    default_col = "ecg2" if "ecg2" in raw_df.columns else numeric_cols[0]
    ecg_column = st.selectbox("ECG column", options=numeric_cols, index=numeric_cols.index(default_col))

    ecg_series = raw_df[ecg_column].dropna().reset_index(drop=True)
    duration_seconds = len(ecg_series) / sampling_rate
    st.write(f"Detected usable ECG samples: {len(ecg_series):,} ({duration_seconds:.1f} seconds)")

    if duration_seconds < window_seconds:
        st.error(
            f"Need at least {window_seconds} seconds of data for one prediction window. "
            f"Current duration is {duration_seconds:.1f} seconds."
        )
        return

    with st.spinner("Loading model and extracting ECG features..."):
        try:
            model = load_model(model_path)
        except Exception as exc:
            st.error(f"Could not load model: {exc}")
            return

        features_df = extract_window_features(
            ecg_raw=ecg_series,
            sampling_rate=int(sampling_rate),
            window_seconds=int(window_seconds),
            step_seconds=int(step_seconds),
        )

    if features_df.empty:
        st.error("No valid windows were extracted. Check signal quality, column choice, or window settings.")
        return

    prediction_df = predict_windows(model, features_df)
    summary_df = aggregate_prediction(prediction_df)
    final_label = summary_df.iloc[0]["class"]
    final_score = summary_df.iloc[0]["mean_probability"]

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Final Prediction")
        st.metric("Predicted cognitive load", final_label)
        st.metric("Top confidence", f"{final_score:.3f}")
        st.dataframe(summary_df, use_container_width=True)

    with col2:
        st.subheader("Signal Preview")
        chart_df = pd.DataFrame(
            {
                "time_sec": ecg_series.index / sampling_rate,
                "ecg": ecg_series.values,
            }
        ).set_index("time_sec")
        st.line_chart(chart_df, use_container_width=True)

    st.subheader("Window Predictions")
    display_cols = ["window_start_sec", "window_end_sec", *DEFAULT_FEATURE_COLUMNS, "predicted_label"]
    prob_cols = [col for col in prediction_df.columns if col.startswith("prob_")]
    st.dataframe(prediction_df[display_cols + prob_cols], use_container_width=True)

    st.download_button(
        "Download window predictions as CSV",
        data=prediction_df.to_csv(index=False).encode("utf-8"),
        file_name="ecg_window_predictions.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
