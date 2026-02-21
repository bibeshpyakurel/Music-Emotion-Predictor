from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from music_emotion_predictor.pipeline import PipelineResult, run_pipeline
from music_emotion_predictor.validation import DatasetError


def _holdout_metrics_table(result: PipelineResult) -> pd.DataFrame:
    rows = [
        {
            "model": "Majority Baseline",
            "key": "majority_baseline",
            "accuracy": result.majority_baseline_metrics.accuracy,
            "macro_f1": result.majority_baseline_metrics.macro_f1,
            "weighted_f1": result.majority_baseline_metrics.weighted_f1,
        },
        {
            "model": "Decision Tree",
            "key": "decision_tree",
            "accuracy": result.decision_tree_metrics.accuracy,
            "macro_f1": result.decision_tree_metrics.macro_f1,
            "weighted_f1": result.decision_tree_metrics.weighted_f1,
        },
        {
            "model": "KNN",
            "key": "knn",
            "accuracy": result.knn_metrics.accuracy,
            "macro_f1": result.knn_metrics.macro_f1,
            "weighted_f1": result.knn_metrics.weighted_f1,
        },
        {
            "model": "Logistic Regression",
            "key": "logistic_regression",
            "accuracy": result.logistic_regression_metrics.accuracy,
            "macro_f1": result.logistic_regression_metrics.macro_f1,
            "weighted_f1": result.logistic_regression_metrics.weighted_f1,
        },
    ]
    holdout = pd.DataFrame(rows)
    baseline_accuracy = float(
        holdout.loc[holdout["key"] == "majority_baseline", "accuracy"].iloc[0]
    )
    holdout["delta_vs_majority"] = holdout["accuracy"] - baseline_accuracy
    holdout["rank"] = holdout["accuracy"].rank(method="dense", ascending=False).astype(int)
    return holdout.sort_values(["rank", "model"]).set_index("model")


def _classification_report_table(report: dict[str, object]) -> pd.DataFrame:
    rows = [
        {**metrics, "label": label}
        for label, metrics in report.items()
        if isinstance(metrics, dict)
    ]
    return pd.DataFrame(rows).set_index("label")


def _model_reports(result: PipelineResult) -> list[tuple[str, dict[str, object]]]:
    return [
        ("Majority Baseline", result.majority_baseline_metrics.classification_report),
        ("Decision Tree", result.decision_tree_metrics.classification_report),
        ("KNN", result.knn_metrics.classification_report),
        ("Logistic Regression", result.logistic_regression_metrics.classification_report),
    ]


def _render_run_registry(artifacts_dir: Path) -> None:
    index_path = artifacts_dir / "index.jsonl"
    if not index_path.exists():
        st.caption("No run registry found yet (`index.jsonl`).")
        return
    records: list[dict[str, object]] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if not records:
        st.caption("Run registry is empty.")
        return

    rows: list[dict[str, object]] = []
    for record in records[-20:]:
        metrics = record.get("metrics", {})
        params = record.get("parameters", {})
        if not isinstance(metrics, dict) or not isinstance(params, dict):
            continue
        rows.append(
            {
                "timestamp_utc": record.get("timestamp_utc"),
                "random_state": params.get("random_state"),
                "cv_folds": params.get("cv_folds"),
                "decision_tree_acc": metrics.get("decision_tree", {}).get("accuracy"),
                "knn_acc": metrics.get("knn", {}).get("accuracy"),
                "log_reg_acc": metrics.get("logistic_regression", {}).get("accuracy"),
                "kmeans_ari": metrics.get("kmeans_ari"),
            }
        )

    if rows:
        st.dataframe(pd.DataFrame(rows).iloc[::-1], use_container_width=True)
        st.caption(f"Showing {len(rows)} most recent run(s) from `{index_path}`.")


def main() -> None:
    st.set_page_config(page_title="Music Emotion Predictor", layout="wide")
    st.title("Music Emotion Predictor")
    st.caption(
        "Upload a CSV of audio features and run the full evaluation pipeline "
        "(Majority, Decision Tree, KNN, Logistic Regression, and KMeans ARI)."
    )

    with st.sidebar:
        st.header("Run Configuration")
        test_size = st.slider("Test size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input("Random state", min_value=0, value=42, step=1)
        cv_folds = st.number_input("CV folds", min_value=2, value=5, step=1)
        nan_strategy = st.selectbox("NaN strategy", options=["drop", "impute"], index=0)
        save_artifacts = st.checkbox("Save artifacts", value=False)
        artifacts_dir = st.text_input("Artifacts directory", value="artifacts")
        show_registry = st.checkbox("Show run registry", value=True)
        show_full_json = st.checkbox("Show full metrics JSON", value=False)

    uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])
    sample_path = Path("data/sample_music_dataset.csv")
    use_sample = uploaded_file is None and sample_path.exists()

    if uploaded_file is None and not use_sample:
        st.info("Upload a CSV to run the model.")
        return

    source_label = uploaded_file.name if uploaded_file is not None else str(sample_path)
    st.write(f"**Dataset source:** `{source_label}`")

    try:
        if uploaded_file is not None:
            uploaded_file.seek(0)
            preview_df = pd.read_csv(uploaded_file)
            uploaded_file.seek(0)
        else:
            preview_df = pd.read_csv(sample_path)
    except Exception as exc:  # pragma: no cover
        st.error(f"Could not parse dataset preview: {exc}")
        return

    profile_cols = st.columns(4)
    profile_cols[0].metric("Rows", f"{len(preview_df):,}")
    profile_cols[1].metric("Columns", len(preview_df.columns))
    profile_cols[2].metric("Missing Cells", int(preview_df.isna().sum().sum()))
    profile_cols[3].metric(
        "Labels Present",
        "Yes" if "labels" in preview_df.columns else "No",
    )

    with st.expander("Dataset Preview", expanded=False):
        st.dataframe(preview_df.head(20), use_container_width=True)
        st.caption("Showing first 20 rows.")

    if st.button("Run Pipeline", type="primary"):
        validation_logs: list[str] = []

        def validation_log(message: str) -> None:
            validation_logs.append(message)

        try:
            if uploaded_file is not None:
                uploaded_file.seek(0)
                dataset = pd.read_csv(uploaded_file)
            else:
                dataset = sample_path

            with st.spinner("Running pipeline..."):
                result = run_pipeline(
                    dataset=dataset,
                    test_size=float(test_size),
                    random_state=int(random_state),
                    cv_folds=int(cv_folds),
                    artifacts_dir=Path(artifacts_dir) if save_artifacts else None,
                    nan_strategy=nan_strategy,
                    logger=validation_log,
                )
        except (DatasetError, ValueError) as exc:
            st.error(f"Run failed: {exc}")
            return
        except Exception as exc:  # pragma: no cover
            st.exception(exc)
            return

        st.success("Run complete.")
        if validation_logs:
            with st.expander("Validation logs", expanded=False):
                for line in validation_logs:
                    st.write(f"- {line}")

        holdout_df = _holdout_metrics_table(result)
        best_model = holdout_df["accuracy"].idxmax()
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.subheader("Model Leaderboard (Holdout)")
            st.dataframe(
                holdout_df[
                    ["rank", "accuracy", "macro_f1", "weighted_f1", "delta_vs_majority"]
                ].style.format("{:.4f}"),
                use_container_width=True,
            )
        with col2:
            st.subheader("KMeans")
            st.metric("ARI", f"{result.kmeans_ari:.4f}")
        with col3:
            st.subheader("Best Model")
            st.metric("By accuracy", best_model)

        st.subheader("Cross-Validation (mean/std)")
        cv_df = pd.DataFrame(result.cross_validation).T
        st.dataframe(cv_df.style.format("{:.4f}"), use_container_width=True)

        st.subheader("Top Features")
        st.dataframe(result.top_features.rename("importance").to_frame(), use_container_width=True)
        st.bar_chart(result.top_features)

        st.subheader("Energy Cluster Summary")
        st.dataframe(result.energy_cluster_summary.style.format("{:.4f}"), use_container_width=True)

        st.subheader("Per-Class Reports")
        reports = _model_reports(result)
        tabs = st.tabs([name for name, _ in reports])
        for tab, (_, report) in zip(tabs, reports):
            with tab:
                st.dataframe(
                    _classification_report_table(report).style.format("{:.4f}"),
                    use_container_width=True,
                )

        st.subheader("Downloads")
        metrics_payload = result.as_dict()
        metrics_json = json.dumps(metrics_payload, indent=2)
        holdout_csv = holdout_df.reset_index().to_csv(index=False)
        cv_csv = cv_df.reset_index().rename(columns={"index": "model"}).to_csv(index=False)
        top_features_csv = result.top_features.rename("importance").to_csv()

        d1, d2, d3, d4 = st.columns(4)
        d1.download_button(
            "metrics.json",
            data=metrics_json,
            file_name="metrics.json",
            mime="application/json",
        )
        d2.download_button(
            "holdout_metrics.csv",
            data=holdout_csv,
            file_name="holdout_metrics.csv",
            mime="text/csv",
        )
        d3.download_button(
            "cross_validation.csv",
            data=cv_csv,
            file_name="cross_validation.csv",
            mime="text/csv",
        )
        d4.download_button(
            "top_features.csv",
            data=top_features_csv,
            file_name="top_features.csv",
            mime="text/csv",
        )

        if result.artifact_dir:
            st.info(f"Artifacts saved to: `{result.artifact_dir}`")

        if show_registry:
            st.subheader("Run Registry")
            _render_run_registry(Path(artifacts_dir))

        if show_full_json:
            st.subheader("Full Metrics JSON")
            st.json(metrics_payload)


if __name__ == "__main__":
    main()
