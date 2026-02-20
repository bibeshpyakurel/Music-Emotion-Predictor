from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import DatasetError
from .pipeline import run_pipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Music emotion modeling pipeline")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/sample_music_dataset.csv"),
        help="Path to the CSV dataset",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--top-features", type=int, default=8)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory where run artifacts are written",
    )
    parser.add_argument(
        "--no-save-artifacts",
        action="store_true",
        help="Disable writing model and metrics artifacts to disk",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full metrics as JSON",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        result = run_pipeline(
            dataset=args.dataset,
            test_size=args.test_size,
            random_state=args.random_state,
            cv_folds=args.cv_folds,
            artifacts_dir=None if args.no_save_artifacts else args.artifacts_dir,
        )
    except (DatasetError, ValueError) as exc:
        print(f"Error: {exc}")
        return 1

    print("Holdout Metrics")
    print(
        f"- Decision Tree: accuracy={result.decision_tree_metrics.accuracy:.4f}, "
        f"macro_f1={result.decision_tree_metrics.macro_f1:.4f}, "
        f"weighted_f1={result.decision_tree_metrics.weighted_f1:.4f}"
    )
    print(
        f"- KNN: accuracy={result.knn_metrics.accuracy:.4f}, "
        f"macro_f1={result.knn_metrics.macro_f1:.4f}, "
        f"weighted_f1={result.knn_metrics.weighted_f1:.4f}"
    )
    print(f"- KMeans ARI: {result.kmeans_ari:.4f}")

    print("\nCross-Validation (mean +/- std)")
    for model_name, metrics in result.cross_validation.items():
        print(
            f"- {model_name}: "
            f"accuracy={metrics['accuracy_mean']:.4f} +/- {metrics['accuracy_std']:.4f}, "
            f"macro_f1={metrics['macro_f1_mean']:.4f} +/- {metrics['macro_f1_std']:.4f}"
        )

    print("\nTop Features")
    print(result.top_features.head(args.top_features).to_string())

    print("\nEnergy Cluster Summary")
    print(result.energy_cluster_summary.to_string())

    if result.artifact_dir:
        print(f"\nArtifacts saved to: {result.artifact_dir}")

    if args.json:
        print("\nFull Metrics JSON")
        print(json.dumps(result.as_dict(), indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
