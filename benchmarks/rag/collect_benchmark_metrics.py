# SPDX-License-Identifier: Apache-2.0
# Standard
import argparse
import re
from pathlib import Path
from typing import Optional

# Third Party
import pandas as pd

METRIC_NAMES = (
    "average_f1_tokenizer",
    "average_precision",
    "average_recall",
    "average_f1_word",
    "average_tpot",
    "average_ttft",
    "throughput",
    "average_quality",
)

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
FLOAT_RE_TEMPLATE = r"{metric}\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Collect benchmark metrics from benchmark.log files in logs subfolders "
            "and save a CSV summary."
        )
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=script_dir / "logs",
        help="Path to logs directory (default: benchmarks/rag/logs).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "benchmark_metrics_summary.csv",
        help="Path to output CSV file.",
    )
    return parser.parse_args()


def _extract_metric(log_text: str, metric_name: str) -> Optional[float]:
    match = re.search(FLOAT_RE_TEMPLATE.format(metric=re.escape(metric_name)), log_text)
    if match is None:
        return None
    return float(match.group(1))


def _parse_metrics_from_log(log_path: Path) -> dict[str, Optional[float]]:
    raw_text = log_path.read_text(encoding="utf-8", errors="ignore")
    normalized_text = ANSI_ESCAPE_RE.sub("", raw_text)

    return {
        metric_name: _extract_metric(normalized_text, metric_name)
        for metric_name in METRIC_NAMES
    }


def build_metrics_dataframe(logs_dir: Path) -> pd.DataFrame:
    """
    Build a dataframe with benchmark metrics for each logs subfolder.

    Args:
        logs_dir: Directory that contains benchmark run subdirectories.

    Returns:
        DataFrame with columns: folder_name and all metric columns.
    """
    if not logs_dir.exists():
        raise FileNotFoundError(f"Logs directory does not exist: {logs_dir}")

    if not logs_dir.is_dir():
        raise NotADirectoryError(f"Logs path is not a directory: {logs_dir}")

    rows: list[dict[str, Optional[float] | str]] = []
    for subdir in sorted(path for path in logs_dir.iterdir() if path.is_dir()):
        benchmark_log = subdir / "benchmark.log"
        if not benchmark_log.exists():
            continue

        row: dict[str, Optional[float] | str] = {"folder_name": subdir.name}
        row.update(_parse_metrics_from_log(benchmark_log))
        rows.append(row)

    columns = ["folder_name", *METRIC_NAMES]
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    """Entry point."""
    args = parse_args()

    dataframe = build_metrics_dataframe(args.logs_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(args.output, index=False)

    print(f"Saved {len(dataframe)} rows to {args.output}")


if __name__ == "__main__":
    main()
