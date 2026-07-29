"""Render the evaluation CSV as an interactive Bokeh dashboard.

Usage:
    python scripts/visualize.py
    python scripts/visualize.py --input evaluation_results.csv --show
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.palettes import Category20
from bokeh.plotting import figure

# Columns that must be present for the CSV to be usable at all.
REQUIRED_METRICS = ["wer", "cer", "bleu", "avg_time_s", "rtf"]

# Columns added later. Plotted when present, skipped when a row predates them.
OPTIONAL_METRICS = [
    "wer_normalized",
    "cer_normalized",
    "p50_time_s",
    "p95_time_s",
    "model_size_mb",
]

# Lower is better for everything except BLEU.
HIGHER_IS_BETTER = {"bleu"}


def load_data(csv_path: Path) -> pd.DataFrame:
    """Read the results CSV and add a display label per row."""
    df = pd.read_csv(csv_path)
    required = {"model_name", "num_samples", *REQUIRED_METRICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {', '.join(sorted(missing))}")

    df = df.copy()
    df["label"] = (
        df["model_name"].astype(str) + " (samples: " + df["num_samples"].astype(str) + ")"
    )
    return df


def metrics_to_plot(df: pd.DataFrame) -> list[str]:
    """Required metrics, plus any optional ones that actually hold data."""
    present = [m for m in OPTIONAL_METRICS if m in df.columns and df[m].notna().any()]
    return REQUIRED_METRICS + present


def build_dashboard(df: pd.DataFrame):
    """Build a stacked column of one bar chart per metric."""
    model_names = df["model_name"].astype(str).tolist()
    unique_models = list(dict.fromkeys(model_names))
    palette = Category20[20]
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(unique_models)}
    colors = [color_map[name] for name in model_names]

    plots = []
    for metric in metrics_to_plot(df):
        direction = "higher is better" if metric in HIGHER_IS_BETTER else "lower is better"
        source = ColumnDataSource(
            {
                "label": df["label"].tolist(),
                "model_name": model_names,
                "num_samples": df["num_samples"].tolist(),
                "value": df[metric].tolist(),
                "color": colors,
            }
        )
        p = figure(
            x_range=source.data["label"],
            height=320,
            width=1100,
            title=f"{metric} ({direction})",
            toolbar_location="above",
            tools="pan,wheel_zoom,box_zoom,reset,save",
        )
        p.vbar(
            x="label",
            top="value",
            source=source,
            width=0.8,
            color="color",
            legend_field="model_name",
        )
        p.xgrid.grid_line_color = None
        p.y_range.start = 0
        # Labels are long and the legend already identifies each bar.
        p.xaxis.major_label_text_font_size = "0pt"
        p.xaxis.axis_label = ""
        p.yaxis.axis_label = metric
        p.legend.click_policy = "hide"
        p.add_layout(p.legend[0], "right")
        p.add_tools(
            HoverTool(
                tooltips=[
                    ("model", "@model_name"),
                    ("samples", "@num_samples"),
                    (metric, "@value{0.0000}"),
                ]
            )
        )
        plots.append(p)
    return column(*plots)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive evaluation metrics dashboard")
    parser.add_argument(
        "--input",
        default="evaluation_results.csv",
        help="Path to evaluation CSV (default: evaluation_results.csv)",
    )
    parser.add_argument(
        "--output",
        default="evaluation_dashboard.html",
        help="Output HTML file path (default: evaluation_dashboard.html)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the dashboard in a browser after writing it.",
    )
    args = parser.parse_args()

    df = load_data(Path(args.input))
    layout = build_dashboard(df)

    output_file(Path(args.output), title="Whisper ASR evaluation")
    # save(), not show(): show() opens a browser, which fails headless and in CI.
    save(layout)
    print(f"Wrote {args.output} ({len(df)} rows, {len(metrics_to_plot(df))} metrics)")

    if args.show:
        show(layout)


if __name__ == "__main__":
    main()
