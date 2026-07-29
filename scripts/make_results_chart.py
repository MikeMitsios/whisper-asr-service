"""Render the evaluation CSV as a static SVG for the README.

GitHub does not execute the Bokeh dashboard's JavaScript in Markdown, so the
headline comparison needs a plain SVG. Hand-built rather than pulled from a
plotting library: no extra dependency, deterministic output, and it inherits the
reader's light or dark theme via `currentColor`.

Usage:
    python scripts/make_results_chart.py
    python scripts/make_results_chart.py --input evaluation_results.csv \\
        --output docs/assets/results.svg
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

# (column, human label, "lower"|"higher")
PANELS = [
    ("wer", "Word Error Rate", "lower"),
    ("avg_time_s", "Mean latency (s)", "lower"),
    ("rtf", "Real-Time Factor", "lower"),
]

# Layout, in SVG user units.
ROW_HEIGHT = 26
BAR_HEIGHT = 15
LABEL_WIDTH = 250
PANEL_GAP = 34
PLOT_WIDTH = 330
VALUE_WIDTH = 78
TITLE_HEIGHT = 26

ACCENT = "#4f7fd4"
ACCENT_BEST = "#3aa06a"


def short_name(model_name: str) -> str:
    """Trim the repo prefix so labels fit: openai/whisper-large-v3_bfloat16 -> bfloat16."""
    name = model_name.split("/")[-1]
    for prefix in ("whisper-large-v3_", "whisper-large-v3"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    return (name.lstrip("_") or model_name).replace("_", " ")


def read_rows(csv_path: Path) -> list[dict]:
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {csv_path}")
    return rows


def comparable_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split rows into the dominant sample count and everything else.

    Runs over different sample counts are not comparable, and a single outlier
    row also flattens every other bar to nothing. Returns
    ``(kept, dropped)`` so the caller can report what it excluded rather than
    truncating silently.
    """
    counts = Counter(row.get("num_samples", "") for row in rows)
    dominant, _ = counts.most_common(1)[0]
    kept = [r for r in rows if r.get("num_samples", "") == dominant]
    dropped = [r for r in rows if r.get("num_samples", "") != dominant]
    return kept, dropped


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_svg(rows: list[dict]) -> str:
    panels = [p for p in PANELS if any(r.get(p[0]) for r in rows)]
    if not panels:
        raise ValueError("None of the expected metric columns hold data")

    width = LABEL_WIDTH + PLOT_WIDTH + VALUE_WIDTH + 20
    panel_height = TITLE_HEIGHT + ROW_HEIGHT * len(rows)
    height = len(panels) * (panel_height + PANEL_GAP) + 30

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="Whisper ASR evaluation results" fill="currentColor">',
        "<style>"
        ".t{font:600 13px system-ui,-apple-system,Segoe UI,sans-serif}"
        ".l{font:12px ui-monospace,SFMono-Regular,Consolas,monospace;opacity:.85}"
        ".v{font:600 12px ui-monospace,SFMono-Regular,Consolas,monospace}"
        ".n{font:11px system-ui,-apple-system,sans-serif;opacity:.6}"
        "</style>",
    ]

    y = 22
    for column, title, direction in panels:
        values = [float(r[column]) for r in rows]
        best = min(values) if direction == "lower" else max(values)
        scale_max = max(values) or 1.0

        parts.append(f'<text class="t" x="0" y="{y}">{_escape(title)}</text>')
        parts.append(
            f'<text class="n" x="{width - 20}" y="{y}" text-anchor="end">'
            f"{direction} is better</text>"
        )
        y += 14

        for row, value in zip(rows, values, strict=True):
            bar = max(2.0, PLOT_WIDTH * value / scale_max)
            color = ACCENT_BEST if value == best else ACCENT
            label = _escape(short_name(row["model_name"]))
            text_y = y + BAR_HEIGHT - 3
            parts.append(
                f'<text class="l" x="{LABEL_WIDTH - 10}" y="{text_y}" '
                f'text-anchor="end">{label}</text>'
            )
            parts.append(
                f'<rect x="{LABEL_WIDTH}" y="{y}" width="{bar:.1f}" '
                f'height="{BAR_HEIGHT}" rx="2" fill="{color}"/>'
            )
            parts.append(
                f'<text class="v" x="{LABEL_WIDTH + bar + 8:.1f}" y="{text_y}">'
                f"{value:.4f}</text>"
            )
            y += ROW_HEIGHT

        y += PANEL_GAP - 14

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render evaluation results as a static SVG")
    parser.add_argument("--input", default="evaluation_results.csv")
    parser.add_argument("--output", default="docs/assets/results.svg")
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    kept, dropped = comparable_rows(rows)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_svg(kept), encoding="utf-8", newline="\n")

    samples = kept[0].get("num_samples", "?")
    print(f"Wrote {output} ({len(kept)} rows at n={samples})")
    for row in dropped:
        print(
            f"  excluded {row['model_name']} "
            f"(n={row.get('num_samples')}, not comparable at n={samples})"
        )


if __name__ == "__main__":
    main()
