#!/usr/bin/env python3
from __future__ import annotations

import argparse
from html import escape
from pathlib import Path
from typing import Dict, List


# Paper-reported Macro-F1 values (%) from:
# - Table 1: Aligned scenario
# - Table 4: Hybrid scenario with modality retention probability p
PAPER_F1: Dict[str, Dict[str, float]] = {
    "CrisisMMD": {
        "aligned_iid": 77.2,
        "aligned_alpha_0_1": 61.4,
        "hybrid_p60_iid": 62.5,
        "hybrid_p60_alpha_0_1": 42.8,
    },
    "Hateful Memes": {
        "aligned_iid": 73.8,
        "aligned_alpha_0_1": 52.7,
        "hybrid_p60_iid": 54.8,
        "hybrid_p60_alpha_0_1": 30.8,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the coupled amplification bar chart from the paper using "
            "the original Table 1 and Table 4 Macro-F1 values."
        )
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("analysis/figures"),
        help="Directory to save the generated figure files.",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="coupled_heterogeneity_amplification",
        help="Base filename without extension.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=240,
        help="PNG export DPI.",
    )
    return parser.parse_args()


def build_plot_records() -> Dict[str, Dict[str, List[float]]]:
    records: Dict[str, Dict[str, List[float]]] = {}
    for dataset, scores in PAPER_F1.items():
        baseline = scores["aligned_iid"]
        f1_values = [
            scores["aligned_alpha_0_1"],
            scores["hybrid_p60_iid"],
            scores["hybrid_p60_alpha_0_1"],
        ]
        drops = [round(baseline - value, 1) for value in f1_values]
        records[dataset] = {
            "baseline": [baseline],
            "f1": f1_values,
            "drop": drops,
        }
    return records


def plot_with_matplotlib(records: Dict[str, Dict[str, List[float]]], output_dir: Path, basename: str, dpi: int) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    labels = ["Aligned\nα=0.1", "Hybrid 60%\nIID", "Hybrid 60%\nα=0.1"]
    colors = ["#5E81A2", "#BC7A53", "#8F3E63"]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, dataset in zip(axes, ["CrisisMMD", "Hateful Memes"]):
        x = list(range(len(labels)))
        drops = records[dataset]["drop"]
        f1_values = records[dataset]["f1"]

        bars = ax.bar(x, drops, width=0.62, color=colors, edgecolor="none")
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 51)
        ax.set_yticks([0, 10, 20, 30, 40, 50])
        ax.set_title(dataset, fontsize=13, fontweight="bold", pad=8)
        ax.set_ylabel("Performance drop from Aligned-IID (%)", fontsize=11)
        ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.35)

        for bar, drop, f1_value in zip(bars, drops, f1_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"-{drop:.1f}\nF1={f1_value:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    fig.tight_layout(w_pad=1.2)

    output_paths = [
        output_dir / f"{basename}.png",
        output_dir / f"{basename}.svg",
        output_dir / f"{basename}.pdf",
    ]
    fig.savefig(output_paths[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(output_paths[1], bbox_inches="tight")
    fig.savefig(output_paths[2], bbox_inches="tight")
    plt.close(fig)
    return output_paths


def plot_as_svg(records: Dict[str, Dict[str, List[float]]], output_dir: Path, basename: str) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    width = 1120
    height = 560
    margin_left = 80
    margin_right = 35
    margin_top = 52
    margin_bottom = 118
    gutter = 40
    panel_width = (width - margin_left - margin_right - gutter) / 2
    panel_height = height - margin_top - margin_bottom
    ymax = 50.0
    tick_values = [0, 10, 20, 30, 40, 50]
    labels = ["Aligned\nα=0.1", "Hybrid 60%\nIID", "Hybrid 60%\nα=0.1"]
    colors = ["#5E81A2", "#BC7A53", "#8C3D61"]

    def y_to_svg(value: float) -> float:
        return margin_top + panel_height - (value / ymax) * panel_height

    def build_panel(panel_index: int, dataset: str) -> List[str]:
        left = margin_left + panel_index * (panel_width + gutter)
        right = left + panel_width
        elements: List[str] = []

        for tick in tick_values:
            y = y_to_svg(tick)
            elements.append(
                f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{right:.1f}" y2="{y:.1f}" '
                'stroke="#B9BDC5" stroke-width="1" stroke-dasharray="4 4" opacity="0.65" />'
            )
            if panel_index == 0:
                elements.append(
                    f'<text x="{left - 14:.1f}" y="{y + 4:.1f}" text-anchor="end" '
                    f'font-size="15" font-family="DejaVu Serif">{tick}</text>'
                )

        elements.append(
            f'<line x1="{left:.1f}" y1="{margin_top:.1f}" x2="{left:.1f}" y2="{margin_top + panel_height:.1f}" '
            'stroke="#222222" stroke-width="1.2" />'
        )
        elements.append(
            f'<line x1="{left:.1f}" y1="{margin_top + panel_height:.1f}" x2="{right:.1f}" y2="{margin_top + panel_height:.1f}" '
            'stroke="#222222" stroke-width="1.2" />'
        )

        elements.append(
            f'<text x="{left + panel_width / 2:.1f}" y="{margin_top - 14:.1f}" text-anchor="middle" '
            'font-size="24" font-weight="700" font-family="DejaVu Serif">'
            f"{escape(dataset)}</text>"
        )

        label_x = left - 58
        label_y = margin_top + panel_height / 2
        elements.append(
            f'<g transform="translate({label_x:.1f},{label_y:.1f}) rotate(-90)">'
            '<text text-anchor="middle" font-size="17" font-family="DejaVu Serif">'
            'Performance drop from Aligned-IID (%)</text></g>'
        )

        drops = records[dataset]["drop"]
        f1_values = records[dataset]["f1"]
        centers = [
            left + panel_width * 0.17,
            left + panel_width * 0.50,
            left + panel_width * 0.83,
        ]
        bar_width = panel_width * 0.21

        for center, label, color, drop, f1_value in zip(centers, labels, colors, drops, f1_values):
            top = y_to_svg(drop)
            bottom = y_to_svg(0)
            bar_left = center - bar_width / 2
            bar_height = bottom - top

            elements.append(
                f'<rect x="{bar_left:.1f}" y="{top:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" '
                f'fill="{color}" rx="0" ry="0" />'
            )
            elements.append(
                f'<text x="{center:.1f}" y="{top - 10:.1f}" text-anchor="middle" '
                'font-size="16" font-family="DejaVu Serif">'
                f"-{drop:.1f}</text>"
            )
            elements.append(
                f'<text x="{center:.1f}" y="{top + 9:.1f}" text-anchor="middle" '
                'font-size="16" font-family="DejaVu Serif">'
                f"F1={f1_value:.1f}</text>"
            )

            label_lines = label.split("\n")
            for line_index, line in enumerate(label_lines):
                elements.append(
                    f'<text x="{center:.1f}" y="{bottom + 20 + line_index * 18:.1f}" text-anchor="middle" '
                    'font-size="16" font-family="DejaVu Serif">'
                    f"{escape(line)}</text>"
                )

        return elements

    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white" />',
    ]
    svg_parts.extend(build_panel(0, "CrisisMMD"))
    svg_parts.extend(build_panel(1, "Hateful Memes"))
    svg_parts.append("</svg>")

    output_path = output_dir / f"{basename}.svg"
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")
    return [output_path]


def plot_figure(records: Dict[str, Dict[str, List[float]]], output_dir: Path, basename: str, dpi: int) -> List[Path]:
    try:
        return plot_with_matplotlib(records=records, output_dir=output_dir, basename=basename, dpi=dpi)
    except ModuleNotFoundError as exc:
        missing_name = str(exc).split("'")
        package_name = missing_name[1] if len(missing_name) >= 2 else "matplotlib"
        print(f"[plot] {package_name} not available; falling back to pure-SVG export.")
        return plot_as_svg(records=records, output_dir=output_dir, basename=basename)


def main() -> int:
    args = parse_args()
    records = build_plot_records()
    output_paths = plot_figure(
        records=records,
        output_dir=args.output_dir,
        basename=args.basename,
        dpi=args.dpi,
    )

    print("Figure generated from paper values:")
    for dataset, values in records.items():
        print(
            f"  {dataset}: baseline={values['baseline'][0]:.1f}, "
            f"drops={values['drop']}, f1={values['f1']}"
        )
    for path in output_paths:
        print(f"saved: {path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
