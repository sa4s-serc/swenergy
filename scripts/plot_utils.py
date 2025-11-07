from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCAFFOLD_DISPLAY_NAMES: Dict[str, str] = {
    "acr_gemma": "ACR (Gemma)",
    "acr_qwen": "ACR (Qwen)",
    "swe_agent_gemma": "SWE Agent (Gemma)",
    "swe_agent_qwen": "SWE Agent (Qwen)",
    "mini_swe_agent_gemma": "Mini SWE Agent (Gemma)",
    "mini_swe_agent_qwen": "Mini SWE Agent (Qwen)",
    "openhands_gemma": "OpenHands (Gemma)",
    "openhands_qwen": "OpenHands (Qwen)",
}


def ensure_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Install it via `pip install matplotlib`."
        ) from exc
    return plt


def annotate_without_overlap(
    ax,
    labels: Sequence[str],
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    fontsize: int = 8,
    padding: float = 4.0,
    initial_radius: float = 0.02,
    radial_growth: float = 0.005,
    offset_overrides: Optional[Mapping[str, Tuple[float, float]]] = None,
):
    if not labels:
        return []

    fig = ax.figure
    x_values = np.asarray(xs, dtype=float)
    y_values = np.asarray(ys, dtype=float)
    x_range = float(np.max(x_values) - np.min(x_values) or 1.0)
    y_range = float(np.max(y_values) - np.min(y_values) or 1.0)
    offsets = np.linspace(0, 2 * math.pi, num=len(labels), endpoint=False)

    texts = []
    for idx, (label, x, y) in enumerate(zip(labels, xs, ys)):
        radius = initial_radius + radial_growth * idx
        x_offset = math.cos(offsets[idx]) * radius * x_range
        y_offset = math.sin(offsets[idx]) * radius * y_range
        texts.append(
            ax.text(
                x + x_offset,
                y + y_offset,
                label,
                fontsize=fontsize,
                ha="left",
                va="bottom",
                zorder=3,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75, ec="none"),
            )
        )

    transform_to_display = ax.transData.transform
    transform_to_data = ax.transData.inverted().transform

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    max_iterations = 400
    for _ in range(max_iterations):
        moved = False
        bboxes = [text.get_window_extent(renderer) for text in texts]
        centers: List[Tuple[float, float]] = [
            (bbox.x0 + bbox.width / 2.0, bbox.y0 + bbox.height / 2.0) for bbox in bboxes
        ]
        for i in range(len(texts)):
            bbox_i = bboxes[i]
            center_i = centers[i]
            for j in range(i + 1, len(texts)):
                bbox_j = bboxes[j]
                if not bbox_i.overlaps(bbox_j):
                    continue
                center_j = centers[j]
                moved = True

                overlap_x = min(bbox_i.x1, bbox_j.x1) - max(bbox_i.x0, bbox_j.x0)
                overlap_y = min(bbox_i.y1, bbox_j.y1) - max(bbox_i.y0, bbox_j.y0)

                if overlap_x <= overlap_y:
                    direction = 1.0 if center_j[0] >= center_i[0] else -1.0
                    shift_amount = (overlap_x + padding) / 2.0
                    shift_vec = np.array([direction * shift_amount, 0.0])
                else:
                    direction = 1.0 if center_j[1] >= center_i[1] else -1.0
                    shift_amount = (overlap_y + padding) / 2.0
                    shift_vec = np.array([0.0, direction * shift_amount])

                pos_i_display = np.asarray(transform_to_display(texts[i].get_position()))
                pos_j_display = np.asarray(transform_to_display(texts[j].get_position()))

                new_pos_i = pos_i_display - shift_vec
                new_pos_j = pos_j_display + shift_vec

                texts[i].set_position(transform_to_data(new_pos_i))
                texts[j].set_position(transform_to_data(new_pos_j))

        if not moved:
            break
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    if offset_overrides:
        for label, text in zip(labels, texts):
            override = offset_overrides.get(label)
            if override:
                dx, dy = override
                x, y = text.get_position()
                text.set_position((x + dx, y + dy))

    x_positions = [text.get_position()[0] for text in texts]
    y_positions = [text.get_position()[1] for text in texts]
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    ax.set_xlim(min(xmin, min(x_positions)), max(xmax, max(x_positions)))
    ax.set_ylim(min(ymin, min(y_positions)), max(ymax, max(y_positions)))

    for (x, y), text in zip(zip(xs, ys), texts):
        tx, ty = text.get_position()
        if abs(tx - x) < 1e-9 and abs(ty - y) < 1e-9:
            continue
        ax.plot(
            [x, tx],
            [y, ty],
            linestyle=":",
            linewidth=0.8,
            color="#888888",
            alpha=0.6,
            zorder=2,
            clip_on=False,
        )

    return texts


def format_scaffold_label(scaffold: str) -> str:
    """Return a polished display string for a scaffold key."""
    if scaffold in SCAFFOLD_DISPLAY_NAMES:
        return SCAFFOLD_DISPLAY_NAMES[scaffold]
    cleaned = scaffold.replace("_", " ").title()
    # Ensure acronyms like SWE remain uppercase.
    cleaned = cleaned.replace("Swe", "SWE").replace("Acr", "ACR")
    return cleaned
