"""PG-Quant 可视化：单元级热图、颜色分布图与状态叠图。

定位可视化（overlay_grid/roi_debug）回答"点找得准不准"；
本模块回答"每个单元读出来的数值/颜色长什么样、哪些单元不可信"：

- render_unit_heatmap: 任意单元数值字段的网格热图（强度、SNR 等）；
- render_unit_color_map: 每格填充校正后颜色的分布图；
- draw_quant_overlay: 矫正图上的 ROI/背景环与可靠性标记叠图。

只依赖 OpenCV 与 NumPy，渲染均为确定性绘制，便于测试与移动端移植。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# 不可靠单元的 flags 缩写，用于叠图标注。
_FLAG_CODES = {
    "saturated": "S",
    "under_exposed": "U",
    "low_snr": "N",
    "roi_out_of_bounds": "B",
    "non_uniform": "H",
    "background_anomaly": "A",
    "imputed_position": "I",
}


def _write_image(path: str | Path, image: np.ndarray) -> None:
    """写图（兼容非 ASCII 路径；避免反向依赖 pg_grid）。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        raise RuntimeError(f"图像编码失败：{path}")
    encoded.tofile(str(path))


def _grid_canvas(grid_size: int) -> tuple[np.ndarray, dict[str, object], int, int]:
    """构建带标题区/行列标签区/色标区的画布，返回 (画布, 布局, 左边距, 上边距)。

    布局中的 cell_centers 按行优先给出每个单元格中心的画布坐标，
    供调用方与测试做像素级取样。
    """

    cell = max(28, min(64, 640 // grid_size))
    left, top, right, bottom = 40, 48, 16, 64
    width = left + grid_size * cell + right
    height = top + grid_size * cell + bottom
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)

    centers = []
    for row in range(grid_size):
        for col in range(grid_size):
            centers.append((left + col * cell + cell // 2, top + row * cell + cell // 2))

    # 行列索引标签。
    for col in range(grid_size):
        cv2.putText(canvas, str(col), (left + col * cell + cell // 2 - 6, top - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)
    for row in range(grid_size):
        cv2.putText(canvas, str(row), (8, top + row * cell + cell // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 90, 90), 1)

    layout: dict[str, object] = {"cell_centers": centers, "cell_size": cell}
    return canvas, layout, left, top


def _draw_unreliable_cross(canvas: np.ndarray, x0: int, y0: int, cell: int) -> None:
    """在单元格上画黑色叉线，标记该单元读数不可靠。"""

    inset = 4
    cv2.line(canvas, (x0 + inset, y0 + inset), (x0 + cell - inset, y0 + cell - inset), (0, 0, 0), 3)
    cv2.line(canvas, (x0 + cell - inset, y0 + inset), (x0 + inset, y0 + cell - inset), (0, 0, 0), 3)


def render_unit_heatmap(
    records: list[dict[str, object]],
    grid_size: int,
    value_key: str,
    title: str = "",
    value_format: str = "{:.1f}",
) -> tuple[np.ndarray, dict[str, object]]:
    """把逐单元数值字段渲染为网格热图。

    - 颜色映射使用 Viridis，按全体单元的 min/max 归一；
    - 不可靠单元（quant_reliable=False）打黑色叉线；
    - grid_size <= 10 时在格内标注数值；
    - 底部绘制色标与 min/max 数值。
    返回 (BGR 图像, 布局信息)。
    """

    canvas, layout, left, top = _grid_canvas(grid_size)
    cell = int(layout["cell_size"])

    values = np.array([float(np.nan_to_num(r.get(value_key, 0.0))) for r in records], dtype=np.float64)
    vmin, vmax = float(values.min()), float(values.max())
    span = max(vmax - vmin, 1e-9)
    normalized = ((values - vmin) / span * 255.0).astype(np.uint8)
    colors = cv2.applyColorMap(normalized.reshape(-1, 1), cv2.COLORMAP_VIRIDIS).reshape(-1, 3)

    for index, record in enumerate(records):
        row, col = int(record["row"]), int(record["col"])
        x0, y0 = left + col * cell, top + row * cell
        canvas[y0:y0 + cell, x0:x0 + cell] = colors[index]
        if grid_size <= 10:
            # 数值文字放在格子左上角，避开中心，便于测试与人工取样格心颜色。
            text_color = (255, 255, 255) if normalized[index] < 140 else (0, 0, 0)
            cv2.putText(canvas, value_format.format(values[index]), (x0 + 3, y0 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, text_color, 1)
        if not bool(record.get("quant_reliable", True)):
            _draw_unreliable_cross(canvas, x0, y0, cell)

    # 网格线与标题。
    for k in range(grid_size + 1):
        cv2.line(canvas, (left + k * cell, top), (left + k * cell, top + grid_size * cell), (230, 230, 230), 1)
        cv2.line(canvas, (left, top + k * cell), (left + grid_size * cell, top + k * cell), (230, 230, 230), 1)
    cv2.putText(canvas, title or value_key, (left, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)

    # 底部色标。
    bar_y = top + grid_size * cell + 16
    bar_w = grid_size * cell
    gradient = np.tile(np.linspace(0, 255, bar_w, dtype=np.uint8), (14, 1))
    canvas[bar_y:bar_y + 14, left:left + bar_w] = cv2.applyColorMap(gradient, cv2.COLORMAP_VIRIDIS)
    cv2.putText(canvas, value_format.format(vmin), (left, bar_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)
    max_label = value_format.format(vmax)
    cv2.putText(canvas, max_label, (left + bar_w - 9 * len(max_label), bar_y + 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)

    return canvas, layout


def render_unit_color_map(
    records: list[dict[str, object]],
    grid_size: int,
    title: str = "corrected color",
) -> tuple[np.ndarray, dict[str, object]]:
    """每个单元格直接填充校正后颜色 (corr_b, corr_g, corr_r)。

    这是"每个单元的颜色分布"最直观的呈现方式；
    不可靠单元同样打叉标记。返回 (BGR 图像, 布局信息)。
    """

    canvas, layout, left, top = _grid_canvas(grid_size)
    cell = int(layout["cell_size"])

    for record in records:
        row, col = int(record["row"]), int(record["col"])
        x0, y0 = left + col * cell, top + row * cell
        color = (
            int(np.clip(float(record.get("corr_b", 0.0)), 0, 255)),
            int(np.clip(float(record.get("corr_g", 0.0)), 0, 255)),
            int(np.clip(float(record.get("corr_r", 0.0)), 0, 255)),
        )
        canvas[y0:y0 + cell, x0:x0 + cell] = color
        if not bool(record.get("quant_reliable", True)):
            _draw_unreliable_cross(canvas, x0, y0, cell)

    for k in range(grid_size + 1):
        cv2.line(canvas, (left + k * cell, top), (left + k * cell, top + grid_size * cell), (230, 230, 230), 1)
        cv2.line(canvas, (left, top + k * cell), (left + grid_size * cell, top + k * cell), (230, 230, 230), 1)
    cv2.putText(canvas, title, (left, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)

    return canvas, layout


def draw_quant_overlay(
    rectified: np.ndarray,
    records: list[dict[str, object]],
    meta: dict[str, object],
) -> np.ndarray:
    """在矫正图上叠加定量状态：ROI 圆、背景环与可靠性标记。

    绿色 ROI 圆 = 读数可靠；红色 ROI 圆 = 不可靠，并在旁边标注 flags 缩写
    （S 过曝 / U 欠曝 / N 低信噪比 / B 越界 / H 异质 / A 背景异常 / I 补位）。
    """

    overlay = rectified.copy()
    roi_radius = int(round(float(meta.get("roi_radius_px", 10.0))))
    annulus_outer = int(round(float(meta.get("annulus_outer_px", roi_radius * 2.4))))

    for record in records:
        center = (int(round(float(record["x"]))), int(round(float(record["y"]))))
        reliable = bool(record.get("quant_reliable", True))
        color = (0, 200, 0) if reliable else (0, 0, 255)
        cv2.circle(overlay, center, roi_radius, color, 2)
        cv2.circle(overlay, center, annulus_outer, (160, 160, 160), 1)
        if not reliable:
            codes = "".join(_FLAG_CODES.get(flag, "?") for flag in record.get("flags", []))
            cv2.putText(overlay, codes, (center[0] + roi_radius + 2, center[1] - roi_radius),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 1)

    legend = "green=reliable  red=unreliable  S=sat U=under N=snr B=bounds H=hetero A=bg I=imputed"
    cv2.putText(overlay, legend, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)
    return overlay


def write_quant_visualizations(
    output_dir: str | Path,
    rectified: np.ndarray,
    records: list[dict[str, object]],
    meta: dict[str, object],
) -> dict[str, str]:
    """写出四个定量可视化文件，返回 {逻辑名: 路径}。"""

    output_dir = Path(output_dir)
    grid_size = int(meta["grid_size"])

    overlay = draw_quant_overlay(rectified, records, meta)
    heatmap_intensity, _ = render_unit_heatmap(
        records, grid_size, "corr_signal_gray", title="corrected signal (gray)"
    )
    heatmap_snr, _ = render_unit_heatmap(records, grid_size, "snr", title="SNR")
    color_map, _ = render_unit_color_map(records, grid_size)

    outputs = {
        "quant_overlay": output_dir / "quant_overlay.jpg",
        "quant_heatmap_intensity": output_dir / "quant_heatmap_intensity.jpg",
        "quant_heatmap_snr": output_dir / "quant_heatmap_snr.jpg",
        "quant_color_map": output_dir / "quant_color_map.jpg",
    }
    _write_image(outputs["quant_overlay"], overlay)
    _write_image(outputs["quant_heatmap_intensity"], heatmap_intensity)
    _write_image(outputs["quant_heatmap_snr"], heatmap_snr)
    _write_image(outputs["quant_color_map"], color_map)
    return {key: str(path) for key, path in outputs.items()}
