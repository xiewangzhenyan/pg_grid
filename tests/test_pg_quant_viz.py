"""PG-Quant 可视化测试：热图、颜色分布图、单元状态叠图。

可视化的正确性用像素级取样验证：热图单元格颜色随数值单调变化、
颜色图单元格反映校正后颜色、不可靠单元在叠图中有醒目标记。
"""

from pathlib import Path

import numpy as np


def _make_records(grid_size: int = 10):
    """构造带已知数值的合成定量记录。"""
    records = []
    for row in range(grid_size):
        for col in range(grid_size):
            index = row * grid_size + col
            records.append(
                {
                    "row": row,
                    "col": col,
                    "x": 112.0 + col * 64.0,
                    "y": 112.0 + row * 64.0,
                    "corr_b": 40.0 + index,
                    "corr_g": 90.0,
                    "corr_r": 200.0 - index,
                    "corr_signal_gray": float(index),
                    "snr": float(10 + index),
                    "flags": [] if index != 37 else ["low_snr"],
                    "quant_reliable": index != 37,
                }
            )
    return records


def test_heatmap_cell_colors_follow_values():
    """热图必须按数值着色：数值大的单元格颜色应与数值小的明显不同。"""
    from pg_quant_viz import render_unit_heatmap

    records = _make_records()
    image, layout = render_unit_heatmap(records, grid_size=10, value_key="corr_signal_gray", title="intensity")

    assert image.ndim == 3 and image.dtype == np.uint8
    # 取第一个（值 0）与最后一个（值 99）单元格中心的颜色，差异必须显著。
    (x0, y0) = layout["cell_centers"][0]
    (x1, y1) = layout["cell_centers"][99]
    color_low = image[y0, x0].astype(int)
    color_high = image[y1, x1].astype(int)
    assert int(np.abs(color_low - color_high).sum()) > 120


def test_heatmap_marks_unreliable_units():
    """不可靠单元的格子必须带有可见标记（打叉），且不影响其他格子。"""
    from pg_quant_viz import render_unit_heatmap

    records = _make_records()
    image, layout = render_unit_heatmap(records, grid_size=10, value_key="snr", title="snr")

    # 标记以深色叉线绘制：不可靠格子内的最低亮度应远低于同值域邻格。
    bad_center = layout["cell_centers"][37]
    good_center = layout["cell_centers"][38]
    half = layout["cell_size"] // 2 - 2
    bad_patch = image[bad_center[1] - half:bad_center[1] + half, bad_center[0] - half:bad_center[0] + half]
    good_patch = image[good_center[1] - half:good_center[1] + half, good_center[0] - half:good_center[0] + half]
    assert int(bad_patch.min()) < int(good_patch.min()) - 30


def test_color_map_cells_show_corrected_unit_colors():
    """颜色分布图的每个格子必须填充该单元校正后的 BGR 颜色。"""
    from pg_quant_viz import render_unit_color_map

    records = _make_records()
    image, layout = render_unit_color_map(records, grid_size=10)

    for index in (0, 55, 99):
        cx, cy = layout["cell_centers"][index]
        expected = np.array(
            [records[index]["corr_b"], records[index]["corr_g"], records[index]["corr_r"]]
        )
        sampled = image[cy, cx].astype(np.float64)
        assert float(np.abs(sampled - expected).max()) <= 3.0, index


def test_quant_overlay_marks_unreliable_units_in_red():
    """状态叠图中不可靠单元必须出现红色标记，可靠单元附近为绿色标记。"""
    from pg_quant_viz import draw_quant_overlay

    records = _make_records()
    rectified = np.full((800, 800, 3), 180, dtype=np.uint8)
    meta = {"roi_radius_px": 11.5, "annulus_inner_px": 19.2, "annulus_outer_px": 28.2}

    overlay = draw_quant_overlay(rectified, records, meta)

    bad = records[37]
    bx, by = int(bad["x"]), int(bad["y"])
    patch = overlay[by - 16:by + 16, bx - 16:bx + 16].astype(int)
    # 红色标记：存在 R 通道显著高于 G/B 的像素。
    redness = patch[:, :, 2] - np.maximum(patch[:, :, 0], patch[:, :, 1])
    assert int(redness.max()) > 60

    good = records[36]
    gx, gy = int(good["x"]), int(good["y"])
    patch_good = overlay[gy - 16:gy + 16, gx - 16:gx + 16].astype(int)
    greenness = patch_good[:, :, 1] - np.maximum(patch_good[:, :, 0], patch_good[:, :, 2])
    assert int(greenness.max()) > 60


def test_process_image_writes_quant_visualizations(tmp_path):
    """完整 pipeline 必须落盘四个定量可视化文件并登记在 outputs 中。"""
    from pg_grid import process_image

    example = Path(__file__).resolve().parent.parent / "examples" / "synthetic_10x10_dark_squares.png"
    result = process_image(image_path=example, grid_size=10, output_dir=tmp_path)

    expected = {
        "quant_overlay": "quant_overlay.jpg",
        "quant_heatmap_intensity": "quant_heatmap_intensity.jpg",
        "quant_heatmap_snr": "quant_heatmap_snr.jpg",
        "quant_color_map": "quant_color_map.jpg",
    }
    for key, name in expected.items():
        assert (tmp_path / name).exists(), name
        assert result["outputs"][key] == str(tmp_path / name), key
