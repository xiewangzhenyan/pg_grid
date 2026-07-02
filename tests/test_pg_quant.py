"""PG-Quant V1.0 测试：ROI+背景环定量、平场校正、单元级质量标记。

测试全部使用合成矫正图（内存数组，不经过有损压缩），
信号真值已知，验证"读数是否稳定"而不只是"点找得漂不漂亮"。
"""

import json
from pathlib import Path

import numpy as np


def _make_units_image(
    size: int = 800,
    grid_size: int = 10,
    start: float = 112.0,
    pitch: float = 64.0,
    unit_half: int = 10,
    background: float = 140.0,
    deltas=None,
    field_x=None,
):
    """构造矫正图风格的合成阵列：背景 + 方形单元（灰度 = 背景 + delta）。

    field_x: 可选乘性光照场，输入为 (f_left, f_right)，沿 x 线性变化，
    同时作用于背景与单元，模拟单侧打光。
    返回 (BGR 图像, 点位数组 N×2, 每单元 delta 数组)。
    """
    if deltas is None:
        deltas = np.full(grid_size * grid_size, 60.0, dtype=np.float64)
    deltas = np.asarray(deltas, dtype=np.float64)

    base = np.full((size, size), background, dtype=np.float64)
    points = []
    for row in range(grid_size):
        for col in range(grid_size):
            idx = row * grid_size + col
            cx = start + col * pitch
            cy = start + row * pitch
            points.append((cx, cy))
            x0, x1 = int(round(cx)) - unit_half, int(round(cx)) + unit_half
            y0, y1 = int(round(cy)) - unit_half, int(round(cy)) + unit_half
            base[y0:y1, x0:x1] = background + deltas[idx]

    if field_x is not None:
        f_left, f_right = field_x
        ramp = np.linspace(f_left, f_right, size, dtype=np.float64)
        base = base * ramp[None, :]

    gray = np.clip(base, 0, 255).astype(np.uint8)
    image = np.stack([gray, gray, gray], axis=2)
    return image, np.asarray(points, dtype=np.float32), deltas


def test_annulus_background_subtraction_recovers_known_signals():
    """背景环差分必须还原每个单元的已知信号强度。"""
    from pg_quant import quantify_rectified

    deltas = np.array([10.0 + (i % 10) * 8.0 for i in range(100)])
    image, points, _ = _make_units_image(deltas=deltas)

    records, meta = quantify_rectified(image, points, grid_size=10)

    assert len(records) == 100
    assert meta["unit_count"] == 100
    for record, delta in zip(records, deltas):
        assert abs(float(record["signal_gray"]) - delta) <= 2.0, (record["row"], record["col"])
        assert abs(float(record["bg_gray_median"]) - 140.0) <= 2.0


def test_flat_field_correction_stabilizes_readings_under_gradient():
    """乘性光照梯度下，平场校正后的读数离散度必须远小于原始读数。"""
    from pg_quant import quantify_rectified

    image, points, _ = _make_units_image(field_x=(0.75, 1.25))
    records, meta = quantify_rectified(image, points, grid_size=10)

    raw = np.array([float(r["roi_gray_median"]) for r in records])
    corr = np.array([float(r["corr_gray"]) for r in records])
    corr_signal = np.array([float(r["corr_signal_gray"]) for r in records])
    ratio = np.array([float(r["signal_ratio"]) for r in records])

    raw_spread = (raw.max() - raw.min()) / raw.mean()
    corr_spread = (corr.max() - corr.min()) / corr.mean()
    corr_signal_spread = (corr_signal.max() - corr_signal.min()) / abs(corr_signal.mean())
    ratio_spread = (ratio.max() - ratio.min()) / abs(ratio.mean())

    assert raw_spread > 0.25, "光照梯度本身必须在原始读数中可见"
    assert corr_spread < 0.05
    assert corr_signal_spread < 0.08
    assert ratio_spread < 0.05
    assert meta["illumination_model"] == "poly2"
    assert meta["illumination_uniformity"] < 0.8


def test_saturated_unit_flagged_and_neighbors_unaffected():
    """过曝单元必须被标记为不可靠，且不得污染其他单元的读数。"""
    import cv2
    from pg_quant import quantify_rectified

    image_a, points, _ = _make_units_image()
    image_b = image_a.copy()
    bad_idx = 4 * 10 + 4
    bx, by = points[bad_idx]
    cv2.circle(image_b, (int(bx), int(by)), 14, (255, 255, 255), -1)

    records_a, _ = quantify_rectified(image_a, points, grid_size=10)
    records_b, _ = quantify_rectified(image_b, points, grid_size=10)

    bad = records_b[bad_idx]
    assert "saturated" in bad["flags"]
    assert bad["quant_reliable"] is False

    for idx, (ra, rb) in enumerate(zip(records_a, records_b)):
        if idx == bad_idx:
            continue
        assert abs(float(ra["signal_gray"]) - float(rb["signal_gray"])) <= 1.5, idx


def test_low_snr_unit_flagged_under_noise():
    """信号淹没在噪声里的单元必须被标记为低信噪比。"""
    from pg_quant import quantify_rectified

    rng = np.random.default_rng(11)
    deltas = np.full(100, 60.0)
    weak_idx = 3 * 10 + 6
    deltas[weak_idx] = 1.0

    image, points, _ = _make_units_image(deltas=deltas)
    noisy = np.clip(
        image.astype(np.float64) + rng.normal(0.0, 3.0, size=image.shape), 0, 255
    ).astype(np.uint8)

    records, _ = quantify_rectified(noisy, points, grid_size=10)

    weak = records[weak_idx]
    assert "low_snr" in weak["flags"]
    assert weak["quant_reliable"] is False

    strong_flags = [r for i, r in enumerate(records) if i != weak_idx and "low_snr" in r["flags"]]
    assert len(strong_flags) == 0, "正常信号单元不应被误标低信噪比"


def test_roi_out_of_bounds_unit_flagged():
    """ROI 或背景环明显越出图像边界的点必须被标记。"""
    from pg_quant import quantify_rectified

    image, _, _ = _make_units_image(size=400, grid_size=2, start=150.0, pitch=100.0)
    points = np.array(
        [[12.0, 150.0], [250.0, 150.0], [150.0, 250.0], [250.0, 250.0]],
        dtype=np.float32,
    )

    records, _ = quantify_rectified(image, points, grid_size=2)

    assert "roi_out_of_bounds" in records[0]["flags"]
    assert records[0]["quant_reliable"] is False
    assert "roi_out_of_bounds" not in records[3]["flags"]


def test_non_uniform_unit_flagged_on_partial_occlusion():
    """ROI 一半被异物覆盖的单元必须被均匀性检查标记。"""
    import cv2
    from pg_quant import quantify_rectified

    image, points, _ = _make_units_image()
    idx = 6 * 10 + 2
    cx, cy = points[idx]
    cv2.rectangle(
        image,
        (int(cx), int(cy) - 12),
        (int(cx) + 12, int(cy) + 12),
        (12, 12, 12),
        -1,
    )

    records, _ = quantify_rectified(image, points, grid_size=10)

    assert "non_uniform" in records[idx]["flags"]
    assert records[idx]["quant_reliable"] is False


def test_unit_slightly_larger_roi_does_not_trigger_non_uniform():
    """ROI 略大于单元（少量背景混入）不得被误判为异质。

    中位数读数对少于一半的混入是鲁棒的，均匀性检查只应拦截
    足以破坏中位数的大面积异质（如遮挡约一半 ROI）。
    """
    import cv2
    from pg_quant import quantify_rectified

    # 圆形单元半径 10，ROI 半径约 0.18*64=11.5：单元覆盖 ROI 约 76%，
    # 边缘背景混入约 24%——与自带 15x15 样例的几何一致。
    image = np.full((800, 800, 3), 140, dtype=np.uint8)
    points = []
    for row in range(10):
        for col in range(10):
            cx, cy = 112 + col * 64, 112 + row * 64
            points.append((float(cx), float(cy)))
            cv2.circle(image, (cx, cy), 10, (200, 200, 200), -1)
    points = np.asarray(points, dtype=np.float32)

    records, meta = quantify_rectified(image, points, grid_size=10)

    flagged = [r for r in records if "non_uniform" in r["flags"]]
    assert len(flagged) == 0, f"{len(flagged)} 个正常单元被误标 non_uniform"
    assert meta["summary"]["reliable_ratio"] >= 0.95


def test_bundled_15x15_example_units_are_reliable(tmp_path):
    """自带 15x15 亮点样例（圆点略小于 ROI）绝大多数单元必须判为可靠。"""
    import json as json_module
    from pg_grid import process_image

    example = Path(__file__).resolve().parent.parent / "examples" / "synthetic_15x15_bright_points.png"
    result = process_image(image_path=example, grid_size=15, output_dir=tmp_path)

    with (tmp_path / "quant_result.json").open("r", encoding="utf-8") as f:
        quant = json_module.load(f)
    assert quant["unit_count"] == 225
    assert quant["summary"]["reliable_ratio"] >= 0.95
    signals = np.array([float(u["signal_gray"]) for u in quant["units"]])
    assert float(np.median(signals)) > 10.0


def test_process_image_integration_writes_quant_outputs(tmp_path):
    """完整 pipeline 必须落盘 quant_values.csv 与 quant_result.json，旧输出不受影响。"""
    from pg_grid import process_image

    example = Path(__file__).resolve().parent.parent / "examples" / "synthetic_10x10_dark_squares.png"
    result = process_image(image_path=example, grid_size=10, output_dir=tmp_path)

    # 旧输出全部保留
    for name in ["overlay_grid.jpg", "rectified_chip.jpg", "roi_debug.jpg", "values.csv", "result.json"]:
        assert (tmp_path / name).exists(), name

    # 新输出
    quant_csv = tmp_path / "quant_values.csv"
    quant_json = tmp_path / "quant_result.json"
    assert quant_csv.exists()
    assert quant_json.exists()
    assert result["outputs"]["quant_values_csv"] == str(quant_csv)
    assert result["outputs"]["quant_result_json"] == str(quant_json)

    with quant_json.open("r", encoding="utf-8") as f:
        quant = json.load(f)
    assert quant["schema"] == "pg-quant-v1"
    assert quant["unit_count"] == 100
    assert len(quant["units"]) == 100
    for unit in quant["units"]:
        assert isinstance(unit["flags"], list)
        assert isinstance(unit["quant_reliable"], bool)

    # 10x10 样例是暗单元亮背景：信号应显著为负
    signals = np.array([float(u["signal_gray"]) for u in quant["units"]])
    assert float(np.median(signals)) < -50.0

    summary = result["quant_summary"]
    assert summary["unit_count"] == 100
    assert 0.0 <= summary["reliable_ratio"] <= 1.0
