from pathlib import Path

import numpy as np


def test_generate_grid_points_returns_expected_count_and_order():
    """验证物理规则网格：点数、左上角、右下角都必须可预测。"""
    from pg_grid import generate_grid_points

    points = generate_grid_points(3, 90, 90, margin_ratio=0.1)

    assert points.shape == (9, 2)
    assert tuple(points[0]) == (9.0, 9.0)
    assert tuple(points[-1]) == (81.0, 81.0)


def test_extract_roi_measurements_returns_one_record_per_grid_point():
    """验证每个理论网格点都能输出一条 ROI 定量记录。"""
    from pg_grid import extract_roi_measurements, generate_grid_points

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :] = (10, 20, 30)
    points = generate_grid_points(2, 100, 100, margin_ratio=0.25)

    records = extract_roi_measurements(image, points, grid_size=2, roi_radius=3)

    assert len(records) == 4
    assert records[0]["row"] == 0
    assert records[0]["col"] == 0
    assert records[0]["b_mean"] == 10.0
    assert records[0]["g_mean"] == 20.0
    assert records[0]["r_mean"] == 30.0
    assert 0.0 <= records[0]["saturation_ratio"] <= 1.0


def test_refine_grid_points_prefers_10x10_square_center_over_dark_line_distractor():
    """10x10 局部二次精修应贴近方形目标中心，而不是被暗线/边缘阴影拉偏。"""
    from pg_grid import refine_grid_points

    image = np.full((100, 100, 3), 220, dtype=np.uint8)
    image[44:56, 44:56] = (70, 70, 70)
    image[38:63, 33:37] = (0, 0, 0)

    initial_points = np.array([[47.0, 50.0]], dtype=np.float32)
    refined = refine_grid_points(image, initial_points, grid_size=10, radius=25)

    assert abs(float(refined[0, 0]) - 50.0) <= 2.0
    assert abs(float(refined[0, 1]) - 50.0) <= 2.0


def test_process_image_writes_grid_points_on_synthetic_image(tmp_path):
    """用合成规则阵列图验证完整 pipeline 输出 grid_points。"""
    import cv2
    from pg_grid import process_image, write_image_unicode

    image = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.rectangle(image, (120, 120), (520, 520), (210, 210, 210), -1)
    for row in range(5):
        for col in range(5):
            center = (170 + col * 80, 170 + row * 80)
            cv2.circle(image, center, 8, (245, 245, 245), -1)

    image_path = tmp_path / "synthetic_grid.jpg"
    write_image_unicode(image_path, image)
    result = process_image(image_path=image_path, grid_size=5, output_dir=tmp_path / "out")

    assert result["grid_size"] == 5
    assert result["point_count"] == 25
    assert len(result["grid_points"]) == 25
    assert (tmp_path / "out" / "result.json").exists()
