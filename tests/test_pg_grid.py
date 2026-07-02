import json
import time
from pathlib import Path

import numpy as np


def _make_affine_lattice_points(grid_size: int, start: float, pitch: float, shear: float) -> np.ndarray:
    """构造带轻微剪切的仿射晶格点（row 优先排列），模拟矫正残差下的真实网格。"""
    points = []
    for row in range(grid_size):
        for col in range(grid_size):
            x = start + pitch * col + shear * row
            y = start + shear * col + pitch * row
            points.append((x, y))
    return np.asarray(points, dtype=np.float32)


def _make_synthetic_rectified_10x10(size: int = 800, start: float = 112.0, pitch: float = 64.0, square: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """构造类似矫正图的 10x10 暗方块阵列。

    网格起点/间距刻意偏离按外框均分（margin_ratio=0.085 时 linspace 起点为 68、
    间距约 73.8），以便区分“真的检测到了方块”与“退化为均分网格”。
    返回 (图像, 方块像素质心真值)。
    """
    import cv2

    image = np.full((size, size, 3), 200, dtype=np.uint8)
    centers = []
    for row in range(10):
        for col in range(10):
            cx = start + col * pitch
            cy = start + row * pitch
            x0 = int(round(cx - square / 2))
            y0 = int(round(cy - square / 2))
            image[y0:y0 + square, x0:x0 + square] = 90
            centers.append((x0 + (square - 1) / 2.0, y0 + (square - 1) / 2.0))
    return image, np.asarray(centers, dtype=np.float32)


def _make_synthetic_photo_10x10() -> tuple[np.ndarray, np.ndarray]:
    """构造端到端合成场景：暗背景 + 亮板 + 偏移暗方块阵列 + 干扰结构。

    亮板占整图约 5.4%，落在主区域检测的面积先验（0.2%-20%）内；
    方块阵列刻意不居中于亮板，防止按外框均分侥幸通过。
    返回 (图像, 方块像素质心真值[原图坐标])。
    """
    import cv2

    size = 1200
    image = np.full((size, size, 3), 8, dtype=np.uint8)
    plate_x, plate_y = 460, 430
    plate_side = 280
    cv2.rectangle(image, (plate_x, plate_y), (plate_x + plate_side, plate_y + plate_side), (205, 205, 205), -1)

    start, pitch, square = 32.0, 24.0, 9
    centers = []
    for row in range(10):
        for col in range(10):
            cx = plate_x + start + col * pitch
            cy = plate_y + start + row * pitch
            x0 = int(round(cx - square / 2))
            y0 = int(round(cy - square / 2))
            image[y0:y0 + square, x0:x0 + square] = 95
            centers.append((x0 + (square - 1) / 2.0, y0 + (square - 1) / 2.0))

    # 干扰：列间隙处的螺丝状暗圆、板上缘细长暗线、板外高亮反光弧。
    cv2.circle(image, (plate_x + 116, plate_y + 140), 4, (40, 40, 40), -1)
    cv2.rectangle(image, (plate_x + 10, plate_y + 6), (plate_x + 270, plate_y + 9), (60, 60, 60), -1)
    cv2.ellipse(image, (600, 600), (560, 560), 0, -30, 60, (70, 70, 70), 12)
    return image, np.asarray(centers, dtype=np.float32)


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


def test_enforce_lattice_consistency_snaps_outliers_back_to_lattice():
    """被局部干扰拉偏的点必须被全局规则晶格约束吸附回模型预测位置。"""
    from pg_grid import enforce_lattice_consistency

    rng = np.random.default_rng(7)
    lattice = _make_affine_lattice_points(10, start=100.0, pitch=62.0, shear=2.5)
    # inlier 加入 ±0.8px 的“制造公差”噪声，模拟真实精修结果。
    noisy = lattice + rng.uniform(-0.8, 0.8, size=lattice.shape).astype(np.float32)

    outlier_a = 3 * 10 + 4
    outlier_b = 7 * 10 + 1
    noisy[outlier_a] += (25.0, -6.0)
    noisy[outlier_b] += (-4.0, 18.0)

    corrected, info = enforce_lattice_consistency(noisy, grid_size=10)

    assert corrected.shape == (100, 2)
    assert info["applied"] is True
    assert info["outlier_count"] == 2

    # outlier 必须回到晶格模型附近（真晶格 1.5px 内），inlier 保持精修结果不变。
    for idx in (outlier_a, outlier_b):
        assert float(np.linalg.norm(corrected[idx] - lattice[idx])) <= 1.5
    inlier_mask = np.ones(100, dtype=bool)
    inlier_mask[[outlier_a, outlier_b]] = False
    assert np.allclose(corrected[inlier_mask], noisy[inlier_mask], atol=1e-4)


def test_enforce_lattice_consistency_keeps_regular_grid_unchanged():
    """没有异常点时，一致性校正不得移动任何点。"""
    from pg_grid import enforce_lattice_consistency

    lattice = _make_affine_lattice_points(15, start=90.0, pitch=45.0, shear=1.0)
    corrected, info = enforce_lattice_consistency(lattice, grid_size=15)

    assert info["applied"] is True
    assert info["outlier_count"] == 0
    assert corrected.shape == lattice.shape
    assert np.allclose(corrected, lattice, atol=1e-4)


def test_enforce_lattice_consistency_skips_small_or_mismatched_grids():
    """网格太小或点数不匹配时应原样返回，不做不可靠的拟合。"""
    from pg_grid import enforce_lattice_consistency

    tiny = np.array([[10.0, 10.0], [50.0, 10.0], [10.0, 50.0], [50.0, 50.0]], dtype=np.float32)
    corrected, info = enforce_lattice_consistency(tiny, grid_size=2)
    assert info["applied"] is False
    assert np.allclose(corrected, tiny)

    mismatched = np.zeros((7, 2), dtype=np.float32)
    corrected, info = enforce_lattice_consistency(mismatched, grid_size=10)
    assert info["applied"] is False
    assert corrected.shape == (7, 2)


def test_select_regular_axis_survives_many_noise_clusters_quickly():
    """大量杂散簇（螺丝/边缘碎片）下，轴选择必须仍然找到规则晶格且不能组合爆炸。"""
    from pg_grid import _select_regular_axis_from_clusters

    length = 800
    # 10 个规则簇：支持度高（每列约有 10 个方块成员）。
    regular = [(112.0 + i * 64.0, 50.0, 8) for i in range(10)]
    # 14 个杂散簇：支持度低、间距无规律。
    noise = [(60.0 + i * 31.7, 3.0, 1) for i in range(14)]
    clusters = sorted(regular + noise, key=lambda item: item[0])

    started = time.monotonic()
    axis = _select_regular_axis_from_clusters(clusters, count=10, length=length)
    elapsed = time.monotonic() - started

    assert elapsed < 5.0, f"轴选择耗时 {elapsed:.1f}s，组合枚举没有被有效约束"
    assert axis is not None
    expected = np.array([112.0 + i * 64.0 for i in range(10)], dtype=np.float32)
    assert np.allclose(axis, expected, atol=1.0)


def test_fit_grid_points_10x10_does_not_degenerate_to_uniform_split():
    """10x10 拟合必须贴合真实方块中心，而不是按外框均分。"""
    from pg_grid import fit_grid_points, generate_grid_points

    image, true_centers = _make_synthetic_rectified_10x10()
    fitted = fit_grid_points(image, grid_size=10)

    fit_errors = np.linalg.norm(fitted - true_centers, axis=1)
    uniform = generate_grid_points(10, image.shape[1], image.shape[0], margin_ratio=0.085)
    uniform_errors = np.linalg.norm(uniform - true_centers, axis=1)

    assert fitted.shape == (100, 2)
    assert float(fit_errors.mean()) < 3.0
    # linspace 均分网格在该布局下平均偏差超过 30px，若二者接近说明退化。
    assert float(uniform_errors.mean()) > 30.0


def test_refine_grid_points_resists_corner_blob_distractor():
    """10x10 局部精修不得被角落更暗的大块结构（如螺丝/阴影）拉偏。"""
    from pg_grid import refine_grid_points

    image = np.full((100, 100, 3), 220, dtype=np.uint8)
    image[44:56, 44:56] = (70, 70, 70)
    image[20:34, 20:34] = (10, 10, 10)

    initial_points = np.array([[48.0, 49.0]], dtype=np.float32)
    refined = refine_grid_points(image, initial_points, grid_size=10, radius=25)

    assert abs(float(refined[0, 0]) - 50.0) <= 2.0
    assert abs(float(refined[0, 1]) - 50.0) <= 2.0


def test_process_image_and_evaluation_end_to_end_10x10(tmp_path):
    """端到端：完整 pipeline 定位偏移网格 + 评价工具消费 result.json。

    覆盖以下验收要求：
    - result.json 落盘且包含完整 grid_points，数量等于 grid_size*grid_size；
    - 网格不退化为按外框均分；
    - 评价工具能读取预测点并生成三个报告文件。
    """
    import cv2
    from pg_grid import generate_grid_points, process_image, write_image_unicode
    from pg_grid_eval import write_annotation_json, write_evaluation_report

    image, true_centers = _make_synthetic_photo_10x10()
    image_path = tmp_path / "synthetic_10x10.jpg"
    write_image_unicode(image_path, image)

    out_dir = tmp_path / "out"
    result = process_image(image_path=image_path, grid_size=10, output_dir=out_dir)

    with (out_dir / "result.json").open("r", encoding="utf-8") as f:
        saved = json.load(f)
    assert saved["point_count"] == 100
    assert len(saved["grid_points"]) == saved["grid_size"] ** 2
    assert "lattice_consistency" in saved

    # 把方块质心真值映射到矫正坐标系，与预测点逐一对比。
    region = np.asarray(result["chip_region"]["points"], dtype=np.float32)
    rect_size = int(result["rectified_size"])
    dst = np.array(
        [[0, 0], [rect_size - 1, 0], [rect_size - 1, rect_size - 1], [0, rect_size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(region, dst)
    true_rect = cv2.perspectiveTransform(true_centers.reshape(-1, 1, 2), matrix).reshape(-1, 2)

    predicted = np.asarray([[p["x"], p["y"]] for p in saved["grid_points"]], dtype=np.float32)
    fit_errors = np.linalg.norm(predicted - true_rect, axis=1)
    uniform = generate_grid_points(10, rect_size, rect_size, margin_ratio=0.085)
    uniform_errors = np.linalg.norm(uniform - true_rect, axis=1)

    assert float(fit_errors.mean()) < 5.0
    assert float(uniform_errors.mean()) > 25.0

    # 用真值作为标注，验证评价工具与 result.json 的兼容性。
    annotation_points = [
        {"row": index // 10, "col": index % 10, "x": float(x), "y": float(y)}
        for index, (x, y) in enumerate(true_rect)
    ]
    annotation_path = tmp_path / "annotation.json"
    write_annotation_json(
        annotation_path,
        image_path=out_dir / "rectified_chip.jpg",
        grid_size=10,
        points=annotation_points,
    )
    report_dir = tmp_path / "report"
    metrics = write_evaluation_report(
        annotation_path=annotation_path,
        prediction_path=out_dir / "result.json",
        image_path=out_dir / "rectified_chip.jpg",
        output_dir=report_dir,
    )

    assert float(metrics["mean_error_px"]) < 5.0
    assert (report_dir / "localization_metrics.json").exists()
    assert (report_dir / "localization_errors.csv").exists()
    assert (report_dir / "localization_error_overlay.jpg").exists()
