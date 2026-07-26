import json
import math
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


def test_select_regular_axis_completes_missing_edge_cluster():
    """缺一个边缘行/列簇时，轴选择必须按间距外推补全，而不是放弃。

    9 个连续簇 + 网格近似居中（矫正对称外扩的物理先验）时，
    缺失位置唯一可辨：补在前端才能保持左右边距对称。
    """
    from pg_grid import _select_regular_axis_from_clusters

    # 真值：10 个中心 131..668（pitch 59.7，边距对称）；第 0 个缺失。
    full = [131.0 + 59.7 * i for i in range(10)]
    clusters = [(center, 40.0, 8) for center in full[1:]]

    axis = _select_regular_axis_from_clusters(clusters, count=10, length=800)

    assert axis is not None, "缺一个簇不应导致整个轴选择失败"
    assert np.allclose(axis, full, atol=2.0)


def test_select_regular_axis_completes_missing_middle_cluster():
    """缺一个中间行/列簇时，间距跳变本身就标识了缺口位置。"""
    from pg_grid import _select_regular_axis_from_clusters

    full = [112.0 + 64.0 * i for i in range(10)]
    observed = full[:4] + full[5:]  # 缺第 4 个
    clusters = [(center, 40.0, 8) for center in observed]

    axis = _select_regular_axis_from_clusters(clusters, count=10, length=800)

    assert axis is not None
    assert np.allclose(axis, full, atol=2.0)


def test_fit_grid_points_10x10_survives_fully_occluded_row():
    """整行单元被遮挡时，候选晶格拟合必须外推缺失行而不是退化到投影路径。"""
    import cv2
    from pg_grid import fit_grid_points

    image, truth = _make_synthetic_rectified_10x10()
    # 抹掉第 0 行全部方块，并放置遮挡异物（比方块大、会被候选检测过滤）。
    for col in range(10):
        cx, cy = truth[col]
        x_lo, y_lo = int(round(cx - 14)), int(round(cy - 14))
        image[y_lo:y_lo + 28, x_lo:x_lo + 28] = 200
        cv2.circle(image, (int(cx), int(cy)), 34, (60, 60, 60), -1)

    fitted = fit_grid_points(image, grid_size=10)
    errors = np.linalg.norm(fitted - truth, axis=1)

    assert fitted.shape == (100, 2)
    # 未被遮挡的 90 个单元必须精确；被遮挡行的外推位置允许稍宽容差。
    assert float(errors[10:].mean()) < 3.0
    assert float(errors[:10].mean()) < 8.0


def test_fit_grid_points_10x10_handles_rotated_lattice():
    """矫正残差导致点阵相对图像坐标轴有几度旋转时，拟合仍须贴合真实中心。

    透视矫正只保证面板四角对齐；如果点阵与面板边缘存在装配旋转差，
    或主区域检测的外接矩形略有偏差，矫正图中的晶格会带小角度旋转。
    轴对齐的行列聚类在这种情况下会给出系统性偏差。
    """
    import cv2
    from pg_grid import fit_grid_points

    size = 800
    angle = math.radians(3.0)
    image = np.full((size, size, 3), 200, dtype=np.uint8)
    center = size / 2.0
    centers = []
    for row in range(10):
        for col in range(10):
            base_x = 112.0 + col * 64.0 - center
            base_y = 112.0 + row * 64.0 - center
            cx = center + base_x * math.cos(angle) - base_y * math.sin(angle)
            cy = center + base_x * math.sin(angle) + base_y * math.cos(angle)
            x0 = int(round(cx - 12))
            y0 = int(round(cy - 12))
            image[y0:y0 + 24, x0:x0 + 24] = 90
            centers.append((x0 + 11.5, y0 + 11.5))
    centers = np.asarray(centers, dtype=np.float32)

    fitted = fit_grid_points(image, grid_size=10)
    errors = np.linalg.norm(fitted - centers, axis=1)

    assert fitted.shape == (100, 2)
    assert float(errors.mean()) < 3.0


def _make_polarity_scene(size: int, grid_size: int, start: float, pitch: float,
                         unit_half: int, background: int, unit_value: int, circular: bool):
    """构造指定极性的合成矫正图（亮底暗块 或 暗底亮点）。"""
    import cv2

    image = np.full((size, size, 3), background, dtype=np.uint8)
    centers = []
    for row in range(grid_size):
        for col in range(grid_size):
            cx = start + col * pitch
            cy = start + row * pitch
            if circular:
                cv2.circle(image, (int(round(cx)), int(round(cy))), unit_half,
                           (unit_value, unit_value, unit_value), -1)
                centers.append((round(cx), round(cy)))
            else:
                x0 = int(round(cx - unit_half))
                y0 = int(round(cy - unit_half))
                image[y0:y0 + 2 * unit_half, x0:x0 + 2 * unit_half] = unit_value
                centers.append((x0 + unit_half - 0.5, y0 + unit_half - 0.5))
    return image, np.asarray(centers, dtype=np.float32)


def test_fit_grid_points_15x15_dark_units_on_bright_panel():
    """15x15 也可能是"亮面板 + 暗单元"（背光成像），极性必须自动检测。

    若按旧的"grid>=15 即亮点"假设，投影峰会落在单元间的发光间隙上，
    整个网格偏移约半个间距。
    """
    from pg_grid import fit_grid_points

    image, truth = _make_polarity_scene(
        size=1200, grid_size=15, start=110.0, pitch=70.0,
        unit_half=12, background=200, unit_value=90, circular=False,
    )
    fitted = fit_grid_points(image, grid_size=15)
    errors = np.linalg.norm(fitted - truth, axis=1)

    assert fitted.shape == (225, 2)
    assert float(errors.mean()) < 3.0


def test_fit_grid_points_10x10_bright_units_on_dark_panel():
    """10x10 也可能是"暗面板 + 亮单元"，极性同样必须自动检测。"""
    from pg_grid import fit_grid_points

    image, truth = _make_polarity_scene(
        size=800, grid_size=10, start=112.0, pitch=64.0,
        unit_half=11, background=30, unit_value=220, circular=True,
    )
    fitted = fit_grid_points(image, grid_size=10)
    errors = np.linalg.norm(fitted - truth, axis=1)

    assert fitted.shape == (100, 2)
    assert float(errors.mean()) < 3.0


def test_polarity_order_prefers_significant_hint_over_marginal_count_gap():
    """统计提示显著时，不得被噪声级别的候选数差压过。

    实拍图上同一张图的候选数会因形态学过滤的偶然性大幅波动
    （实测 79→164），而"内部均值 vs 中位数"的提示符号在全部样本上
    都正确。因此提示显著时它必须是主判据。
    """
    from pg_grid import _order_polarities_by_evidence

    # 暗单元真值：提示显著为 dark，但亮候选数偶然更接近理论值。
    order = _order_polarities_by_evidence(
        dark_count=93, bright_count=95, expected=100, hint="dark", hint_strength=18.9
    )
    assert order[0] == "dark"

    # 提示同样显著时，另一个方向也必须被尊重。
    order = _order_polarities_by_evidence(
        dark_count=95, bright_count=93, expected=100, hint="bright", hint_strength=12.0
    )
    assert order[0] == "bright"


def test_polarity_order_falls_back_to_counts_when_hint_is_weak():
    """提示很弱时（小亮点、低对比），候选数接近度重新成为主判据。"""
    from pg_grid import _order_polarities_by_evidence

    # 合成亮点图实测提示仅 +0.93，此时候选数（0 vs 225）才是可靠证据。
    order = _order_polarities_by_evidence(
        dark_count=0, bright_count=225, expected=225, hint="bright", hint_strength=0.93
    )
    assert order[0] == "bright"

    # 弱提示指向 dark，但候选数强烈指向 bright：应信候选数。
    order = _order_polarities_by_evidence(
        dark_count=3, bright_count=224, expected=225, hint="dark", hint_strength=0.5
    )
    assert order[0] == "bright"


def test_real_photo_10x10_keeps_polarity_when_cropped(tmp_path):
    """实拍 10x10 第二张图裁到芯片占 35% 时曾极性判反，必须保持正确。"""
    from pg_grid import detect_chip_region, process_image, read_image_unicode, write_image_unicode

    photo = Path(__file__).resolve().parent.parent / "examples" / "手机通过短塔正式拍摄-EL板发光(10×10芯片)02.jpg"
    full = read_image_unicode(photo)
    region = detect_chip_region(full)
    xs, ys = region.points[:, 0], region.points[:, 1]
    chip_w = float(xs.max() - xs.min())
    chip_h = float(ys.max() - ys.min())

    x0 = int(max(0, xs.min() - chip_w * 0.35))
    y0 = int(max(0, ys.min() - chip_h * 0.35))
    x1 = int(min(full.shape[1], xs.max() + chip_w * 0.35))
    y1 = int(min(full.shape[0], ys.max() + chip_h * 0.35))
    cropped_path = tmp_path / "cropped_02.png"
    write_image_unicode(cropped_path, full[y0:y1, x0:x1])

    result = process_image(image_path=cropped_path, grid_size=10, output_dir=tmp_path / "out")
    lattice = result["lattice_consistency"]

    assert result["unit_polarity"] == "dark"
    assert float(lattice["candidate_support_ratio"]) >= 0.7
    assert lattice["trusted"] is True


def test_grid_polarity_contrast_distinguishes_true_lattice_from_complement():
    """网格-间隙对比度必须能区分真实晶格与互补晶格。

    这是极性歧义的最终仲裁依据：真实晶格的点落在单元上，互补晶格的点
    落在单元间隙上，两者的"点位 vs 间隙"对比度符号相反。相比候选数量
    或全局统计量，它直接读取图像在候选网格位置上的证据。
    """
    import cv2
    from pg_grid import _measure_grid_polarity_contrast

    size, pitch, start = 800, 70.0, 95.0
    image = np.full((size, size, 3), 100, dtype=np.uint8)
    true_points, shifted_points = [], []
    for row in range(10):
        for col in range(10):
            cx, cy = start + col * pitch, start + row * pitch
            cv2.circle(image, (int(cx), int(cy)), 12, (220, 220, 220), -1)
            true_points.append((cx, cy))
            # 半格错位是投影拟合失误的实际形态（沿单轴偏移半个间距）。
            shifted_points.append((cx + pitch / 2.0, cy))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    true_contrast = _measure_grid_polarity_contrast(gray, np.asarray(true_points, dtype=np.float32), 10)
    shifted_contrast = _measure_grid_polarity_contrast(gray, np.asarray(shifted_points, dtype=np.float32), 10)

    # 亮单元的真实晶格必须给出显著正对比度，且明显强于半格错位的晶格
    # ——仲裁取的正是"证据更强"的那个。
    assert true_contrast > 20.0
    assert shifted_contrast < true_contrast * 0.5


def test_bright_units_survive_blur_that_collapses_the_hint():
    """重模糊把统计提示压到接近 0 时，投影回退必须听候选数证据。

    模糊会让内部均值与中位数几乎重合（实测强度衰减到 0.07），此时提示
    符号是噪声；若投影回退仍盲信提示，亮点阵列会被当成暗单元处理，
    落到互补晶格上产生半格错位（Spearman 变负）。
    """
    import cv2
    from pg_grid import _fit_grid_points_with_polarity

    size = 1200
    image = np.full((size, size, 3), 120, dtype=np.uint8)
    centers = []
    for row in range(15):
        for col in range(15):
            cx, cy = 110 + col * 70, 110 + row * 70
            cv2.circle(image, (cx, cy), 8, (215, 215, 215), -1)
            centers.append((float(cx), float(cy)))
    centers = np.asarray(centers, dtype=np.float32)
    blurred = cv2.GaussianBlur(image, (41, 41), 9.0)

    fitted, polarity = _fit_grid_points_with_polarity(blurred, grid_size=15)
    errors = np.linalg.norm(fitted - centers, axis=1)

    assert polarity == "bright"
    # 半格错位约 35px；正确结果应远小于此。
    assert float(errors.mean()) < 12.0


def test_polarity_not_fooled_by_complementary_lattice_under_blur():
    """重模糊杀死两个候选检测器时，不得掉进"互补晶格陷阱"。

    暗单元阵列的亮间隙同样构成规则投影峰：盲目尝试反极性投影会得到
    自信但整体偏移半个间距的网格。极性必须由图像统计提示决定。
    """
    import cv2
    from pg_grid import _fit_grid_points_with_polarity

    image, truth = _make_synthetic_rectified_10x10()
    blurred = cv2.GaussianBlur(image, (61, 61), 10.0)

    fitted, polarity = _fit_grid_points_with_polarity(blurred, grid_size=10)
    errors = np.linalg.norm(fitted - truth, axis=1)

    assert polarity == "dark"
    # 半格错位约为 0.5*pitch=32px；正确结果应远小于 0.25*pitch。
    assert float(errors.mean()) < 16.0


def test_real_photo_10x10_localizes_with_candidate_support(tmp_path):
    """实拍 10x10 图（EL 背光、含连接器/通道线干扰）必须获得高候选支撑。"""
    from pg_grid import process_image

    photo = Path(__file__).resolve().parent.parent / "examples" / "手机通过短塔正式拍摄-EL板发光(10×10芯片).jpg"
    result = process_image(image_path=photo, grid_size=10, output_dir=tmp_path)

    lattice = result["lattice_consistency"]
    assert result["chip_region"]["method"] != "fallback_center"
    assert result["point_count"] == 100
    assert result["unit_polarity"] == "dark"
    assert lattice["applied"] is True
    assert lattice["support_check"] == "ok"
    assert float(lattice["candidate_support_ratio"]) >= 0.75
    assert lattice["trusted"] is True


def test_real_photo_15x15_localizes_with_candidate_support(tmp_path):
    """实拍 15x15 图（亮面板 + 暗单元）必须被极性检测正确处理并获得支撑。"""
    from pg_grid import process_image

    photo = Path(__file__).resolve().parent.parent / "examples" / "手机通过短塔正式拍摄-EL板发光(15×15芯片)04.jpg"
    result = process_image(image_path=photo, grid_size=15, output_dir=tmp_path)

    lattice = result["lattice_consistency"]
    assert result["chip_region"]["method"] != "fallback_center"
    assert result["point_count"] == 225
    assert result["unit_polarity"] == "dark"
    assert lattice["applied"] is True
    assert lattice["support_check"] == "ok"
    assert float(lattice["candidate_support_ratio"]) >= 0.6
    assert lattice["trusted"] is True


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


def _make_panel_scene(image_w: int, image_h: int, panel_box: tuple[int, int, int, int]) -> np.ndarray:
    """暗背景 + 亮面板（面板内部带亮度不均），用于裁切鲁棒性测试。

    面板内部刻意做出亮度梯度与暗单元：高分位阈值一旦被迫抬高，
    就会切在面板内部而不是面板边界，从而暴露"阈值依赖目标占图比例"的缺陷。
    """
    import cv2

    image = np.full((image_h, image_w, 3), 10, dtype=np.uint8)
    x0, y0, x1, y1 = panel_box
    panel_w, panel_h = x1 - x0, y1 - y0
    ramp = np.linspace(190, 245, panel_w, dtype=np.float64)
    panel = np.repeat(ramp[None, :], panel_h, axis=0)
    image[y0:y1, x0:x1] = np.stack([panel] * 3, axis=2).astype(np.uint8)
    # 面板内规则暗单元，进一步破坏"面板内部一致明亮"的假设。
    for row in range(10):
        for col in range(10):
            cx = x0 + int(panel_w * (0.08 + 0.093 * col))
            cy = y0 + int(panel_h * (0.08 + 0.093 * row))
            cv2.rectangle(image, (cx - 5, cy - 5), (cx + 5, cy + 5), (95, 95, 95), -1)
    return image


def test_refine_survives_points_far_outside_image():
    """点位落在图像外时不得崩溃。

    负坐标尤其危险：`min(width, x + radius)` 会得到负数，而 numpy 把
    负切片索引解释为"从末尾算起"，于是空窗口检查被绕过，随后按字面
    范围构造索引网格就会因负维度崩溃。极端动态范围下全局模型可能把
    未被观测到的单元外推到图外，这条路径必须安全。
    """
    from pg_grid import bundle_adjust_lattice, refine_grid_points

    # 需要有纹理：均匀图像会让加权质心的权重和为 0 而提前退出，
    # 掩盖真正的索引越界问题。
    rng = np.random.default_rng(0)
    image = rng.integers(20, 200, size=(200, 200, 3), dtype=np.uint8)
    # 危险区间是"负偏移的绝对值小于图像尺寸"：此时 min(size, x+radius+1)
    # 得到的负数会被 numpy 当作有效的从末尾计数的索引，切片非空。
    outside = np.array(
        [[80.0, -40.0], [-40.0, 80.0], [-30.0, -30.0], [230.0, 90.0], [90.0, 240.0], [50.0, 50.0]],
        dtype=np.float32,
    )

    # 亮极性走质心分支（构造索引网格的那条路径）。
    refined = refine_grid_points(image, outside, grid_size=15, radius=12)
    assert refined.shape == outside.shape
    assert np.all(np.isfinite(refined))

    # 暗极性走连通域分支。
    refined_dark = refine_grid_points(image, outside, grid_size=10, radius=12)
    assert refined_dark.shape == outside.shape
    assert np.all(np.isfinite(refined_dark))

    # 平差入口同样不得崩溃（3x3 需要 9 个点）。
    points = np.array(
        [[-40.0, -40.0], [90.0, -35.0], [230.0, -30.0],
         [-35.0, 90.0], [90.0, 90.0], [235.0, 95.0],
         [-30.0, 230.0], [95.0, 235.0], [240.0, 240.0]],
        dtype=np.float32,
    )
    adjusted, info, meta = bundle_adjust_lattice(image, points, grid_size=3, radius=12, polarity="bright")
    assert adjusted.shape == points.shape
    assert len(meta) == points.shape[0]
    assert np.all(np.isfinite(adjusted))


def test_select_region_solution_keeps_first_when_margin_is_thin():
    """分差不足时必须保留首选假设，不做无依据的切换。

    仲裁分只是证据的加权和，本身带噪声；分差零点几个百分点不足以支持
    换掉按可靠性排在前面的假设。实测有案例在 0.428 vs 0.503 的分差下
    切换后误差从 118% 恶化到 669%。
    """
    from pg_grid import select_region_solution

    first = {"score": 0.62, "tag": "first"}
    close = {"score": 0.68, "tag": "close"}
    assert select_region_solution([first, close])["tag"] == "first"

    clear = {"score": 0.85, "tag": "clear"}
    assert select_region_solution([first, clear])["tag"] == "clear"


def test_select_region_solution_keeps_first_when_all_evidence_is_weak():
    """所有假设证据都弱时保留首选，不把"证据缺失"当成"证据为负"。

    候选检测在某个矫正图里整体失效会让该假设的覆盖率与支撑率同时归零，
    分数随之归零——但那只说明无从判断，不说明它更差。此时切换是赌博。
    """
    from pg_grid import select_region_solution

    dead = {"score": 0.0, "tag": "first_no_evidence"}
    weak = {"score": 0.47, "tag": "weak"}
    assert select_region_solution([dead, weak])["tag"] == "first_no_evidence"

    strong = {"score": 0.78, "tag": "strong"}
    assert select_region_solution([dead, strong])["tag"] == "strong"


def test_region_hypotheses_include_both_substrate_and_dot_cloud():
    """荧光图必须同时产出基底轮廓与发光点云两个区域假设供后续仲裁。

    固定优先级是错的：基底轮廓在多数图上更好，但它会被棋盘/稀疏图案
    和不均匀基底带偏；发光点云在基底不可见时反而是唯一线索。两者必须
    都拿出来，由下游用图像证据选。
    """
    import json

    from pg_grid import enumerate_region_hypotheses, read_image_unicode

    case = Path(__file__).resolve().parent.parent / "examples" / "fluo" / "trusted_wrong_typical_0024"
    image = read_image_unicode(case / "input.jpg")

    hypotheses = enumerate_region_hypotheses(image)
    methods = [h.method for h in hypotheses]

    assert "opencv_faint_substrate" in methods
    assert "opencv_emitting_dots" in methods
    # 兜底框不应混在候选里（它不是证据，是没有证据时的最后手段）。
    assert "fallback_center" not in methods


def test_grid_coverage_ratio_penalises_truncated_region():
    """原图点云包围率必须能识别"区域框切掉了部分阵列"。

    支撑率只看"网格点旁边有没有候选"，对被切掉的整行完全无感——那些
    单元根本没进矫正图。包围率反过来看"原图里检测到的单元有没有被网格
    覆盖"，因此能抓住区域框截断。
    """
    import cv2
    from pg_grid import measure_grid_coverage_ratio

    size = 900
    image = np.full((size, size, 3), 6, dtype=np.uint8)
    centers = []
    for row in range(10):
        for col in range(10):
            cx, cy = 150 + col * 60, 150 + row * 60
            cv2.rectangle(image, (cx - 9, cy - 9), (cx + 9, cy + 9), (200, 210, 200), -1)
            centers.append((float(cx), float(cy)))
    centers = np.asarray(centers, dtype=np.float32)

    full = measure_grid_coverage_ratio(image, centers, "bright", pitch=60.0)
    # 只覆盖下面 7 行（模拟区域框把上面 3 行切掉）。
    truncated = measure_grid_coverage_ratio(image, centers[30:], "bright", pitch=60.0)

    assert full > 0.95
    # 判据只统计网格邻域内的候选，因此截断越深"看得见"的比例反而越小；
    # 关键是截断必须让包围率明显下降，用于假设之间的相对比较。
    assert truncated < 0.92
    assert full - truncated > 0.08


def test_detect_chip_region_rejects_tiny_bright_cluster_in_favour_of_substrate():
    """极小的亮块不得被当作目标区域，应让位给微弱基底轮廓。

    这是 examples/fluo 里的真实失败样本：只有少数单元很亮时它们连成
    一小块（实测占整图 0.6%）通过了亮区路径，而能容纳整个阵列的目标
    面板不可能只占这么小。真值曾因此几乎全部落在矫正图之外。
    """
    import json

    import cv2
    from pg_grid import detect_chip_region, read_image_unicode

    case = Path(__file__).resolve().parent.parent / "examples" / "fluo" / "region_collapse_extreme_0025"
    image = read_image_unicode(case / "input.jpg")
    with (case / "ground_truth.json").open("r", encoding="utf-8") as f:
        truth = json.load(f)
    points = np.asarray(truth["points"], dtype=np.float64)

    region = detect_chip_region(image)
    xs, ys = region.points[:, 0], region.points[:, 1]

    # 框必须包住整个阵列，而不是最亮的那一小簇。
    assert float(xs.min()) <= points[:, 0].min()
    assert float(ys.min()) <= points[:, 1].min()
    assert float(xs.max()) >= points[:, 0].max()
    assert float(ys.max()) >= points[:, 1].max()


def test_detect_chip_region_finds_faint_substrate_under_bright_emitters():
    """发光阵列下方的微弱基底轮廓必须能被检出，且优先于发光点云。

    荧光成像里基底自发荧光只比暗背景亮 1-2 个灰度级，单像素上淹没在
    噪声里；但它是几十万像素的连续区域，用大于单元尺寸的形态学开运算
    擦掉亮单元后即可分离。这个几何参考独立于单元亮度分布——发光点云
    路径会因为暗单元检测不到而框偏，基底轮廓不会。
    """
    import cv2
    from pg_grid import detect_chip_region

    size = 1400
    rng = np.random.default_rng(3)
    image = rng.normal(0.6, 0.5, (size, size, 3)).clip(0, 255).astype(np.uint8)
    # 基底：只比背景亮约 2 个灰度级。
    panel_x0, panel_y0, panel_side = 400, 400, 620
    image[panel_y0:panel_y0 + panel_side, panel_x0:panel_x0 + panel_side] += 2

    # 发光单元：亮度沿行递增，最上面几行接近背景（检测不到）。
    margin, grid_size = 52.0, 15
    pitch = (panel_side - 2 * margin) / (grid_size - 1)
    centers = []
    for row in range(grid_size):
        level = int(2 + 240 * (row / (grid_size - 1)) ** 2)
        for col in range(grid_size):
            cx = int(round(panel_x0 + margin + col * pitch))
            cy = int(round(panel_y0 + margin + row * pitch))
            cv2.rectangle(image, (cx - 8, cy - 8), (cx + 8, cy + 8), (level, level, level), -1)
            centers.append((cx, cy))
    centers = np.asarray(centers, dtype=np.float64)

    region = detect_chip_region(image)

    assert region.method == "opencv_faint_substrate", region.method
    xs, ys = region.points[:, 0], region.points[:, 1]
    # 必须完整包住整个阵列（含最暗的首行），四边都要有余量。
    assert float(xs.min()) <= centers[:, 0].min()
    assert float(ys.min()) <= centers[:, 1].min()
    assert float(xs.max()) >= centers[:, 0].max()
    assert float(ys.max()) >= centers[:, 1].max()
    # 且贴合面板而非整幅图。
    assert float(xs.max() - xs.min()) < panel_side * 1.5


def test_detect_chip_region_handles_emitting_dots_without_bright_panel():
    """暗背景 + 离散发光点阵列（无连续亮面板）时必须能定出主区域。

    荧光成像下面板本身不发光，只有单元发射：图中最大的连续亮区就是
    单个发光点（占比远小于面积下限），"找亮矩形区域"的策略完全失效。
    此时主区域应由发光点云自身的分布决定。
    """
    import cv2
    from pg_grid import detect_chip_region

    size = 1200
    image = np.full((size, size, 3), 3, dtype=np.uint8)
    start, pitch = 300.0, 40.0
    centers = []
    for row in range(15):
        for col in range(15):
            cx = int(round(start + col * pitch))
            cy = int(round(start + row * pitch))
            cv2.rectangle(image, (cx - 7, cy - 7), (cx + 7, cy + 7), (30, 210, 60), -1)
            centers.append((cx, cy))
    centers = np.asarray(centers, dtype=np.float64)

    region = detect_chip_region(image)

    assert region.method != "fallback_center"
    xs, ys = region.points[:, 0], region.points[:, 1]
    # 检测框必须包住整个发光点阵列。
    assert float(xs.min()) <= centers[:, 0].min() + 8
    assert float(ys.min()) <= centers[:, 1].min() + 8
    assert float(xs.max()) >= centers[:, 0].max() - 8
    assert float(ys.max()) >= centers[:, 1].max() - 8
    # 且不能过度膨胀（不应该接近整幅图）。
    assert float(xs.max() - xs.min()) < size * 0.85


def test_detect_chip_region_stable_when_background_is_cropped_away():
    """裁掉四周背景不得改变主区域检测结果。

    人工裁切是完全合理的操作（甚至减少了干扰），但固定高分位阈值隐含
    "目标只占整图一小部分"的假设：背景被裁掉后阈值被迫抬高，掩膜会
    切在面板内部而非面板边界。检测必须对目标占图比例不敏感。
    """
    from pg_grid import detect_chip_region

    panel_box = (700, 700, 1700, 1700)
    full = _make_panel_scene(2400, 2400, panel_box)
    panel_w = panel_h = 1000

    reference = detect_chip_region(full)
    ref_xs, ref_ys = reference.points[:, 0], reference.points[:, 1]
    ref_size = (float(ref_xs.max() - ref_xs.min()) + float(ref_ys.max() - ref_ys.min())) / 2.0
    assert reference.method != "fallback_center"
    # 未裁切时检测框应覆盖整个面板（含 1.10 外扩，允许 15% 容差）。
    assert abs(ref_size - panel_w * 1.10) < panel_w * 0.15

    # 逐级裁掉背景：面板占比从 17% 一路升到 78%。
    for pad in (400, 200, 100, 60):
        x0, y0 = panel_box[0] - pad, panel_box[1] - pad
        x1, y1 = panel_box[2] + pad, panel_box[3] + pad
        cropped = full[y0:y1, x0:x1]
        region = detect_chip_region(cropped)
        assert region.method != "fallback_center", pad

        xs, ys = region.points[:, 0], region.points[:, 1]
        size = (float(xs.max() - xs.min()) + float(ys.max() - ys.min())) / 2.0
        # 检测框尺寸必须与未裁切时一致（面板本身没变），容差 12%。
        assert abs(size - ref_size) < ref_size * 0.12, (pad, size, ref_size)
        # 检测框中心必须落在面板中心附近。
        center_x = (float(xs.min()) + float(xs.max())) / 2.0
        center_y = (float(ys.min()) + float(ys.max())) / 2.0
        expected_x = panel_box[0] - x0 + panel_w / 2.0
        expected_y = panel_box[1] - y0 + panel_h / 2.0
        assert math.hypot(center_x - expected_x, center_y - expected_y) < panel_w * 0.06, pad


def test_real_photo_localization_survives_background_cropping(tmp_path):
    """实拍图裁掉四周背景后，定位质量不得塌陷。"""
    import cv2
    from pg_grid import detect_chip_region, process_image, read_image_unicode, write_image_unicode

    photo = Path(__file__).resolve().parent.parent / "examples" / "手机通过短塔正式拍摄-EL板发光(15×15芯片).jpg"
    full = read_image_unicode(photo)
    region = detect_chip_region(full)
    xs, ys = region.points[:, 0], region.points[:, 1]
    chip_w = float(xs.max() - xs.min())
    chip_h = float(ys.max() - ys.min())

    # 只保留芯片外 10% 芯片尺寸的背景：芯片占裁切图约 70%。
    x0 = int(max(0, xs.min() - chip_w * 0.10))
    y0 = int(max(0, ys.min() - chip_h * 0.10))
    x1 = int(min(full.shape[1], xs.max() + chip_w * 0.10))
    y1 = int(min(full.shape[0], ys.max() + chip_h * 0.10))
    cropped_path = tmp_path / "cropped.png"
    write_image_unicode(cropped_path, full[y0:y1, x0:x1])

    result = process_image(image_path=cropped_path, grid_size=15, output_dir=tmp_path / "out")
    lattice = result["lattice_consistency"]

    assert result["chip_region"]["method"] != "fallback_center"
    assert result["unit_polarity"] == "dark"
    assert float(lattice["candidate_support_ratio"]) >= 0.85
    assert lattice["trusted"] is True
    assert float(lattice["mean_confidence"]) >= 0.85


def test_detect_chip_region_finds_large_rotated_panel():
    """主区域检测必须能处理占近半幅面的旋转亮面板，而不是落入中央兜底框。"""
    import cv2
    from pg_grid import detect_chip_region, order_quad_points

    image = np.full((900, 900, 3), 18, dtype=np.uint8)
    rect = ((450.0, 450.0), (620.0, 620.0), -3.5)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(image, box, (214, 214, 214))

    region = detect_chip_region(image)

    assert region.method != "fallback_center"
    # 检测四角应贴近真实面板四角。检测本身带 1.10 设计外扩
    # （对角方向约 44px）加形态学膨胀数 px，因此容差取 60px；
    # 若落入中央兜底框，角点误差会达到数百 px，断言仍有区分力。
    expected = order_quad_points(box.astype(np.float32))
    corner_errors = np.linalg.norm(region.points - expected, axis=1)
    assert float(corner_errors.max()) < 60.0


def _assert_bundled_example_localizes(tmp_path, image_name: str, grid_size: int, true_centers: np.ndarray, mean_error_limit: float):
    """对仓库自带样例图跑完整 pipeline，并断言定位误差与输出完整性。"""
    import cv2
    from pg_grid import process_image

    example_path = Path(__file__).resolve().parent.parent / "examples" / image_name
    result = process_image(image_path=example_path, grid_size=grid_size, output_dir=tmp_path / "out")

    assert result["chip_region"]["method"] != "fallback_center"
    assert result["point_count"] == grid_size * grid_size
    assert len(result["grid_points"]) == grid_size * grid_size

    # 把原图坐标系的真值中心映射到矫正坐标系后逐点比较。
    region = np.asarray(result["chip_region"]["points"], dtype=np.float32)
    rect_size = int(result["rectified_size"])
    dst = np.array(
        [[0, 0], [rect_size - 1, 0], [rect_size - 1, rect_size - 1], [0, rect_size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(region, dst)
    true_rect = cv2.perspectiveTransform(true_centers.reshape(-1, 1, 2), matrix).reshape(-1, 2)

    predicted = np.asarray([[p["x"], p["y"]] for p in result["grid_points"]], dtype=np.float32)
    errors = np.linalg.norm(predicted - true_rect, axis=1)
    assert float(errors.mean()) < mean_error_limit


def test_process_image_on_bundled_10x10_example(tmp_path):
    """仓库自带 10x10 样例（旋转面板 + 细线干扰）必须被准确定位。"""
    xs = np.linspace(210, 690, 10)
    ys = np.linspace(210, 690, 10)
    centers = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            # 与 examples/generate_synthetic_samples.py 中的抖动公式保持一致。
            cx = round(x + (col % 3 - 1) * 1.5)
            cy = round(y + (row % 3 - 1) * 1.2)
            centers.append((float(cx), float(cy)))
    _assert_bundled_example_localizes(
        tmp_path,
        "synthetic_10x10_dark_squares.png",
        grid_size=10,
        true_centers=np.asarray(centers, dtype=np.float32),
        mean_error_limit=5.0,
    )


def test_process_image_on_bundled_15x15_example(tmp_path):
    """仓库自带 15x15 亮点样例必须被准确定位。"""
    xs = np.linspace(190, 810, 15)
    ys = np.linspace(190, 810, 15)
    centers = []
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            cx = round(x + (col % 4 - 1.5) * 0.8)
            cy = round(y + (row % 4 - 1.5) * 0.8)
            centers.append((float(cx), float(cy)))
    _assert_bundled_example_localizes(
        tmp_path,
        "synthetic_15x15_bright_points.png",
        grid_size=15,
        true_centers=np.asarray(centers, dtype=np.float32),
        mean_error_limit=5.0,
    )


def _make_projective_10x10(gx: float = 3.5e-5, gy: float = 3.5e-5, shear: float = 0.02):
    """构造中心带透视畸变的 10x10 暗方块矫正图。

    真值点位由 仿射晶格 + 单应透视项 生成：透视分量是仿射模型
    无法表达的系统性弯曲，用于验证光束法平差的必要性。
    返回 (图像, 真值中心)。
    """
    image = np.full((800, 800, 3), 200, dtype=np.uint8)
    centers = []
    for row in range(10):
        for col in range(10):
            x0 = 112.0 + 64.0 * col + shear * (112.0 + 64.0 * row)
            y0 = 112.0 + 64.0 * row
            w = 1.0 + gx * (x0 - 400.0) + gy * (y0 - 400.0)
            cx, cy = x0 / w, y0 / w
            x_lo = int(round(cx - 12))
            y_lo = int(round(cy - 12))
            image[y_lo:y_lo + 24, x_lo:x_lo + 24] = 90
            centers.append((x_lo + 11.5, y_lo + 11.5))
    return image, np.asarray(centers, dtype=np.float32)


def test_bundle_adjust_recovers_perspective_distorted_lattice():
    """光束法平差必须恢复带透视残差的晶格；仿射模型对该场景有系统性残差。"""
    from pg_grid import bundle_adjust_lattice, fit_grid_points

    image, truth = _make_projective_10x10()
    initial = fit_grid_points(image, grid_size=10)
    adjusted, info, point_meta = bundle_adjust_lattice(image, initial, grid_size=10)

    errors = np.linalg.norm(adjusted - truth, axis=1)
    assert adjusted.shape == (100, 2)
    assert len(point_meta) == 100
    assert info["applied"] is True
    assert info["model"] == "homography"
    assert float(errors.mean()) < 1.0

    # 对照：仿射晶格模型对同一真值的最优拟合仍留下明显系统残差，
    # 证明升级到单应模型是必要的而不是锦上添花。
    indices = np.arange(100)
    design = np.stack([np.ones(100), indices % 10, indices // 10], axis=1)
    affine_coef, _, _, _ = np.linalg.lstsq(design, truth.astype(np.float64), rcond=None)
    affine_residual = np.linalg.norm(truth - design @ affine_coef, axis=1)
    assert float(affine_residual.mean()) > 1.8


def test_bundle_adjust_marks_occluded_units_imputed_but_accurate():
    """被抹除的单元必须标记为模型补位（imputed），且补位坐标仍贴近真实晶格。"""
    from pg_grid import bundle_adjust_lattice, fit_grid_points

    image, truth = _make_synthetic_rectified_10x10()
    erased = [2 * 10 + 3, 5 * 10 + 5, 8 * 10 + 1]
    for idx in erased:
        cx, cy = truth[idx]
        x_lo, y_lo = int(round(cx - 14)), int(round(cy - 14))
        image[y_lo:y_lo + 28, x_lo:x_lo + 28] = 200

    initial = fit_grid_points(image, grid_size=10)
    adjusted, info, point_meta = bundle_adjust_lattice(image, initial, grid_size=10)

    for idx in erased:
        assert point_meta[idx]["source"] == "model_imputed", idx
        assert "imputed_position" in point_meta[idx]["flags"]
        assert float(np.linalg.norm(adjusted[idx] - truth[idx])) < 3.0

    supported = [m for i, m in enumerate(point_meta) if i not in erased]
    refined_count = sum(1 for m in supported if m["source"] == "candidate_refined")
    assert refined_count >= 90
    assert info["candidate_support_ratio"] >= 0.9
    assert info["trusted"] is True


def test_bundle_adjust_support_check_inconclusive_under_heavy_blur():
    """重模糊使候选检测器失效但精修证据充分时，支撑率检查应判"不可用"而非误报不可信。

    支撑率是防"规则但错误网格"的外部证据；当检测器本身几乎无候选
    （如重模糊）而逐点精修证据充分时，缺席的是检查手段而不是网格质量。
    """
    import cv2
    from pg_grid import bundle_adjust_lattice, fit_grid_points

    image, truth = _make_synthetic_rectified_10x10()
    # sigma=8 时黑帽候选检测完全失效（实测 0 候选），而质心精修仍然精确。
    blurred = cv2.GaussianBlur(image, (49, 49), 8.0)

    initial = fit_grid_points(blurred, grid_size=10)
    adjusted, info, _ = bundle_adjust_lattice(blurred, initial, grid_size=10)

    assert info["applied"] is True
    assert info["support_check"] == "inconclusive"
    assert info["trusted"] is True
    errors = np.linalg.norm(adjusted - truth, axis=1)
    assert float(errors.mean()) < 3.0


def test_bundle_adjust_rejects_grid_without_image_evidence():
    """空白图像上的均分网格没有任何候选支撑，必须被判定为不可信而非静默通过。"""
    from pg_grid import bundle_adjust_lattice, generate_grid_points

    image = np.full((800, 800, 3), 200, dtype=np.uint8)
    points = generate_grid_points(10, 800, 800, margin_ratio=0.085)

    adjusted, info, point_meta = bundle_adjust_lattice(image, points, grid_size=10)

    assert info["trusted"] is False
    assert float(info["candidate_support_ratio"]) == 0.0
    # 无证据时不得虚构调整：坐标原样返回，点数不变。
    assert adjusted.shape == (100, 2)
    assert np.allclose(adjusted, points, atol=1e-3)


def test_process_image_grid_points_carry_confidence_source_flags(tmp_path):
    """result.json 的每个 grid_point 必须携带 confidence/source/flags 增量字段。"""
    from pg_grid import process_image

    example = Path(__file__).resolve().parent.parent / "examples" / "synthetic_10x10_dark_squares.png"
    result = process_image(image_path=example, grid_size=10, output_dir=tmp_path)

    points = result["grid_points"]
    assert len(points) == 100
    refined_count = 0
    for point in points:
        # 旧字段完整保留
        assert set(["row", "col", "x", "y"]).issubset(point.keys())
        assert 0.0 <= float(point["confidence"]) <= 1.0
        assert point["source"] in ("candidate_refined", "model_imputed", "unadjusted")
        assert isinstance(point["flags"], list)
        if point["source"] == "candidate_refined":
            refined_count += 1
    assert refined_count >= 90

    lattice = result["lattice_consistency"]
    assert lattice["model"] == "homography"
    # 样例图带细线干扰：与线粘连的方块会被全图候选检测的严格过滤排除
    # （局部精修仍能找到它们），支撑率断言按"绝大多数有证据"校准为 0.8，
    # 高于 trusted 阈值 0.6。
    assert float(lattice["candidate_support_ratio"]) >= 0.8
    assert lattice["trusted"] is True
    assert "reasons" in result["quality"]


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


def test_bright_dot_gates_scale_with_grid_size_not_canvas_side():
    """斑点几何门限必须按单元间距归一化，而不是按画幅边长。

    _default_rectified_size 给 10x10 用 800、给 15x15 用 1200，两者的
    单元间距同为 80px、单元尺寸同为约 34px。按边长归一化时 max_area
    是 704 与 1584，同样大的单元在 10x10 被判超限——实测 100 个孔只
    剩 19-24 个候选，且被拒的正是最亮最可靠的大斑点。
    """
    from pg_grid import _detect_bright_dot_candidates

    def scene(size: int, grid_size: int) -> np.ndarray:
        """按管线自身的边距约定铺满 grid_size² 个同样大小的亮点。"""
        image = np.full((size, size, 3), 20, dtype=np.uint8)
        margin = size * 0.085
        pitch = (size - 2.0 * margin) / (grid_size - 1)
        half = max(2, int(round(pitch * 0.42 / 2)))
        for row in range(grid_size):
            for col in range(grid_size):
                cx = int(round(margin + col * pitch))
                cy = int(round(margin + row * pitch))
                image[cy - half:cy + half, cx - half:cx + half] = 210
        return image

    # 单元的像素尺寸在两种规格下几乎相同（约 30px），检出率也必须相同。
    ratios = []
    for size, grid_size in [(800, 10), (1200, 15)]:
        found = _detect_bright_dot_candidates(scene(size, grid_size), grid_size=grid_size)
        ratio = found.shape[0] / float(grid_size * grid_size)
        ratios.append(ratio)
        assert ratio >= 0.9, f"{grid_size}x{grid_size} 只检出 {found.shape[0]}/{grid_size**2}"
    assert abs(ratios[0] - ratios[1]) < 0.1, f"两种规格检出率不一致: {ratios}"


def test_axis_completion_tolerates_whole_dark_rows():
    """整行低于检测地板时必须仍能补全轴，而不是放弃候选晶格路径。

    自发光阵列的稀释序列里最暗的若干行整体不可见是常态：实测 log 稀释
    序列 10x10 只有 6/10 行、15x15 只有 9/15 行产生轴向簇。原先固定
    "最多缺 2 个"的上限会让这类图退到投影回退并给出整体错位的网格。
    """
    from pg_grid import _select_regular_axis_from_clusters

    # 10 行晶格，起点 75、间距 72；只有后 6 行可见（前 4 行整体不发光）。
    pitch, start, count = 72.0, 75.0, 10
    visible = [(start + pitch * i, 500.0, 8) for i in range(4, count)]
    axis = _select_regular_axis_from_clusters(visible, count, 800)

    assert axis is not None, "缺 4 行时轴补全不应放弃"
    assert axis.shape == (count,)
    expected = start + pitch * np.arange(count)
    assert float(np.abs(axis - expected).max()) < 3.0


def test_axis_completion_uses_other_axis_pitch_as_prior():
    """方阵两轴间距应相等，已确定的一轴是另一轴最强的可用先验。"""
    from pg_grid import _select_regular_axis_from_clusters

    pitch, start, count = 70.0, 110.0, 15
    visible = [(start + pitch * i, 400.0, 10) for i in range(6, count)]

    with_prior = _select_regular_axis_from_clusters(visible, count, 1200, pitch)
    assert with_prior is not None
    expected = start + pitch * np.arange(count)
    assert float(np.abs(with_prior - expected).max()) < 3.0


def test_axis_pitch_bounds_are_count_normalised_and_guard_compressed_lattices():
    """轴间距区间必须按行列数归一化，且下界不能放宽。

    历史写法 [length*0.045, length*0.095] 隐含 count≈15：grid=10 时上界
    76 相对名义间距 73.8 只剩 3% 余量，区域框稍一外扩轴拟合就整体被拒。
    但区间不能改成围绕名义间距对称——下界是防"压缩/互补晶格"的门。实测
    把 grid=15 下界从 54 放到 44 后，examples/fluo/gradient_failure_0034
    拟合出间距 44.4 的压缩晶格，错 99.9%pitch 却拿到 0.91 支撑率并被判
    可信，正是"绝不静默失败"契约要拦的情形。
    """
    from pg_grid import _axis_pitch_bounds, _expected_axis_pitch

    # 参考规格 15x15 上按构造逐值等于历史值。
    low, high = _axis_pitch_bounds(1200, 15)
    assert abs(low - 1200 * 0.045) < 1e-6
    assert abs(high - 1200 * 0.095) < 1e-6

    # 其余规格继承同一套相对倍率，余量不再随规格漂移。
    for length, count in [(800, 10), (1200, 15)]:
        low, high = _axis_pitch_bounds(length, count)
        expected = _expected_axis_pitch(length, count)
        assert abs(high / expected - 1.602) < 0.01, f"{count}: 上余量偏离参考规格"
        assert abs(low / expected - 0.759) < 0.01, f"{count}: 下界偏离参考规格"

    # 压缩晶格（约 0.6 倍名义间距）必须落在区间外。
    low, _ = _axis_pitch_bounds(1200, 15)
    assert low > _expected_axis_pitch(1200, 15) * 0.62


def _phantom_lattice(grid_size, col_pitch, row_pitch, start=100.0):
    """构造规则晶格，行/列间距可不同（行优先排列）。"""
    points = []
    for row in range(grid_size):
        for col in range(grid_size):
            points.append((start + col * col_pitch, start + row * row_pitch))
    return np.asarray(points, dtype=np.float64)


def _candidates_on_rows(lattice, grid_size, rows):
    """在指定的若干网格行位置放置候选，返回 N×3。"""
    lat = lattice.reshape(grid_size, grid_size, 2)
    picked = lat[list(rows), :, :].reshape(-1, 2)
    return np.column_stack([picked, np.full(picked.shape[0], 100.0)])


def test_phantom_edge_detects_whole_lattice_shift():
    """整格错位必须被检出——支撑率对它结构性免疫。

    构造与实测错位一致：网格最后一行伸到候选点云之外、自身无支撑，
    内侧邻行支撑良好，且该轴被拉伸（两轴间距失配）。
    """
    from pg_grid import detect_phantom_lattice_edge

    g, pitch = 15, 70.0
    # 行间距被拉伸 6%：错位拟合把 15 行硬塞进只容得下 14 行的跨度
    lattice = _phantom_lattice(g, col_pitch=pitch, row_pitch=pitch * 1.06)
    # 候选覆盖前 14 行，最后一行是凭空多出来的
    candidates = _candidates_on_rows(lattice, g, range(g - 1))

    diag = detect_phantom_lattice_edge(lattice, g, candidates, pitch)
    assert diag["flagged"] == "row_hi", f"未检出行向错位: {diag}"
    assert diag["anisotropy"] > 0.03


def test_phantom_edge_ignores_a_merely_dark_edge_row():
    """边缘行只是检不到候选（强光照梯度）时不得误报。

    实测 fluo/trusted_wrong_gradient 两例定位 0.9%/1.0% pitch，最暗的
    一行没有候选，满足"外伸 + 自身无支撑 + 内侧有支撑"三条却完全正确。
    区分靠第四条：网格正确时两轴间距一致。

    与上一个用例的候选布局完全相同，唯一差别就是几何是否被拉伸——
    这正是判据真正依赖的那个量。
    """
    from pg_grid import detect_phantom_lattice_edge

    g, pitch = 15, 70.0
    lattice = _phantom_lattice(g, col_pitch=pitch, row_pitch=pitch)  # 两轴一致
    candidates = _candidates_on_rows(lattice, g, range(g - 1))       # 同样缺最后一行

    diag = detect_phantom_lattice_edge(lattice, g, candidates, pitch)
    assert diag["flagged"] is None, f"暗边缘行被误报为幻影: {diag}"
    assert diag["anisotropy"] < 0.01


def test_phantom_edge_stays_silent_without_candidates():
    """检测器整体失效时不得判错——那是检查手段缺席，不是网格错位。"""
    from pg_grid import detect_phantom_lattice_edge

    g, pitch = 15, 70.0
    truth = _phantom_lattice(g, pitch, 100.0)
    diag = detect_phantom_lattice_edge(truth, g, np.empty((0, 3)), pitch)

    assert diag["flagged"] is None
    assert diag["available"] is False


def test_phantom_edge_downgrades_trust_end_to_end(tmp_path):
    """接进管线：真实错位样本必须报 trusted=false 并给出可读原因。"""
    import json
    from pathlib import Path
    from pg_grid import process_image

    case = Path(__file__).resolve().parent.parent / "examples" / "fluo" / "trusted_wrong_gradient_0015"
    result = process_image(image_path=case / "input.jpg", grid_size=15, output_dir=tmp_path / "ok")
    lattice = result["lattice_consistency"]

    # 这张图定位正确，不得被幻影判据误伤
    assert lattice["phantom_edge"]["flagged"] is None
    assert lattice["trusted"] is True
