"""荧光成像仿真器测试：可复现性、真值一致性、物理正确性。"""

import json
from pathlib import Path

import numpy as np


def test_wavelength_to_rgb_maps_530nm_to_green():
    """530nm 必须映射为绿色主导（G 最大、B 最小）。"""
    from pg_fluoro_sim import wavelength_to_rgb

    r, g, b = wavelength_to_rgb(530.0)
    assert g == max(r, g, b)
    assert g > 0.9
    assert b < 0.05
    # 530nm 偏黄绿：红分量存在但远小于绿。
    assert 0.1 < r < 0.5

    # 相邻波段的方向性：更长波长更偏黄/红，更短波长更偏青。
    assert wavelength_to_rgb(570.0)[0] > r
    assert wavelength_to_rgb(500.0)[2] > b


def test_render_is_reproducible_with_seed():
    """同一 seed 必须逐字节复现；不同 seed 必须不同。"""
    from pg_fluoro_sim import FluorescenceConfig, render_fluorescence_scene

    config = FluorescenceConfig(grid_size=10, image_size=600, seed=3)
    image_a, truth_a = render_fluorescence_scene(config)
    image_b, truth_b = render_fluorescence_scene(config)
    image_c, _ = render_fluorescence_scene(FluorescenceConfig(grid_size=10, image_size=600, seed=4))

    assert np.array_equal(image_a, image_b)
    assert np.allclose(truth_a["points"], truth_b["points"])
    assert not np.array_equal(image_a, image_c)


def test_ground_truth_points_land_on_emitting_wells():
    """真值点必须落在发光孔上：孔中心亮度显著高于孔间背景。"""
    from pg_fluoro_sim import FluorescenceConfig, render_fluorescence_scene

    config = FluorescenceConfig(
        grid_size=10, image_size=800, intensity_pattern="uniform",
        intensity_max=0.6, radial_k1=0.0, perspective_strength=0.0,
        rotation_deg=0.0, seed=1,
    )
    image, truth = render_fluorescence_scene(config)
    gray = image.mean(axis=2)
    points = np.asarray(truth["points"], dtype=np.float64)

    assert points.shape == (100, 2)
    well_values = np.array([gray[int(round(y)), int(round(x))] for x, y in points])
    # 孔间中点作为背景参照（相邻两孔连线中点）。
    lattice = points.reshape(10, 10, 2)
    gaps = (lattice[:, :-1] + lattice[:, 1:]) / 2.0
    gap_values = np.array([gray[int(round(y)), int(round(x))] for x, y in gaps.reshape(-1, 2)])

    assert float(well_values.mean()) > float(gap_values.mean()) + 40.0


def test_intensity_ordering_is_preserved_in_rendered_image():
    """渲染后的孔亮度必须与设定强度单调相关（未饱和区间）。"""
    from pg_fluoro_sim import FluorescenceConfig, render_fluorescence_scene

    config = FluorescenceConfig(
        grid_size=10, image_size=800, intensity_pattern="log_series",
        intensity_min=0.02, intensity_max=0.5, radial_k1=0.0,
        perspective_strength=0.0, rotation_deg=0.0, vignetting=0.0, seed=2,
    )
    image, truth = render_fluorescence_scene(config)
    gray = image.mean(axis=2).astype(np.float64)
    points = np.asarray(truth["points"], dtype=np.float64)
    intensities = np.asarray(truth["intensities"], dtype=np.float64)

    measured = []
    for x, y in points:
        xi, yi = int(round(x)), int(round(y))
        measured.append(float(gray[yi - 2:yi + 3, xi - 2:xi + 3].mean()))
    measured = np.asarray(measured)

    rank_true = np.argsort(np.argsort(intensities)).astype(np.float64)
    rank_measured = np.argsort(np.argsort(measured)).astype(np.float64)
    spearman = float(np.corrcoef(rank_true, rank_measured)[0, 1])
    assert spearman > 0.95


def test_sensor_noise_and_saturation_behave_physically():
    """散粒噪声必须随信号增强；饱和孔必须被截断到 255 附近。"""
    from pg_fluoro_sim import FluorescenceConfig, render_fluorescence_scene

    config = FluorescenceConfig(
        grid_size=10, image_size=800, intensity_pattern="log_series",
        intensity_min=0.001, intensity_max=3.0, radial_k1=0.0,
        perspective_strength=0.0, rotation_deg=0.0, vignetting=0.0, seed=5,
    )
    image, truth = render_fluorescence_scene(config)
    # 单色荧光只在对应发射通道上饱和：530nm 的蓝分量近 0，
    # 用三通道均值会低估亮度，因此取主发射通道（BGR 的 G）。
    green = image[:, :, 1].astype(np.float64)
    points = np.asarray(truth["points"], dtype=np.float64)
    intensities = np.asarray(truth["intensities"], dtype=np.float64)

    def patch_stats(index):
        x, y = points[index]
        xi, yi = int(round(x)), int(round(y))
        patch = green[yi - 3:yi + 4, xi - 3:xi + 4]
        return float(patch.mean()), float(patch.std())

    order = np.argsort(intensities)
    dim_mean, dim_std = patch_stats(order[5])
    mid_mean, mid_std = patch_stats(order[len(order) // 2])

    # 散粒噪声：强信号处绝对噪声更大。
    assert mid_std > dim_std
    # 强度设为 3.0（远超满阱）的孔必须饱和。
    bright_mean, _ = patch_stats(order[-1])
    assert bright_mean > 245.0


def test_write_fluorescence_sample_emits_image_and_ground_truth(tmp_path):
    """落盘必须同时产出图像与可复现的真值 JSON。"""
    from pg_fluoro_sim import FluorescenceConfig, write_fluorescence_sample

    config = FluorescenceConfig(grid_size=15, image_size=900, seed=7)
    paths = write_fluorescence_sample(config, tmp_path, name="sample")

    image_path = Path(paths["image"])
    truth_path = Path(paths["ground_truth"])
    assert image_path.exists() and truth_path.exists()

    with truth_path.open("r", encoding="utf-8") as f:
        truth = json.load(f)
    assert truth["schema"] == "pg-fluoro-truth-v1"
    assert truth["grid_size"] == 15
    assert len(truth["points"]) == 225
    assert len(truth["intensities"]) == 225
    assert truth["emission_nm"] == 530.0
    # 配置完整回写，保证可复现。
    assert truth["config"]["seed"] == 7


def test_pipeline_localizes_generated_fluorescence_image(tmp_path):
    """生成的荧光图必须能被现有管线正确定位（极性判为 bright）。"""
    from pg_fluoro_sim import FluorescenceConfig, write_fluorescence_sample
    from pg_grid import process_image

    config = FluorescenceConfig(
        grid_size=15, image_size=1400, intensity_pattern="log_series",
        intensity_min=0.05, intensity_max=0.9, seed=11,
    )
    paths = write_fluorescence_sample(config, tmp_path, name="fluoro15")
    result = process_image(image_path=paths["image"], grid_size=15, output_dir=tmp_path / "out")

    assert result["unit_polarity"] == "bright"
    assert result["point_count"] == 225
    assert result["chip_region"]["method"] != "fallback_center"

    metrics = _localization_error(paths["ground_truth"], result)
    assert metrics["mean_pct_pitch"] < 8.0


def _localization_error(truth_path, result):
    """把真值点映射到矫正坐标系并计算误差（测试辅助）。"""
    import cv2

    with Path(truth_path).open("r", encoding="utf-8") as f:
        truth = json.load(f)
    points = np.asarray(truth["points"], dtype=np.float32).reshape(-1, 1, 2)
    region = np.asarray(result["chip_region"]["points"], dtype=np.float32)
    size = int(result["rectified_size"])
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(region, dst)
    mapped = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)

    predicted = np.asarray([[p["x"], p["y"]] for p in result["grid_points"]], dtype=np.float64)
    errors = np.linalg.norm(predicted - mapped, axis=1)
    pitch = float(result["lattice_consistency"]["pitch_px"])
    return {"mean_px": float(errors.mean()), "mean_pct_pitch": float(errors.mean() / pitch * 100.0)}


def test_region_selection_is_not_dominated_by_one_hypothesis(tmp_path):
    """区域假设必须由证据选出，不能由某一条路径垄断。

    固定优先级的症状是"某条路径被用在几乎所有图上"——那说明它是被
    顺序选中的，而不是被证据选中的。这里要求在失败样本集上至少有两条
    路径各自胜出过，且诊断里记录了完整的假设比较过程。
    """
    from pg_grid import process_image

    case_dir = Path(__file__).resolve().parent.parent / "examples" / "fluo"
    cases = sorted(p for p in case_dir.iterdir() if (p / "ground_truth.json").exists())[:12]

    chosen, multi_hypothesis = [], 0
    for case in cases:
        with (case / "ground_truth.json").open("r", encoding="utf-8") as f:
            grid_size = int(json.load(f)["grid_size"])
        result = process_image(
            image_path=case / "input.jpg", grid_size=grid_size, output_dir=tmp_path / case.name
        )
        lattice = result["lattice_consistency"]
        hypotheses = lattice.get("region_hypotheses", [])
        assert hypotheses, "诊断中必须记录区域假设比较过程"
        assert sum(1 for h in hypotheses if h["selected"]) == 1
        assert "grid_coverage_ratio" in lattice
        chosen.append(result["chip_region"]["method"])
        if len(hypotheses) > 1:
            multi_hypothesis += 1

    assert multi_hypothesis >= 6, "多数样本应至少产出两个可比较的区域假设"
    assert len(set(chosen)) >= 2, f"区域路径被单一假设垄断：{set(chosen)}"


def test_bundled_fluo_failure_cases_never_fail_silently(tmp_path):
    """examples/fluo 的失败样本集：绝不允许"宣称可信但定位错"。

    这是最重要的一条契约——定位失败可以接受（极暗单元、大量失效孔本身
    就存在物理极限），但算法必须如实报告。历史上这批样本里有 8 例
    trusted=True 却整体错位一个间距（支撑率对整数格平移是盲的）。
    同时要求多数样本能被正确定位，防止用"全部报失败"来通过本测试。
    """
    import cv2
    from pg_grid import process_image

    case_dir = Path(__file__).resolve().parent.parent / "examples" / "fluo"
    cases = sorted(p for p in case_dir.iterdir() if (p / "ground_truth.json").exists())
    assert len(cases) >= 20, f"失败样本集过小：{len(cases)}"

    silent_failures, accurate = [], 0
    for case in cases:
        with (case / "ground_truth.json").open("r", encoding="utf-8") as f:
            truth = json.load(f)
        grid_size = int(truth["grid_size"])
        result = process_image(
            image_path=case / "input.jpg", grid_size=grid_size, output_dir=tmp_path / case.name
        )

        region = np.asarray(result["chip_region"]["points"], dtype=np.float32)
        size = int(result["rectified_size"])
        matrix = cv2.getPerspectiveTransform(
            region,
            np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32),
        )
        mapped = cv2.perspectiveTransform(
            np.asarray(truth["points"], dtype=np.float32).reshape(-1, 1, 2), matrix
        ).reshape(-1, 2)
        predicted = np.asarray([[p["x"], p["y"]] for p in result["grid_points"]], dtype=np.float64)
        pitch = max(float(result["lattice_consistency"]["pitch_px"]), 1e-6)
        error_pct = float(np.linalg.norm(predicted - mapped, axis=1).mean() / pitch * 100.0)

        if error_pct <= 10.0:
            accurate += 1
        elif bool(result["lattice_consistency"]["trusted"]):
            silent_failures.append((case.name, round(error_pct, 1)))

    assert not silent_failures, f"静默错误（trusted 但定位错）：{silent_failures}"
    assert accurate >= 15, f"仅 {accurate}/{len(cases)} 例被正确定位"


def test_evaluate_generated_sample_reports_per_decade_detection(tmp_path):
    """评估入口必须按强度分档报告定位与定量表现。"""
    from pg_fluoro_sim import FluorescenceConfig, evaluate_generated_sample, write_fluorescence_sample

    config = FluorescenceConfig(
        grid_size=10, image_size=1200, intensity_pattern="log_series",
        intensity_min=0.002, intensity_max=1.2, seed=13,
    )
    paths = write_fluorescence_sample(config, tmp_path, name="eval10")
    report = evaluate_generated_sample(paths["image"], paths["ground_truth"], tmp_path / "eval_out")

    assert report["grid_size"] == 10
    assert report["point_count"] == 100
    assert "mean_error_pct_pitch" in report
    assert len(report["decades"]) >= 3
    for row in report["decades"]:
        assert {"intensity_low", "intensity_high", "count", "mean_error_px", "supported_ratio"} <= set(row)
