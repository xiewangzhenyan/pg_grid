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
    assert truth["schema"] == "pg-fluoro-truth-v2"
    assert truth["grid_size"] == 15
    assert len(truth["points"]) == 225
    assert len(truth["intensities"]) == 225
    assert truth["emission_nm"] == 530.0
    # 配置完整回写，保证可复现。
    assert truth["config"]["seed"] == 7
    # 未启用浓度模式时浓度键必须存在且为 None——下游可以无条件取键判断。
    assert truth["concentrations"] is None
    assert truth["concentration_unit"] is None


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


def test_region_selection_is_evidence_driven(tmp_path):
    """区域假设必须经过比较后选出，且选择过程可复核。

    注意这里不强求"多条路径各自胜出"：若某条路径在这批图上确实都拿到
    更强证据，全选它就是正确结果。防止"顺序即选择"靠的是仲裁规则本身
    （见 select_region_solution 的单元测试），这里要保证的是机制确实在
    运行——多个假设被实际求解、证据被记录、且恰好一个被选中。
    """
    from pg_grid import process_image

    case_dir = Path(__file__).resolve().parent.parent / "examples" / "fluo"
    cases = sorted(p for p in case_dir.iterdir() if (p / "ground_truth.json").exists())[:10]

    multi_hypothesis = 0
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
        for hypothesis in hypotheses:
            assert {"method", "coverage", "score", "selected"} <= set(hypothesis)
        if len(hypotheses) > 1:
            multi_hypothesis += 1

    assert multi_hypothesis >= 5, "多数样本应至少产出两个可比较的区域假设"


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


def test_emission_is_non_monotonic_so_relative_labels_are_ambiguous():
    """浓度→强度在量程内非单调，这正是真值必须记绝对浓度的理由。

    发射强度到达峰值后因内滤效应回落，拐点两侧存在读数相同的两个浓度。
    若真值只记"占最大浓度的百分比"，同一个百分比标签在不同 C_max 下对应
    的物理位置会漂移，标注与图像里烘焙的非线性就对不上了。

    拐点位置由 A = ε·c·l 决定，所以本测试分两段：模型必须能表达拐点
    （物理事实），而默认的 0.5 mm 液层必须把它推到 1 nM–100 µM 之外
    （实际可反演性）。液层加厚会把拐点拉进量程——一并断言，避免有人
    改了 path_length_cm 却没意识到量程上半段已经不可反演。
    """
    from dataclasses import replace

    from pg_fluoro_sim import Photophysics, concentration_to_emission

    physics = Photophysics()
    concentration = np.logspace(-3, 3.5, 4000)
    emission = concentration_to_emission(concentration, physics)

    peak = int(np.argmax(emission))
    assert 0 < peak < len(concentration) - 1, "模型必须能表达内滤拐点"
    assert emission[-1] < emission[peak], "越过拐点后强度必须回落"

    # 拐点左侧存在与量程上端读数相同的浓度：单凭读数无法区分二者。
    twin = int(np.argmin(np.abs(emission[:peak] - emission[-1])))
    assert concentration[twin] < concentration[peak] < concentration[-1]
    assert abs(emission[twin] - emission[-1]) / emission[-1] < 0.02

    # 默认 0.5 mm 液层：1 nM–100 µM 全程严格单调，因而可反演。
    working = np.logspace(-3, 2, 4000)
    assert np.all(np.diff(concentration_to_emission(working, physics)) > 0)
    assert concentration[peak] > 100.0

    # 液层加厚到 3 mm，拐点前移进量程，上半段不再可反演。
    thick = concentration_to_emission(working, replace(physics, path_length_cm=0.30))
    assert np.any(np.diff(thick) < 0)


def test_linear_mode_is_monotonic_and_continuous_with_nonlinear_at_dilute_limit():
    """关掉内滤后必须严格单调，且低浓度端与非线性模式数值连续。

    连续性是刻意的：两种模式在稀释极限重合，才能把"线性假设下训练的
    模型在真实非线性数据上掉多少"归因于非线性本身，而不是归因于两套
    数据用了不同的比例系数。
    """
    from pg_fluoro_sim import Photophysics, concentration_to_emission

    concentration = np.logspace(-3, 2, 500)
    nonlinear = concentration_to_emission(concentration, Photophysics())
    linear = concentration_to_emission(
        concentration, Photophysics(inner_filter=False, self_quenching=False))

    assert np.all(np.diff(linear) > 0)
    # 线性模式下读数与浓度严格成比例。
    ratio = linear / concentration
    assert float(ratio.std() / ratio.mean()) < 1e-9
    # 稀释极限（≤10 nM）两模式相对偏差 < 0.1%。
    dilute = concentration <= 0.01
    assert np.allclose(linear[dilute], nonlinear[dilute], rtol=1e-3)


def test_turnover_tracks_path_length_and_is_absent_when_linear():
    """拐点位置必须随液层厚度按 A = ε·c·l 移动，线性模式下不存在。

    这是"量程上界该定在哪"的依据：改了光程或换了荧光团，可反演的上界
    就跟着变，不该靠人记住。
    """
    from dataclasses import replace

    from pg_fluoro_sim import Photophysics, turnover_concentration_uM

    physics = Photophysics()
    expected = {3.0: 51.5, 1.0: 151.6, 0.5: 295.4, 0.2: 690.2}
    for mm, reference in expected.items():
        measured = turnover_concentration_uM(replace(physics, path_length_cm=mm / 10.0))
        assert measured is not None
        assert abs(measured - reference) / reference < 0.02

    # 拐点 ∝ 1/(ε·l)：加倍消光系数等价于加倍光程。
    doubled_epsilon = turnover_concentration_uM(replace(physics, epsilon_M_cm=160000.0))
    doubled_path = turnover_concentration_uM(replace(physics, path_length_cm=0.10))
    assert abs(doubled_epsilon - doubled_path) / doubled_path < 0.02

    # 线性模式严格单调，没有拐点可报。
    assert turnover_concentration_uM(
        Photophysics(inner_filter=False, self_quenching=False)) is None


def test_truth_carries_absolute_concentration_and_photophysics(tmp_path):
    """浓度模式的真值必须含绝对浓度、单位、派生比例与光物理参数。"""
    from pg_fluoro_sim import FluorescenceConfig, write_fluorescence_sample

    config = FluorescenceConfig(
        grid_size=10, image_size=700, concentration_pattern="log_series",
        concentration_min_uM=0.001, concentration_max_uM=100.0, seed=17,
    )
    paths = write_fluorescence_sample(config, tmp_path, name="conc")
    with Path(paths["ground_truth"]).open("r", encoding="utf-8") as f:
        truth = json.load(f)

    concentrations = np.asarray(truth["concentrations"], dtype=np.float64)
    fractions = np.asarray(truth["concentration_fraction"], dtype=np.float64)
    assert truth["concentration_unit"] == "uM"
    assert concentrations.shape == (100,)
    assert concentrations.min() >= 0.0 and concentrations.max() <= 200.0
    # 派生比例必须与绝对浓度一致，且以本图最大浓度为 1。
    assert np.isclose(fractions.max(), 1.0)
    assert np.allclose(fractions, concentrations / concentrations.max(), atol=1e-6)

    physics = truth["photophysics"]
    assert physics["fluorophore"] == "fluorescein"
    assert physics["inner_filter"] is True
    assert truth["exposure_gain"] > 0.0
    # 强度是导出量，必须与浓度同为 0 / 同为正。
    intensities = np.asarray(truth["intensities"], dtype=np.float64)
    assert np.array_equal(concentrations > 0, intensities > 0)


def test_intensity_is_derived_from_concentration_not_the_reverse():
    """浓度模式下改光物理参数必须改变强度：因果方向是浓度→强度。"""
    from dataclasses import replace

    from pg_fluoro_sim import FluorescenceConfig, Photophysics

    base = FluorescenceConfig(
        grid_size=10, concentration_pattern="log_series",
        concentration_min_uM=0.001, concentration_max_uM=100.0,
        concentration_jitter=0.0, exposure_gain=1.5, seed=3,
    )
    linear = replace(base, photophysics=Photophysics(inner_filter=False, self_quenching=False))

    nonlinear_values = base.resolved_intensities()
    linear_values = linear.resolved_intensities()
    assert not np.allclose(nonlinear_values, linear_values)
    # 高浓度端被内滤压缩，低浓度端两者一致。
    assert nonlinear_values[-1] < linear_values[-1]
    assert np.isclose(nonlinear_values[0], linear_values[0], rtol=1e-3)


def test_fixed_exposure_gain_keeps_concentrations_comparable_across_images():
    """整批共用曝光增益后，同一浓度在不同图上必须给出同一强度。

    这是"绝对浓度"能跨图汇总的前提。反例同时验证：自动定标下每张图按
    自己的峰值归一化，同一浓度读数不同——公共纵轴消失，多张图的散点
    没法合成一条标定曲线。
    """
    from dataclasses import replace

    from pg_fluoro_sim import FluorescenceConfig

    narrow = FluorescenceConfig(
        grid_size=10, concentration_pattern="log_series",
        concentration_min_uM=0.001, concentration_max_uM=10.0,
        concentration_jitter=0.0, seed=1,
    )
    wide = replace(narrow, concentration_max_uM=100.0)
    # 两图最低浓度相同（0.001 µM），是可比性的探针。
    assert np.isclose(narrow.resolved_concentrations()[0], wide.resolved_concentrations()[0])

    # 自动定标：同一浓度读数不同。
    assert not np.isclose(
        narrow.resolved_intensities()[0], wide.resolved_intensities()[0], rtol=0.05)

    # 共用增益：同一浓度必须给出同一强度。
    gain = narrow.resolved_exposure_gain()
    assert np.isclose(
        replace(narrow, exposure_gain=gain).resolved_intensities()[0],
        replace(wide, exposure_gain=gain).resolved_intensities()[0],
        rtol=1e-9,
    )


def test_summarize_calibration_pairs_finds_lod_and_inner_filter_turnover():
    """汇总必须同时定位检出下限与内滤拐点。

    拐点（默认 0.5 mm 液层下约 295 µM）与量程上端同处 [100,1000) 一个十倍
    档内，因此不能靠档间比较中位数发现它——本测试正是针对这一点。
    """
    from pg_fluoro_sim import Photophysics, concentration_to_emission, summarize_calibration_pairs

    rng = np.random.default_rng(0)
    physics = Photophysics()
    concentration = np.logspace(-3, 3, 900)
    gray = concentration_to_emission(concentration, physics) * 300.0
    gray = gray + rng.normal(0.0, 0.4, gray.size)

    rows = [{"concentration_uM": float(c), "measured_gray": float(g)}
            for c, g in zip(concentration, gray)]
    rows += [{"concentration_uM": 0.0, "measured_gray": float(v)}
             for v in rng.normal(0.0, 0.4, 40)]

    summary = summarize_calibration_pairs(rows)
    assert summary["detection_limit_uM"] is not None
    # 检出下限必须显著低于拐点：可用区间要能跨至少一个半数量级，
    # 否则"可反演"就成了空话。
    assert 0.001 <= summary["detection_limit_uM"] <= 10.0
    assert summary["monotonic_max_uM"] / summary["detection_limit_uM"] > 30.0
    assert summary["turnover_detected"] is True
    # 理论拐点 295 µM；容差覆盖噪声与滑动中位数窗口的影响。
    assert 150.0 <= summary["monotonic_max_uM"] <= 600.0
    assert summary["spearman_usable"] > 0.95


def test_build_concentration_dataset_pools_pairs_and_gates_on_localization(tmp_path):
    """数据集必须汇总浓度-灰度配对、共用曝光增益、并按定位精度筛图。"""
    import csv

    from pg_fluoro_sim import build_concentration_dataset

    # 两个实测约束（均为既有管线行为，与浓度改动无关），故意用 10×10 这个
    # 更严苛的规格：900px 时孔斑过小、支撑率跌到 0.01–0.27；ideal 预设在
    # 10×10 上系统性失败（支撑 0.03–0.10，锐利合成方块反而难检出），而在
    # 15×15 上四种预设都是 3/3。
    manifest = build_concentration_dataset(
        tmp_path, count=3, grid_size=10, concentration_pattern="plate_series",
        difficulties=("typical", "hard"), blank_wells=6,
        base_overrides={"image_size": 1200},
    )

    assert manifest["schema"] == "pg-fluoro-dataset-v1"
    assert manifest["concentration_unit"] == "uM"
    assert manifest["image_count"] == 3
    assert manifest["included_image_count"] >= 2
    assert manifest["pair_count"] == manifest["included_image_count"] * 100

    # 曝光增益整批唯一：每张图的真值都必须回写同一个值。
    gains = set()
    for sample in manifest["samples"]:
        with Path(sample["ground_truth"]).open("r", encoding="utf-8") as f:
            gains.add(json.load(f)["exposure_gain"])
    assert len(gains) == 1
    assert abs(gains.pop() - manifest["exposure_gain"]) < 1e-6

    # 每张图的筛选结论必须可复核。
    for sample in manifest["samples"]:
        assert (sample["excluded_reason"] is None) == sample["included"]

    with Path(manifest["outputs"]["calibration_pairs_csv"]).open(encoding="utf-8") as f:
        pairs = list(csv.DictReader(f))
    assert len(pairs) == manifest["pair_count"]
    assert {"concentration_uM", "measured_gray", "reliable", "snr"} <= set(pairs[0])
    # 空白孔（浓度 0）必须进入数据集——没有本底就定不出检出下限。
    assert sum(1 for p in pairs if float(p["concentration_uM"]) == 0.0) >= 6

    calibration = manifest["calibration"]
    assert calibration["detection_limit_uM"] is not None
    assert calibration["blank_std_gray"] >= 0.0
    assert len(calibration["decades"]) >= 3


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
