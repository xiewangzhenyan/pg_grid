"""PG-Colori-Sim 比色仿真测试。"""

import json
from pathlib import Path

import numpy as np
import pytest


def test_transmittance_is_monotonic_and_blank_is_unity():
    """透射率必须随浓度单调下降，且空白恒为 1.0。

    这是比色与荧光最根本的方向差异：荧光有内滤拐点（非单调），
    比色的透射率在任何浓度下都严格单调下降，永远不会回升。
    """
    from pg_colori_sim import Chromophore, concentration_to_channel_transmittance

    concentrations = np.array([0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0])
    transmittance = concentration_to_channel_transmittance(concentrations, Chromophore())

    assert transmittance.shape == (6, 3)
    # 空白孔三通道全为 1.0
    assert np.allclose(transmittance[0], 1.0, atol=1e-9)
    # 每个通道都严格单调下降
    for channel in range(3):
        diffs = np.diff(transmittance[:, channel])
        assert np.all(diffs <= 1e-12), f"通道 {channel} 出现回升: {transmittance[:, channel]}"
    # 透射率始终在 (0, 1]
    assert np.all(transmittance > 0.0) and np.all(transmittance <= 1.0 + 1e-9)


def test_yellow_chromophore_absorbs_blue_not_red():
    """黄色显色产物必须吸蓝透红——否则就不是"比色"而只是"变暗"。"""
    from pg_colori_sim import Chromophore, apparent_absorbance, concentration_to_channel_transmittance

    transmittance = concentration_to_channel_transmittance(np.array([500.0]), Chromophore())
    absorbance = apparent_absorbance(transmittance)[0]   # RGB 顺序

    assert absorbance[2] > absorbance[1] > absorbance[0], f"通道次序不对: {absorbance}"
    # 蓝通道吸光度应比红通道高一个数量级以上
    assert absorbance[2] > absorbance[0] * 10.0


def test_polychromatic_nonlinearity_compresses_high_absorbance():
    """多色光下高吸光度必须偏离线性（Beer-Lambert 与光谱积分不可交换）。

    用标量 ε 建模会得到一条永远笔直的曲线，把量程上界藏起来。
    """
    from pg_colori_sim import Chromophore, linearity_table

    rows = linearity_table(Chromophore(), np.array([10.0, 100.0, 1000.0, 4000.0]))
    ratios = [r["ratio"] for r in rows]

    # 低浓度端应当几乎线性
    assert ratios[0] > 0.99
    # 高浓度端必须显著压缩，且比值单调下降
    assert ratios[-1] < 0.5
    assert all(ratios[i] >= ratios[i + 1] for i in range(len(ratios) - 1))


def test_path_length_scales_absorbance_linearly():
    """吸光度与光程成正比：0.5 mm 芯片相对 10 mm 比色皿灵敏度低 20 倍。

    这是硬件量程的来源，必须能被仿真如实复现。
    """
    from pg_colori_sim import Chromophore, apparent_absorbance, concentration_to_channel_transmittance

    thin = apparent_absorbance(
        concentration_to_channel_transmittance(np.array([20.0]), Chromophore(path_length_mm=0.5))
    )[0, 2]
    thick = apparent_absorbance(
        concentration_to_channel_transmittance(np.array([20.0]), Chromophore(path_length_mm=5.0))
    )[0, 2]

    # 低吸光度区间尚未进入压缩，比值应接近光程比 10
    assert 9.0 < thick / thin < 10.1


def test_render_produces_bright_wells_on_dark_mask(tmp_path):
    """黑罩挡光、孔透光：孔内必须显著亮于罩体，极性为 bright。"""
    from pg_colori_sim import ColorimetricConfig, render_colorimetric_scene

    config = ColorimetricConfig(grid_size=15, image_size=600, seed=3, layout="masked")
    image, truth = render_colorimetric_scene(config)

    assert image.shape == (600, 600, 3)
    assert truth["schema"] == "pg-colori-truth-v1"
    assert len(truth["points"]) == 225
    assert len(truth["concentrations"]) == 225
    assert np.asarray(truth["absorbance_rgb"]).shape == (225, 3)

    gray = image.mean(axis=2)
    points = np.asarray(truth["points"], dtype=np.float64)
    # 孔中心亮度 vs 相邻孔间隙（罩体）亮度
    well = [gray[int(y), int(x)] for x, y in points
            if 0 <= int(y) < 600 and 0 <= int(x) < 600]
    assert float(np.median(well)) > float(np.median(gray)) * 1.5


def test_bare_layout_makes_spots_darker_than_substrate():
    """亮基底形态：反应区必须比基底**暗**，极性为 dark。

    这是实拍 15x15 芯片的真实样子（白色膜基底 + 深色反应点）。它与黑罩
    形态走的是定位管线里完全不同的检测器（暗方块 vs 亮点），所以两种
    形态都必须能生成，否则验证的是另一条代码路径。
    """
    from pg_colori_sim import ColorimetricConfig, render_colorimetric_scene

    config = ColorimetricConfig(
        grid_size=15, image_size=700, seed=4, layout="bare",
        concentration_pattern="uniform", concentration_max_uM=400.0,
        blank_well_count=0, el_honeycomb=0.0, substrate_texture=0.0,
    )
    image, truth = render_colorimetric_scene(config)

    gray = image.mean(axis=2)
    points = np.asarray(truth["points"], dtype=np.float64)
    spot = [gray[int(y), int(x)] for x, y in points
            if 0 <= int(y) < 700 and 0 <= int(x) < 700]
    panel = float(np.percentile(gray[gray > 30], 75))    # 基底亮度
    assert float(np.median(spot)) < panel * 0.9, "反应区没有比基底暗"


def test_blank_image_is_brighter_than_sample(tmp_path):
    """空白图必须整体亮于样品图——比色的方向与荧光相反。"""
    from pg_colori_sim import ColorimetricConfig, render_colorimetric_scene

    config = ColorimetricConfig(grid_size=15, image_size=500, seed=11)
    sample, sample_truth = render_colorimetric_scene(config, force_blank=False)
    blank, blank_truth = render_colorimetric_scene(config, force_blank=True)

    assert blank.mean() > sample.mean()
    assert all(c == 0.0 for c in blank_truth["concentrations"])
    assert blank_truth["mode"] == "blank"
    assert sample_truth["mode"] == "sample"
    # 空白图逐孔透射率全为 1.0
    assert np.allclose(np.asarray(blank_truth["transmittance_rgb"]), 1.0, atol=1e-9)


def test_write_sample_emits_paired_blank(tmp_path):
    """落盘必须同时给出配对空白图：没有空白就没有吸光度。"""
    from pg_colori_sim import ColorimetricConfig, write_colorimetric_sample

    config = ColorimetricConfig(grid_size=15, image_size=400, seed=5)
    paths = write_colorimetric_sample(config, tmp_path, name="c")

    for key in ("image", "image_truth", "blank", "blank_truth"):
        assert Path(paths[key]).exists(), key
    with Path(paths["blank_truth"]).open(encoding="utf-8") as f:
        assert json.load(f)["mode"] == "blank"


@pytest.mark.parametrize("layout,expected_polarity",
                         [("bare", "dark"), ("masked", "bright")])
def test_pipeline_localizes_colorimetric_image(tmp_path, layout, expected_polarity):
    """定位管线必须能处理**两种**比色版面，且极性判定正确。"""
    import cv2
    from pg_colori_sim import ColorimetricConfig, write_colorimetric_sample
    from pg_grid import process_image

    config = ColorimetricConfig(
        grid_size=15, image_size=1100, seed=2, layout=layout,
        concentration_pattern="uniform", concentration_max_uM=400.0, blank_well_count=0,
        rotation_deg=1.5, perspective_strength=0.01, radial_k1=-0.02,
    )
    paths = write_colorimetric_sample(config, tmp_path, name="loc")
    with Path(paths["image_truth"]).open(encoding="utf-8") as f:
        truth = json.load(f)

    result = process_image(image_path=paths["image"], grid_size=15, output_dir=tmp_path / "out")

    assert result["unit_polarity"] == expected_polarity
    assert result["point_count"] == 225

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

    assert error_pct < 10.0, f"比色图定位误差 {error_pct:.2f} %pitch"


def test_leaky_mask_lifts_dark_field_toward_substrate():
    """半透光罩体必须介于"理想黑罩"与"不装罩"之间，且单调过渡。

    这是判断 PDMS 罩值不值得装的物理基础：罩体透过率是一个连续旋钮，
    T→0 得到黑底亮孔，T→1 退化为不装罩。
    """
    from pg_colori_sim import ColorimetricConfig, render_colorimetric_scene

    def field_level(transmittance: float) -> float:
        config = ColorimetricConfig(
            grid_size=15, image_size=600, seed=8, layout="masked",
            mask_transmittance=transmittance,
            concentration_pattern="uniform", concentration_max_uM=300.0,
            blank_well_count=0, el_honeycomb=0.0, substrate_texture=0.0,
            defocus_sigma_px=0.4, vignetting=0.0, illumination_nonuniformity=0.0,
        )
        image, _ = render_colorimetric_scene(config)
        gray = image.mean(axis=2)
        panel = gray[180:420, 180:420]
        return float(np.percentile(panel, 20))    # 罩体（孔间）亮度

    opaque, leaky, very_leaky = field_level(0.002), field_level(0.2), field_level(0.6)
    assert opaque < leaky < very_leaky, f"罩体亮度非单调: {opaque}, {leaky}, {very_leaky}"


def test_polarity_flip_absorbance_matches_mask_transmittance():
    """极性翻转点 A = −log10(T_mask)：孔比罩体还暗时同图极性不再一致。"""
    from pg_colori_sim import polarity_flip_absorbance

    assert abs(polarity_flip_absorbance(0.01) - 2.0) < 1e-9
    assert abs(polarity_flip_absorbance(0.1) - 1.0) < 1e-9
    # 越透光越早翻转
    assert polarity_flip_absorbance(0.3) < polarity_flip_absorbance(0.05)


def test_plasmon_shift_is_langmuir_not_linear():
    """结合响应必须是 Langmuir 饱和，不是线性。

    专利用线性拟合标准曲线，只在 c ≪ K_D 时成立；跨一个数量级以上
    线性拟合会在高浓度端系统性偏低。
    """
    from pg_colori_sim import PlasmonResonance

    p = PlasmonResonance(shift_max_nm=4.0, kd_uM=0.05)
    c = np.array([0.0, 0.005, 0.05, 0.5, 5.0, 50.0])
    shift = p.shift_for(c)

    assert shift[0] == 0.0
    # K_D 处恰好半饱和
    assert abs(shift[2] - 2.0) < 1e-9
    # 单调递增且饱和到 shift_max
    assert np.all(np.diff(shift) > 0)
    assert shift[-1] < p.shift_max_nm
    assert shift[-1] > p.shift_max_nm * 0.99
    # 明确非线性：浓度涨 10 倍，高浓度端峰移涨幅远小于 10 倍
    assert (shift[4] / shift[3]) < 1.15


def test_plasmon_blank_is_not_transparent():
    """等离激元空白是"未结合的金阵列"，透射率显著小于 1。

    这是与显色剂路径的根本差异：显色剂 c=0 时 T=1（透明），
    金阵列 c=0 时仍有消光。把它当成显色剂建模会得到错误的基线。
    """
    from pg_colori_sim import PlasmonResonance, plasmon_channel_transmittance

    p = PlasmonResonance(peak_nm=630.0, extinction_depth=0.45)
    channels, shift = plasmon_channel_transmittance(np.array([0.0]), p)

    assert shift[0] == 0.0
    assert np.all(channels < 0.999), "空白透射率不应接近 1"
    # 峰落在红端，故 R 通道衰减应强于 B
    assert channels[0, 0] < channels[0, 2]


def test_channel_ratio_cancels_common_mode():
    """通道比值必须对消曝光/光源亮度这类共模缩放。"""
    from pg_colori_sim import channel_ratio

    t = np.array([[0.6, 0.8, 0.9], [0.5, 0.85, 0.92]])
    base = channel_ratio(t, "RG")
    scaled = channel_ratio(t * 0.37, "RG")      # 曝光减少到 37%
    assert np.allclose(base, scaled), "比值未能对消共模缩放"


def test_plasmon_readout_favours_shorter_peak_and_deeper_band():
    """灵敏度排序：深度 > 峰位 ≫ 线宽。

    线宽在**固定深度**下几乎不起作用——它之所以在真实器件上重要，
    是因为等振子强度下变窄即变深，即通过深度间接起作用。
    """
    from pg_colori_sim import PlasmonResonance, plasmon_readout_study

    def sens(peak, fwhm, depth):
        p = PlasmonResonance(peak_nm=peak, fwhm_nm=fwhm, extinction_depth=depth)
        return plasmon_readout_study(p, np.array([0.0]))["sensitivity_per_nm"]

    # 峰位 630 → 560 显著提升
    assert sens(560, 110, 0.45) > sens(630, 110, 0.45) * 2.5
    # 深度提升同样显著
    assert sens(560, 110, 0.65) > sens(560, 110, 0.25) * 2.5
    # 固定深度下线宽影响很小
    narrow, wide = sens(560, 80, 0.45), sens(560, 110, 0.45)
    assert abs(narrow / wide - 1.0) < 0.15


def test_reference_column_is_zero_concentration():
    """参考列必须整列为零浓度，且不受移液抖动影响。"""
    from pg_colori_sim import ColorimetricConfig

    config = ColorimetricConfig(grid_size=15, reference_column=3,
                                blank_well_count=0, concentration_jitter=0.2, seed=6)
    values = config.resolved_concentrations().reshape(15, 15)

    assert np.all(values[:, 3] == 0.0), "参考列存在非零浓度"
    other = np.delete(values, 3, axis=1)
    assert np.count_nonzero(other) > other.size * 0.9, "非参考列不应大面积为零"


def test_plasmon_scene_renders_and_carries_shift_truth(tmp_path):
    """等离激元模式必须渲染成图，并在真值里给出峰移与比值读数。"""
    from pg_colori_sim import ColorimetricConfig, PlasmonResonance, render_colorimetric_scene

    config = ColorimetricConfig(
        grid_size=15, image_size=600, seed=9, layout="bare",
        signal_model="plasmon", plasmon=PlasmonResonance(),
        concentration_pattern="log_series", concentration_min_uM=0.001,
        concentration_max_uM=1.0, reference_column=0, blank_well_count=0,
    )
    image, truth = render_colorimetric_scene(config)

    assert image.shape == (600, 600, 3)
    assert truth["signal_model"] == "plasmon"
    assert len(truth["peak_shift_nm"]) == 225
    assert len(truth["channel_ratio"]) == 225
    assert truth["reference_column"] == 0

    shift = np.asarray(truth["peak_shift_nm"]).reshape(15, 15)
    assert np.all(shift[:, 0] == 0.0), "参考列峰移应为零"
    assert shift.max() > 0.0


def test_gold_array_extinction_cancels_in_the_absorbance_ratio():
    """金阵列的消光必须在 A=−log10(I/I_blank) 中精确对消。

    这是"金纳米芯片上能不能做酶显色定量"的关键：金让整体透过率下降，
    但它在样品图和空白图里完全相同，取比值时消掉，因此不引入偏差——
    代价只体现在信噪比（光子变少），不体现在准确度。
    """
    from pg_colori_sim import (ColorimetricConfig, PlasmonResonance,
                               render_colorimetric_scene)

    base = dict(grid_size=15, image_size=400, layout="bare",
                concentration_pattern="log_series", concentration_min_uM=50.0,
                concentration_max_uM=800.0, concentration_jitter=0.0,
                blank_well_count=0, seed=3)

    def absorbance(array_plasmon):
        config = ColorimetricConfig(**base, array_plasmon=array_plasmon)
        _, sample = render_colorimetric_scene(config, force_blank=False)
        _, blank = render_colorimetric_scene(config, force_blank=True)
        t_s = np.asarray(sample["transmittance_rgb"])
        t_b = np.asarray(blank["transmittance_rgb"])
        return -np.log10(np.clip(t_s, 1e-12, None) / np.clip(t_b, 1e-12, None)), t_b

    gold = PlasmonResonance(peak_nm=630.0, fwhm_nm=110.0, extinction_depth=0.45)
    a_plain, blank_plain = absorbance(None)
    a_gold, blank_gold = absorbance(gold)

    # 金确实压低了整体透过率
    assert blank_gold[0, 2] < blank_plain[0, 2] * 0.99
    # 但吸光度逐点完全一致
    assert np.allclose(a_plain, a_gold, atol=1e-9), "金阵列在吸光度比值中未对消"
