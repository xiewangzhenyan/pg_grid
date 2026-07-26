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
