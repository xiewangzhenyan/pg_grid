"""PG-Colori-Sim 比色仿真测试。"""

import json
from pathlib import Path

import numpy as np


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

    config = ColorimetricConfig(grid_size=15, image_size=600, seed=3)
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


def test_pipeline_localizes_colorimetric_image(tmp_path):
    """定位管线必须能处理比色图（黑罩亮孔阵列）。"""
    import cv2
    from pg_colori_sim import ColorimetricConfig, write_colorimetric_sample
    from pg_grid import process_image

    config = ColorimetricConfig(
        grid_size=15, image_size=1100, seed=2,
        rotation_deg=1.5, perspective_strength=0.01, radial_k1=-0.02,
    )
    paths = write_colorimetric_sample(config, tmp_path, name="loc")
    with Path(paths["image_truth"]).open(encoding="utf-8") as f:
        truth = json.load(f)

    result = process_image(image_path=paths["image"], grid_size=15, output_dir=tmp_path / "out")

    assert result["unit_polarity"] == "bright"
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
