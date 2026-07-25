"""PG-Fluoro-Sim：荧光发光阵列的物理相机仿真与真值生成。

用途：在没有足够实拍样本时，用**带真值**的仿真图评估定位与定量算法在
荧光成像条件下的表现——尤其是"多暗的孔还能被正确定位、多亮开始饱和失效"
这类只能靠受控实验回答的问题。

与背光成像的本质区别（决定了它是独立的仿真而不是换个颜色）：
- 背光图是"亮面板 + 暗单元"，荧光图是"暗背景 + 亮发光孔"，极性相反；
- 荧光信号弱，**噪声成为主导因素**：低强度端由读出噪声决定检测下限，
  高强度端由满阱饱和决定上限，中间由散粒噪声决定信噪比；
- 因此仿真必须建在"光子 → 电子 → 灰度"的物理链路上，而不是直接画色块，
  否则得出的检测下限没有参考价值。

渲染链路（顺序即物理顺序，不可随意调换）：
  1. 场景辐射（线性空间）：孔发射 + 基底自发荧光 + 滤光片漏光
  2. 激发光不均匀（乘性场）
  3. 光学串扰（相邻孔散射）
  4. 几何：旋转 → 透视 → 径向畸变
  5. 光学：离焦 PSF → 渐晕 → 色差
  6. 传感器：曝光增益 → 散粒噪声 → 暗电流/读出噪声 → 热像素 → 满阱截断
  7. 编码：gamma → 8bit 量化 → 可选 JPEG 压缩

真值输出包含每个孔的**几何中心（最终图像坐标）**与**设定强度**，
可直接用于定位误差评估和定量线性度评估。

用法：
python pg_fluoro_sim.py --grid 15 --output sim/fluoro15
python pg_fluoro_sim.py --grid 15 --output sim/f15 --evaluate
python pg_fluoro_sim.py --grid 10 --sweep --output sim/sweep
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------- 波长 → 颜色


def wavelength_to_rgb(nm: float) -> tuple[float, float, float]:
    """把可见光波长（nm）转换为归一化 sRGB 三元组。

    采用 Bruton 分段近似（CIE 色匹配函数的常用工程近似）。
    530nm 落在 510-580 段，得到 R≈0.29、G=1.0、B=0——即偏黄的绿色，
    与常见 FITC/GFP 类荧光的观感一致。
    """

    nm = float(nm)
    if 380.0 <= nm < 440.0:
        r, g, b = -(nm - 440.0) / 60.0, 0.0, 1.0
    elif 440.0 <= nm < 490.0:
        r, g, b = 0.0, (nm - 440.0) / 50.0, 1.0
    elif 490.0 <= nm < 510.0:
        r, g, b = 0.0, 1.0, -(nm - 510.0) / 20.0
    elif 510.0 <= nm < 580.0:
        r, g, b = (nm - 510.0) / 70.0, 1.0, 0.0
    elif 580.0 <= nm < 645.0:
        r, g, b = 1.0, -(nm - 645.0) / 65.0, 0.0
    elif 645.0 <= nm <= 780.0:
        r, g, b = 1.0, 0.0, 0.0
    else:
        return 0.0, 0.0, 0.0

    # 视见函数在可见光两端衰减。
    if 380.0 <= nm < 420.0:
        falloff = 0.3 + 0.7 * (nm - 380.0) / 40.0
    elif 700.0 < nm <= 780.0:
        falloff = 0.3 + 0.7 * (780.0 - nm) / 80.0
    else:
        falloff = 1.0
    return r * falloff, g * falloff, b * falloff


# ---------------------------------------------------------------- 配置


@dataclass
class FluorescenceConfig:
    """荧光仿真参数。默认值对应"手机 + 暗箱 + 荧光滤光片"的典型场景。

    强度语义：1.0 = 恰好填满传感器满阱（即刚好饱和）。因此
    intensity_min=0.002 表示信号只有满阱的 0.2%，接近读出噪声水平。
    """

    # --- 阵列几何 ---
    grid_size: int = 15
    image_size: int = 1600
    panel_fill: float = 0.62          # 面板边长 / 图像边长
    well_fill: float = 0.42           # 孔边长 / 孔间距
    well_shape: str = "square"        # square | circle
    supersample: int = 2              # 超采样倍数，保证亚像素级中心精度

    # --- 发射光谱 ---
    emission_nm: float = 530.0        # 荧光发射峰值波长
    spectral_bleed: float = 0.06      # 相邻通道串色（滤光片非理想）

    # --- 逐孔强度 ---
    intensity_pattern: str = "log_series"  # log_series | gradient | uniform | random | checker
    intensity_min: float = 0.004
    intensity_max: float = 1.10       # >1.0 使最亮的孔进入饱和
    dead_well_count: int = 0          # 完全不发光的孔（模拟漏加/失效）
    intensity_jitter: float = 0.05    # 逐孔相对随机波动（移液误差）

    # --- 背景 ---
    substrate_autofluorescence: float = 0.012   # 基底自发荧光
    background_texture: float = 0.004           # 基底纹理起伏
    filter_leak: float = 0.003                  # 滤光片漏光（全场均匀底噪）
    outside_panel_level: float = 0.001          # 面板外（暗箱）残余亮度

    # --- 激发与光学 ---
    excitation_nonuniformity: float = 0.22      # 激发光强度梯度幅度
    excitation_angle_deg: float = 35.0          # 梯度方向
    crosstalk_sigma_ratio: float = 0.05         # 孔间光学串扰（相对孔间距）
    defocus_sigma_px: float = 1.1               # 离焦/衍射 PSF
    vignetting: float = 0.30                    # 边缘渐晕强度
    chromatic_aberration_px: float = 0.6        # 横向色差（通道间缩放差）

    # --- 几何畸变 ---
    rotation_deg: float = 2.0
    perspective_strength: float = 0.018         # 角点位移 / 图像边长
    radial_k1: float = -0.055                   # 负值为桶形畸变

    # --- 传感器 ---
    full_well_e: float = 12000.0                # 满阱电子数
    read_noise_e: float = 6.0                   # 读出噪声（决定弱信号检测下限）
    dark_current_e: float = 4.0                 # 暗电流
    hot_pixel_ratio: float = 3e-6               # 热像素比例
    prnu: float = 0.008                         # 像素响应非均匀性（固定图案噪声）
    gamma: float = 1.0                          # 1.0=线性；手机 JPEG 通常 2.2
    jpeg_quality: int | None = 92               # None 表示输出 PNG 不压缩

    seed: int = 0

    def resolved_intensities(self) -> np.ndarray:
        """按 pattern 生成逐孔强度真值（行优先）。"""

        rng = np.random.default_rng(self.seed + 9973)
        count = self.grid_size * self.grid_size
        low = max(self.intensity_min, 1e-6)
        high = max(self.intensity_max, low * 1.001)

        if self.intensity_pattern == "uniform":
            values = np.full(count, high, dtype=np.float64)
        elif self.intensity_pattern == "gradient":
            values = np.linspace(low, high, count, dtype=np.float64)
        elif self.intensity_pattern == "random":
            values = np.exp(rng.uniform(math.log(low), math.log(high), count))
        elif self.intensity_pattern == "checker":
            values = np.where(np.arange(count) % 2 == 0, high, low).astype(np.float64)
        else:
            # 对数稀释序列：跨越多个数量级，一张图同时覆盖
            # "淹没在噪声里"到"饱和溢出"的全部区间。
            values = np.exp(np.linspace(math.log(low), math.log(high), count))

        if self.intensity_jitter > 0:
            values = values * (1.0 + rng.normal(0.0, self.intensity_jitter, count))
        values = np.clip(values, 0.0, None)

        if self.dead_well_count > 0:
            dead = rng.choice(count, size=min(self.dead_well_count, count), replace=False)
            values[dead] = 0.0
        return values


# ---------------------------------------------------------------- 几何工具


def _radial_distort_points(points: np.ndarray, center: np.ndarray, k1: float, norm_r: float) -> np.ndarray:
    """把理想坐标前向映射到畸变后坐标：r' = r(1 + k1·r²)。"""

    if abs(k1) < 1e-9:
        return points.copy()
    delta = points - center
    r2 = (delta ** 2).sum(axis=1) / (norm_r ** 2)
    return center + delta * (1.0 + k1 * r2)[:, None]


def _radial_distort_image(image: np.ndarray, k1: float, norm_r: float) -> np.ndarray:
    """按同一畸变模型重采样图像。

    remap 需要"目标像素 → 源像素"的逆映射，而模型是前向的，
    因此用不动点迭代求解 src·(1 + k1·|src|²/R²) = dst。
    小畸变下几次迭代即收敛（与 OpenCV undistortPoints 同法）。
    """

    if abs(k1) < 1e-9:
        return image
    height, width = image.shape[:2]
    cx, cy = width / 2.0, height / 2.0
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    sx, sy = dx.copy(), dy.copy()
    for _ in range(8):
        factor = 1.0 + k1 * (sx * sx + sy * sy) / (norm_r ** 2)
        factor = np.where(np.abs(factor) < 1e-6, 1e-6, factor)
        sx, sy = dx / factor, dy / factor
    return cv2.remap(
        image, (sx + cx).astype(np.float32), (sy + cy).astype(np.float32),
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def _geometric_matrix(size: int, rotation_deg: float, perspective_strength: float,
                      rng: np.random.Generator) -> np.ndarray:
    """构造 旋转 + 透视 的 3×3 变换矩阵。"""

    center = (size / 2.0, size / 2.0)
    rotation = cv2.getRotationMatrix2D(center, rotation_deg, 1.0)
    matrix = np.vstack([rotation, [0.0, 0.0, 1.0]]).astype(np.float64)

    if perspective_strength > 0:
        shift = perspective_strength * size
        src = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
        dst = src + rng.uniform(-shift, shift, size=(4, 2)).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst).astype(np.float64) @ matrix
    return matrix


# ---------------------------------------------------------------- 渲染


def _render_emission_map(config: FluorescenceConfig, intensities: np.ndarray,
                         canvas: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """在超采样画布上渲染线性辐射图与孔中心（理想坐标）。"""

    grid = config.grid_size
    scene = np.full((canvas, canvas), config.outside_panel_level, dtype=np.float64)

    panel_side = canvas * config.panel_fill
    panel_x0 = (canvas - panel_side) / 2.0
    panel_y0 = (canvas - panel_side) / 2.0
    px0, py0 = int(round(panel_x0)), int(round(panel_y0))
    px1, py1 = int(round(panel_x0 + panel_side)), int(round(panel_y0 + panel_side))

    # 基底自发荧光 + 纹理起伏（低频，模拟材质不均）。
    substrate = np.full((py1 - py0, px1 - px0), config.substrate_autofluorescence, dtype=np.float64)
    if config.background_texture > 0:
        coarse = rng.normal(0.0, config.background_texture, (16, 16))
        texture = cv2.resize(coarse, (px1 - px0, py1 - py0), interpolation=cv2.INTER_CUBIC)
        substrate = np.clip(substrate + texture, 0.0, None)
    scene[py0:py1, px0:px1] = substrate

    # 孔阵列：pitch 由面板内可用区域与孔数决定。
    margin = panel_side * 0.085
    usable = panel_side - 2.0 * margin
    pitch = usable / max(grid - 1, 1)
    half = max(1.0, pitch * config.well_fill / 2.0)

    centers = np.empty((grid * grid, 2), dtype=np.float64)
    for row in range(grid):
        for col in range(grid):
            index = row * grid + col
            cx = panel_x0 + margin + col * pitch
            cy = panel_y0 + margin + row * pitch
            centers[index] = (cx, cy)
            value = float(intensities[index])
            if value <= 0.0:
                continue
            if config.well_shape == "circle":
                cv2.circle(scene, (int(round(cx)), int(round(cy))), int(round(half)), value, -1, lineType=cv2.LINE_AA)
            else:
                cv2.rectangle(
                    scene,
                    (int(round(cx - half)), int(round(cy - half))),
                    (int(round(cx + half)), int(round(cy + half))),
                    value, -1,
                )

    # 光学串扰：邻孔光在样品/盖片中散射，表现为轻度扩散。
    sigma = config.crosstalk_sigma_ratio * pitch
    if sigma > 0.3:
        ksize = int(sigma * 6) | 1
        scene = cv2.GaussianBlur(scene, (ksize, ksize), sigma)

    # 激发光不均匀：沿指定方向的线性梯度（乘性）。
    if config.excitation_nonuniformity > 0:
        angle = math.radians(config.excitation_angle_deg)
        yy, xx = np.mgrid[0:canvas, 0:canvas].astype(np.float64)
        projection = (xx * math.cos(angle) + yy * math.sin(angle)) / canvas
        projection = (projection - projection.min()) / max(projection.max() - projection.min(), 1e-9)
        field = 1.0 + config.excitation_nonuniformity * (projection - 0.5)
        scene = scene * field

    # 滤光片漏光：全场均匀底噪，抬高暗背景、压低对比度。
    scene = scene + config.filter_leak
    return scene, centers


def render_fluorescence_scene(config: FluorescenceConfig) -> tuple[np.ndarray, dict[str, object]]:
    """渲染一张荧光仿真图，返回 (BGR uint8 图像, 真值字典)。"""

    rng = np.random.default_rng(config.seed)
    size = int(config.image_size)
    scale = max(1, int(config.supersample))
    canvas = size * scale

    intensities = config.resolved_intensities()
    scene, centers = _render_emission_map(config, intensities, canvas, rng)

    # --- 几何：旋转 + 透视 ---
    matrix = _geometric_matrix(canvas, config.rotation_deg, config.perspective_strength, rng)
    scene = cv2.warpPerspective(scene, matrix, (canvas, canvas), flags=cv2.INTER_LINEAR, borderValue=0.0)
    centers = cv2.perspectiveTransform(
        centers.reshape(-1, 1, 2).astype(np.float32), matrix.astype(np.float32)
    ).reshape(-1, 2).astype(np.float64)

    # --- 几何：径向畸变 ---
    norm_r = canvas / 2.0
    center_pt = np.array([canvas / 2.0, canvas / 2.0], dtype=np.float64)
    scene = _radial_distort_image(scene, config.radial_k1, norm_r)
    centers = _radial_distort_points(centers, center_pt, config.radial_k1, norm_r)

    # --- 降采样到目标分辨率（抗锯齿，保证亚像素中心精度）---
    if scale > 1:
        scene = cv2.resize(scene, (size, size), interpolation=cv2.INTER_AREA)
        centers = centers / scale

    # --- 光谱：单色发射映射到 RGB 通道 ---
    r_gain, g_gain, b_gain = wavelength_to_rgb(config.emission_nm)
    bleed = config.spectral_bleed
    gains = np.array([b_gain, g_gain, r_gain], dtype=np.float64)  # OpenCV BGR 顺序
    gains = gains * (1.0 - bleed) + bleed * gains.mean()
    gains = np.clip(gains, 0.0, None)
    stack = scene[:, :, None] * gains[None, None, :]

    # --- 横向色差：通道间轻微缩放差 ---
    if config.chromatic_aberration_px > 0:
        shifts = {0: config.chromatic_aberration_px, 2: -config.chromatic_aberration_px}
        for channel, shift_px in shifts.items():
            factor = 1.0 + shift_px / max(size, 1)
            warp = cv2.getRotationMatrix2D((size / 2.0, size / 2.0), 0.0, factor)
            stack[:, :, channel] = cv2.warpAffine(stack[:, :, channel], warp, (size, size), flags=cv2.INTER_LINEAR)

    # --- 离焦 PSF ---
    if config.defocus_sigma_px > 0.05:
        ksize = int(config.defocus_sigma_px * 6) | 1
        stack = cv2.GaussianBlur(stack, (ksize, ksize), config.defocus_sigma_px)

    # --- 渐晕：镜头边缘光强衰减 ---
    if config.vignetting > 0:
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
        radius = np.hypot(xx - size / 2.0, yy - size / 2.0) / (size / 2.0 * math.sqrt(2.0))
        stack = stack * (1.0 - config.vignetting * radius[:, :, None] ** 2)

    # --- 传感器：辐射 → 电子 ---
    electrons = np.clip(stack, 0.0, None) * config.full_well_e
    if config.prnu > 0:
        # 像素响应非均匀性是固定图案噪声：同一传感器每帧一致。
        prnu_rng = np.random.default_rng(config.seed + 4242)
        electrons = electrons * (1.0 + prnu_rng.normal(0.0, config.prnu, electrons.shape))
    electrons = np.clip(electrons, 0.0, None)

    # 散粒噪声（信号相关，泊松）。大信号用高斯近似避免溢出。
    small = electrons < 1e6
    noisy = np.empty_like(electrons)
    noisy[small] = rng.poisson(electrons[small])
    noisy[~small] = electrons[~small] + rng.normal(0.0, np.sqrt(electrons[~small]))

    # 暗电流（泊松）+ 读出噪声（高斯）：决定弱信号检测下限。
    noisy = noisy + rng.poisson(config.dark_current_e, noisy.shape)
    noisy = noisy + rng.normal(0.0, config.read_noise_e, noisy.shape)

    # 热像素：少量像素始终接近饱和。
    if config.hot_pixel_ratio > 0:
        hot_count = int(size * size * config.hot_pixel_ratio)
        if hot_count > 0:
            hot_rng = np.random.default_rng(config.seed + 777)
            ys = hot_rng.integers(0, size, hot_count)
            xs = hot_rng.integers(0, size, hot_count)
            noisy[ys, xs, :] = config.full_well_e

    # 满阱截断 → 归一化 → gamma → 8bit 量化
    normalized = np.clip(noisy / config.full_well_e, 0.0, 1.0)
    if abs(config.gamma - 1.0) > 1e-6:
        normalized = normalized ** (1.0 / config.gamma)
    image = np.clip(normalized * 255.0 + 0.5, 0, 255).astype(np.uint8)

    truth: dict[str, object] = {
        "schema": "pg-fluoro-truth-v1",
        "grid_size": int(config.grid_size),
        "emission_nm": float(config.emission_nm),
        "image_size": size,
        "points": [[round(float(x), 4), round(float(y), 4)] for x, y in centers],
        "intensities": [round(float(v), 8) for v in intensities],
        "config": asdict(config),
    }
    return image, truth


# ---------------------------------------------------------------- 落盘与评估


def write_fluorescence_sample(config: FluorescenceConfig, output_dir: str | Path,
                              name: str = "fluoro") -> dict[str, str]:
    """渲染并写出图像与真值 JSON，返回路径字典。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image, truth = render_fluorescence_scene(config)

    suffix = ".jpg" if config.jpeg_quality is not None else ".png"
    image_path = output_dir / f"{name}{suffix}"
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(config.jpeg_quality)] if config.jpeg_quality is not None else []
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise RuntimeError(f"图像编码失败：{image_path}")
    encoded.tofile(str(image_path))

    truth_path = output_dir / f"{name}_truth.json"
    with truth_path.open("w", encoding="utf-8") as f:
        json.dump(truth, f, ensure_ascii=False, indent=2)
    return {"image": str(image_path), "ground_truth": str(truth_path)}


def evaluate_generated_sample(image_path: str | Path, truth_path: str | Path,
                              output_dir: str | Path) -> dict[str, object]:
    """在仿真图上跑完整管线，按强度分档报告定位与支撑表现。

    这是"多暗的孔还能被正确定位"的直接答案：按强度十倍档分组，
    给出该档的定位误差与候选支撑率。
    """

    from pg_grid import process_image

    with Path(truth_path).open("r", encoding="utf-8") as f:
        truth = json.load(f)
    grid_size = int(truth["grid_size"])
    intensities = np.asarray(truth["intensities"], dtype=np.float64)
    true_points = np.asarray(truth["points"], dtype=np.float32)

    result = process_image(image_path=image_path, grid_size=grid_size, output_dir=output_dir)

    # 真值映射到矫正坐标系（用管线自己的四角单应，保证坐标系一致）。
    region = np.asarray(result["chip_region"]["points"], dtype=np.float32)
    size = int(result["rectified_size"])
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(region, dst)
    mapped = cv2.perspectiveTransform(true_points.reshape(-1, 1, 2), matrix).reshape(-1, 2).astype(np.float64)

    points = result["grid_points"]
    predicted = np.asarray([[p["x"], p["y"]] for p in points], dtype=np.float64)
    errors = np.linalg.norm(predicted - mapped, axis=1)
    supported = np.asarray([p["source"] == "candidate_refined" for p in points], dtype=bool)
    pitch = max(float(result["lattice_consistency"]["pitch_px"]), 1e-6)

    # 定量读数（若定量结果可用）：用于评估强度线性度。
    measured = None
    quant_path = Path(output_dir) / "quant_result.json"
    if quant_path.exists():
        with quant_path.open("r", encoding="utf-8") as f:
            quant = json.load(f)
        measured = np.asarray([float(u["corr_signal_gray"]) for u in quant["units"]], dtype=np.float64)

    # 按十倍强度档分组统计。
    decades: list[dict[str, object]] = []
    positive = intensities[intensities > 0]
    if positive.size:
        low_exp = math.floor(math.log10(positive.min()))
        high_exp = math.ceil(math.log10(positive.max()))
        for exponent in range(int(low_exp), int(high_exp)):
            lo, hi = 10.0 ** exponent, 10.0 ** (exponent + 1)
            mask = (intensities >= lo) & (intensities < hi)
            if not mask.any():
                continue
            row: dict[str, object] = {
                "intensity_low": lo, "intensity_high": hi,
                "count": int(mask.sum()),
                "mean_error_px": round(float(errors[mask].mean()), 4),
                "mean_error_pct_pitch": round(float(errors[mask].mean() / pitch * 100.0), 4),
                "supported_ratio": round(float(supported[mask].mean()), 4),
            }
            if measured is not None and int(mask.sum()) >= 3:
                sub_true, sub_meas = intensities[mask], measured[mask]
                rank_t = np.argsort(np.argsort(sub_true)).astype(np.float64)
                rank_m = np.argsort(np.argsort(sub_meas)).astype(np.float64)
                if rank_t.std() > 1e-9 and rank_m.std() > 1e-9:
                    row["spearman"] = round(float(np.corrcoef(rank_t, rank_m)[0, 1]), 4)
            decades.append(row)

    dead = intensities <= 0
    report: dict[str, object] = {
        "schema": "pg-fluoro-eval-v1",
        "image_path": str(image_path),
        "grid_size": grid_size,
        "point_count": int(len(points)),
        "unit_polarity": result["unit_polarity"],
        "chip_method": result["chip_region"]["method"],
        "trusted": bool(result["lattice_consistency"]["trusted"]),
        "candidate_support_ratio": result["lattice_consistency"]["candidate_support_ratio"],
        "pitch_px": round(pitch, 4),
        "mean_error_px": round(float(errors.mean()), 4),
        "mean_error_pct_pitch": round(float(errors.mean() / pitch * 100.0), 4),
        "p90_error_pct_pitch": round(float(np.percentile(errors, 90) / pitch * 100.0), 4),
        "max_error_pct_pitch": round(float(errors.max() / pitch * 100.0), 4),
        "supported_ratio": round(float(supported.mean()), 4),
        "dead_well_count": int(dead.sum()),
        "quality_status": result["quality"]["status"],
        "decades": decades,
    }
    with (Path(output_dir) / "fluoro_eval.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


# ---------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="PG-Fluoro-Sim 荧光阵列仿真与真值生成")
    parser.add_argument("--grid", type=int, default=15, choices=[10, 15], help="阵列规格")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--name", default=None, help="输出文件名前缀")
    parser.add_argument("--emission", type=float, default=530.0, help="发射波长 nm（默认 530 绿色）")
    parser.add_argument("--size", type=int, default=1600, help="输出图像边长")
    parser.add_argument("--pattern", default="log_series",
                        choices=["log_series", "gradient", "uniform", "random", "checker"],
                        help="逐孔强度分布")
    parser.add_argument("--imin", type=float, default=0.004, help="最低强度（1.0=满阱饱和）")
    parser.add_argument("--imax", type=float, default=1.10, help="最高强度（>1.0 进入饱和）")
    parser.add_argument("--dead", type=int, default=0, help="完全不发光的孔数")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--png", action="store_true", help="输出无损 PNG 而非 JPEG")
    parser.add_argument("--evaluate", action="store_true", help="生成后立即跑管线并报告分档表现")
    parser.add_argument("--sweep", action="store_true", help="生成一组难度递增的样本并汇总评估")
    parser.add_argument("--intensity-scan", action="store_true",
                        help="扫描整体亮度水平，找出算法可用的最低发光强度")
    return parser.parse_args()


def _print_report(tag: str, report: dict[str, object]) -> None:
    """打印单张样本的评估摘要。"""

    print(f"[{tag}] 极性={report['unit_polarity']} 可信={report['trusted']} "
          f"支撑率={report['candidate_support_ratio']} 质量={report['quality_status']}")
    print(f"       定位误差 平均={report['mean_error_pct_pitch']:.2f}%pitch "
          f"P90={report['p90_error_pct_pitch']:.2f}% 最大={report['max_error_pct_pitch']:.2f}% "
          f"候选支撑点占比={report['supported_ratio']:.2f}")
    print(f"       {'强度档':>16s}{'孔数':>6s}{'定位误差%pitch':>16s}{'候选支撑':>10s}{'保序性':>9s}")
    for row in report["decades"]:
        spearman = f"{row['spearman']:.3f}" if "spearman" in row else "   -  "
        print(f"       {row['intensity_low']:>7.4f}-{row['intensity_high']:<8.4f}{row['count']:>6d}"
              f"{row['mean_error_pct_pitch']:>16.2f}{row['supported_ratio']:>10.2f}{spearman:>9s}")


def main() -> None:
    """命令行入口。"""

    args = parse_args()
    output_dir = Path(args.output)

    def build(**overrides) -> FluorescenceConfig:
        base = dict(
            grid_size=args.grid, image_size=args.size, emission_nm=args.emission,
            intensity_pattern=args.pattern, intensity_min=args.imin, intensity_max=args.imax,
            dead_well_count=args.dead, seed=args.seed,
            jpeg_quality=None if args.png else 92,
        )
        base.update(overrides)
        return FluorescenceConfig(**base)

    if args.intensity_scan:
        # 逐档降低整体发光强度，找出定位开始失效的位置。
        # 每档内部保留 4 倍动态范围（接近真实稀释序列），
        # 这样报告的是"整版信号水平"而不是"单孔极值"。
        print(f"PG-Fluoro-Sim 强度扫描（{args.grid}x{args.grid}，发射 {args.emission}nm）")
        print("目的：找出算法可用的最低发光强度（1.0 = 传感器满阱）\n")
        print(f"{'峰值强度':>10s}{'8bit灰度':>10s}{'可信':>6s}{'支撑率':>8s}{'候选支撑':>10s}"
              f"{'定位%pitch':>12s}{'保序性':>8s}{'可靠率':>8s}")
        levels = [1.0, 0.5, 0.25, 0.12, 0.06, 0.03, 0.015, 0.008, 0.004]
        for peak in levels:
            config = build(intensity_min=peak / 4.0, intensity_max=peak, intensity_pattern="log_series")
            paths = write_fluorescence_sample(config, output_dir, name=f"scan_{peak:.4f}")
            report = evaluate_generated_sample(paths["image"], paths["ground_truth"],
                                               output_dir / f"eval_{peak:.4f}")
            # 峰值强度对应的近似 8bit 灰度（绿通道主导 + 亮度加权）。
            approx_gray = peak * 255.0 * 0.65
            spearman = [r["spearman"] for r in report["decades"] if "spearman" in r]
            spearman_text = f"{max(spearman):.3f}" if spearman else "   -  "
            quant_path = Path(output_dir) / f"eval_{peak:.4f}" / "quant_result.json"
            reliable = "-"
            if quant_path.exists():
                with quant_path.open("r", encoding="utf-8") as f:
                    reliable = f"{json.load(f)['summary']['reliable_ratio']:.2f}"
            print(f"{peak:>10.4f}{approx_gray:>10.1f}{str(report['trusted']):>6s}"
                  f"{report['candidate_support_ratio']:>8.2f}{report['supported_ratio']:>10.2f}"
                  f"{report['mean_error_pct_pitch']:>12.2f}{spearman_text:>8s}{reliable:>8s}")
        print(f"\n输出目录: {output_dir.resolve()}")
        return

    if args.sweep:
        # 难度递增：理想 → 典型 → 困难 → 极端，用于找算法失效边界。
        presets = {
            "ideal": dict(defocus_sigma_px=0.6, read_noise_e=3.0, vignetting=0.1,
                          excitation_nonuniformity=0.05, rotation_deg=0.5,
                          perspective_strength=0.005, radial_k1=-0.01),
            "typical": dict(),
            "hard": dict(defocus_sigma_px=2.2, read_noise_e=12.0, vignetting=0.45,
                         excitation_nonuniformity=0.40, rotation_deg=5.0,
                         perspective_strength=0.035, radial_k1=-0.09, dead_well_count=5),
            "extreme": dict(defocus_sigma_px=3.5, read_noise_e=25.0, vignetting=0.60,
                            excitation_nonuniformity=0.55, rotation_deg=8.0,
                            perspective_strength=0.055, radial_k1=-0.13,
                            dead_well_count=10, jpeg_quality=70),
        }
        print(f"PG-Fluoro-Sim 难度扫描（{args.grid}x{args.grid}，发射 {args.emission}nm）\n")
        for tag, overrides in presets.items():
            config = build(**overrides)
            paths = write_fluorescence_sample(config, output_dir, name=f"{args.grid}x{args.grid}_{tag}")
            report = evaluate_generated_sample(paths["image"], paths["ground_truth"], output_dir / f"eval_{tag}")
            _print_report(tag, report)
            print()
        print(f"输出目录: {output_dir.resolve()}")
        return

    name = args.name or f"fluoro_{args.grid}x{args.grid}"
    config = build()
    paths = write_fluorescence_sample(config, output_dir, name=name)
    print("PG-Fluoro-Sim 生成完成")
    print(f"发射波长: {args.emission}nm  RGB增益: {tuple(round(v, 3) for v in wavelength_to_rgb(args.emission))}")
    print(f"图像: {paths['image']}")
    print(f"真值: {paths['ground_truth']}")

    if args.evaluate:
        report = evaluate_generated_sample(paths["image"], paths["ground_truth"], output_dir / "eval")
        print()
        _print_report(name, report)


if __name__ == "__main__":
    main()
