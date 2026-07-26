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

浓度模式（--concentration）另外把**绝对浓度**写进真值，用于生成荧光素
浓度标定数据集。因果方向是"浓度 → 光物理模型 → 强度 → 成像链路"，
不是反过来给强度再贴标签；模型含内滤效应与自猝灭，因此强度对浓度
**非单调**——按默认的 0.5 mm 液层拐点在约 295 µM，1 nM–100 µM 全程仍
严格单调可反演，但液层加厚到 3 mm 时拐点前移到 51 µM，量程上半段就
落进不可反演区。这也是真值记绝对浓度而不是"占最大浓度百分比"的原因：
拐点绑定在绝对浓度上，相对刻度既无法表达它，也无法让不同量程的图
共用一条标定曲线。详见 Photophysics。

用法：
python pg_fluoro_sim.py --grid 15 --output sim/fluoro15
python pg_fluoro_sim.py --grid 15 --output sim/f15 --evaluate
python pg_fluoro_sim.py --grid 10 --sweep --output sim/sweep
python pg_fluoro_sim.py --grid 15 --concentration --evaluate --output sim/conc
python pg_fluoro_sim.py --grid 15 --concentration --dataset 8 --output sim/dataset
python pg_fluoro_sim.py --grid 15 --concentration --dataset 8 --linear --output sim/linear
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, field, replace
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


# ---------------------------------------------------------------- 浓度 → 发射


@dataclass
class Photophysics:
    """浓度 → 发射强度的显式光物理模型。默认值对应荧光素钠（pH≈9）。

    这一层单独建模、且真值记录**绝对浓度**而非"占最大浓度的百分比"，
    原因是荧光强度对浓度并非线性，而非线性的拐点位置由绝对浓度决定：

    - 稀溶液（吸光度 A ≲ 0.05）：F ∝ c，标定曲线是过原点的直线；
    - 中等浓度：一次内滤效应——激发光在液层前段就被吸收殆尽，
      F ∝ (1 − 10^(−A))，曲线开始压缩；
    - 高浓度：二次内滤（发射光被自身重吸收；荧光素 Stokes 位移小、
      自吸收显著）叠加自猝灭，**F 随浓度回落**。

    最后一条是关键：越过峰值后同一灰度对应两个浓度。若真值只记百分比，
    这个拐点会随用户事后选的最大浓度漂移，标注就和图像里已经烘焙进去的
    物理对不上了；不同 C_max 的图也失去公共横轴，无法汇总成一条标定曲线。

    拐点位置由 A = ε·c·l 决定，因此**孔内液层厚度直接决定可反演量程**：

        光程     拐点      1nM–100µM 全程单调
        3.0 mm   51.5 µM   否（20 µM 以上灵敏度已劣化）
        1.0 mm  151.5 µM   否
        0.5 mm  295.3 µM   是 ← 默认值
        0.2 mm  690.1 µM   是

    默认取 0.5 mm。薄液层把拐点推远、换来全量程单调，代价是吸收的激发光
    更少、整体信号更弱，检出下限随之抬高——两者是同一个 A 的两面，调
    path_length_cm 时要一起看。自猝灭在 100 µM 处仅贡献约 1% 衰减，拐点
    几乎全部来自内滤效应（单独开自猝灭时曲线单调无拐点）。
    """

    fluorophore: str = "fluorescein"
    epsilon_M_cm: float = 80000.0       # 激发波长处摩尔消光系数 M⁻¹cm⁻¹
    quantum_yield: float = 0.92         # 量子产率（pH 9）
    path_length_cm: float = 0.05        # 孔内液层光程（0.5 mm）
    emission_overlap: float = 0.06      # ε(发射)/ε(激发)，决定二次内滤强度
    self_quench_kd_uM: float = 10000.0  # 自猝灭半猝灭浓度（Stern-Volmer 形式）
    inner_filter: bool = True
    self_quenching: bool = True


def concentration_to_emission(concentrations_uM, physics: Photophysics) -> np.ndarray:
    """浓度（µM）→ 相对发射强度（任意单位，尚未乘曝光增益）。

        F = Φ_eff · (1 − 10^(−A_ex)) · 10^(−A_em)
        A_ex  = ε·c·l            一次内滤：激发光被吸收的比例
        A_em  = overlap · A_ex   二次内滤：出射路径上的自吸收
        Φ_eff = Φ / (1 + c/Kd)   自猝灭

    关掉内滤时退化为稀释极限的线性展开 ln10·A_ex，而不是另起一套比例
    系数——这样线性数据集与非线性数据集在低浓度端量纲和数值都连续，
    两者可以直接对比"线性假设下训练的模型在真实非线性数据上掉多少"。
    """

    concentration = np.asarray(concentrations_uM, dtype=np.float64)
    absorbance = physics.epsilon_M_cm * (concentration * 1e-6) * physics.path_length_cm

    if physics.inner_filter:
        absorbed = 1.0 - np.power(10.0, -absorbance)
        reabsorbed = np.power(10.0, -physics.emission_overlap * absorbance)
    else:
        absorbed = math.log(10.0) * absorbance
        reabsorbed = np.ones_like(absorbance)

    quantum_yield = np.full_like(absorbance, physics.quantum_yield)
    if physics.self_quenching and physics.self_quench_kd_uM > 0:
        quantum_yield = quantum_yield / (1.0 + concentration / physics.self_quench_kd_uM)

    return quantum_yield * absorbed * reabsorbed


@dataclass
class FluorescenceConfig:
    """荧光仿真参数。默认值对应"手机 + 暗箱 + 荧光滤光片"的典型场景。

    强度语义：1.0 = 恰好填满传感器满阱（即刚好饱和）。因此
    intensity_min=0.002 表示信号只有满阱的 0.2%，接近读出噪声水平。

    两种驱动方式，由 concentration_pattern 是否为 None 区分：
    - None（默认）：直接指定逐孔强度，用于定位/成像链路的受控实验；
    - 非 None：由逐孔**浓度**经 Photophysics 模型导出强度，用于生成
      带浓度真值的标定数据集。此时 intensity_* 仅参与曝光自动定标。
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

    # --- 逐孔浓度（设定 concentration_pattern 即切换到浓度驱动）---
    concentration_pattern: str | None = None   # log_series | plate_series | gradient | uniform | random | checker
    concentration_min_uM: float = 0.001        # 1 nM
    concentration_max_uM: float = 100.0        # 100 µM
    concentration_jitter: float = 0.05         # 逐孔移液误差（相对）
    exposure_gain: float | None = None         # None = 按本图峰值自动定标
    photophysics: Photophysics = field(default_factory=Photophysics)

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

    def resolved_concentrations(self) -> np.ndarray | None:
        """按 pattern 生成逐孔浓度真值（µM，行优先）。未启用浓度模式返回 None。"""

        if self.concentration_pattern is None:
            return None

        rng = np.random.default_rng(self.seed + 5501)
        count = self.grid_size * self.grid_size
        low = max(self.concentration_min_uM, 1e-12)
        high = max(self.concentration_max_uM, low * 1.001)

        if self.concentration_pattern == "uniform":
            values = np.full(count, high, dtype=np.float64)
        elif self.concentration_pattern == "gradient":
            values = np.linspace(low, high, count, dtype=np.float64)
        elif self.concentration_pattern == "random":
            values = np.exp(rng.uniform(math.log(low), math.log(high), count))
        elif self.concentration_pattern == "checker":
            values = np.where(np.arange(count) % 2 == 0, high, low).astype(np.float64)
        elif self.concentration_pattern == "plate_series":
            # 逐行等比稀释、行内同浓度：最接近实际移液排布，也让"同浓度
            # 重复孔"的读数离散度可以直接量化（标定曲线的误差棒来源）。
            #
            # 行序随机打乱，有两个独立的理由：
            # 其一是实验设计——浓度若沿版面单调排列，就与激发光梯度、
            # 渐晕这些同样沿版面单调的效应完全混杂，标定曲线分不清读数
            # 变化有多少来自浓度、多少来自位置。
            # 其二是实测的定位失效——单调排列会让最暗的几行连成一条贴着
            # 版面边缘的暗带，格点拟合看不到边界行，在错误的一侧补行，
            # 整版错位一个间距（实测 15×15 平均误差 ~110% pitch）。
            per_row = np.exp(np.linspace(math.log(high), math.log(low), self.grid_size))
            per_row = per_row[rng.permutation(self.grid_size)]
            values = np.repeat(per_row, self.grid_size)
        else:
            # 对数稀释序列：一张图跨越全部数量级，同时覆盖检测下限与内滤压缩区。
            values = np.exp(np.linspace(math.log(low), math.log(high), count))

        if self.concentration_jitter > 0:
            values = values * (1.0 + rng.normal(0.0, self.concentration_jitter, count))
        values = np.clip(values, 0.0, None)

        if self.dead_well_count > 0:
            dead = rng.choice(count, size=min(self.dead_well_count, count), replace=False)
            values[dead] = 0.0
        return values

    def resolved_exposure_gain(self, emission: np.ndarray | None = None) -> float:
        """相对发射强度 → 满阱分数的曝光增益。

        显式给定时原样返回。**跨图共用同一增益是把多张图汇总成一条标定
        曲线的前提**——各图按自己的峰值归一化会让公共纵轴消失，正是绝对
        浓度想避免的那个问题。未给定时按本图峰值定标到 intensity_max，
        只适合单图快速查看。
        """

        if self.exposure_gain is not None:
            return float(self.exposure_gain)
        if emission is None:
            concentrations = self.resolved_concentrations()
            if concentrations is None:
                return 1.0
            emission = concentration_to_emission(concentrations, self.photophysics)
        peak = float(np.max(emission)) if np.size(emission) else 0.0
        return float(self.intensity_max / peak) if peak > 1e-12 else 1.0

    def resolved_intensities(self) -> np.ndarray:
        """逐孔强度真值（满阱分数，行优先）。

        浓度模式下强度是**导出量**：浓度 → 光物理模型 → 相对发射强度 →
        乘曝光增益。因果方向不能反过来，否则图像里的非线性和真值标注就
        各说各话了。
        """

        concentrations = self.resolved_concentrations()
        if concentrations is not None:
            emission = concentration_to_emission(concentrations, self.photophysics)
            return emission * self.resolved_exposure_gain(emission)

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

    concentrations = config.resolved_concentrations()
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

    # 浓度块：未启用浓度模式时各字段为 None，而不是省略键——下游消费者
    # 可以无条件取键并据此判断这张图有没有浓度标注。
    concentration_block: dict[str, object] = {
        "concentrations": None,
        "concentration_unit": None,
        "concentration_fraction": None,
        "photophysics": None,
        "exposure_gain": None,
    }
    if concentrations is not None:
        peak = float(np.max(concentrations)) if concentrations.size else 0.0
        concentration_block = {
            "concentrations": [round(float(v), 9) for v in concentrations],
            "concentration_unit": "uM",
            # 派生的相对刻度，仅作方便标签。反演必须用绝对浓度：
            # 强度-浓度关系非单调，相对刻度不能跨图移植。
            "concentration_fraction": [
                round(float(v) / peak, 8) if peak > 0 else 0.0 for v in concentrations
            ],
            "photophysics": asdict(config.photophysics),
            "exposure_gain": round(float(config.resolved_exposure_gain()), 8),
        }

    truth: dict[str, object] = {
        "schema": "pg-fluoro-truth-v2",
        "grid_size": int(config.grid_size),
        "emission_nm": float(config.emission_nm),
        "image_size": size,
        "points": [[round(float(x), 4), round(float(y), 4)] for x, y in centers],
        "intensities": [round(float(v), 8) for v in intensities],
        **concentration_block,
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

    # 浓度标定：图上有绝对浓度且定量可用时，顺带给出这张图的 LOD 与内滤拐点。
    calibration = None
    concentrations = truth.get("concentrations")
    if concentrations is not None and measured is not None and len(concentrations) == measured.size:
        fractions = truth.get("concentration_fraction") or [0.0] * len(concentrations)
        calibration = summarize_calibration_pairs([
            {"concentration_uM": concentrations[i], "concentration_fraction": fractions[i],
             "measured_gray": float(measured[i])}
            for i in range(len(concentrations))
        ])

    dead = intensities <= 0
    report: dict[str, object] = {
        "schema": "pg-fluoro-eval-v2",
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
        "concentration_unit": truth.get("concentration_unit"),
        "calibration": calibration,
    }
    with (Path(output_dir) / "fluoro_eval.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


# ------------------------------------------------------------ 浓度标定数据集


# 难度预设：--sweep 与 --dataset 共用，保证两条路径描述的是同一组条件。
DIFFICULTY_PRESETS: dict[str, dict[str, object]] = {
    "ideal": dict(defocus_sigma_px=0.6, read_noise_e=3.0, vignetting=0.1,
                  excitation_nonuniformity=0.05, rotation_deg=0.5,
                  perspective_strength=0.005, radial_k1=-0.01),
    "typical": dict(),
    "hard": dict(defocus_sigma_px=2.2, read_noise_e=12.0, vignetting=0.45,
                 excitation_nonuniformity=0.40, rotation_deg=5.0,
                 perspective_strength=0.035, radial_k1=-0.09),
    "extreme": dict(defocus_sigma_px=3.5, read_noise_e=25.0, vignetting=0.60,
                    excitation_nonuniformity=0.55, rotation_deg=8.0,
                    perspective_strength=0.055, radial_k1=-0.13, jpeg_quality=70),
}

_PAIR_FIELDS = [
    "sample", "difficulty", "seed", "well_index", "row", "col",
    "concentration_uM", "concentration_fraction", "intensity_true",
    "measured_gray", "snr", "saturation_ratio", "reliable",
]


def detectable_concentration_window(
    config: FluorescenceConfig,
    exposure_gain: float | None = None,
    floor_sigma: float = 3.0,
    background_contrast: float = 0.5,
    ceiling_intensity: float = 0.98,
) -> dict[str, float]:
    """给定曝光下真正读得出的浓度区间。

    存在的理由是一个实测结论：把 1 nM–100 µM 这 5 个数量级压进**一张图**
    时，任何能让 100 µM 不饱和的曝光都会让底部约 3 个数量级落到读出噪声
    以下——1 nM 荧光素在满阱 12000 e⁻ 的传感器上只有约 1 个电子。这不是
    仿真的缺陷，是真实的物理上限（实拍同样看不到）。

    后果不止"那些孔没信号"：整版三分之一的孔不可见会让区域检测与格点
    拟合直接失败，图连定位都过不了，浓度-灰度配对无从谈起。因此数据集
    按本函数把每张图的浓度范围裁到可读窗口内，并把裁掉的部分如实记进
    清单，而不是生成一批注定失败的图。要覆盖窗口以外的浓度，得换曝光
    再拍一组——这也正是真实实验的做法。
    """

    physics = config.photophysics
    low = max(config.concentration_min_uM, 1e-12)
    high = max(config.concentration_max_uM, low * 1.001)

    grid = np.logspace(math.log10(low), math.log10(high), 4000)
    emission = concentration_to_emission(grid, physics)
    peak = float(emission.max()) if emission.size else 0.0
    if exposure_gain is None:
        # 让量程内的峰值恰好落在饱和之下：拐点必须看得见，否则数据集
        # 无法体现"同一灰度对应两个浓度"这一最关键的非线性特征。
        exposure_gain = float(ceiling_intensity / peak) if peak > 1e-12 else 1.0

    intensity = emission * exposure_gain
    # 下限由**基底本底**决定，不只是读出噪声。基底自发荧光加滤光片漏光
    # 在默认参数下约合 0.015 满阱，比读出噪声（6/12000 ≈ 5e-4）高一个数量
    # 级；孔要被检出必须在这个本底之上有可分辨的增量。只按读出噪声定下限
    # 会把大量"淹没在基底里"的孔算作可用——实测后果是整行不可见、格点
    # 拟合缺行，整版错位一个间距。
    background = config.substrate_autofluorescence + config.filter_leak
    floor = max(floor_sigma * config.read_noise_e / max(config.full_well_e, 1e-9),
                background_contrast * background)
    usable = intensity >= floor
    if not usable.any():
        return {"low_uM": low, "high_uM": high, "exposure_gain": float(exposure_gain),
                "floor_intensity": float(floor), "clamped": False}

    first = int(np.argmax(usable))
    return {
        "low_uM": float(grid[first]),
        "high_uM": float(high),
        "exposure_gain": float(exposure_gain),
        "floor_intensity": float(floor),
        "clamped": bool(grid[first] > low * 1.001),
    }


def build_concentration_dataset(
    output_dir: str | Path,
    count: int = 12,
    grid_size: int = 15,
    concentration_pattern: str = "plate_series",
    difficulties: tuple[str, ...] = ("ideal", "typical", "hard"),
    blank_wells: int = 8,
    exposure_gain: float | None = None,
    base_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    """生成一批带绝对浓度真值的样本，汇总成可直接回归的标定数据集。

    三个设计点决定了产出的数据能不能真的用来反演浓度：

    1. **整批共用一个曝光增益**。逐图归一化会让每张图有各自的纵轴，汇总
       后的散点就是若干条互不重合的曲线。增益在这里解析一次、写进每张图
       的真值，跨图可比。
    2. **含真正的空白孔**（浓度恰为 0）。没有空白就没有本底，检测下限只能
       靠猜；LOD 的定义本身就依赖空白的均值与离散度。
    3. **按定位精度筛图，而不是照单全收**。定位错位一格时，第 k 个定量读数
       对应的是第 k+1 个孔，浓度-灰度配对整体偏移——这种污染在散点图上看
       不出来，却会把标定曲线拧歪。因此每张图先用真值校核定位误差，超阈值
       的整张丢弃并记录在案。

    分辨率下限：``image_size`` 低于约 1200px 时 10×10 的孔斑过小，候选支撑率
    跌到 0.01–0.27，多数图过不了第 3 点的校核（既有管线行为）。实拍手机照片
    远高于此，不受影响。
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_overrides = dict(base_overrides or {})

    def make_config(seed: int, difficulty: str) -> FluorescenceConfig:
        params: dict[str, object] = dict(
            grid_size=grid_size,
            concentration_pattern=concentration_pattern,
            dead_well_count=blank_wells,
            seed=seed,
        )
        params.update(DIFFICULTY_PRESETS.get(difficulty, {}))
        params.update(base_overrides)
        return FluorescenceConfig(**params)

    # 曝光增益与可读窗口整批解析一次（见文档字符串第 1 点）。
    reference = make_config(0, difficulties[0])
    window = detectable_concentration_window(reference, exposure_gain=exposure_gain)
    exposure_gain = window["exposure_gain"]
    requested = (reference.concentration_min_uM, reference.concentration_max_uM)

    rows: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    for index in range(count):
        difficulty = difficulties[index % len(difficulties)]
        name = f"conc_{index:03d}_{difficulty}"
        config = replace(
            make_config(index, difficulty),
            concentration_min_uM=window["low_uM"], concentration_max_uM=window["high_uM"],
            exposure_gain=exposure_gain,
        )

        paths = write_fluorescence_sample(config, output_dir / "images", name=name)
        eval_dir = output_dir / "eval" / name
        report = evaluate_generated_sample(paths["image"], paths["ground_truth"], eval_dir)

        with Path(paths["ground_truth"]).open("r", encoding="utf-8") as f:
            truth = json.load(f)
        concentrations = truth["concentrations"]
        fractions = truth["concentration_fraction"]
        intensities = truth["intensities"]

        quant_path = eval_dir / "quant_result.json"
        units = []
        if quant_path.exists():
            with quant_path.open("r", encoding="utf-8") as f:
                units = json.load(f)["units"]

        error_pct = float(report["mean_error_pct_pitch"])
        usable = error_pct <= 10.0 and len(units) == len(concentrations)
        samples.append({
            "sample": name, "difficulty": difficulty, "seed": index,
            "image": paths["image"], "ground_truth": paths["ground_truth"],
            "mean_error_pct_pitch": round(error_pct, 4),
            "trusted": bool(report["trusted"]),
            "quant_unit_count": len(units),
            "included": bool(usable),
            "excluded_reason": None if usable else (
                "quant_unit_count_mismatch" if len(units) != len(concentrations)
                else f"localization_error_{error_pct:.1f}pct_pitch"
            ),
        })
        if not usable:
            continue

        for well_index, unit in enumerate(units):
            rows.append({
                "sample": name,
                "difficulty": difficulty,
                "seed": index,
                "well_index": well_index,
                "row": well_index // grid_size,
                "col": well_index % grid_size,
                "concentration_uM": concentrations[well_index],
                "concentration_fraction": fractions[well_index],
                "intensity_true": intensities[well_index],
                "measured_gray": unit["corr_signal_gray"],
                "snr": unit["snr"],
                "saturation_ratio": unit["saturation_ratio"],
                "reliable": int(bool(unit["quant_reliable"])),
            })

    pairs_path = output_dir / "calibration_pairs.csv"
    with pairs_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PAIR_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_calibration_pairs(rows)
    manifest: dict[str, object] = {
        "schema": "pg-fluoro-dataset-v1",
        "grid_size": grid_size,
        "concentration_unit": "uM",
        "concentration_pattern": concentration_pattern,
        "exposure_gain": round(float(exposure_gain), 8),
        "requested_range_uM": [requested[0], requested[1]],
        "effective_range_uM": [round(window["low_uM"], 9), round(window["high_uM"], 9)],
        "range_clamped": window["clamped"],
        "clamp_note": (
            f"低于 {window['low_uM']:.4g} µM 的孔在本曝光下淹没于基底本底与读出噪声"
            f"（强度 < {window['floor_intensity']:.2e} 满阱），已移出量程；"
            "整版三分之一孔不可见会让格点拟合缺行，因此不生成这批注定失败的图。"
            "覆盖更低浓度需要另一组更长曝光、或更低基底自发荧光的图。"
        ) if window["clamped"] else None,
        "photophysics": asdict(reference.photophysics),
        "blank_wells_per_image": blank_wells,
        "image_count": count,
        "included_image_count": sum(1 for s in samples if s["included"]),
        "pair_count": len(rows),
        # 剔除不是随机发生的：定位最容易失败的恰是"最暗的几行落在版面
        # 边缘"的排布，而边缘孔受渐晕压制更重。因此幸存图像在低浓度端
        # 略微偏亮，LOD 会被乐观估计。这个偏倚无法靠丢弃策略消除（用真值
        # 格点去定量就不再是端到端评测了），只能靠加大样本量稀释——
        # 如实记在这里，而不是留给使用者自己发现。
        "selection_bias_note": (
            f"{count - sum(1 for s in samples if s['included'])}/{count} 张因定位未过校核被剔除；"
            "剔除偏向暗行贴边的排布，低浓度端读数与 LOD 可能偏乐观。"
        ) if any(not s["included"] for s in samples) else None,
        "samples": samples,
        "calibration": summary,
        "outputs": {"calibration_pairs_csv": str(pairs_path)},
    }
    manifest_path = output_dir / "dataset_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    manifest["outputs"]["dataset_manifest_json"] = str(manifest_path)
    return manifest


def summarize_calibration_pairs(rows: list[dict[str, object]]) -> dict[str, object]:
    """从浓度-灰度配对中提炼标定曲线的两个可用边界。

    ``detection_limit_uM``：读数首次显著高于空白的浓度档（本底均值 +
    3×本底标准差，即常规 LOD 定义）。低于它的孔不是"测不准"，是根本
    没有信号。

    ``monotonic_max_uM``：分档中位读数**停止上升**的浓度。越过它以后
    灰度随浓度下降，同一读数对应两个浓度——反演在这里必须停手或改用
    双解处理。这正是相对百分比标注无法表达的信息：拐点绑定在绝对浓度
    上，而不是绑定在"占最大浓度的百分之几"上。
    """

    if not rows:
        return {"decades": [], "detection_limit_uM": None, "monotonic_max_uM": None,
                "turnover_detected": False, "blank_mean_gray": None, "spearman_usable": None}

    concentration = np.asarray([float(r["concentration_uM"]) for r in rows])
    measured = np.asarray([float(r["measured_gray"]) for r in rows])

    blank = measured[concentration <= 0]
    blank_mean = float(blank.mean()) if blank.size else 0.0
    blank_std = float(blank.std()) if blank.size > 1 else 0.0
    threshold = blank_mean + 3.0 * blank_std

    positive = concentration > 0
    decades: list[dict[str, object]] = []
    if positive.any():
        low_exp = math.floor(math.log10(concentration[positive].min()))
        high_exp = math.ceil(math.log10(concentration[positive].max()))
        for exponent in range(int(low_exp), int(high_exp)):
            lo, hi = 10.0 ** exponent, 10.0 ** (exponent + 1)
            mask = (concentration >= lo) & (concentration < hi)
            if not mask.any():
                continue
            decades.append({
                "concentration_low_uM": lo,
                "concentration_high_uM": hi,
                "count": int(mask.sum()),
                "median_concentration_uM": round(float(np.median(concentration[mask])), 6),
                "median_measured_gray": round(float(np.median(measured[mask])), 4),
                "measured_gray_cv": round(
                    float(measured[mask].std() / max(abs(measured[mask].mean()), 1e-9)), 4),
                "above_blank": bool(float(np.median(measured[mask])) > threshold),
            })

    detection_limit = next(
        (d["median_concentration_uM"] for d in decades if d["above_blank"]), None)

    # 拐点检测在**排序后的原始配对**上做，不按十倍档比较中位数：默认参数下
    # 拐点在 51 µM 附近，与 100 µM 同处 [10,100) 这一个档内，档间比较看不到它。
    monotonic_max, turnover_detected = None, False
    usable_mask = concentration > 0
    if detection_limit is not None:
        usable_mask = usable_mask & (concentration >= detection_limit)
    if int(usable_mask.sum()) >= 8:
        order = np.argsort(concentration[usable_mask])
        sorted_concentration = concentration[usable_mask][order]
        sorted_measured = measured[usable_mask][order]
        # 滑动中位数抑噪：单点噪声不该被当成拐点。
        window = max(5, (sorted_concentration.size // 20) | 1)
        half = window // 2
        smoothed = np.array([
            np.median(sorted_measured[max(0, i - half): i + half + 1])
            for i in range(sorted_concentration.size)
        ])
        peak = int(np.argmax(smoothed))
        monotonic_max = round(float(sorted_concentration[peak]), 6)
        # 峰值需明显退出尾部才算真拐点，否则只是"到量程边界仍在上升"。
        turnover_detected = peak < sorted_concentration.size - window

    # 可用区间内的保序性：标定曲线能否单调反演的直接指标。
    spearman = None
    if detection_limit is not None and monotonic_max is not None:
        window = (concentration >= detection_limit) & (concentration <= monotonic_max)
        if int(window.sum()) >= 3:
            rank_c = np.argsort(np.argsort(concentration[window])).astype(np.float64)
            rank_m = np.argsort(np.argsort(measured[window])).astype(np.float64)
            if rank_c.std() > 1e-9 and rank_m.std() > 1e-9:
                spearman = round(float(np.corrcoef(rank_c, rank_m)[0, 1]), 4)

    return {
        "decades": decades,
        "blank_mean_gray": round(blank_mean, 4),
        "blank_std_gray": round(blank_std, 4),
        "detection_threshold_gray": round(threshold, 4),
        "detection_limit_uM": detection_limit,
        "monotonic_max_uM": monotonic_max,
        "turnover_detected": turnover_detected,
        "spearman_usable": spearman,
    }


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
    parser.add_argument("--concentration", action="store_true",
                        help="按浓度驱动（强度由光物理模型导出，真值附带绝对浓度）")
    parser.add_argument("--cpattern", default="log_series",
                        choices=["log_series", "plate_series", "gradient", "uniform", "random", "checker"],
                        help="逐孔浓度分布；plate_series 为逐行等比稀释")
    parser.add_argument("--cmin", type=float, default=0.001, help="最低浓度 µM（默认 1nM）")
    parser.add_argument("--cmax", type=float, default=100.0, help="最高浓度 µM（默认 100µM）")
    parser.add_argument("--linear", action="store_true",
                        help="关闭内滤与自猝灭，强制 F ∝ c（用于对照）")
    parser.add_argument("--dataset", type=int, default=0, metavar="N",
                        help="生成 N 张带浓度真值的样本并汇总为标定数据集")
    parser.add_argument("--blanks", type=int, default=8, help="每张图的空白孔数（浓度 0）")
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

    physics = Photophysics(inner_filter=not args.linear, self_quenching=not args.linear)

    def build(**overrides) -> FluorescenceConfig:
        base = dict(
            grid_size=args.grid, image_size=args.size, emission_nm=args.emission,
            intensity_pattern=args.pattern, intensity_min=args.imin, intensity_max=args.imax,
            dead_well_count=args.dead, seed=args.seed,
            jpeg_quality=None if args.png else 92,
        )
        if args.concentration:
            base.update(
                concentration_pattern=args.cpattern,
                concentration_min_uM=args.cmin, concentration_max_uM=args.cmax,
                photophysics=physics,
            )
        base.update(overrides)
        return FluorescenceConfig(**base)

    if args.dataset > 0:
        overrides: dict[str, object] = dict(
            image_size=args.size, emission_nm=args.emission,
            concentration_min_uM=args.cmin, concentration_max_uM=args.cmax,
            photophysics=physics, jpeg_quality=None if args.png else 92,
        )
        print(f"PG-Fluoro-Sim 浓度标定数据集（{args.grid}x{args.grid}，"
              f"{args.cmin}–{args.cmax} µM，{args.dataset} 张）")
        print(f"荧光团: {physics.fluorophore}  内滤: {physics.inner_filter}  "
              f"自猝灭: {physics.self_quenching}\n")
        manifest = build_concentration_dataset(
            output_dir, count=args.dataset, grid_size=args.grid,
            concentration_pattern=args.cpattern, blank_wells=args.blanks,
            base_overrides=overrides,
        )
        calibration = manifest["calibration"]
        print(f"请求量程: {manifest['requested_range_uM'][0]:g}–"
              f"{manifest['requested_range_uM'][1]:g} µM"
              f"   实际量程: {manifest['effective_range_uM'][0]:.4g}–"
              f"{manifest['effective_range_uM'][1]:.4g} µM")
        if manifest["clamp_note"]:
            print(f"  ⚠ {manifest['clamp_note']}")
        print(f"入选图像: {manifest['included_image_count']}/{manifest['image_count']}"
              f"   浓度-灰度配对: {manifest['pair_count']}")
        print(f"曝光增益（整批共用）: {manifest['exposure_gain']:.4f}")
        for sample in manifest["samples"]:
            if not sample["included"]:
                print(f"  [剔除] {sample['sample']}: {sample['excluded_reason']}")
        if manifest["selection_bias_note"]:
            print(f"  ⚠ {manifest['selection_bias_note']}")

        if not manifest["pair_count"]:
            print("\n没有任何图通过定位校核，无法给出标定曲线。")
            print(f"清单: {output_dir.resolve() / 'dataset_manifest.json'}")
            return

        print(f"\n空白本底: {calibration['blank_mean_gray']:.2f} ± "
              f"{calibration['blank_std_gray']:.2f} 灰度"
              f"   检出阈值: {calibration['detection_threshold_gray']:.2f}")
        print(f"\n{'浓度档 µM':>20s}{'孔数':>7s}{'中位读数':>10s}{'CV':>8s}{'高于空白':>10s}")
        for row in calibration["decades"]:
            print(f"{row['concentration_low_uM']:>9.4g}-{row['concentration_high_uM']:<10.4g}"
                  f"{row['count']:>7d}{row['median_measured_gray']:>10.2f}"
                  f"{row['measured_gray_cv']:>8.3f}{str(row['above_blank']):>10s}")
        print(f"\n检出下限 LOD: {calibration['detection_limit_uM']} µM")
        turnover = ("内滤拐点" if calibration["turnover_detected"]
                    else "量程上界，未观察到拐点")
        print(f"单调上限: {calibration['monotonic_max_uM']} µM（{turnover}）"
              f"   可用区间保序性: {calibration['spearman_usable']}")
        print(f"\n配对数据: {manifest['outputs']['calibration_pairs_csv']}")
        print(f"清单: {output_dir.resolve() / 'dataset_manifest.json'}")
        return

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
        # 失效孔数不属于成像条件，因此不在 DIFFICULTY_PRESETS 里，在此叠加。
        dead_wells = {"hard": 5, "extreme": 10}
        print(f"PG-Fluoro-Sim 难度扫描（{args.grid}x{args.grid}，发射 {args.emission}nm）\n")
        for tag, preset in DIFFICULTY_PRESETS.items():
            overrides = dict(preset)
            if tag in dead_wells:
                overrides["dead_well_count"] = dead_wells[tag]
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
