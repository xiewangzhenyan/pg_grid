"""PG-Colori-Sim：比色（透射吸光）阵列的物理成像仿真，带绝对浓度真值。

对应硬件（多模态传感器 v4.2 比色模式）：

    D1 22×22 mm 白色 EL 面板 → 0.3 mm PET 漫射膜
    → D2 空 → D3 芯片 + 黑罩(中心距 1.0 mm) → D4 空 → 短塔 → 手机主摄

与 pg_fluoro_sim 的根本差异只有一处：**辐射来源**。荧光是自发光；
比色是背光透射，浓度越高越**暗**——方向与荧光相反。

版面形态见 ColorimetricConfig.layout / mask_transmittance。不装黑罩时是
"亮的散射基底 + 深色反应点"（polarity=dark，当前实拍芯片的样子）；装
理想不透光黑罩时是"黑底 + 亮孔"（polarity=bright，v4.2 文档的形态）。
半透光的 PDMS 罩落在两者之间，且会在图内翻转极性——见 --mask-study。

成像链（几何 → 光学 → 传感器）与荧光完全共用 pg_fluoro_sim 的实现，
不另写一份，避免两种模式的噪声/满阱/量化口径各自漂移。

为什么必须逐波长积分而不能给每个通道一个标量 ε：
Beer-Lambert 与光谱积分**不可交换**。相机通道有几十纳米带宽，通带内
ε(λ) 变化很大，实际读数是

    I_ch = ∫ S(λ)·R_ch(λ)·10^(−ε(λ)·c·l) dλ

高浓度时积分被通带内 ε **最小**的那些波长主导，于是表观吸光度
A = −log10(I/I_blank) 偏离线性并趋于饱和。这就是比色版的"内滤拐点"，
是量程上界的真正来源；用标量 ε 会得到一条永远笔直的曲线，把这个上界
藏起来。
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from pg_fluoro_sim import (
    _geometric_matrix,
    _radial_distort_image,
    _radial_distort_points,
    apply_optics_and_sensor,
)

# 波长采样栅格：400-700 nm，2 nm 步长。步长要明显小于吸收带宽度，
# 否则积分本身就会把多色非线性抹平。
LAMBDA_NM = np.arange(400.0, 701.0, 2.0)


# ---------------------------------------------------------------- 光谱模型


def _gaussian_band(peak_nm: float, fwhm_nm: float) -> np.ndarray:
    """归一化到峰值 1.0 的高斯谱带。"""

    sigma = max(float(fwhm_nm), 1e-6) / 2.3548200450309493
    return np.exp(-0.5 * ((LAMBDA_NM - float(peak_nm)) / sigma) ** 2)


@dataclass
class Chromophore:
    """显色产物的吸收光谱。

    默认值为 **TMB 酸终止后的黄色二亚胺**（ELISA 最常见的终点）：
    λmax 450 nm，ε≈5.9×10⁴ M⁻¹cm⁻¹。黄色意味着吸蓝透红，因此浓度升高时
    B 通道显著变暗而 R 通道几乎不变——这才是"比色"，而不只是"变暗"。
    """

    name: str = "TMB-acid (yellow diimine)"
    peak_nm: float = 450.0
    fwhm_nm: float = 100.0
    epsilon_peak: float = 5.9e4      # M⁻¹cm⁻¹，峰值摩尔吸光系数
    # 次级谱带（很多显色剂有肩峰）；amplitude=0 时不生效。
    second_peak_nm: float = 370.0
    second_fwhm_nm: float = 70.0
    second_amplitude: float = 0.0
    path_length_mm: float = 0.5      # 芯片液层厚度，对应 20×20×0.5 芯片

    def epsilon_spectrum(self) -> np.ndarray:
        """ε(λ)，单位 M⁻¹cm⁻¹。"""

        band = _gaussian_band(self.peak_nm, self.fwhm_nm)
        if self.second_amplitude > 0:
            band = band + self.second_amplitude * _gaussian_band(
                self.second_peak_nm, self.second_fwhm_nm
            )
        return self.epsilon_peak * band

    def absorbance_spectrum(self, concentration_uM: float | np.ndarray) -> np.ndarray:
        """A(λ) = ε(λ)·c·l。浓度 µM，光程 mm → 统一到 M 与 cm。"""

        molar = np.asarray(concentration_uM, dtype=np.float64) * 1e-6
        path_cm = self.path_length_mm * 0.1
        return self.epsilon_spectrum()[None, :] * molar[:, None] * path_cm


# 常用显色产物预设。颜色由**吸收峰**决定，看到什么颜色就是它没吸掉的那部分：
# 450 nm 吸蓝 → 呈黄；652 nm 吸红 → 呈蓝青。
CHROMOPHORES: dict[str, Chromophore] = {
    # TMB 酸终止后的黄色二亚胺，ELISA 最常见终点，读 450 nm。
    "tmb-yellow": Chromophore(
        name="TMB-acid (yellow diimine)", peak_nm=450.0, fwhm_nm=100.0, epsilon_peak=5.9e4
    ),
    # TMB 未终止的蓝色电荷转移复合物，读 652 nm。实拍芯片上呈青/蓝绿的
    # 反应点对应的就是这一种——吸红透蓝绿。
    "tmb-blue": Chromophore(
        name="TMB (blue radical)", peak_nm=652.0, fwhm_nm=110.0, epsilon_peak=3.9e4
    ),
    # ABTS 绿色阳离子自由基，读 420 nm（另有 650/734 肩峰）。
    "abts-green": Chromophore(
        name="ABTS radical cation", peak_nm=420.0, fwhm_nm=90.0, epsilon_peak=3.6e4,
        second_peak_nm=650.0, second_fwhm_nm=120.0, second_amplitude=0.45,
    ),
    # 金纳米颗粒比色（聚集变色），读 520 nm 呈红。
    "aunp-red": Chromophore(
        name="AuNP (520 nm plasmon)", peak_nm=520.0, fwhm_nm=80.0, epsilon_peak=2.7e8
    ),
}


def _lorentz_band(peak_nm: float, fwhm_nm: float) -> np.ndarray:
    """归一化到峰值 1.0 的洛伦兹带。等离激元共振是阻尼振子，尾巴比高斯重。"""

    half = max(float(fwhm_nm), 1e-6) / 2.0
    return 1.0 / (1.0 + ((LAMBDA_NM - float(peak_nm)) / half) ** 2)


@dataclass
class PlasmonResonance:
    """金纳米结构阵列的局域表面等离激元共振带（无标记 LSPR 传感）。

    与 Chromophore 的物理**完全不同**，不是同一个模型换参数：

    - 显色剂：吸收带位置固定，浓度让它**加深**。读数是吸光度，A ∝ c。
    - 等离激元：消光带深度基本固定，结合让它**移动**。读数必须是通道比值；
      对峰移取 −log10(I/I₀) 会把位移和幅度揉进一个数，信息反而丢失。

    剂量响应是抗原抗体亲和结合，服从 Langmuir 等温式而不是线性关系：

        Δλ(c) = Δλ_max · c / (K_D + c)

    只有 c ≪ K_D 时才近似线性。跨一个数量级以上仍用线性拟合，高浓度端
    会系统性偏低，这是标定曲线最常见的一个错误。
    """

    name: str = "annealed Au island array"
    peak_nm: float = 630.0            # 未结合时的共振峰位
    fwhm_nm: float = 110.0            # 线宽
    extinction_depth: float = 0.45    # 峰处消光深度 1 − T_min
    shift_max_nm: float = 4.0         # 饱和结合时的峰移
    kd_uM: float = 0.05               # 解离常数（决定动态范围位置）

    def shift_for(self, concentrations_uM: np.ndarray) -> np.ndarray:
        """Langmuir 结合 → 峰移 (nm)。"""

        c = np.clip(np.asarray(concentrations_uM, dtype=np.float64), 0.0, None)
        return self.shift_max_nm * c / (self.kd_uM + c)

    def transmittance_spectrum(self, shift_nm: np.ndarray) -> np.ndarray:
        """逐孔透射谱 T(λ)，形状 (N, len(LAMBDA_NM))。"""

        shift = np.asarray(shift_nm, dtype=np.float64).reshape(-1, 1)
        half = max(self.fwhm_nm, 1e-6) / 2.0
        detune = (LAMBDA_NM[None, :] - (self.peak_nm + shift)) / half
        return 1.0 - self.extinction_depth / (1.0 + detune ** 2)


def plasmon_channel_transmittance(
    concentrations_uM: np.ndarray,
    plasmon: PlasmonResonance,
    source: np.ndarray | None = None,
    response: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """浓度 → (逐通道透射率 (N,3) RGB, 逐孔峰移 (N,))。

    与显色剂路径的关键差异：c=0 时透射率**不是 1.0**——金阵列本身就有
    消光。所以"空白"是"未结合的金阵列"，而不是"什么都没有"。
    """

    if source is None:
        source = el_source_spectrum()
    if response is None:
        response = camera_channel_response()

    shift = plasmon.shift_for(concentrations_uM)
    spectra = plasmon.transmittance_spectrum(shift)      # (N, L)
    weight = source[None, :] * response                  # (3, L)
    channels = spectra @ weight.T / weight.sum(axis=1)[None, :]
    return channels, shift


def channel_ratio(transmittance: np.ndarray, pair: str = "RG") -> np.ndarray:
    """通道比值读数。峰移改变通道间的相对强弱，比值即信号。

    比值天然对消曝光、光源亮度、照明梯度这些共模量——它们同等地作用于
    两个通道。这正是等离激元读数应当用比值而不是绝对值的原因。
    """

    index = {"R": 0, "G": 1, "B": 2}
    t = np.asarray(transmittance, dtype=np.float64)
    numerator = t[:, index[pair[0]]]
    denominator = np.clip(t[:, index[pair[1]]], 1e-12, None)
    return numerator / denominator


def el_source_spectrum(blue_peak_nm: float = 490.0, blue_fwhm_nm: float = 70.0,
                       amber_peak_nm: float = 590.0, amber_fwhm_nm: float = 90.0,
                       amber_ratio: float = 0.75) -> np.ndarray:
    """白色 EL 面板的相对辐射光谱。

    EL 白不是平坦光源，通常是蓝绿主峰加一个琥珀次峰。这一点必须建模：
    有效的逐通道吸光系数是 ε(λ) 对 **S(λ)·R_ch(λ)** 加权的结果，换一块
    不同配方的 EL 板，同一显色反应的 A_B/A_G 比值就会变。真机标定时应
    实测这条谱，默认值只是量级正确的占位。
    """

    return _gaussian_band(blue_peak_nm, blue_fwhm_nm) + amber_ratio * _gaussian_band(
        amber_peak_nm, amber_fwhm_nm
    )


def camera_channel_response() -> np.ndarray:
    """手机 CMOS 的 R/G/B 光谱响应近似，形状 (3, len(LAMBDA_NM))。

    用高斯近似拜耳滤色阵列的通带。真机应以实测替换；这里只需要"通带有
    几十纳米宽度"这一性质成立，多色非线性就会如实出现。
    """

    red = _gaussian_band(600.0, 90.0) + 0.15 * _gaussian_band(680.0, 60.0)
    green = _gaussian_band(540.0, 90.0)
    blue = _gaussian_band(460.0, 80.0)
    return np.stack([red, green, blue], axis=0)


def concentration_to_channel_transmittance(
    concentrations_uM: np.ndarray,
    chromophore: Chromophore,
    source: np.ndarray | None = None,
    response: np.ndarray | None = None,
) -> np.ndarray:
    """逐孔浓度 → 逐通道透射率 T_ch ∈ (0,1]，形状 (N, 3) 且顺序为 RGB。

    T_ch(c) = ∫S·R_ch·10^(−ε(λ)cl) dλ / ∫S·R_ch dλ

    c=0 时恒为 1.0（空白），因此空白图天然是最亮的一张——与荧光相反。
    """

    concentrations = np.asarray(concentrations_uM, dtype=np.float64).reshape(-1)
    if source is None:
        source = el_source_spectrum()
    if response is None:
        response = camera_channel_response()

    absorbance = chromophore.absorbance_spectrum(concentrations)      # (N, L)
    transmitted = np.power(10.0, -absorbance)                          # (N, L)
    weight = source[None, :] * response                                # (3, L)
    numerator = transmitted @ weight.T                                 # (N, 3)
    denominator = weight.sum(axis=1)[None, :]                          # (1, 3)
    return numerator / np.maximum(denominator, 1e-12)


def apparent_absorbance(transmittance: np.ndarray) -> np.ndarray:
    """A = −log10(T)。透射率已相对空白归一化，故直接取负对数。"""

    return -np.log10(np.clip(np.asarray(transmittance, dtype=np.float64), 1e-12, None))


# ---------------------------------------------------------------- 仿真配置


@dataclass
class ColorimetricConfig:
    """比色仿真参数。默认几何直接取自 v4.2 硬件定稿。

    强度语义与荧光模块一致：1.0 = 恰好填满满阱。但方向相反——
    **空白孔最亮**，因此曝光应让空白落在略低于满阱处（blank_level）。
    """

    # --- 阵列几何（v4.2：15×15，中心距 1.0 mm）---
    grid_size: int = 15
    image_size: int = 1600
    panel_fill: float = 0.62          # 面板边长 / 图像边长
    well_fill: float = 0.40           # 反应区边长 / 中心距
    well_shape: str = "square"        # 方形反应区
    supersample: int = 2

    # 版面形态。Beer-Lambert 物理两者完全相同，差别只在**有没有遮罩层**：
    #
    # - "bare"（默认）：不装黑罩。亮的散射基底整片透光，反应区因显色而
    #   变暗，定位管线判定 polarity=dark。这是当前实拍芯片的样子。
    # - "masked"：装黑罩。罩体透过率由 mask_transmittance 给出，孔内为 1.0。
    #
    # 两者对定位算法是**不同的检测器通路**（暗方块 vs 亮点），所以必须能
    # 生成实际会拍到的那一种，否则验证的是另一条代码路径。
    layout: str = "bare"              # bare | masked
    # 罩体透过率。理想 1 mm 哑光黑 PMMA 约 0.002；PDMS 3D 打印的黑罩会
    # 明显透光，实测量级通常在 0.1–0.4，这正是"装了还不如不装"的原因：
    # 当显色透射率降到罩体透过率以下（A > −log10(T_mask)）时，该孔变得比
    # 罩体还暗，同一张图里极性不再一致。用 --mask-study 可以直接扫出
    # 自己那批 PDMS 罩需要压到多少才有增益。
    mask_transmittance: float = 0.002

    # --- 显色化学 ---
    concentration_pattern: str = "log_series"  # log_series | plate_series | gradient | uniform | random | checker
    # 量程由 0.5 mm 液层决定，不能照搬 96 孔板的 µM 级习惯：
    # A = ε·c·l，l 只有 0.5 mm 时 A_B=0.1 需要 47 µM、A_B=1.0 需要 471 µM。
    # 默认区间刻意两端都越界——低端落到检出限以下、高端进入多色压缩区，
    # 一张图就能同时看到"测不出"和"压缩失真"两个边界。
    concentration_min_uM: float = 5.0
    concentration_max_uM: float = 800.0
    concentration_jitter: float = 0.05
    blank_well_count: int = 8         # 浓度恰为 0 的空白孔，标定必需
    chromophore: Chromophore = field(default_factory=Chromophore)

    # 信号模型。两者物理不同，不是同一模型换参数：
    # - "dye"：显色剂，吸收带固定、浓度让它加深，读吸光度。
    # - "plasmon"：无标记 LSPR，共振峰随结合而移动，读通道比值。
    #   金纳米阵列芯片属于后者——空白不是"透明"，而是"未结合的金阵列"。
    signal_model: str = "dye"         # dye | plasmon
    plasmon: PlasmonResonance = field(default_factory=PlasmonResonance)
    ratio_pair: str = "RG"            # 比值读数用哪两个通道

    # 参考列：该列只通缓冲液、不加抗原，作为同帧空间参考。
    # 交叉通道版型（横向加抗体、纵向加抗原）让这件事几乎免费——牺牲
    # 15/225 = 6.7% 通量，换来照明漂移、曝光波动、温度、行间基线差异
    # 的共模对消，而且参考点与样品点在**同一帧**里。
    # 时序参考（加抗原前后各拍一张）要跨越孵育、清洗、重新装夹，
    # 中间的漂移往往比 1-5 nm 的信号还大。
    reference_column: int | None = None

    # --- 曝光 ---
    blank_level: float = 0.85         # 空白孔的满阱分数；留 15% 余量防饱和

    # --- 背照、基底与黑罩 ---
    mask_leak: float = 0.0025         # 黑罩非理想遮光（PMMA 边缘散射 + 杂散光）
    outside_panel_level: float = 0.001
    substrate_transmittance: float = 0.92   # 白色散射膜本体透射率（substrate 形态）
    substrate_texture: float = 0.03         # 膜的纤维纹理起伏（实拍可见）
    el_honeycomb: float = 0.05        # EL 蜂窝纹残留幅度（漫射膜抑制后）
    el_honeycomb_mm: float = 1.6      # 蜂窝周期，mm
    panel_side_mm: float = 20.0       # 黑罩物理边长，用于把 mm 换算到像素
    illumination_nonuniformity: float = 0.18   # EL 面板亮度梯度
    illumination_angle_deg: float = 35.0

    # --- 光学 ---
    crosstalk_sigma_ratio: float = 0.05   # 孔间串扰（黑罩出射锥 14°-27°）
    defocus_sigma_px: float = 1.1
    vignetting: float = 0.30
    chromatic_aberration_px: float = 0.6

    # --- 几何畸变 ---
    rotation_deg: float = 2.0
    perspective_strength: float = 0.018
    radial_k1: float = -0.055

    # --- 传感器（与荧光模块同口径）---
    full_well_e: float = 12000.0
    read_noise_e: float = 6.0
    dark_current_e: float = 4.0
    hot_pixel_ratio: float = 3e-6
    prnu: float = 0.008
    gamma: float = 1.0
    jpeg_quality: int | None = 92

    seed: int = 0

    def resolved_concentrations(self) -> np.ndarray:
        """逐孔浓度真值（µM，行优先）。"""

        rng = np.random.default_rng(self.seed + 3301)
        count = self.grid_size * self.grid_size
        low = max(self.concentration_min_uM, 1e-9)
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
            # 逐行等比稀释 + 行序打乱。打乱的理由与荧光模块相同：浓度若沿
            # 版面单调排列，就与照明梯度和渐晕完全混杂，标定曲线分不清
            # 读数变化有多少来自浓度、多少来自位置。
            per_row = np.exp(np.linspace(math.log(high), math.log(low), self.grid_size))
            per_row = per_row[rng.permutation(self.grid_size)]
            values = np.repeat(per_row, self.grid_size)
        else:
            values = np.exp(np.linspace(math.log(low), math.log(high), count))

        if self.concentration_jitter > 0:
            values = values * (1.0 + rng.normal(0.0, self.concentration_jitter, count))
        values = np.clip(values, 0.0, None)

        if self.blank_well_count > 0:
            blanks = rng.choice(count, size=min(self.blank_well_count, count), replace=False)
            values[blanks] = 0.0

        if self.reference_column is not None:
            # 参考列只通缓冲液：整列浓度置零，且不受移液抖动影响。
            column = int(self.reference_column) % self.grid_size
            values.reshape(self.grid_size, self.grid_size)[:, column] = 0.0
        return values


# ---------------------------------------------------------------- 场景渲染


def _render_transmission_map(
    config: ColorimetricConfig,
    transmittance: np.ndarray,
    canvas: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """渲染逐通道线性辐射图（超采样画布）与孔中心真值。

    统一模型：先渲染"背光穿过散射基底、再被反应区吸收"的亮场，
    然后可选地乘上一层**遮罩**——孔内透过率 1.0，罩体透过率
    mask_transmittance。这样三种情况是同一套公式的三个取值：

        无罩            → 不乘遮罩层，亮基底 + 暗斑点（polarity=dark）
        理想不透光黑罩  → T_mask≈0，黑底 + 亮孔（polarity=bright）
        半透光 PDMS 罩  → T_mask 居中，对比度被压，且可能中途翻转极性

    翻转发生在显色透射率降到罩体透过率以下时，即 A > −log10(T_mask)：
    此时该孔比罩体还暗，同一张图里低浓度孔是亮点、高浓度孔是暗点，
    而极性判定是**全局单一决策**，必然判错一半。
    """

    grid = config.grid_size
    masked = config.layout == "masked"
    scene = np.full((canvas, canvas, 3), config.outside_panel_level, dtype=np.float64)

    panel_side = canvas * config.panel_fill
    panel_x0 = (canvas - panel_side) / 2.0
    panel_y0 = (canvas - panel_side) / 2.0
    px0, py0 = int(round(panel_x0)), int(round(panel_y0))
    px1, py1 = int(round(panel_x0 + panel_side)), int(round(panel_y0 + panel_side))

    # 背照场：EL 面板亮度分布，先在整幅画布上算好，再按孔采样。
    # 蜂窝纹是 EL 的固有像素结构，漫射膜只能压低不能消除；周期取 mm 再
    # 换算成像素，这样改变 image_size 不会改变它的物理周期。
    yy, xx = np.mgrid[0:canvas, 0:canvas].astype(np.float64)
    field = np.ones((canvas, canvas), dtype=np.float64)
    if config.illumination_nonuniformity > 0:
        angle = math.radians(config.illumination_angle_deg)
        projection = (xx * math.cos(angle) + yy * math.sin(angle)) / canvas
        projection = (projection - projection.min()) / max(projection.max() - projection.min(), 1e-9)
        field = field * (1.0 + config.illumination_nonuniformity * (projection - 0.5))
    if config.el_honeycomb > 0:
        px_per_mm = panel_side / max(config.panel_side_mm, 1e-6)
        period = max(config.el_honeycomb_mm * px_per_mm, 2.0)
        honeycomb = (np.cos(2.0 * math.pi * xx / period)
                     * np.cos(2.0 * math.pi * yy / period))
        field = field * (1.0 + config.el_honeycomb * honeycomb)

    margin = panel_side * 0.085
    usable = panel_side - 2.0 * margin
    pitch = usable / max(grid - 1, 1)
    half = max(1.0, pitch * config.well_fill / 2.0)

    # 1) 亮基底：整片面板被背光穿透，亮度受照明场与膜透射率调制。
    scene[py0:py1, px0:px1, :] = (
        field[py0:py1, px0:px1, None] * config.blank_level * config.substrate_transmittance
    )
    if config.substrate_texture > 0:
        coarse = rng.normal(0.0, config.substrate_texture, (48, 48))
        texture = cv2.resize(coarse, (px1 - px0, py1 - py0), interpolation=cv2.INTER_CUBIC)
        scene[py0:py1, px0:px1, :] *= np.clip(1.0 + texture, 0.0, None)[:, :, None]

    # 2) 反应区吸收：在亮基底上做乘法，基底的梯度与纹理因此会透出来。
    #    同时记录孔的覆盖掩膜，供第 3 步的遮罩层使用。
    aperture = np.zeros((canvas, canvas), dtype=np.float64)
    centers = np.empty((grid * grid, 2), dtype=np.float64)
    for row in range(grid):
        for col in range(grid):
            index = row * grid + col
            cx = panel_x0 + margin + col * pitch
            cy = panel_y0 + margin + row * pitch
            centers[index] = (cx, cy)

            x0, x1 = int(round(cx - half)), int(round(cx + half))
            y0, y1 = int(round(cy - half)), int(round(cy + half))
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, canvas), min(y1, canvas)
            if x1 <= x0 or y1 <= y0:
                continue

            if config.well_shape == "circle":
                patch = np.zeros((y1 - y0, x1 - x0), dtype=np.float64)
                cv2.circle(patch, ((x1 - x0) // 2, (y1 - y0) // 2),
                           int(round(half)), 1.0, -1, lineType=cv2.LINE_AA)
            else:
                patch = np.ones((y1 - y0, x1 - x0), dtype=np.float64)
            aperture[y0:y1, x0:x1] = np.maximum(aperture[y0:y1, x0:x1], patch)
            for ch in range(3):
                factor = 1.0 + patch * (float(transmittance[index, ch]) - 1.0)
                scene[y0:y1, x0:x1, ch] *= factor

    # 3) 遮罩层：孔内 1.0、罩体 mask_transmittance。理想黑罩把罩体压到近乎
    #    全黑（于是变成"黑底亮孔"），半透光 PDMS 罩只是把它压暗一些。
    if masked:
        mask_layer = np.full((canvas, canvas), config.mask_transmittance, dtype=np.float64)
        mask_layer[py0:py1, px0:px1] = config.mask_transmittance
        mask_layer = mask_layer + aperture * (1.0 - config.mask_transmittance)
        scene = scene * mask_layer[:, :, None]

    # 孔间串扰：黑罩出射锥半角 14°-27°，零间距贴合时锥脚会溢到邻孔。
    sigma = config.crosstalk_sigma_ratio * pitch
    if sigma > 0.3:
        ksize = int(sigma * 6) | 1
        scene = cv2.GaussianBlur(scene, (ksize, ksize), sigma)

    return scene, centers


def render_colorimetric_scene(
    config: ColorimetricConfig,
    force_blank: bool = False,
) -> tuple[np.ndarray, dict[str, object]]:
    """渲染一张比色仿真图，返回 (BGR uint8 图像, 真值字典)。

    force_blank=True 时把所有孔按浓度 0 渲染，用于生成配对空白图。
    空白图必须与样品图**共用同一随机种子**，这样 PRNU、热像素等固定
    图案噪声一致，A=−log10(I/I_blank) 才能把它们约掉。
    """

    rng = np.random.default_rng(config.seed)
    size = int(config.image_size)
    scale = max(1, int(config.supersample))
    canvas = size * scale

    concentrations = config.resolved_concentrations()
    if force_blank:
        concentrations = np.zeros_like(concentrations)

    plasmon_mode = config.signal_model == "plasmon"
    if plasmon_mode:
        transmittance, peak_shift = plasmon_channel_transmittance(concentrations, config.plasmon)
    else:
        transmittance = concentration_to_channel_transmittance(concentrations, config.chromophore)
        peak_shift = None

    scene, centers = _render_transmission_map(config, transmittance, canvas, rng)

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

    if scale > 1:
        scene = cv2.resize(scene, (size, size), interpolation=cv2.INTER_AREA)
        centers = centers / scale

    # RGB → OpenCV 的 BGR 通道序
    stack = scene[:, :, ::-1].copy()
    image = apply_optics_and_sensor(stack, config, rng, size)

    absorbance = apparent_absorbance(transmittance)
    truth: dict[str, object] = {
        "schema": "pg-colori-truth-v1",
        "mode": "blank" if force_blank else "sample",
        "grid_size": int(config.grid_size),
        "image_size": size,
        "points": [[round(float(x), 4), round(float(y), 4)] for x, y in centers],
        # intensities 保留与荧光真值同名的键，取绿通道透射率作为
        # "单通道亮度"，使既有的定位评估代码可以直接复用。
        "intensities": [round(float(v), 8) for v in transmittance[:, 1]],
        "concentrations": [round(float(v), 9) for v in concentrations],
        "concentration_unit": "uM",
        "transmittance_rgb": [[round(float(v), 8) for v in row] for row in transmittance],
        "absorbance_rgb": [[round(float(v), 8) for v in row] for row in absorbance],
        "chromophore": asdict(config.chromophore),
        "config": asdict(config),
    }
    if plasmon_mode:
        # 等离激元路径的被测量是**峰移**，吸光度只是顺带保留的派生量。
        # ratio 是实际读数：峰移改变通道相对强弱，比值即信号。
        truth.update({
            "signal_model": "plasmon",
            "peak_shift_nm": [round(float(v), 6) for v in peak_shift],
            "ratio_pair": config.ratio_pair,
            "channel_ratio": [round(float(v), 8) for v in channel_ratio(transmittance, config.ratio_pair)],
            "plasmon": asdict(config.plasmon),
            "reference_column": config.reference_column,
        })
    else:
        truth["signal_model"] = "dye"
    return image, truth


def write_colorimetric_sample(
    config: ColorimetricConfig,
    output_dir: str | Path,
    name: str = "colori",
) -> dict[str, str]:
    """渲染样品图 + 配对空白图并落盘，返回路径字典。

    空白图不是可选项：比色的被测量是 A=−log10(I_sample/I_blank)，没有
    空白就没有吸光度，只有一个受照明分布、EL 光谱和曝光共同污染的灰度。
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".png" if config.jpeg_quality is None else ".jpg"

    paths: dict[str, str] = {}
    for tag, blank in (("", False), ("_blank", True)):
        image, truth = render_colorimetric_scene(config, force_blank=blank)
        image_path = output_dir / f"{name}{tag}{suffix}"
        if suffix == ".jpg":
            cv2.imwrite(str(image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), int(config.jpeg_quality)])
        else:
            cv2.imwrite(str(image_path), image)
        truth_path = output_dir / f"{name}{tag}_truth.json"
        with truth_path.open("w", encoding="utf-8") as f:
            json.dump(truth, f, ensure_ascii=False, indent=2)
        key = "blank" if blank else "image"
        paths[key] = str(image_path)
        paths[f"{key}_truth"] = str(truth_path)
    return paths


# ---------------------------------------------------------------- 评估


def evaluate_generated_sample(
    image_path: str | Path,
    blank_path: str | Path,
    truth_path: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """在仿真图上跑完整定位管线，并按真值校核吸光度读数。

    比色的评估比荧光多一步：定位对不对之外，还要问**吸光度算得对不对**，
    而后者必须用配对空白图逐孔归一化。
    """

    from pg_grid import process_image

    with Path(truth_path).open("r", encoding="utf-8") as f:
        truth = json.load(f)
    grid_size = int(truth["grid_size"])
    true_points = np.asarray(truth["points"], dtype=np.float32)
    true_absorbance = np.asarray(truth["absorbance_rgb"], dtype=np.float64)
    concentrations = np.asarray(truth["concentrations"], dtype=np.float64)

    output_dir = Path(output_dir)
    result = process_image(image_path=image_path, grid_size=grid_size, output_dir=output_dir / "sample")
    blank_result = process_image(image_path=blank_path, grid_size=grid_size, output_dir=output_dir / "blank")

    # 定位误差（与荧光评估同口径：真值经管线自己的四角单应映射到矫正系）
    region = np.asarray(result["chip_region"]["points"], dtype=np.float32)
    size = int(result["rectified_size"])
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(region, dst)
    mapped = cv2.perspectiveTransform(true_points.reshape(-1, 1, 2), matrix).reshape(-1, 2).astype(np.float64)
    predicted = np.asarray([[p["x"], p["y"]] for p in result["grid_points"]], dtype=np.float64)
    errors = np.linalg.norm(predicted - mapped, axis=1)
    pitch = max(float(result["lattice_consistency"]["pitch_px"]), 1e-6)

    # 吸光度读数：逐孔、逐通道用空白图归一化
    def read_channels(res: dict[str, object], out: Path) -> np.ndarray | None:
        quant_path = out / "quant_values.csv"
        if not quant_path.exists():
            return None
        import csv
        rows = list(csv.DictReader(quant_path.open(encoding="utf-8")))
        # 刻意取**原始** ROI 中位数，而不是 PG-Quant 的自参照平场校正值：
        # 比色的参考是配对空白图，它已经承载了照明分布、EL 光谱和渐晕；
        # 再叠一层自参照校正等于把同一个照明场扣两次。
        keys = ("roi_b_median", "roi_g_median", "roi_r_median")
        if not rows or not all(k in rows[0] for k in keys):
            return None
        # 输出为 RGB 顺序，与真值一致
        return np.asarray(
            [[float(r[keys[2]]), float(r[keys[1]]), float(r[keys[0]])] for r in rows],
            dtype=np.float64,
        )

    sample_rgb = read_channels(result, output_dir / "sample")
    blank_rgb = read_channels(blank_result, output_dir / "blank")

    absorbance_report: dict[str, object] | None = None
    if sample_rgb is not None and blank_rgb is not None and sample_rgb.shape == blank_rgb.shape:
        measured = -np.log10(np.clip(sample_rgb, 1e-6, None) / np.clip(blank_rgb, 1e-6, None))
        channel_names = ("R", "G", "B")
        per_channel = {}
        for ch in range(3):
            valid = np.isfinite(measured[:, ch]) & np.isfinite(true_absorbance[:, ch])
            if int(valid.sum()) < 5:
                continue
            bias = float(np.mean(measured[valid, ch] - true_absorbance[valid, ch]))
            rmse = float(np.sqrt(np.mean((measured[valid, ch] - true_absorbance[valid, ch]) ** 2)))
            rank_t = np.argsort(np.argsort(true_absorbance[valid, ch])).astype(np.float64)
            rank_m = np.argsort(np.argsort(measured[valid, ch])).astype(np.float64)
            spearman = (float(np.corrcoef(rank_t, rank_m)[0, 1])
                        if rank_t.std() > 1e-9 and rank_m.std() > 1e-9 else float("nan"))
            per_channel[channel_names[ch]] = {
                "bias": round(bias, 5),
                "rmse": round(rmse, 5),
                "spearman": round(spearman, 5),
                "true_max": round(float(true_absorbance[valid, ch].max()), 4),
            }
        # 检出限：空白孔读数的均值 + 3σ，换算回浓度需要标定曲线，
        # 这里只报吸光度域的阈值，避免用非线性曲线做外推。
        blanks = concentrations <= 0
        blank_stats = None
        if int(blanks.sum()) >= 3:
            blank_a = measured[blanks, 2]   # B 通道（黄色产物吸蓝）
            blank_stats = {
                "count": int(blanks.sum()),
                "mean_A_B": round(float(np.mean(blank_a)), 5),
                "std_A_B": round(float(np.std(blank_a)), 5),
                "detection_threshold_A_B": round(float(np.mean(blank_a) + 3.0 * np.std(blank_a)), 5),
            }
        absorbance_report = {"per_channel": per_channel, "blank": blank_stats}

    report: dict[str, object] = {
        "schema": "pg-colori-eval-v1",
        "image_path": str(image_path),
        "grid_size": grid_size,
        "unit_polarity": result["unit_polarity"],
        "chip_method": result["chip_region"]["method"],
        "trusted": bool(result["lattice_consistency"]["trusted"]),
        "candidate_support_ratio": result["lattice_consistency"]["candidate_support_ratio"],
        "pitch_px": round(pitch, 4),
        "mean_error_px": round(float(errors.mean()), 4),
        "mean_error_pct_pitch": round(float(errors.mean() / pitch * 100.0), 4),
        "p90_error_pct_pitch": round(float(np.percentile(errors, 90) / pitch * 100.0), 4),
        "quality_status": result["quality"]["status"],
        "absorbance": absorbance_report,
    }
    with (output_dir / "colori_eval.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


DIFFICULTY_PRESETS: dict[str, dict[str, object]] = {
    "ideal": dict(defocus_sigma_px=0.6, read_noise_e=3.0, vignetting=0.1,
                  illumination_nonuniformity=0.05, el_honeycomb=0.02, rotation_deg=0.5,
                  perspective_strength=0.005, radial_k1=-0.01),
    "typical": dict(),
    "hard": dict(defocus_sigma_px=2.2, read_noise_e=12.0, vignetting=0.45,
                 illumination_nonuniformity=0.40, el_honeycomb=0.10, rotation_deg=5.0,
                 perspective_strength=0.035, radial_k1=-0.09),
    "extreme": dict(defocus_sigma_px=3.5, read_noise_e=25.0, vignetting=0.60,
                    illumination_nonuniformity=0.55, el_honeycomb=0.16, rotation_deg=8.0,
                    perspective_strength=0.055, radial_k1=-0.13, jpeg_quality=70),
}


def plasmon_readout_study(
    plasmon: PlasmonResonance,
    concentrations_uM: np.ndarray,
    ratio_pair: str = "RG",
    ratio_noise: float = 0.00198,
) -> dict[str, object]:
    """比值读数的分辨率与动态范围，纯光学、不含成像。

    先回答"比色到底能不能替代光谱"，再决定要不要建完整图像仿真——
    如果这一步就答不可行，图像仿真做得再细也没意义。

    ratio_noise 默认 0.198%（满阱 12000e⁻、ROI≈450px、双点比值的 3σ
    散粒噪声）。真实系统还要叠加漂移与配准误差，把它调大即可看到
    结论如何退化。
    """

    channels, shift = plasmon_channel_transmittance(concentrations_uM, plasmon)
    ratio = channel_ratio(channels, ratio_pair)
    baseline = channel_ratio(
        plasmon_channel_transmittance(np.array([0.0]), plasmon)[0], ratio_pair
    )[0]
    signal = ratio / baseline - 1.0

    # 每 nm 峰移带来多少比值变化（在工作点附近取导数）
    probe, _ = plasmon_channel_transmittance(np.array([0.0]), plasmon)
    shifted = plasmon.transmittance_spectrum(np.array([1.0]))
    weight = el_source_spectrum()[None, :] * camera_channel_response()
    probe_shift = shifted @ weight.T / weight.sum(axis=1)[None, :]
    per_nm = abs(channel_ratio(probe_shift, ratio_pair)[0] / baseline - 1.0)
    resolvable_nm = ratio_noise / per_nm if per_nm > 0 else float("inf")

    full_scale = float(np.max(np.abs(signal))) if signal.size else 0.0
    return {
        "ratio_pair": ratio_pair,
        "baseline_ratio": round(float(baseline), 6),
        "sensitivity_per_nm": round(float(per_nm), 8),
        "resolvable_shift_nm": round(float(resolvable_nm), 4),
        "shift_max_nm": plasmon.shift_max_nm,
        # 可用档位数：满量程峰移能被分成几个可分辨的等级
        "usable_levels": int(plasmon.shift_max_nm / resolvable_nm) if resolvable_nm > 0 else 0,
        "full_scale_ratio_change": round(full_scale, 6),
        "shift_nm": [round(float(v), 4) for v in shift],
        "signal": [round(float(v), 8) for v in signal],
    }


def polarity_flip_absorbance(mask_transmittance: float) -> float:
    """极性翻转发生的吸光度：A > −log10(T_mask) 的孔比罩体还暗。

    这是"半透光罩为什么有害"的定量表述。理想黑罩 T=0.002 时翻转点是
    A=2.7，远在量程之外；PDMS 罩 T=0.25 时只有 A=0.60，量程内就会翻。
    """

    return float(-math.log10(max(float(mask_transmittance), 1e-12)))


def mask_transmittance_study(
    base: ColorimetricConfig,
    levels: tuple[float, ...] = (0.002, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 1.0),
    output_dir: str | Path | None = None,
) -> list[dict[str, object]]:
    """扫描罩体透过率，报告每一档的定位表现与**吸光度读数精度**。

    回答的是一个硬件决策：**这批半透光的 PDMS 罩值不值得装**。

    两项代价方向相反，必须一起看：
    - 装罩的收益在**定位**。罩子给出硬边几何孔径，斑点再淡也有边缘可找；
      不装罩时低浓度斑点在白膜上几乎没有对比度，检测器直接失明。
    - 装罩的代价在**光度**。罩体漏的光经离焦与串扰渗进 ROI，等效于杂散光：
      A = −log10((I+L)/(I_blank+L))，L 越大曲线被压得越平，高吸光端最先失真。

    因此判据不是"罩子看起来黑不黑"，而是漏光把吸光度压偏多少。
    """

    rows: list[dict[str, object]] = []
    concentrations = base.resolved_concentrations()
    transmittance = concentration_to_channel_transmittance(concentrations, base.chromophore)
    # 翻转判据必须按**亮度**算，不能挑某一个通道：极性检测走的是灰度图，
    # 而显色产物的信号通道随染料而变（黄色吸蓝、蓝色吸红）。
    luminance = transmittance @ np.array([0.299, 0.587, 0.114], dtype=np.float64)
    absorbance = apparent_absorbance(luminance)

    for level in levels:
        bare = level >= 1.0
        config = ColorimetricConfig(**{
            **asdict_shallow(base),
            "layout": "bare" if bare else "masked",
            "mask_transmittance": float(level),
        })
        flip_A = polarity_flip_absorbance(level)
        # 量程内有多少孔会翻到"比罩体还暗"
        flipped = int((absorbance > flip_A).sum()) if not bare else 0

        row: dict[str, object] = {
            "mask_transmittance": float(level),
            "layout": "bare" if bare else "masked",
            "flip_absorbance": round(flip_A, 3) if not bare else None,
            "wells_past_flip": flipped,
            "well_count": int(concentrations.size),
        }

        if output_dir is not None:
            out = Path(output_dir) / f"T{level:.3f}"
            paths = write_colorimetric_sample(config, out, name="s")
            report = evaluate_generated_sample(
                paths["image"], paths["blank"], paths["image_truth"], out / "eval"
            )
            row.update({
                "polarity": report["unit_polarity"],
                "error_pct_pitch": report["mean_error_pct_pitch"],
                "support": report["candidate_support_ratio"],
                "trusted": report["trusted"],
            })
            channels = ((report.get("absorbance") or {}).get("per_channel") or {})
            # 取真值吸光度最大的通道作为信号通道——它随染料而变。
            signal = max(channels.items(), key=lambda kv: kv[1]["true_max"], default=(None, None))
            if signal[0] is not None:
                row.update({
                    "signal_channel": signal[0],
                    "absorbance_bias": signal[1]["bias"],
                    "absorbance_rmse": signal[1]["rmse"],
                    "absorbance_spearman": signal[1]["spearman"],
                    "true_max_absorbance": signal[1]["true_max"],
                })
        rows.append(row)
    return rows


def asdict_shallow(config: ColorimetricConfig) -> dict[str, object]:
    """浅拷贝配置字段（保留 Chromophore 对象本身，不展开成 dict）。"""

    return {f: getattr(config, f) for f in config.__dataclass_fields__}


def linearity_table(chromophore: Chromophore, concentrations_uM: np.ndarray) -> list[dict[str, float]]:
    """表观吸光度相对"过原点直线"的偏离，量化多色非线性的量程上界。"""

    transmittance = concentration_to_channel_transmittance(concentrations_uM, chromophore)
    absorbance = apparent_absorbance(transmittance)[:, 2]      # B 通道
    reference = None
    rows = []
    for index, conc in enumerate(concentrations_uM):
        if conc <= 0:
            continue
        if reference is None:
            reference = absorbance[index] / conc
        ideal = reference * conc
        rows.append({
            "concentration_uM": float(conc),
            "apparent_A_B": float(absorbance[index]),
            "linear_A_B": float(ideal),
            "ratio": float(absorbance[index] / ideal) if ideal > 1e-12 else float("nan"),
        })
    return rows


# ---------------------------------------------------------------- 命令行


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PG-Colori-Sim 比色阵列仿真")
    parser.add_argument("--grid", type=int, default=15, choices=[10, 15], help="阵列规格")
    parser.add_argument("--output", required=True, help="输出目录")
    parser.add_argument("--name", default=None, help="输出文件名前缀")
    parser.add_argument("--size", type=int, default=1600, help="输出图像边长")
    parser.add_argument("--pattern", default="log_series",
                        choices=["log_series", "plate_series", "gradient", "uniform", "random", "checker"],
                        help="逐孔浓度排布")
    parser.add_argument("--cmin", type=float, default=5.0, help="最低浓度 µM")
    parser.add_argument("--cmax", type=float, default=800.0, help="最高浓度 µM")
    parser.add_argument("--blanks", type=int, default=8, help="空白孔数（浓度 0）")
    parser.add_argument("--path-mm", type=float, default=0.5, help="液层光程 mm")
    parser.add_argument("--layout", default="bare", choices=["bare", "masked"],
                        help="bare=不装黑罩（实拍芯片），masked=装黑罩")
    parser.add_argument("--mask-transmittance", type=float, default=0.002,
                        help="罩体透过率：黑 PMMA 约 0.002，PDMS 打印约 0.1-0.4")
    parser.add_argument("--mask-study", action="store_true",
                        help="扫描罩体透过率，给出'罩子要多不透光才有增益'的答案")
    parser.add_argument("--dye", default=None, choices=sorted(CHROMOPHORES),
                        help="显色产物预设；给了它就忽略 --peak-nm/--epsilon")
    parser.add_argument("--peak-nm", type=float, default=450.0, help="显色产物吸收峰 nm")
    parser.add_argument("--epsilon", type=float, default=5.9e4, help="峰值摩尔吸光系数 M⁻¹cm⁻¹")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--png", action="store_true", help="输出无损 PNG")
    parser.add_argument("--evaluate", action="store_true", help="生成后立即跑管线并评估")
    parser.add_argument("--sweep", action="store_true", help="按难度预设生成一组样本并评估")
    parser.add_argument("--linearity", action="store_true", help="只打印多色非线性表，不生成图像")
    return parser.parse_args()


def _print_report(tag: str, report: dict[str, object]) -> None:
    print(f"[{tag}] 极性={report['unit_polarity']} 可信={report['trusted']} "
          f"支撑率={report['candidate_support_ratio']} 质量={report['quality_status']}")
    print(f"       定位误差 平均={report['mean_error_pct_pitch']:.2f}%pitch "
          f"P90={report['p90_error_pct_pitch']:.2f}%")
    absorbance = report.get("absorbance")
    if not absorbance:
        return
    print(f"{'通道':>10s}{'偏差':>10s}{'RMSE':>10s}{'保序性':>10s}{'真值A上限':>12s}")
    for channel, stats in absorbance["per_channel"].items():
        print(f"{channel:>10s}{stats['bias']:>10.4f}{stats['rmse']:>10.4f}"
              f"{stats['spearman']:>10.4f}{stats['true_max']:>12.4f}")
    if absorbance.get("blank"):
        b = absorbance["blank"]
        print(f"       空白孔 n={b['count']}  A_B={b['mean_A_B']:.4f}±{b['std_A_B']:.4f}"
              f"  检出阈值 A_B={b['detection_threshold_A_B']:.4f}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)

    if args.dye:
        preset = CHROMOPHORES[args.dye]
        chromophore = Chromophore(
            name=preset.name, peak_nm=preset.peak_nm, fwhm_nm=preset.fwhm_nm,
            epsilon_peak=preset.epsilon_peak, second_peak_nm=preset.second_peak_nm,
            second_fwhm_nm=preset.second_fwhm_nm, second_amplitude=preset.second_amplitude,
            path_length_mm=args.path_mm,
        )
    else:
        chromophore = Chromophore(
            peak_nm=args.peak_nm, epsilon_peak=args.epsilon, path_length_mm=args.path_mm
        )

    if args.linearity:
        grid = np.array([10.0, 25.0, 50.0, 100.0, 200.0, 400.0, 700.0, 1000.0, 2000.0], dtype=np.float64)
        print(f"PG-Colori-Sim 多色非线性（{chromophore.name}，"
              f"ε={chromophore.epsilon_peak:.3g}，l={chromophore.path_length_mm} mm）")
        print("Beer-Lambert 与光谱积分不可交换：通带内 ε 变化使高浓度端表观吸光度偏低。\n")
        print(f"{'浓度 µM':>10s}{'表观 A_B':>12s}{'线性 A_B':>12s}{'比值':>10s}")
        for row in linearity_table(chromophore, grid):
            print(f"{row['concentration_uM']:>10.4g}{row['apparent_A_B']:>12.4f}"
                  f"{row['linear_A_B']:>12.4f}{row['ratio']:>10.4f}")
        return

    def build(**overrides) -> ColorimetricConfig:
        base = dict(
            grid_size=args.grid, image_size=args.size,
            concentration_pattern=args.pattern,
            concentration_min_uM=args.cmin, concentration_max_uM=args.cmax,
            blank_well_count=args.blanks, chromophore=chromophore, seed=args.seed,
            layout=args.layout, mask_transmittance=args.mask_transmittance,
            jpeg_quality=None if args.png else 92,
        )
        base.update(overrides)
        return ColorimetricConfig(**base)

    if args.mask_study:
        config = build()
        print(f"PG-Colori-Sim 黑罩透过率扫描（{args.grid}x{args.grid}，"
              f"{args.cmin}–{args.cmax} µM，{chromophore.name}）")
        print("问题：这批罩子值不值得装？判据是量程内会不会翻转极性、定位还站不站得住。\n")
        rows = mask_transmittance_study(config, output_dir=output_dir)
        print("定位看前半段（装罩给硬边孔径），光度看后半段（漏光等效杂散光压平曲线）。\n")
        print(f"{'罩体透过率':>11s}{'形态':>8s}{'极性':>7s}{'定位%pitch':>12s}"
              f"{'支撑率':>8s}{'可信':>7s}{'信号道':>8s}{'A偏差':>9s}{'A_RMSE':>9s}{'保序性':>9s}")
        for row in rows:
            print(f"{row['mask_transmittance']:>11.3f}{row['layout']:>8s}"
                  f"{str(row.get('polarity','-')):>7s}"
                  f"{row.get('error_pct_pitch', float('nan')):>12.2f}"
                  f"{row.get('support', float('nan')):>8.3f}{str(row.get('trusted','-')):>7s}"
                  f"{str(row.get('signal_channel','-')):>8s}"
                  f"{row.get('absorbance_bias', float('nan')):>9.4f}"
                  f"{row.get('absorbance_rmse', float('nan')):>9.4f}"
                  f"{row.get('absorbance_spearman', float('nan')):>9.4f}")
        truth_max = next((r.get("true_max_absorbance") for r in rows if r.get("true_max_absorbance")), None)
        if truth_max:
            print(f"\n真值吸光度上限 A={truth_max:.3f}；A偏差相对它的占比即漏光造成的系统性压缩。")
        print(f"输出目录: {output_dir.resolve()}")
        return

    if args.sweep:
        print(f"PG-Colori-Sim 难度扫描（{args.grid}x{args.grid}，"
              f"{args.cmin}–{args.cmax} µM，{chromophore.name}）\n")
        for tag, preset in DIFFICULTY_PRESETS.items():
            config = build(**preset)
            paths = write_colorimetric_sample(config, output_dir, name=f"{args.grid}x{args.grid}_{tag}")
            report = evaluate_generated_sample(
                paths["image"], paths["blank"], paths["image_truth"], output_dir / f"eval_{tag}"
            )
            _print_report(tag, report)
            print()
        print(f"输出目录: {output_dir.resolve()}")
        return

    name = args.name or f"colori_{args.grid}x{args.grid}"
    config = build()
    paths = write_colorimetric_sample(config, output_dir, name=name)
    print("PG-Colori-Sim 生成完成")
    print(f"显色产物: {chromophore.name}  峰 {chromophore.peak_nm} nm  "
          f"ε {chromophore.epsilon_peak:.3g}  光程 {chromophore.path_length_mm} mm")
    print(f"样品图: {paths['image']}")
    print(f"空白图: {paths['blank']}")
    print(f"真值:   {paths['image_truth']}")

    if args.evaluate:
        report = evaluate_generated_sample(
            paths["image"], paths["blank"], paths["image_truth"], output_dir / "eval"
        )
        print()
        _print_report(name, report)


if __name__ == "__main__":
    main()
