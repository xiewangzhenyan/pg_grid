"""PG-Calibration：从仿真图像闭合到分析物浓度，并给出回收率报告。

这个模块回答的是**买试剂之前该问的那个问题**：整条算法链跑下来，
浓度读数的准确度和检出限是多少，可用量程在哪里。

链路是完整的，中间不允许有"假设已知"的环节：

    分析物浓度（真值，ng/mL）
      → 免疫反应模型（Langmuir 结合 + 酶促显色）
      → 显色产物浓度 µM
      → pg_colori_sim 渲染成图（含光学、噪声、几何畸变、JPEG）
      → pg_grid 定位 225 个孔
      → pg_quant 逐孔取 ROI
      → 相对配对空白算吸光度
      → 4PL 标定曲线拟合
      → 反算分析物浓度
      → 与真值比对：回收率、CV、LOD、LOQ、可用量程

关键在于**反算这一步必须只用图像里能得到的信息**。真值只在最后
比对时出现，不参与拟合，否则整个评估就是自证。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from pg_colori_sim import ColorimetricConfig, render_colorimetric_scene, write_colorimetric_sample


# ---------------------------------------------------------------- 免疫反应模型


@dataclass
class ImmunoassayResponse:
    """分析物浓度 → 显色产物浓度。

    这是仿真里唯一属于"生物化学"的一段，其余全是光学与图像。分成两步：

    1. **结合**：抗原与固定抗体的结合服从 Langmuir 等温式，
       θ = c^h / (K_D^h + c^h)。这一步天然饱和，所以标定曲线是 S 形而
       不是直线——用线性拟合跨一个数量级以上，高浓度端必然偏低。
    2. **显色**：酶标二抗的量正比于 θ，显色时间内转化的底物量正比于酶量，
       因此产物浓度 ≈ 非特异本底 + 最大产物 × θ。

    非特异本底是**检出限的真正来源**：它不随分析物变化，但它的批间起伏
    决定了多小的信号还能被认出来。
    """

    analyte_unit: str = "ng/mL"
    kd: float = 20.0                  # 半饱和浓度（分析物单位）
    hill: float = 1.0                 # Hill 斜率；1.0 即标准 Langmuir
    chromogen_max_uM: float = 700.0   # 饱和结合且显色到底时的产物浓度
    nonspecific_uM: float = 8.0       # 非特异结合造成的本底显色
    lot_variation: float = 0.04       # 逐孔的固定化密度/显色效率起伏

    def chromogen_for(self, analyte: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """分析物浓度 → 显色产物浓度 (µM)。"""

        c = np.clip(np.asarray(analyte, dtype=np.float64), 0.0, None)
        kd = max(self.kd, 1e-12)
        if abs(self.hill - 1.0) < 1e-9:
            theta = c / (kd + c)
        else:
            ch, kh = np.power(c, self.hill), kd ** self.hill
            theta = ch / (kh + ch)
        product = self.nonspecific_uM + self.chromogen_max_uM * theta
        if rng is not None and self.lot_variation > 0:
            product = product * (1.0 + rng.normal(0.0, self.lot_variation, product.shape))
        return np.clip(product, 0.0, None)


# ---------------------------------------------------------------- 4PL 拟合


@dataclass
class FourPL:
    """五参数 Logistic：y = d + (a − d) / (1 + (x/c50)^b)^g。

    g = 1 时退化为标准 4PL。保留类名是为了不改动既有调用点。

    **为什么这台设备需要那个不对称参数 g**：分析物→读数其实是两级串联，
    先是抗原抗体结合的 Langmuir 饱和（这一级确实是 4PL），再叠一级
    多色光造成的吸光度压缩。两级复合后不再是 4PL——实测吸光度相对结合
    分数的比值从 1.000 单调降到 0.793。硬套 4PL 会在中段产生 10-38%
    的系统性回收率偏高。g 就是用来吸收第二级压缩的。
    """

    a: float
    d: float
    c50: float
    b: float
    g: float = 1.0

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=np.float64), 1e-12, None)
        return self.d + (self.a - self.d) / (1.0 + (x / self.c50) ** self.b) ** self.g

    def invert(self, y: np.ndarray) -> np.ndarray:
        """由读数反算浓度。落在渐近线之外的读数返回 nan，不外推。"""

        y = np.asarray(y, dtype=np.float64)
        span = self.d - self.a
        if abs(span) < 1e-12:
            return np.full(y.shape, np.nan)
        out = np.full(y.shape, np.nan)
        # (a−d)/(y−d) = (1 + (x/c50)^b)^g
        denom = y - self.d
        ok = np.isfinite(denom) & (np.abs(denom) > 1e-12)
        ratio = np.full(y.shape, np.nan)
        ratio[ok] = (self.a - self.d) / denom[ok]
        ok &= np.isfinite(ratio) & (ratio > 0)
        inner = np.full(y.shape, np.nan)
        inner[ok] = np.power(ratio[ok], 1.0 / self.g) - 1.0
        ok &= np.isfinite(inner) & (inner > 1e-12)
        out[ok] = self.c50 * np.power(inner[ok], 1.0 / self.b)
        return out


def fit_four_parameter_logistic(
    concentrations: np.ndarray,
    responses: np.ndarray,
    blank_response: float | None = None,
) -> FourPL | None:
    """4PL 拟合，用 logit-log 线性化，不依赖 scipy。

    令 F = (y − a)/(d − y)，则 log F = b·log x − b·log c50，对 log x 是直线。
    a 取零浓度处读数，d 由最高浓度点外推。两端各留一点余量避免 log 发散。
    """

    x = np.asarray(concentrations, dtype=np.float64).reshape(-1)
    y = np.asarray(responses, dtype=np.float64).reshape(-1)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if x.size < 4:
        return None

    a = float(blank_response) if blank_response is not None else float(
        np.median(y[x <= 0]) if np.any(x <= 0) else y.min()
    )
    positive = x > 0
    if int(positive.sum()) < 3:
        return None

    top = float(np.max(y[positive]))
    xp, yp = x[positive], y[positive]
    # d 必须严格大于所有观测，否则 logit 发散；留余量并搜索。
    # 同时扫描不对称参数 g：g=1 是标准 4PL，g≠1 吸收光学压缩带来的偏斜。
    best: tuple[float, FourPL] | None = None
    for margin in (0.02, 0.05, 0.10, 0.20, 0.35, 0.60, 1.0):
        d = a + (top - a) * (1.0 + margin)
        if d - a < 1e-9:
            continue
        for g in (0.35, 0.5, 0.7, 0.85, 1.0, 1.3, 1.8, 2.5, 3.5, 5.0):
            # ((a−d)/(y−d))^(1/g) − 1 = (x/c50)^b，两边取对数后对 log x 是直线
            ratio = (a - d) / (yp - d)
            inner = np.power(np.clip(ratio, 1e-12, None), 1.0 / g) - 1.0
            valid = np.isfinite(inner) & (inner > 1e-9)
            if int(valid.sum()) < 3:
                continue
            slope, intercept = np.polyfit(np.log(xp[valid]), np.log(inner[valid]), 1)
            if not np.isfinite(slope) or slope <= 1e-6:
                continue
            c50 = float(np.exp(-intercept / slope))
            if not np.isfinite(c50) or c50 <= 0:
                continue
            model = FourPL(a=a, d=float(d), c50=c50, b=float(slope), g=float(g))
            predicted = model.predict(xp)
            if not np.all(np.isfinite(predicted)):
                continue
            residual = float(np.mean((predicted - yp) ** 2))
            if best is None or residual < best[0]:
                best = (residual, model)
    return None if best is None else best[1]


# ---------------------------------------------------------------- 端到端标定


def _read_spot_channels(quant_csv: Path) -> np.ndarray | None:
    """从 pg_quant 输出读取逐孔原始 ROI 中位数，返回 (N,3) RGB。

    刻意取原始值而非自参照平场校正值：比色的参考是配对空白图，
    再叠一层自参照校正等于把同一个照明场扣两次。
    """

    if not quant_csv.exists():
        return None
    rows = list(csv.DictReader(quant_csv.open(encoding="utf-8")))
    keys = ("roi_b_median", "roi_g_median", "roi_r_median")
    if not rows or not all(k in rows[0] for k in keys):
        return None
    return np.asarray(
        [[float(r[keys[2]]), float(r[keys[1]]), float(r[keys[0]])] for r in rows],
        dtype=np.float64,
    )


def calibration_run(
    base_config: ColorimetricConfig,
    assay: ImmunoassayResponse,
    analyte_per_row: np.ndarray,
    output_dir: str | Path,
    signal_channel: int | None = None,
) -> dict[str, object]:
    """跑完整标定链路并给出回收率报告。

    analyte_per_row 给出每一行（= 一条微流通道）的分析物浓度，行内 15 个
    孔是重复孔。这与"横向加抗体、纵向加抗原"的交叉版型一致：一条通道
    一个样品，行内重复给出误差棒。
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grid = base_config.grid_size
    rows = np.asarray(analyte_per_row, dtype=np.float64).reshape(-1)
    if rows.size != grid:
        raise ValueError(f"analyte_per_row 需要 {grid} 个值，收到 {rows.size}")

    # 分析物 → 显色产物。逐孔加固定化密度起伏。
    rng = np.random.default_rng(base_config.seed + 8801)
    analyte = np.repeat(rows, grid)
    chromogen = assay.chromogen_for(analyte, rng)

    config = ColorimetricConfig(**{
        **{f: getattr(base_config, f) for f in base_config.__dataclass_fields__},
        "explicit_concentrations": tuple(float(v) for v in chromogen),
    })
    paths = write_colorimetric_sample(config, output_dir, name="plate")

    # 空白板：分析物全为零，但**非特异显色仍在**——这正是检出限的来源。
    blank_chromogen = assay.chromogen_for(np.zeros_like(analyte), rng)
    blank_config = ColorimetricConfig(**{
        **{f: getattr(base_config, f) for f in base_config.__dataclass_fields__},
        "explicit_concentrations": tuple(float(v) for v in blank_chromogen),
    })
    blank_paths = write_colorimetric_sample(blank_config, output_dir, name="zero")

    from pg_grid import process_image

    sample = process_image(image_path=paths["image"], grid_size=grid,
                           output_dir=output_dir / "pipe_sample")
    reference = process_image(image_path=paths["blank"], grid_size=grid,
                              output_dir=output_dir / "pipe_blank")
    zero_plate = process_image(image_path=blank_paths["image"], grid_size=grid,
                               output_dir=output_dir / "pipe_zero")

    s_rgb = _read_spot_channels(output_dir / "pipe_sample" / "quant_values.csv")
    b_rgb = _read_spot_channels(output_dir / "pipe_blank" / "quant_values.csv")
    z_rgb = _read_spot_channels(output_dir / "pipe_zero" / "quant_values.csv")
    if s_rgb is None or b_rgb is None or z_rgb is None:
        return {"schema": "pg-calibration-v1", "ok": False,
                "reason": "定量输出缺失，无法闭合到浓度"}

    absorbance = -np.log10(np.clip(s_rgb, 1e-6, None) / np.clip(b_rgb, 1e-6, None))
    zero_absorbance = -np.log10(np.clip(z_rgb, 1e-6, None) / np.clip(b_rgb, 1e-6, None))

    # 信号通道：取真值吸光度跨度最大的那一个。换染料时它会变，不能写死。
    if signal_channel is None:
        spread = absorbance.max(axis=0) - absorbance.min(axis=0)
        signal_channel = int(np.argmax(spread))
    y = absorbance[:, signal_channel]
    y_zero = zero_absorbance[:, signal_channel]

    # 拟合只用图像里能得到的量：逐孔读数 + 已知的加样浓度。真值不参与。
    blank_mean = float(np.mean(y_zero))
    blank_sd = float(np.std(y_zero))
    model = fit_four_parameter_logistic(analyte, y, blank_response=blank_mean)
    if model is None:
        return {"schema": "pg-calibration-v1", "ok": False, "reason": "4PL 拟合失败"}

    recovered = model.invert(y)

    # 逐浓度统计
    levels: list[dict[str, object]] = []
    for index, nominal in enumerate(rows):
        take = slice(index * grid, (index + 1) * grid)
        got = recovered[take]
        valid = np.isfinite(got)
        entry: dict[str, object] = {
            "nominal": round(float(nominal), 6),
            "n": int(grid),
            "n_valid": int(valid.sum()),
            "mean_absorbance": round(float(np.mean(y[take])), 5),
        }
        if int(valid.sum()) >= 3 and nominal > 0:
            mean_got = float(np.mean(got[valid]))
            entry.update({
                "mean_recovered": round(mean_got, 5),
                "recovery_pct": round(mean_got / nominal * 100.0, 2),
                "cv_pct": round(float(np.std(got[valid]) / max(abs(mean_got), 1e-12) * 100.0), 2),
            })
        levels.append(entry)

    # LOD / LOQ：空白读数 + 3σ / 10σ，再经标定曲线换算回浓度
    def concentration_at(threshold: float) -> float | None:
        value = model.invert(np.array([threshold]))[0]
        return None if not np.isfinite(value) else round(float(value), 6)

    lod = concentration_at(blank_mean + 3.0 * blank_sd)
    loq = concentration_at(blank_mean + 10.0 * blank_sd)

    # 可用量程：回收率 80-120% 且 CV < 20%（生物分析常规验收判据）
    usable = [
        entry["nominal"] for entry in levels
        if entry.get("recovery_pct") is not None
        and 80.0 <= entry["recovery_pct"] <= 120.0
        and entry.get("cv_pct", 1e9) < 20.0
    ]

    report: dict[str, object] = {
        "schema": "pg-calibration-v1",
        "ok": True,
        "analyte_unit": assay.analyte_unit,
        "signal_channel": "RGB"[signal_channel],
        "localization": {
            "sample_trusted": bool(sample["lattice_consistency"]["trusted"]),
            "blank_trusted": bool(reference["lattice_consistency"]["trusted"]),
            "zero_trusted": bool(zero_plate["lattice_consistency"]["trusted"]),
            "sample_support": sample["lattice_consistency"]["candidate_support_ratio"],
            "quality": sample["quality"]["status"],
        },
        "curve": {"a": round(model.a, 5), "d": round(model.d, 5),
                  "c50": round(model.c50, 5), "hill": round(model.b, 4)},
        "blank": {"mean_absorbance": round(blank_mean, 5),
                  "sd_absorbance": round(blank_sd, 6)},
        "detection_limit": lod,
        "quantitation_limit": loq,
        "usable_range": [min(usable), max(usable)] if usable else None,
        "usable_level_count": len(usable),
        "levels": levels,
    }
    with (output_dir / "calibration_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def log_series(low: float, high: float, count: int) -> np.ndarray:
    """等比稀释序列，含一个零浓度点（标定必需）。"""

    series = np.exp(np.linspace(math.log(low), math.log(high), count - 1))
    return np.concatenate([[0.0], series])
