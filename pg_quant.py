"""PG-Quant V1.0：基于阵列点位的逐单元颜色/亮度定量与单元级质量控制。

定位层（pg_grid）回答"每个阵列单元在哪里"；本模块回答"每个单元读数是多少、
这个读数能不能信"。核心设计：

1. 圆形信号 ROI + 环形背景邻域（annulus）：背景环取自单元与相邻单元之间的
   空隙，为每个单元提供局部背景参照；
2. 自参照平场校正：全部单元的背景环中位数本身就是光照场的采样，
   用二阶多项式拟合光照场后做乘性校正，不依赖外部标定物；
3. 两套特征：颜色特征集（raw/校正后 RGB、色度、Lab）与亮度/强度特征集
   （带符号信号、相对信号、场校正信号、积分信号、SNR）；
4. 单元级质量标记：过曝、欠曝、低信噪比、ROI 越界、ROI 异质、
   背景异常，汇总为 quant_reliable 布尔结论。

本模块不依赖 pg_grid，可独立用于任何"图像 + 点位"输入。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class QuantConfig:
    """PG-Quant 参数。

    roi_radius_ratio: 信号 ROI 半径 / 网格间距。取 0.18，比可视化 ROI（0.23）
        更保守：定量的中位数必须落在单元内部，宁可少采样也不混入背景。
    annulus_inner_ratio / annulus_outer_ratio: 背景环内外半径 / 网格间距。
        内半径 0.30 与信号 ROI 之间留出隔离带，避免单元边缘渐变污染背景；
        外半径 0.44 < 0.5，保证背景环不触及相邻单元。
    saturation_level: 饱和判定灰度，与 pg_grid 现有质量控制（>=250）一致。
    saturation_ratio_limit: ROI 内饱和像素比例超过该值即判过曝。
    under_exposed_level: ROI 中位灰度低于该值即判欠曝。
    snr_min: 低信噪比阈值，采用三倍噪声准则。
    border_clip_limit: ROI 或背景环被图像边界裁掉的比例上限。
    contamination_limit: ROI 内"污染像素"比例上限。污染像素 = 偏离 ROI
        中位数超过噪声容差（max(4×背景噪声, 8 个灰度级)）的像素。
        选它而不是 IQR/MAD 是刻意的：中位数读数在污染 <50% 时仍然正确，
        ROI 略大于单元导致的边缘背景混入（实测 ≤25%）属于正常情况；
        35% 已逼近中位数翻转边界，且大概率意味着结构性异物而非边缘混入。
        注意：污染接近 100%（完全遮挡）时该指标会静默——那种情况由
        background_anomaly（异物通常同时污染背景环）兜底。
    bg_anomaly_z: 背景环中位数相对光照场预测的稳健 z 分数上限，超过说明
        局部背景被阴影/高光/异物污染。
    """

    roi_radius_ratio: float = 0.18
    annulus_inner_ratio: float = 0.30
    annulus_outer_ratio: float = 0.44
    saturation_level: int = 250
    saturation_ratio_limit: float = 0.05
    under_exposed_level: float = 8.0
    snr_min: float = 3.0
    border_clip_limit: float = 0.05
    contamination_limit: float = 0.35
    bg_anomaly_z: float = 3.5


def _estimate_pitch(points: np.ndarray, grid_size: int, image_size: float) -> float:
    """从点位阵列估计网格间距（行内水平间距与列内垂直间距的中位数均值）。"""

    if grid_size >= 2 and points.shape[0] == grid_size * grid_size:
        lattice = points.reshape(grid_size, grid_size, 2)
        dx = np.abs(np.diff(lattice[:, :, 0], axis=1))
        dy = np.abs(np.diff(lattice[:, :, 1], axis=0))
        pitch = (float(np.median(dx)) + float(np.median(dy))) / 2.0
        if math.isfinite(pitch) and pitch > 2.0:
            return pitch
    # 点数异常时的保守兜底：按图像尺寸均分。
    return float(image_size) / max(grid_size + 1, 2)


def _extract_region_pixels(
    image: np.ndarray,
    gray: np.ndarray,
    cx: float,
    cy: float,
    roi_radius: float,
    annulus_inner: float,
    annulus_outer: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """提取单个单元的 ROI 与背景环像素。

    返回 (ROI 彩色像素 N×3, ROI 灰度, 背景环彩色像素, 背景环灰度,
    ROI 越界比例, 背景环越界比例)。越界比例 = 理想掩膜中落在图像外的部分。
    """

    height, width = gray.shape[:2]
    reach = int(math.ceil(annulus_outer)) + 1
    xs = np.arange(int(math.floor(cx)) - reach, int(math.floor(cx)) + reach + 1)
    ys = np.arange(int(math.floor(cy)) - reach, int(math.floor(cy)) + reach + 1)
    grid_x, grid_y = np.meshgrid(xs, ys)
    distance = np.hypot(grid_x - cx, grid_y - cy)

    roi_mask = distance <= roi_radius
    annulus_mask = (distance > annulus_inner) & (distance <= annulus_outer)
    inside = (grid_x >= 0) & (grid_x < width) & (grid_y >= 0) & (grid_y < height)

    roi_total = int(roi_mask.sum())
    annulus_total = int(annulus_mask.sum())
    roi_valid_mask = roi_mask & inside
    annulus_valid_mask = annulus_mask & inside
    roi_clip = 1.0 - int(roi_valid_mask.sum()) / max(roi_total, 1)
    annulus_clip = 1.0 - int(annulus_valid_mask.sum()) / max(annulus_total, 1)

    roi_xy = (grid_y[roi_valid_mask], grid_x[roi_valid_mask])
    annulus_xy = (grid_y[annulus_valid_mask], grid_x[annulus_valid_mask])
    roi_bgr = image[roi_xy].astype(np.float64).reshape(-1, 3)
    roi_gray = gray[roi_xy].astype(np.float64).ravel()
    annulus_bgr = image[annulus_xy].astype(np.float64).reshape(-1, 3)
    annulus_gray = gray[annulus_xy].astype(np.float64).ravel()
    return roi_bgr, roi_gray, annulus_bgr, annulus_gray, float(roi_clip), float(annulus_clip)


def _fit_illumination_field(
    centers: np.ndarray,
    values: np.ndarray,
    image_size: float,
) -> tuple[np.ndarray, str]:
    """用二阶多项式拟合光照场，返回各单元中心处的场预测值。

    values 是每个单元背景环的中位数（逐通道），它们本身就是光照场的采样。
    做一轮残差剔除的鲁棒重拟合；单元太少或拟合退化时回退为常数场。
    """

    unit_count = centers.shape[0]
    channel_count = values.shape[1]
    constant = np.tile(np.median(values, axis=0), (unit_count, 1))
    if unit_count < 12:
        # 二阶模型有 6 个系数，样本不足两倍系数时拟合不可靠。
        return constant, "constant_fallback"

    # 坐标归一化到 [-1, 1]，避免高次项数值病态。
    u = centers[:, 0] / float(image_size) * 2.0 - 1.0
    v = centers[:, 1] / float(image_size) * 2.0 - 1.0
    design = np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], axis=1)

    predictions = np.empty_like(values)
    for channel in range(channel_count):
        target = values[:, channel]
        coeffs, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
        if rank < design.shape[1]:
            return constant, "constant_fallback"
        residual = target - design @ coeffs
        sigma = 1.4826 * float(np.median(np.abs(residual - np.median(residual)))) + 1e-6
        keep = np.abs(residual) <= 3.0 * sigma
        if int(keep.sum()) >= max(12, int(unit_count * 0.6)):
            coeffs, _, _, _ = np.linalg.lstsq(design[keep], target[keep], rcond=None)
        predictions[:, channel] = design @ coeffs

    # 预测出现非正值或过度外推说明拟合失真，此时校正弊大于利。
    medians = np.median(predictions, axis=0)
    if np.any(predictions <= 0.3 * np.maximum(medians, 1e-6)):
        return constant, "constant_fallback"
    return predictions, "poly2"


def quantify_rectified(
    rectified: np.ndarray,
    points: np.ndarray,
    grid_size: int,
    config: QuantConfig | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """对矫正图上的阵列点做逐单元定量。

    返回 (逐单元记录列表, 元信息)。记录与元信息全部为原生类型，可直接
    写入 JSON/CSV。点位数量必须等于 grid_size*grid_size（与 grid_points 约定一致）。
    """

    if rectified.ndim != 3:
        raise ValueError("quantify_rectified 需要 BGR 彩色图像")
    if config is None:
        config = QuantConfig()

    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    expected = grid_size * grid_size
    if points.shape[0] != expected:
        raise ValueError(f"点数应为 {expected}，实际为 {points.shape[0]}")

    height, width = rectified.shape[:2]
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    image_size = float(min(width, height))

    pitch = _estimate_pitch(points, grid_size, image_size)
    roi_radius = max(2.0, pitch * config.roi_radius_ratio)
    annulus_inner = max(roi_radius + 1.0, pitch * config.annulus_inner_ratio)
    annulus_outer = max(annulus_inner + 1.5, pitch * config.annulus_outer_ratio)

    # 第一遍：提取每个单元的 ROI / 背景环统计。
    per_unit: list[dict[str, object]] = []
    bg_medians_bgr = np.zeros((expected, 3), dtype=np.float64)
    bg_medians_gray = np.zeros(expected, dtype=np.float64)
    for index, (cx, cy) in enumerate(points):
        roi_bgr, roi_gray, ann_bgr, ann_gray, roi_clip, ann_clip = _extract_region_pixels(
            rectified, gray, float(cx), float(cy), roi_radius, annulus_inner, annulus_outer
        )
        out_of_bounds = roi_clip > config.border_clip_limit or ann_clip > config.border_clip_limit
        if roi_gray.size == 0 or ann_gray.size == 0:
            # 完全越界：所有统计置零并强制标记，保持记录数量恒定。
            per_unit.append(
                {
                    "roi_bgr_median": np.zeros(3), "roi_gray_median": 0.0,
                    "ann_bgr_median": np.zeros(3), "ann_gray_median": 0.0,
                    "contamination": 0.0, "ann_sigma": 0.0,
                    "saturation_ratio": 0.0, "integrated": 0.0,
                    "out_of_bounds": True,
                }
            )
            continue

        roi_bgr_median = np.median(roi_bgr, axis=0)
        ann_bgr_median = np.median(ann_bgr, axis=0)
        roi_gray_median = float(np.median(roi_gray))
        ann_gray_median = float(np.median(ann_gray))
        ann_sigma = 1.4826 * float(np.median(np.abs(ann_gray - ann_gray_median)))
        # 污染比例：偏离 ROI 中位数超过噪声容差的像素占比（见 QuantConfig 注释）。
        tolerance = max(4.0 * ann_sigma, 8.0)
        contamination = float((np.abs(roi_gray - roi_gray_median) > tolerance).mean())
        saturation_ratio = float((roi_bgr >= config.saturation_level).any(axis=1).mean())
        integrated = float((roi_gray - ann_gray_median).sum())

        bg_medians_bgr[index] = ann_bgr_median
        bg_medians_gray[index] = ann_gray_median
        per_unit.append(
            {
                "roi_bgr_median": roi_bgr_median, "roi_gray_median": roi_gray_median,
                "ann_bgr_median": ann_bgr_median, "ann_gray_median": ann_gray_median,
                "contamination": contamination,
                "ann_sigma": ann_sigma, "saturation_ratio": saturation_ratio,
                "integrated": integrated, "out_of_bounds": out_of_bounds,
            }
        )

    # 第二遍：用全部背景环中位数拟合光照场（灰度 + 三通道一起拟合）。
    valid = np.asarray([not u["out_of_bounds"] for u in per_unit], dtype=bool)
    field_values = np.column_stack([bg_medians_bgr, bg_medians_gray])
    if int(valid.sum()) >= 12:
        field_pred_valid, field_model = _fit_illumination_field(
            points[valid], field_values[valid], image_size
        )
        field_pred = np.tile(np.median(field_pred_valid, axis=0), (expected, 1))
        field_pred[valid] = field_pred_valid
    else:
        field_pred = np.tile(np.median(field_values[valid] if valid.any() else field_values, axis=0), (expected, 1))
        field_model = "constant_fallback"

    field_mean = field_pred[valid].mean(axis=0) if valid.any() else field_pred.mean(axis=0)
    gray_pred = field_pred[:, 3]
    uniformity = 1.0
    if valid.any():
        gp = gray_pred[valid]
        uniformity = float(np.clip(1.0 - (gp.max() - gp.min()) / max(gp.mean(), 1e-6), 0.0, 1.0))

    # 背景异常的稳健尺度：全部单元背景残差的 MAD，下限 1 个灰度级
    #（低于一个量化级的偏差没有物理意义，防止零噪声合成图误标）。
    bg_residual = bg_medians_gray - gray_pred
    if valid.any():
        med_res = float(np.median(bg_residual[valid]))
        resid_sigma = max(1.0, 1.4826 * float(np.median(np.abs(bg_residual[valid] - med_res))))
    else:
        med_res, resid_sigma = 0.0, 1.0

    # 第三遍：组装逐单元特征与质量标记。
    records: list[dict[str, object]] = []
    flag_counts: dict[str, int] = {}
    snr_values: list[float] = []
    for index, unit in enumerate(per_unit):
        row, col = index // grid_size, index % grid_size
        cx, cy = points[index]
        roi_bgr_median = np.asarray(unit["roi_bgr_median"], dtype=np.float64)
        ann_bgr_median = np.asarray(unit["ann_bgr_median"], dtype=np.float64)
        roi_gray_median = float(unit["roi_gray_median"])
        ann_gray_median = float(unit["ann_gray_median"])

        # 乘性平场校正：scale = 场均值 / 该单元处场预测。
        scale_bgr = field_mean[:3] / np.maximum(field_pred[index, :3], 1e-6)
        scale_gray = float(field_mean[3] / max(field_pred[index, 3], 1e-6))
        corr_bgr = roi_bgr_median * scale_bgr
        corr_gray = roi_gray_median * scale_gray

        signal_gray = roi_gray_median - ann_gray_median
        signal_ratio = signal_gray / max(ann_gray_median, 1e-6)
        corr_signal_gray = signal_gray * scale_gray
        snr = min(abs(signal_gray) / (float(unit["ann_sigma"]) + 1e-6), 9999.0)

        # 色度（比例型颜色特征）与 Lab 都基于校正后的颜色计算。
        chroma_total = float(corr_bgr.sum()) + 1e-6
        corr_pixel = np.clip(corr_bgr, 0, 255).astype(np.uint8).reshape(1, 1, 3)
        lab_l, lab_a, lab_b = [float(v) for v in cv2.cvtColor(corr_pixel, cv2.COLOR_BGR2LAB).reshape(3)]

        contamination = float(unit["contamination"])

        flags: list[str] = []
        if float(unit["saturation_ratio"]) > config.saturation_ratio_limit:
            flags.append("saturated")
        if roi_gray_median < config.under_exposed_level:
            flags.append("under_exposed")
        if snr < config.snr_min:
            flags.append("low_snr")
        if bool(unit["out_of_bounds"]):
            flags.append("roi_out_of_bounds")
        if contamination > config.contamination_limit:
            flags.append("non_uniform")
        if abs(float(bg_residual[index]) - med_res) > config.bg_anomaly_z * resid_sigma:
            flags.append("background_anomaly")

        for flag in flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        reliable = len(flags) == 0
        snr_values.append(snr)

        records.append(
            {
                "row": int(row), "col": int(col),
                "x": round(float(cx), 4), "y": round(float(cy), 4),
                "roi_b_median": round(float(roi_bgr_median[0]), 4),
                "roi_g_median": round(float(roi_bgr_median[1]), 4),
                "roi_r_median": round(float(roi_bgr_median[2]), 4),
                "roi_gray_median": round(roi_gray_median, 4),
                "bg_b_median": round(float(ann_bgr_median[0]), 4),
                "bg_g_median": round(float(ann_bgr_median[1]), 4),
                "bg_r_median": round(float(ann_bgr_median[2]), 4),
                "bg_gray_median": round(ann_gray_median, 4),
                "corr_b": round(float(corr_bgr[0]), 4),
                "corr_g": round(float(corr_bgr[1]), 4),
                "corr_r": round(float(corr_bgr[2]), 4),
                "corr_gray": round(corr_gray, 4),
                "chroma_b": round(float(corr_bgr[0]) / chroma_total, 6),
                "chroma_g": round(float(corr_bgr[1]) / chroma_total, 6),
                "chroma_r": round(float(corr_bgr[2]) / chroma_total, 6),
                "lab_l": round(lab_l, 4), "lab_a": round(lab_a, 4), "lab_b": round(lab_b, 4),
                "signal_gray": round(signal_gray, 4),
                "signal_ratio": round(signal_ratio, 6),
                "corr_signal_gray": round(corr_signal_gray, 4),
                "integrated_signal_gray": round(float(unit["integrated"]), 2),
                "snr": round(snr, 4),
                "saturation_ratio": round(float(unit["saturation_ratio"]), 6),
                "roi_contamination": round(contamination, 6),
                "flags": flags,
                "quant_reliable": bool(reliable),
            }
        )

    reliable_count = sum(1 for r in records if r["quant_reliable"])
    meta: dict[str, object] = {
        "schema": "pg-quant-v1",
        "grid_size": int(grid_size),
        "unit_count": int(expected),
        "pitch_px": round(pitch, 4),
        "roi_radius_px": round(roi_radius, 4),
        "annulus_inner_px": round(annulus_inner, 4),
        "annulus_outer_px": round(annulus_outer, 4),
        "illumination_model": field_model,
        "illumination_uniformity": round(uniformity, 6),
        "summary": {
            "reliable_count": int(reliable_count),
            "reliable_ratio": round(reliable_count / expected, 6),
            "mean_abs_snr": round(float(np.mean(snr_values)) if snr_values else 0.0, 4),
            "flag_counts": flag_counts,
        },
    }
    return records, meta


def write_quant_outputs(
    output_dir: str | Path,
    records: list[dict[str, object]],
    meta: dict[str, object],
) -> dict[str, str]:
    """写出 quant_values.csv 与 quant_result.json，返回两个文件路径。"""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not records:
        raise ValueError("没有定量记录可写出")

    csv_path = output_dir / "quant_values.csv"
    json_path = output_dir / "quant_result.json"

    csv_rows = []
    for record in records:
        row = dict(record)
        # CSV 中 flags 以分号连接、布尔转 0/1，便于表格工具直接使用。
        row["flags"] = ";".join(record["flags"])
        row["quant_reliable"] = int(bool(record["quant_reliable"]))
        csv_rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    payload = dict(meta)
    payload["units"] = records
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {"quant_values_csv": str(csv_path), "quant_result_json": str(json_path)}
