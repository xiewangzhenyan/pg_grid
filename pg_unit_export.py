"""PG-Unit-Export：把定位后的阵列单元逐个裁切为独立小图。

设计目标（对应"零背景、零漏裁、零错裁"三个要求）：

1. 零背景 —— 不用固定半径开窗，而是在每个晶格点附近做极性感知的
   局部分割（Otsu），取包含晶格点的连通域的**紧致外接框**导出；
2. 零漏裁 —— 输出数量恒等于 grid_size^2：个别单元分割失败
   （遮挡/弱对比）时，回退为"成功单元的中位尺寸盒 + 晶格点居中"，
   并在清单中如实标注来源；
3. 零错裁 —— 分割窗口半宽 < 0.5×间距（物理上不可能圈进相邻单元），
   紧致框还要通过尺寸/中心偏移校验，可疑结果宁可回退也不输出错框；
   随裁切输出 crop_manifest.json（逐单元几何与来源）和
   crop_contact_sheet.jpg（按行列排布的拼贴校验图，红框=兜底单元），
   一眼即可核对全部裁切。

用法：
python pg_unit_export.py --result outputs/sample/result.json --output crops/sample
python pg_unit_export.py --image examples/xxx.jpg --grid 15 --output crops/xxx
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pg_grid import process_image, read_image_unicode, write_image_unicode


@dataclass
class CropConfig:
    """裁切参数。

    window_ratio: 分割窗口半宽 / 网格间距。0.45 < 0.5 保证窗口物理上
        触不到相邻单元中心，杜绝"圈进邻居"的错裁。
    max_center_shift_ratio: 分割结果中心相对晶格点的最大偏移 / 间距。
        晶格点来自光束法平差，偏移过大说明分割抓到了别的结构。
    min_size_ratio / max_size_ratio: 紧致框边长相对中位单元尺寸的
        允许区间，越界视为分割失败并回退。
    inset_px: 导出前把紧致框向内收缩的像素数。默认 0（按分割边界导出）；
        对边界有轻微光晕的图可设 1-2 进一步确保零背景。
    """

    window_ratio: float = 0.45
    max_center_shift_ratio: float = 0.20
    min_size_ratio: float = 0.40
    max_size_ratio: float = 1.60
    inset_px: int = 0


def _estimate_pitch(points: np.ndarray, grid_size: int, image_size: float) -> float:
    """从点阵估计网格间距（行/列相邻点欧氏距离中位数）。"""

    if grid_size >= 2 and points.shape[0] == grid_size * grid_size:
        lattice = points.reshape(grid_size, grid_size, 2)
        row_steps = np.linalg.norm(np.diff(lattice, axis=1), axis=2)
        col_steps = np.linalg.norm(np.diff(lattice, axis=0), axis=2)
        pitch = (float(np.median(row_steps)) + float(np.median(col_steps))) / 2.0
        if math.isfinite(pitch) and pitch > 4.0:
            return pitch
    return float(image_size) / max(grid_size + 1, 2)


def _segment_unit_box(
    gray: np.ndarray,
    cx: float,
    cy: float,
    half: int,
    polarity: str,
) -> tuple[int, int, int, int] | None:
    """在晶格点附近分割单元，返回紧致外接框 (x0, y0, x1, y1)，失败返回 None。

    框为半开区间 [x0, x1) × [y0, y1)，即 crop = image[y0:y1, x0:x1]。
    """

    height, width = gray.shape[:2]
    wx0, wy0 = max(0, int(round(cx)) - half), max(0, int(round(cy)) - half)
    wx1, wy1 = min(width, int(round(cx)) + half + 1), min(height, int(round(cy)) + half + 1)
    window = gray[wy0:wy1, wx0:wx1]
    if window.size < 25:
        return None

    # 单元相对背景的极性已由管线判定；分割在"单元为前景"的表示下进行。
    foreground = (255 - window) if polarity == "dark" else window
    # 窗口内对比不足（如单元被抹掉）时 Otsu 无意义，直接判失败。
    if int(foreground.max()) - int(foreground.min()) < 12:
        return None
    _, mask = cv2.threshold(foreground, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return None

    # 首选包含晶格点像素的连通域；晶格点像素恰在背景（极少数）时，
    # 取质心离晶格点最近的连通域。
    local_x, local_y = int(round(cx)) - wx0, int(round(cy)) - wy0
    local_x = int(np.clip(local_x, 0, window.shape[1] - 1))
    local_y = int(np.clip(local_y, 0, window.shape[0] - 1))
    chosen = int(labels[local_y, local_x])
    if chosen == 0:
        best_index, best_distance = 0, float("inf")
        for index in range(1, component_count):
            distance = math.hypot(centroids[index][0] - local_x, centroids[index][1] - local_y)
            if stats[index][4] >= 9 and distance < best_distance:
                best_index, best_distance = index, distance
        if best_index == 0:
            return None
        chosen = best_index

    x, y, w, h, area = stats[chosen]
    if area < 9:
        return None
    # 触碰窗口边界说明单元没有被窗口完整包住（或抓到了穿窗结构），
    # 这样的框不可信。
    if x <= 0 or y <= 0 or x + w >= window.shape[1] or y + h >= window.shape[0]:
        return None

    return wx0 + int(x), wy0 + int(y), wx0 + int(x + w), wy0 + int(y + h)


def extract_unit_boxes(
    rectified: np.ndarray,
    points: np.ndarray,
    grid_size: int,
    polarity: str,
    config: CropConfig | None = None,
) -> tuple[list[tuple[int, int, int, int]], list[str], dict[str, object]]:
    """计算全部单元的裁切框。

    两遍策略：第一遍逐单元分割并收集尺寸；第二遍用"中位尺寸区间 +
    中心偏移上限"校验每个分割框，不合格者回退为中位尺寸盒（晶格点居中）。
    返回 (框列表[半开区间], 来源列表, 统计信息)。
    """

    if config is None:
        config = CropConfig()

    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    expected = grid_size * grid_size
    if points.shape[0] != expected:
        raise ValueError(f"点数应为 {expected}，实际为 {points.shape[0]}")

    height, width = rectified.shape[:2]
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    pitch = _estimate_pitch(points, grid_size, float(min(width, height)))
    half = max(6, int(round(pitch * config.window_ratio)))
    max_shift = pitch * config.max_center_shift_ratio

    raw_boxes: list[tuple[int, int, int, int] | None] = []
    for cx, cy in points:
        raw_boxes.append(_segment_unit_box(gray, float(cx), float(cy), half, polarity))

    seg_widths = [b[2] - b[0] for b in raw_boxes if b is not None]
    seg_heights = [b[3] - b[1] for b in raw_boxes if b is not None]
    if seg_widths:
        median_w = int(round(float(np.median(seg_widths))))
        median_h = int(round(float(np.median(seg_heights))))
    else:
        # 极端情况：没有任何单元分割成功。用 0.5×间距的保守盒兜底，
        # 保证输出数量，statistics 中如实报告。
        median_w = median_h = max(6, int(round(pitch * 0.5)))

    boxes: list[tuple[int, int, int, int]] = []
    sources: list[str] = []
    for index, (cx, cy) in enumerate(points):
        box = raw_boxes[index]
        accepted = False
        if box is not None:
            bw, bh = box[2] - box[0], box[3] - box[1]
            bcx, bcy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
            size_ok = (
                config.min_size_ratio * median_w <= bw <= config.max_size_ratio * median_w
                and config.min_size_ratio * median_h <= bh <= config.max_size_ratio * median_h
            )
            shift_ok = math.hypot(bcx - cx, bcy - cy) <= max_shift
            accepted = size_ok and shift_ok

        if accepted:
            final = box
            sources.append("segmented")
        else:
            # 中位尺寸盒兜底：尺寸取全体成功单元的中位数，中心取晶格点。
            x0 = int(round(cx - median_w / 2.0))
            y0 = int(round(cy - median_h / 2.0))
            final = (x0, y0, x0 + median_w, y0 + median_h)
            sources.append("fallback_median_box")

        inset = max(0, int(config.inset_px))
        x0 = int(np.clip(final[0] + inset, 0, width - 2))
        y0 = int(np.clip(final[1] + inset, 0, height - 2))
        x1 = int(np.clip(final[2] - inset, x0 + 1, width))
        y1 = int(np.clip(final[3] - inset, y0 + 1, height))
        boxes.append((x0, y0, x1, y1))

    stats: dict[str, object] = {
        "pitch_px": round(pitch, 4),
        "median_unit_px": [median_w, median_h],
        "segmented_count": int(sources.count("segmented")),
        "fallback_count": int(sources.count("fallback_median_box")),
    }
    return boxes, sources, stats


def _render_contact_sheet(
    rectified: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    sources: list[str],
    grid_size: int,
    cell: int = 56,
) -> np.ndarray:
    """把全部裁切按行列拼贴成一张校验图：绿框=分割成功，红框=兜底。"""

    margin = 28
    sheet = np.full((margin + grid_size * cell, margin + grid_size * cell, 3), 245, dtype=np.uint8)
    for index, (box, source) in enumerate(zip(boxes, sources)):
        row, col = index // grid_size, index % grid_size
        crop = rectified[box[1]:box[3], box[0]:box[2]]
        if crop.size == 0:
            continue
        inner = cell - 8
        resized = cv2.resize(crop, (inner, inner), interpolation=cv2.INTER_NEAREST)
        y0 = margin + row * cell + 4
        x0 = margin + col * cell + 4
        sheet[y0:y0 + inner, x0:x0 + inner] = resized
        color = (0, 170, 0) if source == "segmented" else (0, 0, 255)
        cv2.rectangle(sheet, (x0 - 2, y0 - 2), (x0 + inner + 1, y0 + inner + 1), color, 2)
    for k in range(grid_size):
        cv2.putText(sheet, str(k), (margin + k * cell + cell // 2 - 6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
        cv2.putText(sheet, str(k), (6, margin + k * cell + cell // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (90, 90, 90), 1)
    return sheet


def export_unit_crops(
    rectified: np.ndarray,
    points: np.ndarray,
    grid_size: int,
    polarity: str,
    output_dir: str | Path,
    config: CropConfig | None = None,
) -> dict[str, object]:
    """裁切并导出全部单元小图，返回清单（同时写入 crop_manifest.json）。

    输出：
    - unit_r{row:02d}_c{col:02d}.png：逐单元紧致小图（PNG 无损，
      避免 JPEG 压缩污染后续定量）；
    - crop_manifest.json：逐单元几何、来源与统计；
    - crop_contact_sheet.jpg：拼贴校验图。
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    boxes, sources, stats = extract_unit_boxes(rectified, points, grid_size, polarity, config)

    units: list[dict[str, object]] = []
    for index, (box, source) in enumerate(zip(boxes, sources)):
        row, col = index // grid_size, index % grid_size
        file_name = f"unit_r{row:02d}_c{col:02d}.png"
        crop = rectified[box[1]:box[3], box[0]:box[2]]
        write_image_unicode(output_dir / file_name, crop)
        units.append(
            {
                "row": int(row), "col": int(col),
                "x": round(float(points.reshape(-1, 2)[index][0]), 4),
                "y": round(float(points.reshape(-1, 2)[index][1]), 4),
                "box": [int(v) for v in box],
                "width": int(box[2] - box[0]),
                "height": int(box[3] - box[1]),
                "source": source,
                "file": file_name,
            }
        )

    manifest: dict[str, object] = {
        "schema": "pg-unit-export-v1",
        "grid_size": int(grid_size),
        "unit_polarity": polarity,
        "unit_count": len(units),
        **stats,
        "units": units,
    }
    with (output_dir / "crop_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    sheet = _render_contact_sheet(rectified, boxes, sources, grid_size)
    write_image_unicode(output_dir / "crop_contact_sheet.jpg", sheet)
    return manifest


def export_units_from_result(result_path: str | Path, output_dir: str | Path,
                             config: CropConfig | None = None) -> dict[str, object]:
    """基于已有 result.json（含 grid_points 与矫正图路径）执行裁切导出。"""

    result_path = Path(result_path)
    with result_path.open("r", encoding="utf-8") as f:
        result = json.load(f)

    grid_size = int(result["grid_size"])
    points = np.asarray(
        [[float(p["x"]), float(p["y"])] for p in result["grid_points"]], dtype=np.float64
    )
    rectified_path = result.get("outputs", {}).get("rectified_chip")
    if not rectified_path:
        raise ValueError("result.json 中没有 rectified_chip 路径")
    rectified = read_image_unicode(rectified_path)

    polarity = result.get("unit_polarity")
    if polarity not in ("dark", "bright"):
        # 旧版 result.json 没有极性字段：用内部均值/中位数统计估计。
        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        interior = gray[int(height * 0.1):int(height * 0.9), int(width * 0.1):int(width * 0.9)]
        polarity = "dark" if float(interior.mean()) < float(np.median(interior)) else "bright"

    return export_unit_crops(rectified, points, grid_size, polarity, output_dir, config)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="PG-Unit-Export 逐单元裁切导出")
    parser.add_argument("--result", default=None, help="已有 result.json 路径")
    parser.add_argument("--image", default=None, help="原始图像路径（与 --grid 搭配，先跑定位再裁切）")
    parser.add_argument("--grid", type=int, default=None, choices=[10, 15], help="阵列规格（--image 模式必填）")
    parser.add_argument("--output", required=True, help="小图输出目录")
    parser.add_argument("--inset", type=int, default=0, help="紧致框向内收缩像素数（默认 0）")
    return parser.parse_args()


def main() -> None:
    """命令行入口：支持基于已有定位结果或从原图一步到位。"""

    args = parse_args()
    config = CropConfig(inset_px=args.inset)

    if args.result:
        manifest = export_units_from_result(args.result, args.output, config)
    elif args.image and args.grid:
        pipeline_dir = Path(args.output) / "pipeline"
        result = process_image(image_path=args.image, grid_size=args.grid, output_dir=pipeline_dir)
        manifest = export_units_from_result(pipeline_dir / "result.json", args.output, config)
    else:
        raise SystemExit("请提供 --result，或同时提供 --image 与 --grid")

    print("PG-Unit-Export 完成")
    print(f"单元数量: {manifest['unit_count']}  分割成功: {manifest['segmented_count']}  兜底: {manifest['fallback_count']}")
    print(f"中位单元尺寸: {manifest['median_unit_px']} px  间距: {manifest['pitch_px']} px")
    print(f"输出目录: {Path(args.output).resolve()}")
    print("校验图: crop_contact_sheet.jpg（绿框=分割成功，红框=兜底）")


if __name__ == "__main__":
    main()
