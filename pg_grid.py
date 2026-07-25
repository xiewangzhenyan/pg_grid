"""PG-Grid：物理先验约束的规则阵列阵列定位与定量 demo。

当前版本是 V1.4：OpenCV + 物理规则网格 + 局部微调 + 全局晶格一致性校正。
局部精修之后会用鲁棒仿射晶格模型（见 enforce_lattice_consistency）把被
细长暗线、局部阴影或高光拉偏的点吸附回规则晶格位置。
后续如果要加入 MobileSAM/SAM2/轻量关键点网络，只需要替换
`detect_chip_region()` 这一层，后面的透视矫正、网格生成、ROI 定量
都可以复用。
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from pg_quant import quantify_rectified, write_quant_outputs
from pg_quant_viz import write_quant_visualizations


@dataclass
class ChipRegion:
    """目标平面区域定位结果。

    points 的顺序固定为：左上、右上、右下、左下。
    这个顺序对透视变换非常重要，否则矫正后的目标平面会翻转或扭曲。
    """

    points: np.ndarray
    score: float
    method: str


@dataclass
class PipelineConfig:
    """PG-Grid pipeline 参数。

    grid_size: 阵列边长，当前支持 10 或 15，也可以测试小网格。
    rectified_size: 透视矫正后的标准图像边长；None 时自动按 grid_size 估算。
    margin_ratio: 网格点距离矫正图边缘的比例，避免点落在目标平面边框/流道边界。
    roi_radius_ratio: ROI 半径占网格间距的比例。
    """

    grid_size: int
    rectified_size: int | None = None
    margin_ratio: float = 0.085
    roi_radius_ratio: float = 0.23


def read_image_unicode(image_path: str | Path) -> np.ndarray:
    """读取带中文路径的图像。

    OpenCV 的 imread 在某些 Windows 环境对中文路径支持不稳定，
    因此这里使用 np.fromfile + cv2.imdecode。
    """

    path = Path(image_path)
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"无法读取图像：{path}")
    return image


def write_image_unicode(image_path: str | Path, image: np.ndarray) -> None:
    """写入带中文路径的图像。"""

    path = Path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, image)
    if not ok:
        raise RuntimeError(f"图像编码失败：{path}")
    encoded.tofile(str(path))


def order_quad_points(points: np.ndarray) -> np.ndarray:
    """把四边形点排序为左上、右上、右下、左下。

    规则：
    - x+y 最小的是左上，x+y 最大的是右下；
    - x-y 最大的是右上，x-y 最小的是左下。
    """

    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = pts[:, 0] - pts[:, 1]
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmax(diffs)]
    ordered[3] = pts[np.argmin(diffs)]
    return ordered


def _clip_rect(x: int, y: int, w: int, h: int, image_w: int, image_h: int) -> tuple[int, int, int, int]:
    """把矩形裁剪到图像边界内。"""

    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(image_w, x + w)
    y1 = min(image_h, y + h)
    return x0, y0, max(1, x1 - x0), max(1, y1 - y0)


def detect_chip_region(image: np.ndarray) -> ChipRegion:
    """自动定位目标平面亮区的近似四角。

    当前 V1 不使用神经网络，而是用亮度阈值 + 形态学闭运算把 均匀背景下的
    目标平面区域合并成一个主轮廓。这个函数就是后续接入轻量神经网络的替换点。
    """

    if image.ndim != 3:
        raise ValueError("detect_chip_region 需要 BGR 彩色图像")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # 阈值策略必须与"目标占整图多大比例"无关：用户裁掉四周背景是完全
    # 合理的操作，但会把目标占比从百分之十几推到百分之八十。
    #
    # 高分位阈值按定义只保留最亮的固定比例像素，隐含"目标只占一小部分"
    # 的假设：背景被裁掉后阈值被迫抬高，掩膜会切在面板内部而不是面板
    # 边界，检测框严重缩水。Otsu 阈值最大化类间方差、不预设面积比例，
    # 对"暗背景 + 亮面板"这类双峰分布始终落在两峰之间。
    #
    # 因此这里同时评估两种阈值的全部候选并统一评分，而不是"高分位档
    # 有候选就直接返回"（那样 Otsu 档永远不会被评估——这正是裁切后
    # 定位塌陷的根因）。
    percentile_threshold = float(np.percentile(blurred, 94))
    otsu_threshold, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    attempts: list[tuple[float, float, str]] = [
        # 高分位档：保留历史行为，面积上限 20%（真实成像视野图中目标约占
        # 5%-12%，更大的候选通常是成像视野圆环或暗箱反光区域）。
        (max(22.0, percentile_threshold), 0.20, "opencv_bright_region"),
        # Otsu 档：面积无关阈值，上限放宽到 92% 以接受被裁到几乎只剩目标的图。
        (max(22.0, min(percentile_threshold, float(otsu_threshold))), 0.92, "opencv_bright_region_wide"),
    ]

    kernel_size = max(7, int(min(width, height) * 0.008))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    image_area = width * height

    candidates: list[tuple[float, np.ndarray, str]] = []
    for threshold, max_area_ratio, method in attempts:
        mask = (blurred > threshold).astype(np.uint8) * 255

        # 合并孔洞、流道、局部反光，但不要把圆形视野外圈也合进去。
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            # 面积下限 1%：一个能容纳整个阵列的目标面板不可能更小。
            # 这条下限专门挡住"只有少数单元很亮时它们连成的小块"——实测
            # 荧光样本里这种误检只占整图 0.6%，一旦被采用，真值几乎全部
            # 落到矫正图外。实际场景的占比都远高于此（实拍 12%-14%、
            # 合成样例约 46%、基准场景 5.4%），因此这条门限不会误伤。
            if area < image_area * 0.010 or area > image_area * max_area_ratio:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)
            if not 0.35 <= aspect <= 2.8:
                continue

            rect = cv2.minAreaRect(contour)
            rect_area = float(rect[1][0] * rect[1][1])
            if rect_area < 1.0:
                continue
            # 矩形填充度：目标平面是矩形，阈值切在边界上时轮廓接近矩形；
            # 切在目标内部时轮廓沿亮度等值线破碎，填充度显著下降。
            # 这个判据只看形状，与目标占图比例无关，正是裁切场景所需。
            fill_ratio = float(area / rect_area)

            rect_mask = np.zeros_like(gray, dtype=np.uint8)
            cv2.drawContours(rect_mask, [contour], -1, 255, thickness=-1)
            mean_brightness = cv2.mean(gray, mask=rect_mask)[0]
            # 分数同时考虑面积、亮度和形状完整性：
            # 面积和亮度避免选到暗箱圆环反光，填充度避免选到目标内部的碎块。
            score = math.sqrt(area) * (mean_brightness + 1.0) * (0.35 + 0.65 * fill_ratio)
            candidates.append((score, contour, method))

    if not candidates:
        # 没有连续亮区：荧光/自发光成像下面板本身不发光，只有单元发射，
        # 图中最大的连续亮区就是单个发光点，面积远低于目标区域门限。
        #
        # 退化顺序按"几何参考的可靠性"排列：
        # 1. 微弱基底轮廓——独立于单元亮度分布，最可靠；
        # 2. 发光点云——只有亮单元参与，暗单元成片缺失时框会偏向亮的一侧。
        substrate_region = _detect_region_from_faint_substrate(gray, width, height)
        if substrate_region is not None:
            return substrate_region
        dot_region = _detect_region_from_emitting_dots(gray, width, height)
        if dot_region is not None:
            return dot_region
        return _fallback_center_region(width, height)

    _, best_contour, best_method = max(candidates, key=lambda item: item[0])
    rect = cv2.minAreaRect(best_contour)
    box = cv2.boxPoints(rect)
    ordered = order_quad_points(box)

    # 适度外扩，确保目标点和目标平面边缘不会被裁掉。
    center = ordered.mean(axis=0)
    expanded = center + (ordered - center) * 1.10
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)

    return ChipRegion(points=order_quad_points(expanded), score=float(rect[1][0] * rect[1][1]), method=best_method)


def _detect_region_from_faint_substrate(gray: np.ndarray, width: int, height: int) -> ChipRegion | None:
    """由发光单元下方的微弱基底轮廓界定主区域（荧光成像首选路径）。

    荧光成像中基底自发荧光通常只比暗背景亮 1-2 个灰度级，单像素上完全
    淹没在噪声里；但它是几十万像素的**连续区域**，只要先把亮单元擦掉、
    再做空间平均，这点差异就足够分离。

    为什么用形态学开运算而不是低通滤波：高斯模糊会把亮单元的能量摊到
    基底上，反而抬高基底读数、破坏基底与背景的对比（实测低通方案在
    真实失败样本上 0/24 命中，开运算 22/24）。开运算的结构元只要大于
    单元尺寸，亮单元就被整体腐蚀掉，基底作为大面积平台被保留。

    这个几何参考的关键优势是**独立于单元亮度分布**：发光点云路径会因为
    暗单元检测不到而把框缩到亮单元那一侧，基底轮廓不会。
    """

    side = min(width, height)
    # 结构元必须大于单元尺寸，否则单元擦不干净。
    kernel_size = max(21, int(side * 0.030)) | 1
    opened = cv2.morphologyEx(
        gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    )
    # 中值滤波压掉残余噪声，再按分位数拉伸——基底与背景的差异往
    # 往只有个位数灰度级，不拉伸则 Otsu 无法工作。
    denoised = cv2.medianBlur(opened, 9)
    low = float(np.percentile(denoised, 2))
    high = float(np.percentile(denoised, 98))
    if high - low < 1.0:
        return None
    normalized = np.clip((denoised.astype(np.float32) - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)

    threshold, _ = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(normalized, float(threshold), 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    best = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(best))
    image_area = float(width * height)
    if area < image_area * 0.05 or area > image_area * 0.95:
        return None

    rect = cv2.minAreaRect(best)
    aspect = rect[1][0] / max(rect[1][1], 1e-6)
    if not 0.35 <= aspect <= 2.8:
        return None

    ordered = order_quad_points(cv2.boxPoints(rect))
    # 基底轮廓已经是面板边界（不像点云那样贴着最外圈单元中心），
    # 因此只做与亮面板路径一致的小幅外扩。
    center = ordered.mean(axis=0)
    expanded = center + (ordered - center) * 1.02
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return ChipRegion(
        points=order_quad_points(expanded), score=area, method="opencv_faint_substrate"
    )


def _multi_threshold_blobs(
    response: np.ndarray,
    min_area: int,
    max_area: int,
    min_box: int,
    max_box: int,
    aspect_range: tuple[float, float] = (0.45, 1.80),
    levels: int = 6,
    noise_floor: float | None = None,
) -> list[tuple[float, float, float]]:
    """在一组几何递增的阈值上提取斑点并去重合并。

    单一阈值无法处理跨数量级的亮度分布：自发光/荧光阵列里最亮与最暗
    单元可以相差一两个数量级，全局阈值（Otsu）会牺牲弱单元，局部自适应
    阈值又会被邻近强单元抬高统计量而同样失效。

    这里改为在噪声水平到峰值之间取若干阈值各提取一次，再按中心距离
    去重（保留响应更强者）。强单元在高阈值处被干净地分离，弱单元在低
    阈值处被捕获；当单一阈值已经足够时，各档结果重合，去重后与原来一致。

    返回 [(cx, cy, response)]。
    """

    finite = response[np.isfinite(response)]
    if finite.size == 0:
        return []
    if noise_floor is not None:
        base = float(noise_floor)
    else:
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        base = median + 3.0 * max(1.0, 1.4826 * mad)
    peak = float(np.percentile(finite, 99.9))
    if peak <= base:
        thresholds = [base]
    else:
        thresholds = list(np.geomspace(base, peak, max(2, levels)))

    collected: list[tuple[float, float, float]] = []
    for threshold in thresholds:
        _, mask = cv2.threshold(response, float(threshold), 255, cv2.THRESH_BINARY)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        for index in range(1, count):
            x, y, w, h, area = stats[index]
            if not (min_area <= area <= max_area):
                continue
            if not (min_box <= w <= max_box and min_box <= h <= max_box):
                continue
            if not (aspect_range[0] <= w / max(h, 1) <= aspect_range[1]):
                continue
            values = response[labels == index]
            strength = float(values.mean()) * math.sqrt(float(area))
            collected.append((float(centroids[index][0]), float(centroids[index][1]), strength))

    if not collected:
        return []

    # 去重：同一单元会在多个阈值档各出现一次，保留响应最强的那次。
    collected.sort(key=lambda item: item[2], reverse=True)
    merged: list[tuple[float, float, float]] = []
    for cx, cy, strength in collected:
        if all(math.hypot(cx - mx, cy - my) > min_box for mx, my, _ in merged):
            merged.append((cx, cy, strength))
    return merged


def _detect_region_from_emitting_dots(gray: np.ndarray, width: int, height: int) -> ChipRegion | None:
    """由离散发光点云界定主区域（荧光/自发光成像路径）。

    适用场景：暗背景 + 一片规则排布的发光单元，没有连续亮面板可供
    阈值分割。此时目标区域的物理定义就是"发光单元的分布范围"。

    做法：顶帽增强小亮斑 -> 几何过滤取点云 -> 用点云的最小外接矩形
    作为主区域。为避免把孤立反光/热像素也算进去，要求点数足够多，
    且用中位数绝对偏差剔除远离主簇的离群点。
    """

    side = min(width, height)
    kernel_size = max(15, int(side * 0.030)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
    if int(tophat.max()) < 8:
        return None

    # 多阈值提取：荧光阵列内部亮度可跨一两个数量级，单一阈值会漏掉弱单元，
    # 导致点云只覆盖亮的那部分、外接框严重偏小。
    blobs = _multi_threshold_blobs(
        tophat,
        min_area=max(4, int(side * side * 2e-6)),
        max_area=max(5, int(side * side * 0.004)),
        min_box=max(3, int(side * 0.003)),
        max_box=max(12, int(side * 0.06)),
        aspect_range=(0.3, 3.3),
    )
    points = [(cx, cy) for cx, cy, _ in blobs]

    # 阵列至少 10x10，允许大量漏检，但太少就不足以界定区域。
    if len(points) < 24:
        return None

    cloud = np.asarray(points, dtype=np.float64)
    # 剔除离群点（热像素、孤立反光、图像边缘杂散）。
    # 判据是"最近邻距离"而不是"到中心的距离"或单轴 IQR：阵列内的点
    # 彼此相距一个间距，而热像素/杂散点是孤立的，最近邻距离远大于间距。
    # 这利用了"阵列"这一结构先验，对少量极端离群点也灵敏——散布型判据
    # 会被多数真实点主导而失效（少数几个热像素足以把外接框撑大数百像素）。
    difference = cloud[:, None, :] - cloud[None, :, :]
    distances = np.sqrt((difference ** 2).sum(axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = distances.min(axis=1)
    pitch_estimate = float(np.median(nearest))
    keep = nearest <= max(3.0, pitch_estimate * 2.5)
    if int(keep.sum()) < 24:
        return None
    cloud = cloud[keep].astype(np.float32)

    rect = cv2.minAreaRect(cloud)
    rect_area = float(rect[1][0] * rect[1][1])
    if rect_area < width * height * 0.002 or rect_area > width * height * 0.92:
        return None
    aspect = rect[1][0] / max(rect[1][1], 1e-6)
    if not 0.35 <= aspect <= 2.8:
        return None

    ordered = order_quad_points(cv2.boxPoints(rect))
    # 点云外接框贴着最外圈单元的中心（而不是面板边界），必须外扩出余量，
    # 否则矫正后最外圈单元贴边，会被后续轴选择的位置约束判为不合法。
    # 取 1.205 使阵列中心范围落在矫正图的 8.5%-91.5%，与亮面板路径
    # （面板自带约 8.5% 边距 + 1.10 外扩）后的版面一致。
    expansion = 1.205
    box_center = ordered.mean(axis=0)
    expanded = box_center + (ordered - box_center) * expansion
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    return ChipRegion(
        points=order_quad_points(expanded), score=rect_area, method="opencv_emitting_dots"
    )


def _fallback_center_region(width: int, height: int) -> ChipRegion:
    """目标平面定位失败时使用中央兜底区域。

    这不是最理想的定位，但能保证 demo 不会直接崩溃，并让质量控制给出 warning。
    """

    side_w = width * 0.45
    side_h = height * 0.45
    cx = width / 2
    cy = height / 2
    points = np.array(
        [
            [cx - side_w / 2, cy - side_h / 2],
            [cx + side_w / 2, cy - side_h / 2],
            [cx + side_w / 2, cy + side_h / 2],
            [cx - side_w / 2, cy + side_h / 2],
        ],
        dtype=np.float32,
    )
    return ChipRegion(points=points, score=0.0, method="fallback_center")


def rectify_chip(image: np.ndarray, region: ChipRegion, output_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """根据四角把目标平面区域矫正为标准正视图。

    返回：
    - rectified: 矫正后的目标平面图；
    - matrix: 原图 -> 矫正图 的单应性矩阵；
    - inverse_matrix: 矫正图 -> 原图 的逆单应性矩阵。
    """

    dst = np.array(
        [
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(region.points.astype(np.float32), dst)
    inverse_matrix = cv2.getPerspectiveTransform(dst, region.points.astype(np.float32))
    rectified = cv2.warpPerspective(image, matrix, (output_size, output_size), flags=cv2.INTER_CUBIC)
    return rectified, matrix, inverse_matrix


def generate_grid_points(
    grid_size: int,
    width: int | float,
    height: int | float,
    margin_ratio: float = 0.085,
) -> np.ndarray:
    """生成规则阵列理论点位，按行优先排列。

    这一步体现“物理先验”：目标平面点位/目标点是规则晶格，不应逐点自由漂移。
    """

    if grid_size <= 0:
        raise ValueError("grid_size 必须大于 0")
    if not 0.0 <= margin_ratio < 0.5:
        raise ValueError("margin_ratio 必须在 [0, 0.5) 范围内")

    x_margin = float(width) * margin_ratio
    y_margin = float(height) * margin_ratio
    xs = np.linspace(x_margin, float(width) - x_margin, grid_size, dtype=np.float32)
    ys = np.linspace(y_margin, float(height) - y_margin, grid_size, dtype=np.float32)
    points = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
    return points


def _find_axis_centers(projection: np.ndarray, count: int, length: int, margin_ratio: float) -> np.ndarray | None:
    """从一维投影曲线中寻找规则阵列中心。

    算法是“贪心峰值 + 最小间距约束”。如果峰值不足或分布异常，就返回 None，
    由规则 linspace 网格兜底。
    """

    proj = np.asarray(projection, dtype=np.float32)
    if proj.size < count * 2:
        return None

    # 平滑投影，避免单个脏点/反光点造成虚假峰。
    ksize = max(5, int(length / max(count * 8, 1)))
    if ksize % 2 == 0:
        ksize += 1
    smooth = cv2.GaussianBlur(proj.reshape(1, -1), (ksize, 1), 0).ravel()

    start = int(length * margin_ratio * 0.5)
    end = int(length * (1.0 - margin_ratio * 0.5))
    candidate_indices = np.arange(start, max(start + 1, end))
    if candidate_indices.size == 0:
        return None

    values = smooth[candidate_indices]
    order = candidate_indices[np.argsort(values)[::-1]]
    expected_pitch = (length * (1.0 - 2.0 * margin_ratio)) / max(count - 1, 1)
    min_distance = max(3.0, expected_pitch * 0.45)

    # 多选一些峰而不是恰好 count 个：背景扣除会在面板边缘产生光晕过冲，
    # 其投影峰可能比真实行列峰更强。若贪心恰好取 count 个，边缘假峰会把
    # 真实行列挤出去，导致间距检查失败并退化为均分网格。
    max_peaks = count + 8
    selected: list[float] = []
    for idx in order:
        if all(abs(float(idx) - existing) >= min_distance for existing in selected):
            selected.append(float(idx))
            if len(selected) == max_peaks:
                break

    if len(selected) < count:
        return None

    # 优先用规则晶格子集选择：从候选峰中挑出最符合等间距直线模型的 count 个，
    # 边缘光晕、流道等假峰因破坏规则性而被自然剔除。
    peaks = sorted(selected)
    peak_clusters = [(float(peak), float(smooth[int(round(peak))]), 1) for peak in peaks]
    lattice_axis = _select_regular_axis_from_clusters(peak_clusters, count, length)
    if lattice_axis is not None:
        return lattice_axis.astype(np.float32)

    # 规则子集选择失败（如网格位置超出其先验范围）时退回旧逻辑：
    # 取最强的 count 个峰并检查间距均匀性。
    centers = np.array(sorted(selected[:count]), dtype=np.float32)
    # 如果峰值间距过于不均匀，说明定位到了流道/边缘而非阵列点。
    if count > 2:
        diffs = np.diff(centers)
        if np.std(diffs) / (np.mean(diffs) + 1e-6) > 0.38:
            return None
    return centers


def _detect_dark_square_candidates(rectified: np.ndarray) -> np.ndarray:
    """定位 10x10 暗色目标平面中的暗色方形反应区候选点。

    10x10 实拍图里的有效定位目标不是目标平面外框，而是每个通道上的灰色小方块。
    这些小方块相对 均匀背景更暗，因此用黑帽变换（black-hat）增强“小而暗”的结构，
    再通过连通域过滤掉螺丝孔、外框、划痕等非目标区域。

    返回 N×3 数组：[x, y, weight]，weight 表示候选暗目标的响应强度。
    """

    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    side = min(width, height)

    # 核尺寸要大于反应方块，这样黑帽变换会突出方块而不是大面积背景渐变。
    kernel_size = max(31, int(side * 0.065))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, mask = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    min_box = max(8, int(side * 0.010))
    max_box = max(24, int(side * 0.055))
    min_area = max(50, int(side * side * 0.00012))
    max_area = max(450, int(side * side * 0.00120))
    edge_guard_x = width * 0.035
    edge_guard_y = height * 0.035

    candidates: list[tuple[float, float, float]] = []
    for index in range(1, component_count):
        x, y, w, h, area = stats[index]
        aspect = w / max(h, 1)
        cx, cy = centroids[index]

        # 过滤过小噪声、过大螺丝/边框，以及过扁的通道线。
        if not (min_area <= area <= max_area):
            continue
        if not (min_box <= w <= max_box and min_box <= h <= max_box):
            continue
        if not (0.45 <= aspect <= 1.80):
            continue
        if not (edge_guard_x <= cx <= width - edge_guard_x and edge_guard_y <= cy <= height - edge_guard_y):
            continue

        component_values = blackhat[labels == index]
        response = float(component_values.mean()) * math.sqrt(float(area))
        candidates.append((float(cx), float(cy), response))

    if not candidates:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(candidates, dtype=np.float32)


def _detect_bright_dot_candidates(rectified: np.ndarray) -> np.ndarray:
    """定位 15x15 亮点阵列中的小亮斑候选点。

    与 10x10 暗方块检测对偶：用顶帽变换（top-hat）增强“小而亮”的结构，
    抑制面板亮度本身和大面积光照渐变，再用连通域几何过滤剔除
    面板边缘亮带、高光条等非点状结构。

    返回 N×3 数组：[x, y, weight]，weight 表示候选亮点的响应强度。
    """

    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    side = min(width, height)

    # 核尺寸要大于亮点直径，顶帽变换才会保留整个亮点而不是只留边缘。
    kernel_size = max(25, int(side * 0.050))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    # 多阈值提取：自发光/荧光阵列内部亮度可跨一两个数量级，单一全局阈值
    # （Otsu）会牺牲弱单元，局部自适应阈值又被邻近强单元抬高而同样失效。
    # 噪声基准只从内部区域估计：面板边缘的阶跃过渡在顶帽图中形成远强于
    # 亮点的窄亮带，若参与全图统计会把阈值抬到亮点响应之上导致全部漏检。
    interior = tophat[int(height * 0.10):int(height * 0.90), int(width * 0.10):int(width * 0.90)]
    if interior.size == 0:
        interior = tophat

    # 亮点直径一般在矫正图边长的 0.8%-4.5% 之间；过大者多为高光带/边缘光晕。
    min_box = max(5, int(side * 0.006))
    max_box = max(16, int(side * 0.045))
    min_area = max(20, int(side * side * 0.00004))
    max_area = max(240, int(side * side * 0.00110))

    median = float(np.median(interior))
    mad = float(np.median(np.abs(interior - median)))
    noise_floor = median + 3.0 * max(1.0, 1.4826 * mad)
    blobs = _multi_threshold_blobs(
        tophat,
        min_area=min_area, max_area=max_area,
        min_box=min_box, max_box=max_box,
        noise_floor=noise_floor,
    )

    edge_guard_x = width * 0.035
    edge_guard_y = height * 0.035
    candidates = [
        (cx, cy, strength)
        for cx, cy, strength in blobs
        if edge_guard_x <= cx <= width - edge_guard_x and edge_guard_y <= cy <= height - edge_guard_y
    ]

    if not candidates:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(candidates, dtype=np.float32)


def _rotate_points(points: np.ndarray, angle_deg: float, center: tuple[float, float]) -> np.ndarray:
    """绕给定中心旋转二维点集（角度制，图像坐标系）。"""

    theta = math.radians(angle_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    cx, cy = center
    shifted = np.asarray(points, dtype=np.float64) - (cx, cy)
    rotated = np.empty_like(shifted)
    rotated[:, 0] = shifted[:, 0] * cos_t - shifted[:, 1] * sin_t
    rotated[:, 1] = shifted[:, 0] * sin_t + shifted[:, 1] * cos_t
    return (rotated + (cx, cy)).astype(np.float32)


def _histogram_sharpness(values: np.ndarray, length: float) -> float:
    """一维坐标直方图的“锐度”。

    规则晶格与坐标轴对齐时，行/列坐标聚成少数尖峰，平方和最大；
    带旋转时坐标弥散到更多 bin，平方和降低。
    """

    bin_width = max(4.0, float(length) * 0.01)
    bin_count = max(4, int(round(float(length) / bin_width)))
    hist, _ = np.histogram(values, bins=bin_count, range=(0.0, float(length)))
    return float(np.square(hist.astype(np.float64)).sum())


def _estimate_lattice_rotation(points: np.ndarray, length: float, max_angle_deg: float = 6.0, step_deg: float = 0.25) -> float:
    """估计候选点晶格相对图像坐标轴的小角度旋转。

    透视矫正只对齐面板四角；点阵与面板边缘的装配旋转差、或主区域
    外接矩形的角度偏差，都会让矫正图中的晶格残留几度旋转。
    这里在 ±max_angle_deg 范围内扫描，找到让 x/y 直方图最锐利的角度，
    返回“把晶格转正所需的旋转角”。
    """

    if points.shape[0] < 8:
        return 0.0

    center = (float(length) / 2.0, float(length) / 2.0)
    best_angle = 0.0
    best_score = -1.0
    for angle in np.arange(-max_angle_deg, max_angle_deg + step_deg / 2.0, step_deg):
        rotated = _rotate_points(points, float(angle), center)
        score = _histogram_sharpness(rotated[:, 0], length) + _histogram_sharpness(rotated[:, 1], length)
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return best_angle


def _cluster_axis_candidates(values: np.ndarray, weights: np.ndarray, length: int) -> list[tuple[float, float, int]]:
    """把候选点在单个坐标轴上聚成若干中心簇。

    同一行/列内的暗方块会在投影轴上形成紧密簇；螺丝、边框等误检通常形成
    支持数较少或不满足规则间距的簇，后续会被晶格拟合剔除。
    """

    if values.size == 0:
        return []

    tolerance = max(10.0, length * 0.0225)
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]

    raw_clusters: list[list[tuple[float, float]]] = []
    for value, weight in zip(sorted_values, sorted_weights):
        if not raw_clusters:
            raw_clusters.append([(float(value), float(weight))])
            continue

        current_values = np.asarray([item[0] for item in raw_clusters[-1]], dtype=np.float32)
        if abs(float(value) - float(current_values.mean())) <= tolerance:
            raw_clusters[-1].append((float(value), float(weight)))
        else:
            raw_clusters.append([(float(value), float(weight))])

    clusters: list[tuple[float, float, int]] = []
    for cluster in raw_clusters:
        cluster_values = np.asarray([item[0] for item in cluster], dtype=np.float32)
        cluster_weights = np.asarray([max(item[1], 1e-6) for item in cluster], dtype=np.float32)
        center = float(np.average(cluster_values, weights=cluster_weights))
        support = float(cluster_weights.sum())
        clusters.append((center, support, len(cluster)))
    return clusters


def _complete_axis_with_missing_clusters(
    clusters: list[tuple[float, float, int]],
    count: int,
    length: int,
) -> np.ndarray | None:
    """簇数不足 count 时，按间距锚定索引并外推缺失行/列。

    整行/列单元被遮挡时对应的轴向簇会整体消失，但规则晶格先验让缺失
    位置可推断：
    1. 用相邻簇距的中位数估计间距，把每个簇分配到整数晶格索引
       （中间缺口表现为间距跳变，索引随之跳跃）；
    2. 剩余的缺失量在两端滑动枚举（边缘缺失存在"补前还是补后"的
       二义性），用"左右边距对称"消解——矫正阶段的对称外扩保证
       网格在矫正图中近似居中，这是几何链路自带的物理先验；
    3. 对每个假设做与完整路径相同的间距/位置/残差校验。

    最多容忍 2 个缺失簇：更多缺失时索引分配的错误风险迅速上升。
    """

    missing = count - len(clusters)
    if not 1 <= missing <= 2 or len(clusters) < 3:
        return None

    centers = np.asarray(sorted(item[0] for item in clusters), dtype=np.float64)
    diffs = np.diff(centers)
    pitch_estimate = float(np.median(diffs))
    if not length * 0.045 <= pitch_estimate <= length * 0.095:
        return None

    # 间距跳变 -> 索引跳跃（中间缺口）；累计得到相对索引。
    indices = np.zeros(len(centers), dtype=np.int64)
    for position, diff in enumerate(diffs):
        step = max(1, int(round(diff / pitch_estimate)))
        indices[position + 1] = indices[position] + step
    span = int(indices[-1])
    if span > count - 1:
        return None

    best: tuple[float, np.ndarray] | None = None
    index_axis = np.arange(count, dtype=np.float64)
    min_pitch, max_pitch = length * 0.045, length * 0.095
    min_start, max_start, max_end = length * 0.050, length * 0.300, length * 0.950

    for shift in range(count - span):
        assigned = indices + shift
        pitch, start = np.polyfit(assigned.astype(np.float64), centers, deg=1)
        if not min_pitch <= pitch <= max_pitch:
            continue
        fitted = start + pitch * index_axis
        if fitted[0] < min_start or fitted[0] > max_start or fitted[-1] > max_end:
            continue
        residual = centers - (start + pitch * assigned)
        if float(np.sqrt(np.mean(residual**2))) > length * 0.020:
            continue
        # 边距对称性评分：越接近居中越可信。
        asymmetry = abs(float(fitted[0]) - (length - float(fitted[-1])))
        if best is None or asymmetry < best[0]:
            best = (asymmetry, fitted.astype(np.float32))

    return None if best is None else best[1]


def _select_regular_axis_from_clusters(
    clusters: list[tuple[float, float, int]],
    count: int,
    length: int,
) -> np.ndarray | None:
    """从轴向簇中选择最符合规则晶格的一组中心。

    这里枚举 10 个候选簇的组合，并拟合 `start + pitch * i`。
    评分优先选择残差小、支持强、起止位置合理的组合。
    簇数不足时转入缺失补全路径（整行/列被遮挡的场景）。
    """

    if len(clusters) < count:
        return _complete_axis_with_missing_clusters(clusters, count, length)

    # 防组合爆炸：螺丝、边缘碎片可能产生大量杂散簇，簇数一多，
    # C(N, count) 会迅速失控（例如 C(24,10) 约 200 万组合，实测需要一分钟以上）。
    # 真实阵列的行/列簇有约 grid_size 个成员、支持度远高于孤立误检，
    # 因此按支持度只保留最强的若干簇即可，同时必须恢复位置升序，
    # 因为下面的直线拟合假定组合内的中心是按坐标递增排列的。
    # 上限随 count 自适应：至少给 count 留 4 个假峰余量（边缘光晕可能比真实
    # 行列峰更强，固定上限会把弱的真实行列挤掉），组合数上界约为
    # C(count+4, 4)，count=10 时为 C(16,10)=8008，count=15 时为 C(19,15)=3876。
    max_clusters = max(16, count + 4)
    if len(clusters) > max_clusters:
        strongest = sorted(clusters, key=lambda item: item[1], reverse=True)[:max_clusters]
        clusters = sorted(strongest, key=lambda item: item[0])

    best: tuple[float, np.ndarray] | None = None
    index_axis = np.arange(count, dtype=np.float32)
    min_pitch = length * 0.045
    max_pitch = length * 0.095
    # 起止范围只做弱约束（距边缘至少 5%）：主区域检测的外扩比例和形态学
    # 膨胀会让阵列在矫正图中的位置有 ±3% 左右浮动，过紧的范围会把
    # 合法网格误拒并退化为均分网格。假峰主要靠下面的规则性残差
    # 和支持度评分剔除，而不是靠位置范围。
    min_start = length * 0.050
    max_start = length * 0.300
    max_end = length * 0.950

    for combo in combinations(clusters, count):
        centers = np.asarray([item[0] for item in combo], dtype=np.float32)
        supports = np.asarray([item[1] for item in combo], dtype=np.float32)
        counts = np.asarray([item[2] for item in combo], dtype=np.float32)

        pitch, start = np.polyfit(index_axis, centers, deg=1)
        if not (min_pitch <= pitch <= max_pitch):
            continue

        fitted = start + pitch * index_axis
        if fitted[0] < min_start or fitted[0] > max_start or fitted[-1] > max_end:
            continue

        residual = centers - fitted
        rmse = float(np.sqrt(np.mean(residual * residual)))
        if rmse > length * 0.020:
            continue

        # 支持数越高越可信；但规则性比支持数更重要，避免螺丝/边框高响应误导。
        support_score = float(np.log1p(supports).sum() + counts.sum() * 0.15)
        score = rmse * 8.0 - support_score

        if best is None or score < best[0]:
            best = (score, fitted.astype(np.float32))

    if best is None:
        return None
    return best[1]


def _fit_lattice_from_candidates(candidates: np.ndarray, grid_size: int, width: int, height: int) -> np.ndarray | None:
    """从候选点拟合带旋转补偿的规则晶格。

    步骤：
    1. 估计候选点晶格相对坐标轴的小角度旋转（矫正残差）；
    2. 把候选点旋到与坐标轴对齐后，做行/列聚类和规则等距轴选择；
    3. 由行列中心外积生成网格，再旋回原坐标系。

    轴对齐假设一旦成立，螺丝、边框碎片等误检会因为不满足规则间距
    或支持度不足而被 _select_regular_axis_from_clusters 剔除。
    """

    if candidates.shape[0] < grid_size * 4:
        return None

    length = min(width, height)
    center = (float(length) / 2.0, float(length) / 2.0)
    angle = _estimate_lattice_rotation(candidates[:, :2], length)
    aligned = _rotate_points(candidates[:, :2], angle, center)

    weights = candidates[:, 2]
    x_clusters = _cluster_axis_candidates(aligned[:, 0], weights, width)
    y_clusters = _cluster_axis_candidates(aligned[:, 1], weights, height)
    xs = _select_regular_axis_from_clusters(x_clusters, grid_size, width)
    ys = _select_regular_axis_from_clusters(y_clusters, grid_size, height)

    if xs is None or ys is None:
        return None

    grid = np.array([[x, y] for y in ys for x in xs], dtype=np.float32)
    # 反向旋转，把轴对齐坐标系中的网格映射回矫正图坐标。
    return _rotate_points(grid, -angle, center)


def _fit_dark_square_grid(rectified: np.ndarray, grid_size: int) -> np.ndarray | None:
    """用暗方块候选点拟合 10x10 规则网格。"""

    height, width = rectified.shape[:2]
    candidates = _detect_dark_square_candidates(rectified)
    return _fit_lattice_from_candidates(candidates, grid_size, width, height)


def _fit_bright_dot_grid(rectified: np.ndarray, grid_size: int) -> np.ndarray | None:
    """用亮点候选点拟合 15x15 规则网格。"""

    height, width = rectified.shape[:2]
    candidates = _detect_bright_dot_candidates(rectified)
    return _fit_lattice_from_candidates(candidates, grid_size, width, height)


def _refine_one_10x10_dark_square_center(
    gray: np.ndarray,
    x: float,
    y: float,
    radius: int,
) -> tuple[float, float] | None:
    """在单个理论点附近寻找最可信的暗色方块中心。

    旧版精修直接对低亮度像素做质心，遇到细长暗线、通道边缘或左侧辅助结构时，
    质心会被拉走。这里改为局部连通域分析：只接受近似方形的小暗块，并优先选择
    距离理论点近、形状接近正方形、面积合理的候选。
    """

    height, width = gray.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    # 与 _refine_grid_points_with_status 同理：图外点直接判无证据，
    # 负索引会绕过空窗口检查（见该函数注释）。
    if not (0 <= xi < width and 0 <= yi < height):
        return None
    x0 = max(0, xi - radius)
    x1 = min(width, xi + radius + 1)
    y0 = max(0, yi - radius)
    y1 = min(height, yi + radius + 1)
    window = gray[y0:y1, x0:x1]
    if window.size == 0:
        return None

    # 把“暗结构”转成亮前景，再用 Otsu 自动分割。这样不依赖固定灰度阈值，
    # 对 均匀背景亮度变化和相机曝光差异更稳。
    inverted = 255 - window
    _, mask = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 去掉孤立噪声，同时不明显改变方块几何中心。
    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel, iterations=1)

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if component_count <= 1:
        return None

    min_box = max(5, int(radius * 0.35))
    max_box = max(min_box + 2, int(radius * 1.35))
    min_area = max(20, int((radius * 0.35) ** 2))
    max_area = max(min_area + 10, int((radius * 1.45) ** 2))
    max_shift = max(3.0, radius * 0.72)

    best: tuple[float, float, float] | None = None
    for index in range(1, component_count):
        local_x, local_y, box_w, box_h, area = stats[index]
        if not (min_area <= area <= max_area):
            continue
        if not (min_box <= box_w <= max_box and min_box <= box_h <= max_box):
            continue

        aspect = box_w / max(box_h, 1)
        if not (0.55 <= aspect <= 1.80):
            continue

        cx_local, cy_local = centroids[index]
        cx = float(x0 + cx_local)
        cy = float(y0 + cy_local)
        distance = math.hypot(cx - x, cy - y)
        if distance > max_shift:
            continue

        component_gray = window[labels == index]
        darkness = 255.0 - float(component_gray.mean())
        square_penalty = abs(math.log(aspect))
        # 评分越小越好。距离是主约束，方形度用于排除线状结构，暗度用于区分弱噪声。
        score = distance + square_penalty * radius * 0.35 - darkness * 0.015
        if best is None or score < best[0]:
            best = (score, cx, cy)

    if best is None:
        return None
    return best[1], best[2]


def _legacy_polarity(grid_size: int) -> str:
    """旧版按网格规格假设的极性（仅作向后兼容默认值）。"""

    return "bright" if grid_size >= 15 else "dark"


def _fit_axis_projection(
    gray: np.ndarray,
    polarity: str,
    grid_size: int,
    margin_ratio: float,
) -> np.ndarray | None:
    """按指定极性的能量图做行列投影拟合。"""

    height, width = gray.shape[:2]
    if polarity == "bright":
        energy = gray.astype(np.float32)
    else:
        energy = 255.0 - gray.astype(np.float32)

    # 背景扣除，减弱大面积光照渐变对投影峰的影响。
    blur_size = max(31, int(min(width, height) * 0.08))
    if blur_size % 2 == 0:
        blur_size += 1
    background = cv2.GaussianBlur(energy, (blur_size, blur_size), 0)
    enhanced = cv2.normalize(energy - background, None, 0, 255, cv2.NORM_MINMAX)

    xs = _find_axis_centers(enhanced.sum(axis=0), grid_size, width, margin_ratio)
    ys = _find_axis_centers(enhanced.sum(axis=1), grid_size, height, margin_ratio)
    if xs is None or ys is None:
        return None
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float32)


def _estimate_polarity_hint(gray: np.ndarray) -> str:
    """由图像统计估计单元极性提示（只返回方向，兼容旧调用）。"""

    _, hint = _polarity_hint_with_strength(gray)
    return hint


def _polarity_hint_with_strength(gray: np.ndarray) -> tuple[float, str]:
    """返回 (提示强度, 极性提示)。

    阵列单元只占面板面积的少数（10x10 约 11%、15x15 约 18%）：背景决定
    灰度中位数，单元把均值拉向自己一侧（暗单元 -> 均值 < 中位数）。
    只统计内部区域，避免矫正外扩带来的暗色边缘带干扰。该统计量对模糊
    不敏感，是候选检测器整体失效时仍然可用的极性证据。

    强度 = |均值 - 中位数|。实测全部样例的符号都正确，但强度差异很大
    （实拍暗单元 8.9-34.3；小亮点合成图仅 0.9），因此强度决定这条证据
    能否压过候选数量证据。
    """

    height, width = gray.shape[:2]
    interior = gray[int(height * 0.10):int(height * 0.90), int(width * 0.10):int(width * 0.90)]
    if interior.size == 0:
        interior = gray
    delta = float(interior.mean()) - float(np.median(interior))
    return abs(delta), ("dark" if delta < 0 else "bright")


# 提示强度显著性门限：实测最弱的正确暗单元提示为 8.9，最强的弱提示
# （小亮点合成图）为 0.9，取 3.0 兼顾两侧安全边际。
POLARITY_HINT_SIGNIFICANT = 3.0


def _order_polarities_by_evidence(
    dark_count: int,
    bright_count: int,
    expected: int,
    hint: str,
    hint_strength: float,
) -> list[str]:
    """按证据强度给两种极性排序，返回优先尝试顺序。

    两条证据的可靠性并不对等：
    - 候选数量接近理论单元数：直觉上合理，但形态学过滤的偶然性让它
      在实拍图上噪声很大（同一张图不同裁切下亮候选实测 79->164）；
    - 内部"均值 vs 中位数"统计：物理依据扎实（单元占面积远小于一半），
      实测符号在全部样例上都正确，但强度可能很弱。

    因此：提示显著时以提示为主键、候选数为次键；提示微弱时反过来。
    最终裁决仍是"晶格能否拟合成功"，本函数只决定尝试顺序。
    """

    other = "bright" if hint == "dark" else "dark"
    gaps = {"dark": abs(dark_count - expected), "bright": abs(bright_count - expected)}

    if hint_strength >= POLARITY_HINT_SIGNIFICANT:
        return [hint, other]
    return sorted(("dark", "bright"), key=lambda p: (gaps[p], p != hint))


def _measure_grid_polarity_contrast(gray: np.ndarray, points: np.ndarray, grid_size: int) -> float:
    """测量候选网格的"点位 vs 间隙"对比度。

    正值表示网格点比相邻点位的中点更亮（亮单元），负值表示更暗（暗单元）。

    这是极性歧义的最终仲裁依据：暗单元阵列的亮间隙本身也构成规则晶格
    （互补晶格），投影峰同样整齐，仅凭规则性无法区分；但两者的点位落处
    对比度符号恰好相反。相比候选数量或全局灰度统计，本判据直接读取图像
    在候选网格位置上的证据，因此在候选检测退化（重模糊）时仍然可用。
    """

    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if grid_size < 2 or points.shape[0] != grid_size * grid_size:
        return 0.0
    height, width = gray.shape[:2]
    lattice = points.reshape(grid_size, grid_size, 2)
    # 间隙取相邻点位的中点（水平与垂直两个方向）。
    gaps = np.concatenate(
        [
            ((lattice[:, :-1] + lattice[:, 1:]) / 2.0).reshape(-1, 2),
            ((lattice[:-1, :] + lattice[1:, :]) / 2.0).reshape(-1, 2),
        ],
        axis=0,
    )

    def _sample(sample_points: np.ndarray) -> float:
        xs = np.rint(sample_points[:, 0]).astype(int)
        ys = np.rint(sample_points[:, 1]).astype(int)
        inside = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
        if not inside.any():
            return 0.0
        return float(np.median(gray[ys[inside], xs[inside]]))

    return _sample(points) - _sample(gaps)


def _fit_grid_points_with_polarity(
    rectified: np.ndarray,
    grid_size: int,
    margin_ratio: float = 0.085,
) -> tuple[np.ndarray, str]:
    """拟合阵列点并自动检测单元极性。

    单元极性（比均匀背景暗还是亮）取决于成像方式而不是网格规格：
    同一种阵列在背光成像下是"亮面板 + 暗单元"，在反射照明下可能相反。

    决策规则：
    1. 候选路径两极都尝试（优先数量更接近理论单元数、与统计提示一致
       的一极），晶格拟合成功即裁决——形状过滤过的候选无法从反极性
       的间隙结构里凑出合法晶格，误判风险低；
    2. 投影路径只尝试**综合证据排序后的最优极性**（单一极性，不盲试两极）：
       暗单元阵列的亮间隙同样构成规则投影峰（互补晶格），盲试反极性会
       得到自信但半格错位的网格。这里必须用综合排序而不是单独的统计
       提示——重模糊会把提示强度压到接近 0 并可能翻转其符号，此时
       候选数量才是可靠证据；
    3. 都失败则退回均分网格，极性取排序后的最优极性。
    """

    height, width = rectified.shape[:2]
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    expected = grid_size * grid_size
    hint_strength, hint = _polarity_hint_with_strength(gray)

    candidates_by_polarity = {
        "dark": _detect_dark_square_candidates(rectified),
        "bright": _detect_bright_dot_candidates(rectified),
    }
    order = _order_polarities_by_evidence(
        dark_count=candidates_by_polarity["dark"].shape[0],
        bright_count=candidates_by_polarity["bright"].shape[0],
        expected=expected,
        hint=hint,
        hint_strength=hint_strength,
    )

    for polarity in order:
        lattice = _fit_lattice_from_candidates(candidates_by_polarity[polarity], grid_size, width, height)
        if lattice is not None:
            return lattice, polarity

    # 投影回退：两个极性各拟合一次，用"点位 vs 间隙"对比度直接仲裁。
    # 不能只信排序——候选拟合都失败时说明候选证据本身不可靠（重模糊下
    # 两极候选数都远离理论值），此时排序依据已失去意义；而对比度是在
    # 候选网格位置上读取的图像证据，恰好能区分真实晶格与互补晶格。
    best: tuple[float, np.ndarray, str] | None = None
    for polarity in order:
        projected = _fit_axis_projection(gray, polarity, grid_size, margin_ratio)
        if projected is None:
            continue
        contrast = _measure_grid_polarity_contrast(gray, projected, grid_size)
        # 对比度符号与该极性一致时才算作正向证据。
        evidence = contrast if polarity == "bright" else -contrast
        if best is None or evidence > best[0]:
            best = (evidence, projected, polarity)

    if best is not None and best[0] > 0.0:
        return best[1], best[2]
    if best is not None:
        # 两个极性都没有正向对比度证据：保留排序首选的结果，
        # 后续的候选支撑率检查会把这种低置信情况标记出来。
        preferred = order[0]
        projected = _fit_axis_projection(gray, preferred, grid_size, margin_ratio)
        if projected is not None:
            return projected, preferred
        return best[1], best[2]

    return generate_grid_points(grid_size, width, height, margin_ratio=margin_ratio), order[0]


def fit_grid_points(rectified: np.ndarray, grid_size: int, margin_ratio: float = 0.085) -> np.ndarray:
    """在矫正图中拟合阵列点。

    候选晶格路径（含极性自动检测与旋转补偿）优先；投影路径与
    均分网格作为逐级回退。接口与历史版本保持一致。
    """

    points, _ = _fit_grid_points_with_polarity(rectified, grid_size, margin_ratio=margin_ratio)
    return points


def _refine_grid_points_with_status(
    rectified: np.ndarray,
    points: np.ndarray,
    grid_size: int,
    radius: int,
    polarity: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """局部精修，并返回每个点是否获得了图像证据支持。

    status[i] 为 True 表示该点在局部窗口内找到了可信的结构中心
    （方形组件或质心），False 表示窗口内没有证据或证据被限幅拒绝。
    光束法平差只用有证据的观测拟合全局模型，从机制上避免
    “无证据点占多数时锁定错误网格”的失效模式。

    polarity 指定单元相对背景的明暗极性（None 时沿用旧的按规格假设）。
    暗单元优先用方形组件精修（对任意网格规格），亮单元用高亮质心。
    """

    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    refined = points.copy().astype(np.float32)
    status = np.zeros(points.shape[0], dtype=bool)
    if polarity is None:
        polarity = _legacy_polarity(grid_size)
    polarity_bright = polarity == "bright"

    for idx, (x, y) in enumerate(points):
        if not polarity_bright:
            square_center = _refine_one_10x10_dark_square_center(gray, float(x), float(y), radius)
            if square_center is not None:
                refined[idx] = square_center
                status[idx] = True
                continue

        xi, yi = int(round(x)), int(round(y))
        # 点落在图像外就是"没有图像证据"，直接跳过。
        # 必须显式判断而不能只靠空窗口检查：xi 为负时 min(width, xi+radius+1)
        # 会得到负数，numpy 把它当成从末尾计数的索引，切片反而非空。
        if not (0 <= xi < width and 0 <= yi < height):
            continue
        x0 = max(0, xi - radius)
        x1 = min(width, xi + radius + 1)
        y0 = max(0, yi - radius)
        y1 = min(height, yi + radius + 1)
        window = gray[y0:y1, x0:x1].astype(np.float32)
        if window.size == 0:
            continue

        if polarity_bright:
            # 亮孔：使用高亮像素质心。
            cutoff = np.percentile(window, 78)
            weights = np.clip(window - cutoff, 0, None)
        else:
            # 暗色目标点：使用低亮度像素质心。
            cutoff = np.percentile(window, 32)
            weights = np.clip(cutoff - window, 0, None)

        if float(weights.sum()) <= 1e-6:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1]
        cx = float((xx * weights).sum() / weights.sum())
        cy = float((yy * weights).sum() / weights.sum())

        # 防止局部反光导致点位漂移太远。
        max_shift = max(2.0, radius * 0.65)
        if math.hypot(cx - x, cy - y) <= max_shift:
            refined[idx] = (cx, cy)
            status[idx] = True

    return refined, status


def refine_grid_points(rectified: np.ndarray, points: np.ndarray, grid_size: int, radius: int) -> np.ndarray:
    """在每个理论点周围做局部中心微调。

    微调必须是“小步”的：全局网格位置来自物理模型，局部只允许在小窗口内修正。
    这样可以避免反光、气泡、脏点把某个 ROI 拉到错误位置。
    """

    refined, _ = _refine_grid_points_with_status(rectified, points, grid_size, radius)
    return refined


def enforce_lattice_consistency(
    points: np.ndarray,
    grid_size: int,
    inlier_threshold_ratio: float = 0.18,
    min_threshold_px: float = 2.5,
    max_iterations: int = 5,
) -> tuple[np.ndarray, dict[str, object]]:
    """用全局规则晶格约束修正被局部干扰拉偏的点位。

    注：V2.0 起主管线改由 bundle_adjust_lattice（单应模型 + 证据加权）
    承担此职责；本函数保留为轻量工具与向后兼容接口。

    物理先验：阵列点位构成刚性规则晶格，矫正残差只会表现为整体的
    仿射变形（平移/缩放/旋转/轻微剪切），单个点不应独立漂移。
    因此这里把 (row, col) -> (x, y) 拟合为仿射晶格模型：

        x = a0 + a1*col + a2*row
        y = b0 + b1*col + b2*row

    并做“阈值化迭代重拟合”：残差明显超出阈值的点视为被细长暗线、
    局部阴影、高光或边缘结构拉偏的异常点，剔除后重拟合；收敛后把
    异常点吸附回模型预测位置，其余点保留局部精修结果（允许真实的
    小幅制造/透视残差，不强行摆到完美晶格上）。

    阈值取 max(min_threshold_px, inlier_threshold_ratio * pitch)：
    - 0.18 * pitch 略小于局部精修允许的最大位移（约 0.21-0.23 pitch），
      刚好能抓住“被精修拉到窗口极限”的点；
    - 正常制造与透视残差通常小于 0.05 pitch，不会被误伤；
    - min_threshold_px 防止小间距网格下阈值低于像素量化噪声。

    返回 (修正后的点, 诊断信息)。诊断信息全部为原生类型，可直接写入 JSON。
    """

    points = np.asarray(points, dtype=np.float32)
    expected_count = grid_size * grid_size
    info: dict[str, object] = {
        "applied": False,
        "reason": "",
        "outlier_count": 0,
        "corrected_points": [],
        "pitch_px": 0.0,
        "threshold_px": 0.0,
        "inlier_rmse_px": 0.0,
    }

    # 网格太小时异常点无法与整体变形区分，点数不符说明上游异常，都不做修正。
    if grid_size < 3:
        info["reason"] = "grid_size 小于 3，晶格约束不可靠"
        return points.copy(), info
    if points.shape[0] != expected_count:
        info["reason"] = f"点数 {points.shape[0]} 与 {grid_size}x{grid_size} 不符"
        return points.copy(), info

    indices = np.arange(expected_count)
    rows = (indices // grid_size).astype(np.float64)
    cols = (indices % grid_size).astype(np.float64)
    design = np.stack([np.ones(expected_count), cols, rows], axis=1)
    xy = points.astype(np.float64)

    inliers = np.ones(expected_count, dtype=bool)
    predicted = xy.copy()
    threshold = float(min_threshold_px)
    pitch = 0.0

    for _ in range(max_iterations):
        coefficients, _, _, _ = np.linalg.lstsq(design[inliers], xy[inliers], rcond=None)
        predicted = design @ coefficients
        residual = np.linalg.norm(xy - predicted, axis=1)

        # 由模型系数直接估计间距：列方向步长向量 (a1, b1)、行方向步长向量 (a2, b2)。
        pitch_col = math.hypot(coefficients[1, 0], coefficients[1, 1])
        pitch_row = math.hypot(coefficients[2, 0], coefficients[2, 1])
        pitch = (pitch_col + pitch_row) / 2.0
        threshold = max(float(min_threshold_px), pitch * inlier_threshold_ratio)

        new_inliers = residual <= threshold
        # 保护：inlier 太少说明整体拟合已不可信（如上游整片错位），
        # 此时宁可不动，也不能把大量点吸附到错误模型上。
        if int(new_inliers.sum()) < max(6, expected_count // 2):
            info["reason"] = "晶格模型 inlier 不足，跳过修正"
            return points.copy(), info
        if np.array_equal(new_inliers, inliers):
            inliers = new_inliers
            break
        inliers = new_inliers

    corrected = points.copy()
    outlier_indices = np.flatnonzero(~inliers)
    corrected[outlier_indices] = predicted[outlier_indices].astype(np.float32)

    inlier_residual = np.linalg.norm(xy[inliers] - predicted[inliers], axis=1)
    info.update(
        {
            "applied": True,
            "reason": "ok",
            "outlier_count": int(outlier_indices.size),
            "corrected_points": [
                {
                    "row": int(index // grid_size),
                    "col": int(index % grid_size),
                    "shift_px": round(float(np.linalg.norm(xy[index] - predicted[index])), 4),
                }
                for index in outlier_indices
            ],
            "pitch_px": round(float(pitch), 4),
            "threshold_px": round(float(threshold), 4),
            "inlier_rmse_px": round(float(np.sqrt(np.mean(inlier_residual**2))), 6) if inlier_residual.size else 0.0,
        }
    )
    return corrected, info


def _apply_homography(h_matrix: np.ndarray, src: np.ndarray) -> np.ndarray:
    """把 N×2 点集经单应矩阵映射到目标坐标系。"""

    homogeneous = np.column_stack([src, np.ones(src.shape[0])])
    mapped = homogeneous @ h_matrix.T
    denom = mapped[:, 2:3]
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    return mapped[:, :2] / denom


def _normalize_for_dlt(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hartley 归一化：平移到质心、缩放平均距离为 sqrt(2)，改善 DLT 条件数。"""

    centroid = points.mean(axis=0)
    mean_distance = float(np.linalg.norm(points - centroid, axis=1).mean()) + 1e-12
    scale = math.sqrt(2.0) / mean_distance
    transform = np.array(
        [[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    normalized = (points - centroid) * scale
    return normalized, transform


def _fit_homography_irls(src: np.ndarray, dst: np.ndarray, iterations: int = 5) -> np.ndarray | None:
    """用加权 DLT + Tukey 双权 IRLS 拟合 src -> dst 的 8 自由度单应。

    src 是晶格索引 (col, row)，dst 是像素观测。观测数远大于自由度
    （100/225 对 8），少数被局部干扰拉偏的观测会在迭代中被降权。
    """

    point_count = src.shape[0]
    if point_count < 8:
        return None

    src_n, t_src = _normalize_for_dlt(src.astype(np.float64))
    dst_n, t_dst = _normalize_for_dlt(dst.astype(np.float64))
    weights = np.ones(point_count, dtype=np.float64)
    h_matrix: np.ndarray | None = None

    for _ in range(iterations):
        sx, sy = src_n[:, 0], src_n[:, 1]
        dx, dy = dst_n[:, 0], dst_n[:, 1]
        zeros = np.zeros(point_count)
        ones = np.ones(point_count)
        row_u = np.stack([sx, sy, ones, zeros, zeros, zeros, -dx * sx, -dx * sy, -dx], axis=1)
        row_v = np.stack([zeros, zeros, zeros, sx, sy, ones, -dy * sx, -dy * sy, -dy], axis=1)
        sqrt_w = np.sqrt(weights)[:, None]
        design = np.concatenate([row_u * sqrt_w, row_v * sqrt_w], axis=0)

        try:
            _, _, vt = np.linalg.svd(design, full_matrices=False)
        except np.linalg.LinAlgError:
            return h_matrix
        h_normalized = vt[-1].reshape(3, 3)
        candidate = np.linalg.inv(t_dst) @ h_normalized @ t_src
        if abs(candidate[2, 2]) < 1e-12:
            return h_matrix
        candidate = candidate / candidate[2, 2]
        h_matrix = candidate

        residual = np.linalg.norm(dst - _apply_homography(h_matrix, src), axis=1)
        # Tukey 双权：尺度用 MAD，下限 0.3px 防止零残差合成数据退化。
        sigma = max(0.3, 1.4826 * float(np.median(np.abs(residual - np.median(residual)))))
        tukey_c = 4.685 * sigma
        ratio = np.clip(residual / tukey_c, 0.0, 1.0)
        new_weights = (1.0 - ratio**2) ** 2
        if float(new_weights.sum()) < 8.0:
            break
        weights = new_weights

    return h_matrix


def _estimate_grid_pitch(points: np.ndarray, grid_size: int, fallback: float) -> float:
    """从行内/列内相邻点距估计网格间距，异常时使用兜底值。"""

    if grid_size >= 2 and points.shape[0] == grid_size * grid_size:
        lattice = points.reshape(grid_size, grid_size, 2)
        dx = np.abs(np.diff(lattice[:, :, 0], axis=1))
        dy = np.abs(np.diff(lattice[:, :, 1], axis=0))
        pitch = (float(np.median(dx)) + float(np.median(dy))) / 2.0
        if math.isfinite(pitch) and pitch > 2.0:
            return pitch
    return float(fallback)


def _candidate_support_ratio(
    rectified: np.ndarray,
    points: np.ndarray,
    grid_size: int,
    pitch: float,
    polarity: str | None = None,
) -> tuple[float | None, int]:
    """计算网格点获得真实候选（黑帽/顶帽连通域）支撑的比例。

    这是防"多数锁定"的外部证据：一个规则但错误的网格可以骗过
    任何自洽性检查，却骗不过"点位附近是否真的存在图像结构"。
    没有对应检测器的网格规格返回 (None, 0)（检查不可用，而非通过）。
    返回 (支撑比例, 候选总数)：候选总数供调用方判断检查本身是否可信
    ——重模糊等场景下检测器整体失效，比例为 0 不代表网格错误。
    """

    # 极性已知时按极性选检测器（任意网格规格都可校验）；
    # 未知时沿用旧的按规格映射，保证向后兼容。
    if polarity == "dark":
        candidates = _detect_dark_square_candidates(rectified)
    elif polarity == "bright":
        candidates = _detect_bright_dot_candidates(rectified)
    elif grid_size == 10:
        candidates = _detect_dark_square_candidates(rectified)
    elif grid_size >= 15:
        candidates = _detect_bright_dot_candidates(rectified)
    else:
        return None, 0

    if candidates.shape[0] == 0:
        return 0.0, 0

    tolerance = max(3.0, pitch * 0.25)
    centers = candidates[:, :2]
    supported = 0
    for point in points:
        distances = np.linalg.norm(centers - point, axis=1)
        if float(distances.min()) <= tolerance:
            supported += 1
    return supported / max(points.shape[0], 1), int(candidates.shape[0])


def bundle_adjust_lattice(
    rectified: np.ndarray,
    points: np.ndarray,
    grid_size: int,
    inlier_threshold_ratio: float = 0.18,
    min_threshold_px: float = 2.5,
    radius: int | None = None,
    polarity: str | None = None,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    """晶格光束法平差：用全体阵列点共同估计全局单应几何。

    流程（EM 式两轮）：
    1. 从初始网格做局部精修，只保留有图像证据的观测；
    2. 用这些观测以 Tukey-IRLS 拟合 (col,row) -> (x,y) 的 8 自由度单应
       （相对仿射模型多出的透视项吸收矫正残差）；
    3. 以模型预测为中心重新开窗再精修一轮，再拟合一次得到最终模型
       （窗口重新居中能找回初始网格没盖住的单元；窗口尺寸保持不变，
       因为组件精修的几何限制随窗口缩放，窗口小于单元会拒绝真实目标）；
    4. 有证据且残差在阈值内的点采用观测值（source=candidate_refined），
       其余点吸附到模型预测（source=model_imputed）。

    与 enforce_lattice_consistency 的关键差异：模型只拟合有图像证据的
    观测，无证据点不参与投票，从机制上消除"规则但错误的多数派
    锁定模型"的失效模式；此外还输出候选支撑率作为外部证据校验。

    返回 (最终点位, 诊断信息, 逐点元信息[confidence/source/flags])。
    诊断信息保持 enforce_lattice_consistency 的旧键位并新增
    model/candidate_support_ratio/trusted/observed_ratio 等字段。
    """

    points = np.asarray(points, dtype=np.float32)
    expected_count = grid_size * grid_size
    height, width = rectified.shape[:2]
    fallback_pitch = min(width, height) / max(grid_size + 1, 2)
    pitch = _estimate_grid_pitch(points, grid_size, fallback_pitch)
    threshold = max(float(min_threshold_px), pitch * inlier_threshold_ratio)

    def _passthrough(reason: str, support: float | None) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
        # 平差未执行（证据不足/拟合失败）时不能给出信任背书：
        # 只有"该规格没有支撑检测器"这一种情况保持 trusted=True。
        info = {
            "applied": False,
            "reason": reason,
            "model": "homography",
            "outlier_count": 0,
            "corrected_points": [],
            "pitch_px": round(pitch, 4),
            "threshold_px": round(threshold, 4),
            "inlier_rmse_px": 0.0,
            "candidate_support_ratio": None if support is None else round(float(support), 6),
            "support_check": "unavailable" if support is None else "ok",
            "trusted": bool(support is None),
            "observed_ratio": 0.0,
            "mean_confidence": 0.0,
        }
        meta = [{"confidence": 0.0, "source": "unadjusted", "flags": []} for _ in range(points.shape[0])]
        return points.copy(), info, meta

    if grid_size < 3:
        return _passthrough("grid_size 小于 3，全局模型不可靠", None)
    if points.shape[0] != expected_count:
        return _passthrough(f"点数 {points.shape[0]} 与 {grid_size}x{grid_size} 不符", None)

    indices = np.arange(expected_count)
    lattice_index = np.stack([indices % grid_size, indices // grid_size], axis=1).astype(np.float64)
    min_evidence = max(8, int(expected_count * 0.4))

    # 第一轮：常规窗口精修，收集图像证据。
    # 窗口半径优先用调用方给定值（如 process_image 的名义间距公式），
    # 组件精修的几何上限随窗口缩放，窗口过小会把大尺寸单元误拒。
    radius_pass1 = radius if radius is not None else max(4, int(pitch * 0.32))
    observed1, status1 = _refine_grid_points_with_status(rectified, points, grid_size, radius_pass1, polarity)
    if int(status1.sum()) < min_evidence:
        support, _ = _candidate_support_ratio(rectified, points, grid_size, pitch, polarity)
        return _passthrough("有效图像观测不足，跳过全局平差", support)

    h_first = _fit_homography_irls(lattice_index[status1], observed1[status1].astype(np.float64))
    if h_first is None:
        support, _ = _candidate_support_ratio(rectified, points, grid_size, pitch, polarity)
        return _passthrough("单应拟合失败", support)
    predicted1 = _apply_homography(h_first, lattice_index).astype(np.float32)

    # 第二轮：以模型预测为中心重新开窗精修。窗口尺寸与第一轮一致：
    # 组件精修的方块尺寸上限随窗口缩放，窗口小于单元会把真实目标拒掉。
    observed2, status2 = _refine_grid_points_with_status(rectified, predicted1, grid_size, radius_pass1, polarity)
    if int(status2.sum()) >= min_evidence:
        h_final = _fit_homography_irls(lattice_index[status2], observed2[status2].astype(np.float64))
        if h_final is None:
            h_final, observed2, status2 = h_first, observed1, status1
    else:
        h_final, observed2, status2 = h_first, observed1, status1

    predicted = _apply_homography(h_final, lattice_index).astype(np.float32)
    residual = np.linalg.norm(observed2.astype(np.float64) - predicted.astype(np.float64), axis=1)
    inlier = status2 & (residual <= threshold)
    if int(inlier.sum()) < min_evidence:
        support, _ = _candidate_support_ratio(rectified, points, grid_size, pitch, polarity)
        return _passthrough("晶格模型 inlier 不足，跳过全局平差", support)

    final_points = points.copy()
    final_points[inlier] = observed2[inlier]
    final_points[~inlier] = predicted[~inlier]

    # 逐点元信息：inlier 用残差换算置信度，补位点给固定的中等偏低置信度。
    point_meta: list[dict[str, object]] = []
    corrected: list[dict[str, object]] = []
    for index in range(expected_count):
        if inlier[index]:
            confidence = math.exp(-((residual[index] / threshold) ** 2))
            point_meta.append(
                {"confidence": round(float(confidence), 4), "source": "candidate_refined", "flags": []}
            )
        else:
            # 0.3 的含义：位置来自全局模型，通常准确（模型 RMSE 量级），
            # 但没有局部图像证据背书，下游应降权使用。
            point_meta.append(
                {"confidence": 0.3, "source": "model_imputed", "flags": ["imputed_position"]}
            )
            shift = float(np.linalg.norm(final_points[index] - points[index]))
            corrected.append(
                {
                    "row": int(index // grid_size),
                    "col": int(index % grid_size),
                    "shift_px": round(shift, 4),
                }
            )

    support, candidate_count = _candidate_support_ratio(rectified, final_points, grid_size, pitch, polarity)
    # 支撑率三态判定：
    # - unavailable：该规格没有候选检测器，检查不适用；
    # - inconclusive：检测器整体几乎无候选（如重模糊），而能走到这里说明
    #   逐点精修证据已充分——缺席的是检查手段而非网格质量，不据此判不可信；
    # - ok：候选充足，支撑率 < 0.6 即判不可信（防规则但错误的网格）。
    min_check_candidates = max(6, int(expected_count * 0.3))
    if support is None:
        trusted, support_check = True, "unavailable"
    elif candidate_count < min_check_candidates:
        trusted, support_check = True, "inconclusive"
    else:
        trusted, support_check = support >= 0.6, "ok"
    inlier_residual = residual[inlier]
    info: dict[str, object] = {
        "applied": True,
        "reason": "ok",
        "model": "homography",
        "outlier_count": int((~inlier).sum()),
        "corrected_points": corrected,
        "pitch_px": round(pitch, 4),
        "threshold_px": round(threshold, 4),
        "inlier_rmse_px": round(float(np.sqrt(np.mean(inlier_residual**2))), 6) if inlier_residual.size else 0.0,
        "candidate_support_ratio": None if support is None else round(float(support), 6),
        "support_check": support_check,
        "trusted": bool(trusted),
        "observed_ratio": round(float(status2.sum()) / expected_count, 6),
        "mean_confidence": round(float(np.mean([m["confidence"] for m in point_meta])), 6),
    }
    return final_points, info, point_meta


def extract_roi_measurements(
    image: np.ndarray,
    points: np.ndarray,
    grid_size: int,
    roi_radius: int,
) -> list[dict[str, float | int]]:
    """提取每个阵列点的 ROI 定量特征。

    图像使用 OpenCV BGR 顺序；输出同时给 BGR、Lab、灰度和饱和比例。
    后续安卓端可以先复用这些字段，再决定用 RGB、Lab 还是吸光度模型。
    """

    if image.ndim != 3:
        raise ValueError("extract_roi_measurements 需要 BGR 彩色图像")

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    records: list[dict[str, float | int]] = []

    for index, (x, y) in enumerate(points):
        row = index // grid_size
        col = index % grid_size
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        x0 = max(0, xi - roi_radius)
        x1 = min(width, xi + roi_radius + 1)
        y0 = max(0, yi - roi_radius)
        y1 = min(height, yi + roi_radius + 1)
        roi = image[y0:y1, x0:x1]
        roi_gray = gray[y0:y1, x0:x1]
        roi_lab = lab[y0:y1, x0:x1]

        if roi.size == 0:
            b_mean = g_mean = r_mean = gray_mean = l_mean = a_mean = lab_b_mean = 0.0
            b_median = g_median = r_median = saturation_ratio = local_quality = 0.0
        else:
            # 使用均值和中位数两套指标。中位数对玻璃反光、灰尘更稳。
            b_mean, g_mean, r_mean = [float(v) for v in roi.reshape(-1, 3).mean(axis=0)]
            b_median, g_median, r_median = [float(v) for v in np.median(roi.reshape(-1, 3), axis=0)]
            gray_mean = float(roi_gray.mean())
            l_mean, a_mean, lab_b_mean = [float(v) for v in roi_lab.reshape(-1, 3).mean(axis=0)]
            saturation_ratio = float(np.any(roi >= 250, axis=2).mean())
            local_quality = float(np.clip(roi_gray.std() / 64.0, 0.0, 1.0))

        records.append(
            {
                "row": int(row),
                "col": int(col),
                "x": float(x),
                "y": float(y),
                "b_mean": round(b_mean, 4),
                "g_mean": round(g_mean, 4),
                "r_mean": round(r_mean, 4),
                "b_median": round(b_median, 4),
                "g_median": round(g_median, 4),
                "r_median": round(r_median, 4),
                "gray_mean": round(gray_mean, 4),
                "lab_l_mean": round(l_mean, 4),
                "lab_a_mean": round(a_mean, 4),
                "lab_b_mean": round(lab_b_mean, 4),
                "saturation_ratio": round(saturation_ratio, 6),
                "local_quality": round(local_quality, 6),
            }
        )
    return records


def evaluate_quality(
    original: np.ndarray,
    rectified: np.ndarray,
    records: list[dict[str, float | int]],
    geometry_info: dict[str, object] | None = None,
) -> dict[str, float | str | list[str]]:
    """生成拍摄质量控制指标。

    这些指标用于安卓端提示用户：是否过曝、是否模糊、网格是否可靠。
    geometry_info 为可选的晶格平差诊断（bundle_adjust_lattice 输出）；
    提供时会参与判级并在 reasons 中给出机器可读的降级原因。
    """

    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    saturation_ratio = float(np.any(rectified >= 250, axis=2).mean())
    underexposure_ratio = float((gray <= 5).mean())
    blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    local_quality = float(np.mean([float(r["local_quality"]) for r in records])) if records else 0.0

    exposure_score = 1.0 - min(1.0, saturation_ratio / 0.08)
    blur_score = min(1.0, blur_var / 120.0)
    grid_score = local_quality
    overall = float(np.clip(0.42 * exposure_score + 0.28 * blur_score + 0.30 * grid_score, 0.0, 1.0))

    if overall >= 0.72 and saturation_ratio < 0.06:
        status = "pass"
    elif overall >= 0.45:
        status = "warn"
    else:
        status = "fail"

    # 几何健康度只会下调判级，不会掩盖曝光/清晰度问题。
    reasons: list[str] = []
    severity = {"pass": 0, "warn": 1, "fail": 2}
    if geometry_info is not None:
        support = geometry_info.get("candidate_support_ratio")
        if not bool(geometry_info.get("trusted", True)):
            reasons.append("grid_support_low")
            floor_status = "fail" if (support is not None and float(support) < 0.35) else "warn"
            if severity[floor_status] > severity[status]:
                status = floor_status
        if not bool(geometry_info.get("applied", False)):
            reasons.append("geometry_unadjusted")
            if severity["warn"] > severity[status]:
                status = "warn"
        point_count = max(len(records), 1)
        imputed_ratio = float(geometry_info.get("outlier_count", 0)) / point_count
        if imputed_ratio > 0.2:
            reasons.append("high_imputed_ratio")
            if severity["warn"] > severity[status]:
                status = "warn"

    return {
        "status": status,
        "overall_score": round(overall, 6),
        "exposure_score": round(float(exposure_score), 6),
        "blur_score": round(float(blur_score), 6),
        "grid_score": round(float(grid_score), 6),
        "saturation_ratio": round(saturation_ratio, 6),
        "underexposure_ratio": round(underexposure_ratio, 6),
        "laplacian_variance": round(blur_var, 6),
        "reasons": reasons,
    }


def draw_rectified_debug(rectified: np.ndarray, points: np.ndarray, grid_size: int, roi_radius: int) -> np.ndarray:
    """在矫正图上绘制 ROI 框和序号。"""

    debug = rectified.copy()
    for idx, (x, y) in enumerate(points):
        row = idx // grid_size
        col = idx % grid_size
        center = (int(round(x)), int(round(y)))
        cv2.rectangle(
            debug,
            (center[0] - roi_radius, center[1] - roi_radius),
            (center[0] + roi_radius, center[1] + roi_radius),
            (0, 255, 255),
            1,
        )
        cv2.circle(debug, center, 2, (0, 0, 255), -1)
        if grid_size <= 10:
            cv2.putText(debug, f"{row},{col}", (center[0] + 3, center[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 255, 0), 1)
    return debug


def draw_original_overlay(
    original: np.ndarray,
    region: ChipRegion,
    rectified_points: np.ndarray,
    inverse_matrix: np.ndarray,
    grid_size: int,
) -> np.ndarray:
    """把矫正图中的网格点反投影回原图并绘制。"""

    overlay = original.copy()
    cv2.polylines(overlay, [region.points.astype(np.int32)], isClosed=True, color=(0, 255, 255), thickness=3)

    points = rectified_points.reshape(-1, 1, 2).astype(np.float32)
    projected = cv2.perspectiveTransform(points, inverse_matrix).reshape(-1, 2)

    for idx, (x, y) in enumerate(projected):
        row = idx // grid_size
        col = idx % grid_size
        center = (int(round(float(x))), int(round(float(y))))
        cv2.circle(overlay, center, 5 if grid_size <= 10 else 3, (0, 0, 255), -1)
        if grid_size <= 10:
            cv2.putText(overlay, f"{row},{col}", (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 255, 0), 1)

    return overlay


def write_measurements_csv(csv_path: str | Path, records: Iterable[dict[str, float | int]]) -> None:
    """写出 ROI 定量 CSV。"""

    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(records)
    if not rows:
        raise ValueError("没有 ROI 记录可写出")
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _default_rectified_size(grid_size: int) -> int:
    """根据阵列大小给出矫正图尺寸。

    每个格点约 80 像素，便于可视化和局部 ROI 统计；同时不会太大。
    """

    return int(max(720, min(1600, grid_size * 80)))


def process_image(
    image_path: str | Path,
    grid_size: int,
    output_dir: str | Path,
    config: PipelineConfig | None = None,
) -> dict[str, object]:
    """完整处理单张目标平面图片并输出可视化、CSV 和 JSON。"""

    if config is None:
        config = PipelineConfig(grid_size=grid_size)
    if config.grid_size != grid_size:
        config.grid_size = grid_size

    image_path = Path(image_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = read_image_unicode(image_path)
    rectified_size = config.rectified_size or _default_rectified_size(grid_size)

    region = detect_chip_region(original)
    rectified, _, inverse_matrix = rectify_chip(original, region, rectified_size)

    theoretical_points, unit_polarity = _fit_grid_points_with_polarity(rectified, grid_size, margin_ratio=config.margin_ratio)
    pitch = (rectified_size * (1.0 - 2.0 * config.margin_ratio)) / max(grid_size - 1, 1)
    roi_radius = max(3, int(round(pitch * config.roi_radius_ratio)))
    # V2.0：晶格光束法平差内部完成两轮重定心精修 + 全局单应拟合，
    # 输出逐点置信度/来源，并用候选支撑率做外部证据校验。
    # 窗口半径沿用与历史版本一致的名义间距公式，保持精修行为等价。
    refined_points, lattice_info, point_meta = bundle_adjust_lattice(
        rectified, theoretical_points, grid_size, radius=max(4, int(pitch * 0.32)), polarity=unit_polarity
    )

    records = extract_roi_measurements(rectified, refined_points, grid_size=grid_size, roi_radius=roi_radius)
    quality = evaluate_quality(original, rectified, records, geometry_info=lattice_info)
    grid_points = [
        {
            "row": int(index // grid_size),
            "col": int(index % grid_size),
            "x": round(float(point[0]), 4),
            "y": round(float(point[1]), 4),
            "confidence": point_meta[index]["confidence"],
            "source": point_meta[index]["source"],
            "flags": point_meta[index]["flags"],
        }
        for index, point in enumerate(refined_points)
    ]

    overlay = draw_original_overlay(original, region, refined_points, inverse_matrix, grid_size)
    roi_debug = draw_rectified_debug(rectified, refined_points, grid_size, roi_radius)

    write_image_unicode(output_dir / "overlay_grid.jpg", overlay)
    write_image_unicode(output_dir / "rectified_chip.jpg", rectified)
    write_image_unicode(output_dir / "roi_debug.jpg", roi_debug)
    write_measurements_csv(output_dir / "values.csv", records)

    # PG-Quant V1.0：在定位结果上做逐单元定量与单元级质量标记。
    # 旧输出（values.csv 等）全部保留，定量结果写入独立的 quant_* 文件。
    quant_records, quant_meta = quantify_rectified(rectified, refined_points, grid_size)
    quant_outputs = write_quant_outputs(output_dir, quant_records, quant_meta)
    # 定量可视化：状态叠图 + 强度/SNR 热图 + 校正后颜色分布图。
    quant_outputs.update(write_quant_visualizations(output_dir, rectified, quant_records, quant_meta))

    result: dict[str, object] = {
        "algorithm": "PG-Grid V2.0 bundle-adjusted localization with PG-Quant V1.0",
        "image_path": str(image_path),
        "grid_size": int(grid_size),
        "unit_polarity": unit_polarity,
        "point_count": int(len(refined_points)),
        "grid_points": grid_points,
        "lattice_consistency": lattice_info,
        "quant_summary": {
            "unit_count": quant_meta["unit_count"],
            "reliable_count": quant_meta["summary"]["reliable_count"],
            "reliable_ratio": quant_meta["summary"]["reliable_ratio"],
            "illumination_model": quant_meta["illumination_model"],
            "illumination_uniformity": quant_meta["illumination_uniformity"],
        },
        "roi_radius": int(roi_radius),
        "rectified_size": int(rectified_size),
        "chip_region": {
            "method": region.method,
            "score": float(region.score),
            "points": region.points.tolist(),
        },
        "quality": quality,
        "outputs": {
            "overlay_grid": str(output_dir / "overlay_grid.jpg"),
            "rectified_chip": str(output_dir / "rectified_chip.jpg"),
            "roi_debug": str(output_dir / "roi_debug.jpg"),
            "values_csv": str(output_dir / "values.csv"),
            "result_json": str(output_dir / "result.json"),
            "quant_values_csv": quant_outputs["quant_values_csv"],
            "quant_result_json": quant_outputs["quant_result_json"],
            "quant_overlay": quant_outputs["quant_overlay"],
            "quant_heatmap_intensity": quant_outputs["quant_heatmap_intensity"],
            "quant_heatmap_snr": quant_outputs["quant_heatmap_snr"],
            "quant_color_map": quant_outputs["quant_color_map"],
        },
        "neural_detector_slot": {
            "enabled": False,
            "note": "后续可把 detect_chip_region 替换为 MobileSAM/SAM2/轻量四角关键点模型。",
        },
    }

    with (output_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
