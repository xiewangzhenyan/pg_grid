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

    # 大面积黑背景会拉低整体亮度；成像视野圆环又会产生弱反光。
    # 因此这里使用较高分位数，只保留 均匀背景下真正明亮的目标平面主体。
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    percentile_threshold = float(np.percentile(blurred, 94))
    threshold = max(22.0, percentile_threshold)
    mask = (blurred > threshold).astype(np.uint8) * 255

    # 合并孔洞、流道、局部反光，但不要把圆形视野外圈也合进去。
    kernel_size = max(7, int(min(width, height) * 0.008))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _fallback_center_region(width, height)

    image_area = width * height
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        # 真实目标平面主体在当前成像视野图中约占整图 5%-12%；
        # 大于 20% 的候选通常是成像视野圆环或暗箱反光区域。
        if area < image_area * 0.002 or area > image_area * 0.20:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        aspect = w / max(h, 1)
        if not 0.35 <= aspect <= 2.8:
            continue

        rect_mask = np.zeros_like(gray, dtype=np.uint8)
        cv2.drawContours(rect_mask, [contour], -1, 255, thickness=-1)
        mean_brightness = cv2.mean(gray, mask=rect_mask)[0]
        # 分数同时考虑面积和亮度，避免选到暗箱圆环反光。
        score = math.sqrt(area) * (mean_brightness + 1.0)
        candidates.append((score, contour))

    if not candidates:
        return _fallback_center_region(width, height)

    _, best_contour = max(candidates, key=lambda item: item[0])
    rect = cv2.minAreaRect(best_contour)
    box = cv2.boxPoints(rect)
    ordered = order_quad_points(box)

    # 适度外扩，确保目标点和目标平面边缘不会被裁掉。
    center = ordered.mean(axis=0)
    expanded = center + (ordered - center) * 1.10
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)

    return ChipRegion(points=order_quad_points(expanded), score=float(rect[1][0] * rect[1][1]), method="opencv_bright_region")


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

    selected: list[float] = []
    for idx in order:
        if all(abs(float(idx) - existing) >= min_distance for existing in selected):
            selected.append(float(idx))
            if len(selected) == count:
                break

    if len(selected) < count:
        return None

    centers = np.array(sorted(selected), dtype=np.float32)
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


def _select_regular_axis_from_clusters(
    clusters: list[tuple[float, float, int]],
    count: int,
    length: int,
) -> np.ndarray | None:
    """从轴向簇中选择最符合规则晶格的一组中心。

    这里枚举 10 个候选簇的组合，并拟合 `start + pitch * i`。
    评分优先选择残差小、支持强、起止位置合理的组合。
    """

    if len(clusters) < count:
        return None

    # 防组合爆炸：螺丝、边缘碎片可能产生大量杂散簇，簇数一多，
    # C(N, count) 会迅速失控（例如 C(24,10) 约 200 万组合，实测需要一分钟以上）。
    # 真实阵列的行/列簇有约 grid_size 个成员、支持度远高于孤立误检，
    # 因此按支持度只保留最强的若干簇即可，同时必须恢复位置升序，
    # 因为下面的直线拟合假定组合内的中心是按坐标递增排列的。
    max_clusters = 16
    if len(clusters) > max_clusters:
        strongest = sorted(clusters, key=lambda item: item[1], reverse=True)[:max_clusters]
        clusters = sorted(strongest, key=lambda item: item[0])

    best: tuple[float, np.ndarray] | None = None
    index_axis = np.arange(count, dtype=np.float32)
    min_pitch = length * 0.045
    max_pitch = length * 0.095
    min_start = length * 0.080
    max_start = length * 0.250
    max_end = length * 0.930

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


def _fit_dark_square_grid(rectified: np.ndarray, grid_size: int) -> np.ndarray | None:
    """用暗方块候选点拟合 10x10 规则网格。"""

    height, width = rectified.shape[:2]
    candidates = _detect_dark_square_candidates(rectified)
    if candidates.shape[0] < grid_size * 4:
        return None

    weights = candidates[:, 2]
    x_clusters = _cluster_axis_candidates(candidates[:, 0], weights, width)
    y_clusters = _cluster_axis_candidates(candidates[:, 1], weights, height)
    xs = _select_regular_axis_from_clusters(x_clusters, grid_size, width)
    ys = _select_regular_axis_from_clusters(y_clusters, grid_size, height)

    if xs is None or ys is None:
        return None
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float32)


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
    x0 = max(0, int(round(x)) - radius)
    x1 = min(width, int(round(x)) + radius + 1)
    y0 = max(0, int(round(y)) - radius)
    y1 = min(height, int(round(y)) + radius + 1)
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


def fit_grid_points(rectified: np.ndarray, grid_size: int, margin_ratio: float = 0.085) -> np.ndarray:
    """在矫正图中拟合阵列点。

    15x15 亮点通常表现为亮点；10x10 暗色目标平面上的目标点通常表现为暗色小块。
    因此这里根据 grid_size 选择亮/暗能量图，再用投影曲线找行列中心。
    如果投影不可靠，则回退到纯物理规则网格。
    """

    height, width = rectified.shape[:2]
    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)

    if grid_size == 10:
        dark_square_grid = _fit_dark_square_grid(rectified, grid_size)
        if dark_square_grid is not None:
            return dark_square_grid

    if grid_size >= 15:
        # 15x15 亮点：明亮点位是目标。
        energy = gray.astype(np.float32)
    else:
        # 10x10 目标平面：小色块/目标点一般比 均匀背景暗。
        energy = 255.0 - gray.astype(np.float32)

    # 背景扣除，减弱大面积光照渐变对投影峰的影响。
    blur_size = max(31, int(min(width, height) * 0.08))
    if blur_size % 2 == 0:
        blur_size += 1
    background = cv2.GaussianBlur(energy, (blur_size, blur_size), 0)
    enhanced = cv2.normalize(energy - background, None, 0, 255, cv2.NORM_MINMAX)

    x_projection = enhanced.sum(axis=0)
    y_projection = enhanced.sum(axis=1)
    xs = _find_axis_centers(x_projection, grid_size, width, margin_ratio)
    ys = _find_axis_centers(y_projection, grid_size, height, margin_ratio)

    if xs is None or ys is None:
        return generate_grid_points(grid_size, width, height, margin_ratio=margin_ratio)

    return np.array([[x, y] for y in ys for x in xs], dtype=np.float32)


def refine_grid_points(rectified: np.ndarray, points: np.ndarray, grid_size: int, radius: int) -> np.ndarray:
    """在每个理论点周围做局部中心微调。

    微调必须是“小步”的：全局网格位置来自物理模型，局部只允许在小窗口内修正。
    这样可以避免反光、气泡、脏点把某个 ROI 拉到错误位置。
    """

    gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    refined = points.copy().astype(np.float32)
    polarity_bright = grid_size >= 15

    for idx, (x, y) in enumerate(points):
        if grid_size == 10:
            square_center = _refine_one_10x10_dark_square_center(gray, float(x), float(y), radius)
            if square_center is not None:
                refined[idx] = square_center
                continue

        x0 = max(0, int(round(x)) - radius)
        x1 = min(width, int(round(x)) + radius + 1)
        y0 = max(0, int(round(y)) - radius)
        y1 = min(height, int(round(y)) + radius + 1)
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

    return refined


def enforce_lattice_consistency(
    points: np.ndarray,
    grid_size: int,
    inlier_threshold_ratio: float = 0.18,
    min_threshold_px: float = 2.5,
    max_iterations: int = 5,
) -> tuple[np.ndarray, dict[str, object]]:
    """用全局规则晶格约束修正被局部干扰拉偏的点位。

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


def evaluate_quality(original: np.ndarray, rectified: np.ndarray, records: list[dict[str, float | int]]) -> dict[str, float | str]:
    """生成拍摄质量控制指标。

    这些指标用于安卓端提示用户：是否过曝、是否模糊、网格是否可靠。
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

    return {
        "status": status,
        "overall_score": round(overall, 6),
        "exposure_score": round(float(exposure_score), 6),
        "blur_score": round(float(blur_score), 6),
        "grid_score": round(float(grid_score), 6),
        "saturation_ratio": round(saturation_ratio, 6),
        "underexposure_ratio": round(underexposure_ratio, 6),
        "laplacian_variance": round(blur_var, 6),
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

    theoretical_points = fit_grid_points(rectified, grid_size, margin_ratio=config.margin_ratio)
    pitch = (rectified_size * (1.0 - 2.0 * config.margin_ratio)) / max(grid_size - 1, 1)
    roi_radius = max(3, int(round(pitch * config.roi_radius_ratio)))
    refined_points = refine_grid_points(rectified, theoretical_points, grid_size, radius=max(4, int(pitch * 0.32)))
    # 局部精修可能被细长暗线、阴影或高光拉偏个别点；
    # 最后用全局规则晶格约束把明显偏离的点吸附回晶格模型位置。
    refined_points, lattice_info = enforce_lattice_consistency(refined_points, grid_size)

    records = extract_roi_measurements(rectified, refined_points, grid_size=grid_size, roi_radius=roi_radius)
    quality = evaluate_quality(original, rectified, records)
    grid_points = [
        {
            "row": int(index // grid_size),
            "col": int(index % grid_size),
            "x": round(float(point[0]), 4),
            "y": round(float(point[1]), 4),
        }
        for index, point in enumerate(refined_points)
    ]

    overlay = draw_original_overlay(original, region, refined_points, inverse_matrix, grid_size)
    roi_debug = draw_rectified_debug(rectified, refined_points, grid_size, roi_radius)

    write_image_unicode(output_dir / "overlay_grid.jpg", overlay)
    write_image_unicode(output_dir / "rectified_chip.jpg", rectified)
    write_image_unicode(output_dir / "roi_debug.jpg", roi_debug)
    write_measurements_csv(output_dir / "values.csv", records)

    result: dict[str, object] = {
        "algorithm": "PG-Grid V1.4 OpenCV grid with lattice-consistency correction",
        "image_path": str(image_path),
        "grid_size": int(grid_size),
        "point_count": int(len(refined_points)),
        "grid_points": grid_points,
        "lattice_consistency": lattice_info,
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
        },
        "neural_detector_slot": {
            "enabled": False,
            "note": "后续可把 detect_chip_region 替换为 MobileSAM/SAM2/轻量四角关键点模型。",
        },
    }

    with (output_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result
