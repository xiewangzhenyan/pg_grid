"""生成中性的二维规则阵列样例图。

这些图片只用于测试通用网格定位算法，不包含任何应用背景信息。
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def save_image(path: Path, image: np.ndarray) -> None:
    """写出 PNG 图片，并兼容 Windows 上的非 ASCII 路径。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"Failed to encode image: {path}")
    encoded.tofile(str(path))


def draw_rotated_panel(image: np.ndarray, center: tuple[int, int], size: tuple[int, int], angle: float) -> np.ndarray:
    """绘制带轻微旋转的矩形目标区域，并返回四角坐标。"""

    rect = (center, size, angle)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillConvexPoly(image, box, (214, 214, 214))
    cv2.polylines(image, [box], isClosed=True, color=(150, 150, 150), thickness=3)
    return box


def create_dark_square_grid(path: Path) -> None:
    """生成 10x10 暗色方块阵列样例。"""

    image = np.full((900, 900, 3), 18, dtype=np.uint8)
    draw_rotated_panel(image, center=(450, 450), size=(620, 620), angle=-3.5)

    # 为了保持样例简单，目标点画在近似正视区域内，算法仍会先做区域定位和矫正。
    xs = np.linspace(210, 690, 10)
    ys = np.linspace(210, 690, 10)
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            jitter_x = (col % 3 - 1) * 1.5
            jitter_y = (row % 3 - 1) * 1.2
            cx = int(round(x + jitter_x))
            cy = int(round(y + jitter_y))
            cv2.rectangle(image, (cx - 10, cy - 10), (cx + 10, cy + 10), (88, 104, 118), -1)
            cv2.rectangle(image, (cx - 10, cy - 10), (cx + 10, cy + 10), (70, 82, 94), 1)

    # 加入少量细线干扰，用于测试局部精修不会被线状结构明显拉偏。
    for x in [320, 458, 596]:
        cv2.line(image, (x, 190), (x + 12, 710), (120, 130, 142), 2)

    save_image(path, image)


def create_bright_point_grid(path: Path) -> None:
    """生成 15x15 亮点阵列样例。"""

    image = np.full((1000, 1000, 3), 12, dtype=np.uint8)
    draw_rotated_panel(image, center=(500, 500), size=(680, 680), angle=2.5)

    xs = np.linspace(190, 810, 15)
    ys = np.linspace(190, 810, 15)
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            cx = int(round(x + (col % 4 - 1.5) * 0.8))
            cy = int(round(y + (row % 4 - 1.5) * 0.8))
            cv2.circle(image, (cx, cy), 7, (236, 236, 236), -1)
            cv2.circle(image, (cx, cy), 9, (185, 185, 185), 1)

    save_image(path, image)


def main() -> None:
    """生成全部样例图片。"""

    output_dir = Path(__file__).resolve().parent
    create_dark_square_grid(output_dir / "synthetic_10x10_dark_squares.png")
    create_bright_point_grid(output_dir / "synthetic_15x15_bright_points.png")
    print(f"Synthetic samples written to: {output_dir}")


if __name__ == "__main__":
    main()
