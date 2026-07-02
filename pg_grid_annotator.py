"""PG-Grid 半自动标注工具。

推荐工作流：
1. 先用 `--init-only` 从当前算法预测点生成初始 annotation.json；
2. 再不加 `--init-only` 打开交互窗口，人工拖动明显偏移的点；
3. 保存后的 annotation.json 可交给 `pg_grid_evaluate.py` 计算定位误差。

交互快捷键：
- 鼠标左键：选择并拖动最近点；
- `s`：保存；
- `u`：撤销一步；
- `r`：恢复初始点；
- `q` 或 Esc：退出。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from pg_grid import read_image_unicode
from pg_grid_eval import PointRecord, load_prediction_points, write_annotation_json


def create_initial_annotation(
    image_path: str | Path,
    prediction_path: str | Path,
    grid_size: int,
    output_path: str | Path,
) -> dict[str, object]:
    """从预测点生成初始标注文件。

    这个函数不声称预测点就是真值；它只是把当前算法输出复制成可编辑标注，
    让人工只需要修正偏移点，而不是从零开始逐点点击。
    """

    prediction_grid_size, points = load_prediction_points(prediction_path)
    if prediction_grid_size is not None and int(prediction_grid_size) != int(grid_size):
        raise ValueError(f"预测 grid_size={prediction_grid_size} 与指定 grid_size={grid_size} 不一致")
    return write_annotation_json(output_path, image_path=image_path, grid_size=grid_size, points=points)


class AnnotationSession:
    """OpenCV 交互标注会话。

    只保存点位和简单撤销栈，避免把 GUI 状态和评价逻辑混在一起。
    """

    def __init__(
        self,
        image: np.ndarray,
        image_path: str | Path,
        points: list[PointRecord],
        grid_size: int,
        output_path: str | Path,
    ) -> None:
        self.image = image
        self.image_path = Path(image_path)
        self.points = [dict(point) for point in points]
        self.initial_points = [dict(point) for point in points]
        self.grid_size = int(grid_size)
        self.output_path = Path(output_path)
        self.history: list[list[PointRecord]] = []
        self.selected_index: int | None = None
        self.dragging = False
        self.window_name = "PG-Grid Annotator"

    def _push_history(self) -> None:
        """保存当前点位，用于撤销。"""

        self.history.append([dict(point) for point in self.points])
        if len(self.history) > 50:
            self.history.pop(0)

    def _nearest_point_index(self, x: int, y: int) -> int:
        """找到距离鼠标最近的点。"""

        coordinates = np.asarray([[float(p["x"]), float(p["y"])] for p in self.points], dtype=np.float32)
        distances = np.linalg.norm(coordinates - np.asarray([[x, y]], dtype=np.float32), axis=1)
        return int(np.argmin(distances))

    def _move_selected_point(self, x: int, y: int) -> None:
        """移动当前选中点，并限制在图像范围内。"""

        if self.selected_index is None:
            return
        height, width = self.image.shape[:2]
        self.points[self.selected_index]["x"] = float(np.clip(x, 0, width - 1))
        self.points[self.selected_index]["y"] = float(np.clip(y, 0, height - 1))

    def on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        """OpenCV 鼠标事件回调。"""

        if event == cv2.EVENT_LBUTTONDOWN:
            self._push_history()
            self.selected_index = self._nearest_point_index(x, y)
            self.dragging = True
            self._move_selected_point(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self._move_selected_point(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self._move_selected_point(x, y)
            self.dragging = False

    def render(self) -> np.ndarray:
        """渲染当前标注状态。"""

        canvas = self.image.copy()
        for index, point in enumerate(self.points):
            center = (int(round(float(point["x"]))), int(round(float(point["y"]))))
            color = (0, 0, 255) if index == self.selected_index else (0, 255, 0)
            cv2.circle(canvas, center, 4, color, -1)
            if self.grid_size <= 10:
                label = f"{int(point['row'])},{int(point['col'])}"
                cv2.putText(canvas, label, (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.32, color, 1)

        cv2.putText(
            canvas,
            "s:save  u:undo  r:reset  q/esc:quit",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )
        return canvas

    def save(self) -> None:
        """保存标注文件。"""

        write_annotation_json(self.output_path, image_path=self.image_path, grid_size=self.grid_size, points=self.points)

    def undo(self) -> None:
        """撤销一步。"""

        if self.history:
            self.points = self.history.pop()
            self.selected_index = None

    def reset(self) -> None:
        """恢复到打开窗口时的初始点。"""

        self._push_history()
        self.points = [dict(point) for point in self.initial_points]
        self.selected_index = None

    def run(self) -> None:
        """启动 OpenCV 交互窗口。"""

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        while True:
            cv2.imshow(self.window_name, self.render())
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                self.save()
            elif key == ord("u"):
                self.undo()
            elif key == ord("r"):
                self.reset()
        cv2.destroyWindow(self.window_name)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="PG-Grid 半自动标注工具")
    parser.add_argument("--image", required=True, help="矫正图路径，例如 outputs/10x10_a/rectified_chip.jpg")
    parser.add_argument("--prediction", required=True, help="PG-Grid result.json 或 values.csv")
    parser.add_argument("--grid", required=True, type=int, choices=[1, 10, 15], help="阵列规格")
    parser.add_argument("--output", required=True, help="输出 annotation.json 路径")
    parser.add_argument("--init-only", action="store_true", help="只生成初始标注，不打开交互窗口")
    return parser.parse_args()


def main() -> None:
    """命令行入口。"""

    args = parse_args()
    annotation = create_initial_annotation(
        image_path=args.image,
        prediction_path=args.prediction,
        grid_size=args.grid,
        output_path=args.output,
    )
    if args.init_only:
        print(f"初始标注已写出: {Path(args.output).resolve()}")
        return

    image = read_image_unicode(args.image)
    session = AnnotationSession(
        image=image,
        image_path=args.image,
        points=annotation["points"],
        grid_size=args.grid,
        output_path=args.output,
    )
    session.run()
    print(f"标注文件路径: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
