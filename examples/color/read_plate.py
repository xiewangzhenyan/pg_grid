"""从 examples/color/ 的图像文件读出逐孔浓度，验证信息确实在图里。

这个脚本只吃图像和"每行加了多少标准品"这两样东西，不碰仿真内部状态。
真值 JSON 只在最后比对时打开一次。

    python examples/color/read_plate.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from pg_calibration import fit_four_parameter_logistic          # noqa: E402
from pg_grid import process_image                                # noqa: E402

GRID = 15
# 配板时就知道的量：每一行（一条微流通道）加了哪个浓度的标准品。
ANALYTE_PER_ROW = np.concatenate([[0.0], np.exp(np.linspace(np.log(0.5), np.log(500.0), 14))])


def spot_channels(output_dir: Path) -> np.ndarray:
    """逐孔原始 ROI 中位数，返回 (225, 3) RGB。"""
    import csv
    rows = list(csv.DictReader((output_dir / "quant_values.csv").open(encoding="utf-8")))
    return np.asarray(
        [[float(r["roi_r_median"]), float(r["roi_g_median"]), float(r["roi_b_median"])] for r in rows],
        dtype=np.float64,
    )


def main() -> None:
    work = HERE / "_read"
    # 1) 定位：样品板、配对空白板、零浓度板各跑一次完整管线
    results = {}
    for tag, name in (("sample", "plate.jpg"), ("blank", "plate_blank.jpg"), ("zero", "zero.jpg")):
        results[tag] = process_image(image_path=HERE / name, grid_size=GRID,
                                     output_dir=work / tag)
    lat = results["sample"]["lattice_consistency"]
    print(f"定位: 检出 {results['sample']['point_count']}/225 孔  "
          f"支撑率={lat['candidate_support_ratio']}  可信={lat['trusted']}")

    # 2) 吸光度：样品相对配对空白。金阵列的消光在两图中相同，此处对消。
    s, b, z = (spot_channels(work / t) for t in ("sample", "blank", "zero"))
    absorbance = -np.log10(np.clip(s, 1e-6, None) / np.clip(b, 1e-6, None))
    zero_absorbance = -np.log10(np.clip(z, 1e-6, None) / np.clip(b, 1e-6, None))

    # 信号通道取吸光度跨度最大的一个（换显色剂时它会变，不能写死）
    channel = int(np.argmax(absorbance.max(axis=0) - absorbance.min(axis=0)))
    y = absorbance[:, channel]
    print(f"信号通道: {'RGB'[channel]}   空白板 A = {zero_absorbance[:, channel].mean():.4f}"
          f" ± {zero_absorbance[:, channel].std():.4f}")

    # 3) 标定曲线：横轴是配板时已知的浓度，纵轴是刚从图里读出的吸光度
    analyte = np.repeat(ANALYTE_PER_ROW, GRID)
    model = fit_four_parameter_logistic(analyte, y,
                                        blank_response=float(zero_absorbance[:, channel].mean()))
    print(f"5PL 曲线: c50={model.c50:.2f} hill={model.b:.3f} g={model.g:.2f}\n")

    # 4) 反算浓度并与真值比对
    recovered = model.invert(y).reshape(GRID, GRID)
    truth = np.asarray(json.load((HERE / "plate_truth.json").open(encoding="utf-8"))["concentrations"])

    print(f"{'行':>3s}{'加样 ng/mL':>12s}{'反算均值':>11s}{'回收率':>9s}{'CV':>8s}{'显色产物µM(真值)':>17s}")
    for row in range(GRID):
        nominal = ANALYTE_PER_ROW[row]
        got = recovered[row][np.isfinite(recovered[row])]
        chromogen = truth[row * GRID:(row + 1) * GRID].mean()
        if nominal <= 0 or got.size < 3:
            print(f"{row+1:>3d}{nominal:>12.3g}{'—':>11s}{'—':>9s}{'—':>8s}{chromogen:>17.1f}")
            continue
        mean = float(got.mean())
        print(f"{row+1:>3d}{nominal:>12.3g}{mean:>11.3g}{mean/nominal*100:>8.1f}%"
              f"{got.std()/abs(mean)*100:>7.1f}%{chromogen:>17.1f}")


if __name__ == "__main__":
    main()
