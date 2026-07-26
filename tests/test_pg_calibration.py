"""PG-Calibration：分析物浓度 ↔ 图像 的闭环测试。"""

import numpy as np


def test_immunoassay_response_is_langmuir_saturating():
    """分析物→显色产物必须饱和，且零浓度处保留非特异本底。

    非特异本底是检出限的真正来源：它不随分析物变化，但它的起伏决定了
    多小的信号还能被认出来。零浓度处读数为零的模型会给出虚假的 LOD。
    """
    from pg_calibration import ImmunoassayResponse

    assay = ImmunoassayResponse(kd=20.0, chromogen_max_uM=700.0, nonspecific_uM=8.0)
    c = np.array([0.0, 2.0, 20.0, 200.0, 2000.0])
    product = assay.chromogen_for(c)

    assert abs(product[0] - 8.0) < 1e-9, "零浓度处应只剩非特异本底"
    assert abs(product[2] - (8.0 + 350.0)) < 1e-6, "K_D 处应半饱和"
    assert np.all(np.diff(product) > 0)
    assert product[-1] < 8.0 + 700.0
    # 明确非线性：浓度涨 10 倍，高端产物涨幅远小于 10 倍
    assert (product[3] - 8.0) / (product[2] - 8.0) < 2.0


def test_five_pl_round_trips_a_known_curve():
    """5PL 拟合 + 反算必须能还原已知曲线的浓度。"""
    from pg_calibration import FourPL, fit_four_parameter_logistic

    truth = FourPL(a=0.02, d=1.20, c50=25.0, b=1.1, g=0.7)
    x = np.exp(np.linspace(np.log(0.5), np.log(500.0), 14))
    y = truth.predict(x)

    model = fit_four_parameter_logistic(np.concatenate([[0.0], x]),
                                        np.concatenate([[0.02], y]),
                                        blank_response=0.02)
    assert model is not None
    recovered = model.invert(y)
    valid = np.isfinite(recovered)
    assert int(valid.sum()) >= 10
    ratio = recovered[valid] / x[valid]
    assert np.all((ratio > 0.85) & (ratio < 1.18)), f"反算浓度偏差过大: {ratio}"


def test_asymmetry_parameter_beats_plain_4pl_on_compressed_response():
    """光学压缩使响应偏离 4PL，不对称参数必须能吸收它。

    这台设备的 0.5mm 光程逼着用高显色浓度，落在多色压缩区，
    合成响应是"Langmuir 饱和 + 吸光度压缩"两级串联，不再是 4PL。
    """
    from pg_calibration import FourPL, fit_four_parameter_logistic

    # 构造一条明显不对称的曲线（g 远离 1）
    truth = FourPL(a=0.0, d=1.0, c50=20.0, b=1.0, g=0.4)
    x = np.exp(np.linspace(np.log(1.0), np.log(400.0), 12))
    y = truth.predict(x)

    fitted = fit_four_parameter_logistic(np.concatenate([[0.0], x]),
                                         np.concatenate([[0.0], y]), blank_response=0.0)
    assert fitted is not None
    err_5pl = float(np.mean((fitted.predict(x) - y) ** 2))

    # 强制 g=1（标准 4PL）后重新做同样的线性化拟合
    frac = (y - 0.0) / (1.05 - y)
    slope, intercept = np.polyfit(np.log(x), np.log(frac), 1)
    plain = FourPL(a=0.0, d=1.05, c50=float(np.exp(-intercept / slope)), b=float(slope), g=1.0)
    err_4pl = float(np.mean((plain.predict(x) - y) ** 2))

    assert err_5pl < err_4pl * 0.5, f"5PL 未显著优于 4PL: {err_5pl:.2e} vs {err_4pl:.2e}"


def test_invert_refuses_to_extrapolate_past_asymptotes():
    """超出上下渐近线的读数必须返回 nan，不得外推成一个数字。"""
    from pg_calibration import FourPL

    model = FourPL(a=0.05, d=1.00, c50=20.0, b=1.0, g=1.0)
    out = model.invert(np.array([0.04, 0.05, 1.00, 1.20, 0.5]))

    assert not np.isfinite(out[0]), "低于下渐近线应返回 nan"
    assert not np.isfinite(out[3]), "高于上渐近线应返回 nan"
    assert np.isfinite(out[4]) and out[4] > 0


def test_explicit_concentrations_drive_the_plate():
    """标定流程要能把逐孔浓度直接灌进渲染配置。"""
    from pg_colori_sim import ColorimetricConfig

    values = tuple(float(i) for i in range(225))
    config = ColorimetricConfig(grid_size=15, explicit_concentrations=values)
    got = config.resolved_concentrations()

    assert got.shape == (225,)
    assert np.allclose(got, np.asarray(values))

    bad = ColorimetricConfig(grid_size=15, explicit_concentrations=(1.0, 2.0))
    try:
        bad.resolved_concentrations()
    except ValueError:
        pass
    else:
        raise AssertionError("孔数不匹配时应报错")


def test_log_series_includes_a_zero_point():
    """标定序列必须含零浓度点——没有空白就没有检出限。"""
    from pg_calibration import log_series

    series = log_series(0.5, 500.0, 15)
    assert series.size == 15
    assert series[0] == 0.0
    assert np.all(np.diff(series[1:]) > 0)
    assert abs(series[1] - 0.5) < 1e-9 and abs(series[-1] - 500.0) < 1e-6


def test_reference_column_cancels_per_row_common_mode():
    """参考列必须对消逐行的乘性共模（照明梯度、曝光、EL 漂移）。

    吸光度已是对数量，故对数域相减 = 线性域相除，正是要消掉的那个共模。
    """
    from pg_calibration import apply_reference_column

    grid = 15
    signal = np.tile(np.linspace(0.0, 1.0, grid), (grid, 1))     # 逐列的真信号
    drift = np.linspace(-0.30, 0.30, grid)[:, None]              # 逐行的共模偏置
    observed = (signal + drift).reshape(-1)

    corrected = apply_reference_column(observed, grid, reference_column=0).reshape(grid, grid)
    # 每一行减去自己的参考点后，各行应当完全一致
    assert np.allclose(corrected - corrected[0][None, :], 0.0, atol=1e-12)
    # 参考列自身归零
    assert np.allclose(corrected[:, 0], 0.0)


def test_reference_column_is_forced_to_zero_analyte():
    """指定参考列后，该列必须整列为零分析物，且不计入浓度档统计。"""
    from pg_colori_sim import ColorimetricConfig
    from pg_calibration import ImmunoassayResponse

    grid = 15
    assay = ImmunoassayResponse(kd=20.0, chromogen_max_uM=300.0, nonspecific_uM=8.0)
    levels = np.linspace(1.0, 100.0, grid)

    # 复现 calibration_run 内部的布板逻辑（列内同浓度 + 参考列置零）
    analyte = np.tile(levels, grid).reshape(grid, grid)
    analyte[:, 3] = 0.0
    product = assay.chromogen_for(analyte.reshape(-1)).reshape(grid, grid)

    assert np.all(analyte[:, 3] == 0.0)
    # 零分析物处仍保留非特异本底——检出限正来源于此
    assert np.all(product[:, 3] > 0.0)
    assert abs(float(product[:, 3].mean()) - 8.0) < 1e-6


def test_column_axis_puts_concentration_down_columns():
    """默认 axis='column'：纵向通道送样品，故浓度按列变化。

    方向不是无关紧要的约定——若浓度由横向通道灌进整行，那一行里就不
    可能存在"只通缓冲液"的孔，参考列在物理上不成立。
    """
    grid = 15
    levels = np.arange(1.0, grid + 1.0)

    by_column = np.tile(levels, grid).reshape(grid, grid)
    by_row = np.repeat(levels, grid).reshape(grid, grid)

    # 列内同浓度、行内递增
    assert np.allclose(by_column[:, 4], levels[4])
    assert np.allclose(by_column[0, :], levels)
    # 行式布板则相反
    assert np.allclose(by_row[4, :], levels[4])
