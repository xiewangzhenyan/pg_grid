# Android 迁移指南（Kotlin + Jetpack Compose）

> 适用范围：把 PG-Grid/PG-Quant 管线迁移到 Android 端（Kotlin 为主语言、
> Jetpack Compose UI）。本文只给操作流程与注意事项，不含实现代码。
> 阅读前提：算法侧已完成 V2.0 + PG-Quant V1.0 + 基准体系（见 README 与
> `docs/next_stage_design.md`）。

---

## 0. 总体架构决策

**推荐路线：C++ 算法核心 + 薄 JNI 边界 + Kotlin/Compose 应用层。**

| 路线 | 说明 | 结论 |
|------|------|------|
| A. Kotlin + OpenCV Android SDK 直接重写 | 用 org.opencv 的 Java/Kotlin API 重写全管线 | 不推荐：矩阵运算（IRLS/DLT/SVD）无 NumPy 等价物，需引入 EJML/Multik，行为与 Python 版难以对齐 |
| **B. C++ 重写 + JNI/NDK** | 管线移植为 C++（OpenCV C++ API），JNI 暴露给 Kotlin | **推荐**：`cv2.xxx` 与 `cv::xxx` 几乎一一对应；同一 OpenCV 内核保证行为一致；核心可复用到 iOS/桌面 |
| C. Chaquopy（Android 跑 Python） | 直接打包 Python 解释器 | 仅适合原型验证：包体大（+60MB）、冷启动慢、生产不可用 |

选 B 的根本原因：本项目从 V1 起刻意保持 **OpenCV + NumPy + 标准库** 三件套，
就是为了这条迁移路径。管线中没有任何 Python 独有依赖。

**分层职责**：

```
Kotlin/Compose 层   相机采集、拍摄引导 UI、结果展示（热图/叠图用 Compose Canvas 画）、
                    JSON 解析（kotlinx.serialization）
JNI 边界（极薄）     传入：图像字节 + 网格规格；传出：result/quant JSON 字符串
C++ 核心（一个 .so）  L0-L5 全管线：检测→矫正→光束法平差→定量→QC→JSON
```

JSON 字符串作为 JNI 传出格式是刻意选择：与桌面输出契约完全一致，
parity 测试可以逐字段比对，且避免维护复杂的 JNI 结构体映射。

---

## 1. 迁移前：Python 侧准备（先做完这些再动 C++）

### 1.1 契约冻结与 golden 资产

1. 宣布 `result.json` / `quant_result.json` schema 冻结（当前字段已稳定；
   后续只增不改）。
2. 建立 golden 资产目录（建议 `parity/`）：
   - 输入：`examples/` 两张样例图 + 用 `pg_benchmark.make_scene` 生成的
     若干代表性场景（含旋转、遮挡、光照梯度各一，seed 固定并记录）；
   - 期望输出：当前 Python 版对每个输入产出的 `result.json` 与
     `quant_result.json`。
3. 写明比对容差（见 §6.1）。这套资产就是 C++ 移植和 Android 端的验收标准。

### 1.2 参数集中化

当前阈值散落在函数默认值、`PipelineConfig`、`QuantConfig` 中。移植前收拢为
单一配置文档/结构（一处列出全部数值与依据），C++ 端照抄同一份，防止两端
漂移。重点参数：两档检测阈值与面积上限、候选几何过滤范围、组合枚举上限、
光束法平差阈值（0.18×pitch、支撑率 0.6、inconclusive 候选下限 0.3N）、
定量几何比例（ROI 0.18 / 环 0.30-0.44）、QC 各 flags 阈值。

### 1.3 NumPy → C++ 对应清单

移植 checklist（本项目实际用到的全部 NumPy 能力）：

| Python | C++ 对应 |
|--------|----------|
| `np.linalg.svd`（DLT 求解） | `cv::SVD::compute`（注意 full_matrices 语义差异） |
| `np.linalg.lstsq`（多项式/仿射拟合） | `cv::solve(..., DECOMP_SVD)` 或正规方程 |
| `np.polyfit(deg=1)` | 手写 2×2 正规方程（5 行） |
| `np.median` / `np.percentile` | `std::nth_element`（注意 NumPy 线性插值 percentile 与 nth_element 的取值差异，容差覆盖即可） |
| MAD/加权均值/argsort | 标准库组合 |
| `np.random.default_rng` | 仅基准使用，端侧不需要 |

### 1.4 性能画像（桌面先测）

单线程逐阶段计时（检测/矫正/平差/定量/渲染），对照设计文档预算表
（合计 ≤300ms @1600px 中端 SoC）。Python 慢点（逐点循环精修、组合枚举）
在 C++ 会快 1-2 个数量级，真正要盯的是大图形态学与 warp 的开销。
确定输入策略：**相机 4000×3000 原图先降采样到 ~1600 做检测，矫正图尺寸
维持 800/1200 不变**（矫正图尺寸变化会改变全部像素阈值的语义！）。

---

## 2. C++ 核心移植步骤

按依赖顺序移植并逐模块 parity 验证（每个模块移植完立即和 Python 输出对数）：

1. **几何工具**：`order_quad_points`、单应工具（`_apply_homography`、
   Hartley 归一化、加权 DLT + Tukey IRLS）。用 golden 场景的中间量验证。
2. **主区域检测** `detect_chip_region`（两档阈值逻辑照抄）。
3. **矫正** `rectify_chip`（`cv::getPerspectiveTransform` + `warpPerspective`，
   插值方式保持 `INTER_CUBIC`）。
4. **候选检测**：黑帽/顶帽 + 内部 Otsu + 几何过滤。
5. **晶格拟合**：旋转扫描、轴聚类、规则子集选择（含组合枚举上限与
   **缺失簇补全**）、投影回退路径。
6. **光束法平差** `bundle_adjust_lattice`（含证据加权、支撑率三态判定）。
7. **定量** `pg_quant`（圆形 ROI/环形背景、多项式光照场、flags）。
8. **QC 与 JSON 输出**（字段名逐字对齐 Python 版）。

注意事项：

- **迭代顺序与浮点**：IRLS/迭代重拟合的收敛路径对浮点顺序敏感，C++ 与
  NumPy 的 SVD 实现不同会带来 <0.1px 级差异——这是 parity 容差存在的原因，
  不要试图逐位一致。
- **Otsu/形态学**：OpenCV 同版本下两端结果逐位一致；锁定两端 OpenCV
  大版本号（建议 4.9+）。
- **JPEG 解码差异**：不同平台 libjpeg 解码可能有 ±1 像素值差异，golden
  资产优先用 PNG。
- 端侧不需要移植：`pg_quant_viz`（Compose Canvas 直接画）、`pg_benchmark`
  与标注/评估工具（留在桌面）。

---

## 3. Android 工程搭建

1. **OpenCV 依赖**：官方 `org.opencv:opencv:4.9+`（Maven Central），或
   包体敏感时用 opencv-mobile 精简构建（约 10MB，裁掉视频/DNN 模块——
   本项目只用 core/imgproc，完全够）。
2. **NDK + CMake**：核心管线编为单一 `libpggrid.so`；C++17；
   `-ffast-math` **禁用**（会破坏与桌面的数值一致性）。
3. **JNI 设计**（保持极薄）：
   - 入参：`ByteArray`（编码图像）或 `Bitmap` + grid_size + 可选配置 JSON；
   - 出参：`String`（result JSON，内嵌 quant 结果或分两个方法）；
   - 错误：C++ 异常在 JNI 边界捕获并转为带 error 字段的 JSON，
     绝不让异常穿透 JNI。
4. **Kotlin 侧**：`kotlinx.serialization` 定义与 `result.json` /
   `quant_result.json` 同名字段的 `data class`（字段名已稳定，直接映射）；
   算法调用放 `Dispatchers.Default`，单帧预期 100-300ms。

---

## 4. 相机链路（Android 特有，坑最多的一层）

1. **通道顺序（第一大坑）**：CameraX 输出 `YUV_420_888`，OpenCV 约定
   **BGR**，Android Bitmap 是 **RGBA**。转换链路建议
   `YUV_420_888 → cv::Mat(NV21/I420) → cvtColor(COLOR_YUV2BGR_*)`，
   并在开发期用一张纯红测试卡验证 R/B 没有对调——通道错了颜色定量全错
   但灰度看不出来。
2. **EXIF/旋转**：CameraX 的 `imageInfo.rotationDegrees` 必须在进管线前
   应用；不要依赖算法的 ±6° 旋转扫描去吞 90° 的方向错误。
3. **AE/AWB 锁定（对定量最关键）**：自动曝光/白平衡会让同一目标两次拍摄
   读数不可比。通过 Camera2 interop 在取景稳定后锁定
   `CONTROL_AE_LOCK` / `CONTROL_AWB_LOCK` 再拍摄；condition 允许时进一步
   固定 ISO/快门。这是端侧定量重复性的第一影响因素。
4. **分辨率**：拍摄用高分辨率静态帧（非预览流）；进管线前按 §1.4 策略
   降采样。
5. **实时拍摄引导**：`quality.reasons` 机器码（`tilt_high`、
   `grid_support_low`、过曝等）当初就是为驱动 UI 提示设计的——预览流上
   跑轻量检查（只到检测/QC 层），Compose 显示"请降低倾角/避开反光"。

---

## 5. Compose UI 层

- **结果渲染不移植 pg_quant_viz**：热图/颜色图/叠图全部数据都在 JSON 里
  （grid_points + units 的逐单元数值/flags），用 Compose `Canvas` 按同样
  规则绘制即可，还能做点击单元查看详情等交互。
- 建议展示：矫正图 + ROI 叠层（按 quant_reliable 着色）、强度/SNR 热图、
  校正颜色分布图、QC 状态卡（status + reasons 的人话翻译）。
- 图像展示用 `Bitmap`（从 JNI 返回矫正图字节，或 Kotlin 侧自行 warp）。

---

## 6. 测试策略

### 6.1 Golden parity（生命线）

- 资产放 `androidTest` 的 assets：§1.1 的输入图 + 期望 JSON；
- 设备上跑完整管线，逐字段比对：
  - 坐标容差 **0.5px**；光度量容差 **1 个量化级**；比例/置信度 **0.01**；
  - 布尔与枚举字段（source/flags/status/reasons）**必须完全一致**；
- 桌面 C++ 版先过同一套 parity，再上设备——分两步能区分
  "移植错了"和"平台差异"。

### 6.2 其余

- C++ 核心用 GoogleTest 移植关键单元测试（至少：缺失簇补全、光束法平差
  的透视/遮挡/空白守卫三件套、平场校正梯度稳定性——这些测试的合成场景
  构造方法在 Python 测试里都有，照抄）；
- 设备矩阵：至少一台低端（4GB RAM）+ 一台主流；关注低端机的内存峰值
  （1600px 处理链 float 中间图 ~30MB 级，注意及时 release Mat）；
- 性能回归：androidTest 里对 golden 输入计时，超预算即失败。

---

## 7. 常见坑清单（按踩坑概率排序）

1. BGR/RGBA 通道对调（灰度正常、颜色定量全错，最隐蔽）。
2. YUV 转换用错变体（NV21 vs I420 vs YV12，花屏或色偏）。
3. 忘记 rotationDegrees，靠算法旋转扫描吞方向 → 90° 时直接失败。
4. AE/AWB 未锁定 → 定量重复性差被误判为算法问题。
5. `Bitmap` 与 `Mat` 生命周期：JNI 侧忘记 release → OOM。
6. 两端 OpenCV 版本不一致 → Otsu/形态学细微差异撑破 parity 容差。
7. `-ffast-math` 或 NEON 激进优化改变浮点行为。
8. JPEG 中间落盘引入压缩损失（端上全程内存传递，不落盘）。
9. 哈希/随机：任何"用哈希当种子"的逻辑都要用确定性哈希
   （本仓库在基准模块踩过 `hash()` 跨进程随机化的坑，教训已写入代码注释）。
10. 混淆（R8）剥掉 JNI 入口 → keep 规则要覆盖 native 方法所在类。

---

## 8. 分阶段里程碑

| 阶段 | 内容 | 退出标准 |
|------|------|----------|
| M0 | Python 侧准备（§1） | golden 资产入库；参数清单文档化 |
| M1 | C++ 核心移植（§2） | 桌面 C++ 过全部 parity + GoogleTest |
| M2 | Android 壳（§3） | 设备上 golden parity 全绿 |
| M3 | 相机链路（§4） | 实拍→结果全流程可用；AE/AWB 锁定生效 |
| M4 | UI 与引导（§5） | Compose 渲染 + reasons 驱动的拍摄引导 |
| M5 | 硬化 | 低端机内存/性能达标；性能回归测试入 CI |

风险提示：M1 是主工作量（估全程 60%）；M3 的相机参数锁定在部分国产 ROM
上行为不一致，需要真机验证矩阵。
