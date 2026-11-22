# 传感器约定与标定记录（Sensor Conventions）

本文档说明惯性镜头使用的传感器轴系、角速度单位、时间戳与相机外参，以及固件更换时
的标定流程。文中所有数值示例由 `src/lens/docExamples.ts` 计算生成（`npm run
docs:conventions` 重新生成文末 JSON 块），并由 `tests/conventions.test.ts` 与
`npm run validate` 机器核对——文档与代码不一致时检查失败。

## 图形入口与命令行

- **图形入口（主）**：`npm run dev`，浏览器打开终端打印的地址（默认
  `http://127.0.0.1:5173/`），即惯性镜头的镜头跟踪窗口。惯性辅助的启停、重放、
  标定状态、角标与虚拟地面接触点都在该窗口内。
- **命令行（仅辅助）**：
  - `npm run validate` —— 惯性资料校验与传感约定维护检查（10 项），可在清洁目录执行；
  - `npm run docs:conventions` —— 重新生成本文档的示例 JSON 块；
  - `npm test` —— 全部自动化测试。

## 坐标轴与单位

- **机体系（body）**：x 前、y 左、z 上。估计器内部全部使用机体系。
- **相机系（camera）**：x 右、y 下、z 前。
- **陀螺**：机体系角速度，**rad/s**。
- **加计**：机体系比力，**m/s²**；重力常数取 9.80665 m/s²。
- **时间戳**：相机帧时间为物理时间；IMU 原始样本的时间戳比物理时间晚
  `timeOffset` 秒，即 `物理时刻 = 原始戳 + timeOffset`。切片、积分一律先减
  `timeOffset` 再比较（见 `src/estimation/imu.ts` 的 `imuSlice`）。

每个固件的**传感约定**描述它上报的原始读数如何换算到机体系（`src/core/conventions.ts`）：

```
v_body = R_bs · (v_sensor · 单位比例)
```

- `R_bs`：带符号置换矩阵（|det| = 1）；已固化的约定必须是正常旋转（det = +1）。
- 角速度单位：`rad/s` 或 `deg/s`；加速度单位：`m/s^2` 或 `g`。

### 两款固件

| 固件 | 轴系 | 角速度 | 加速度 |
| --- | --- | --- | --- |
| fw-A（出厂） | 传感轴与机体一致（R_bs = I） | rad/s | m/s² |
| fw-B（换装） | 传感 x = −机体 y，传感 y = +机体 z，传感 z = −机体 x | deg/s | g |

两套约定之间编译出**可逆转换器**（`converterBetween` + `invertConverter`）：
`v_to = R · (v_from · 比例)`，逆转换器精确还原。把换轴、积分与成像串联后，用
2π 闭环转动检查：正确链路姿态闭合、中段成像吻合；若转换器被误用两次
（双重换轴），闭环中段的重投影立即超差报警（校验项 `closed-loop-double-swap`）。

## 相机外参

`R_bc` 把相机坐标映射到机体系，`t_bc` 是相机原点在内的机体坐标；相机中心
`C = p_wb + R_wb · t_bc`。旋转外参可由手眼标定观测，**平移外参不可由旋转观测**，
始终沿用输入值（界面台账如实标注）。

## 标定流程：静止预览，转动固化

同一款固件录制两组素材（`src/lens/record.ts`）：

1. **静止素材**：估计重力方向（|f| ≈ g、比力沿机体 +z）。静止画面无法认出绕竖轴
   的正负，因此加载器保留**全部**一致的轴系候选（含镜像）作为 `preview`，
   此时 `solidifyConvention` 拒绝固化。
2. **绕两条已知轴有序转动的素材**：旋转不可交换，镜像候选积分出的末态姿态与
   已知末态不符，被排除。只剩**唯一**候选且为正常旋转时才允许 `committed`，
   标定记录代际（generation）+1。

标定记录（`CalibrationRecord`）是惯性预积分器（`src/lens/preintegration.ts`）与
快门补偿器（`src/lens/shutter.ts`）**共同引用**的同一份记录：二者都用记录里被
固化的约定换算，未知固件的样本一律丢弃，绝不套用默认轴系。

## 导出 / 读入与过期快照

- `exportCalibration` / `importCalibration`：格式显式校验，缺字段、镜像轴系、
  已固化但候选数不为 1 等情况一律抛错，不用默认值修补。读入后角速度积分与
  快门去斜采用与原件完全相同的被证实约定（校验项 `export-import`）。
- 每个姿态/快照都盖有记录 id 与代际。固件更换后代际递增，**旧换算的迟到快照**
  会被角标（corner marker）与虚拟地面接触点（ground contact）同时拒绝
  （`src/lens/overlays.ts`，校验项 `stale-snapshot`）。

## 数值示例（机器生成，机器核对）

以下数值由 `computeDocExamples()` 计算；改动轴系、单位、时差或外参的代码会使
本块与实现不一致，维护检查随即失败。

<!-- BEGIN EXAMPLES -->
```json
{
  "extrinsics": {
    "R_bc": [
      0,
      0,
      1,
      -1,
      0,
      0,
      0,
      -1,
      0
    ],
    "t_bc": [
      0.015,
      -0.012,
      0.006
    ],
    "camPoint": [
      0.1,
      0.2,
      2
    ],
    "bodyPoint": [
      2.015,
      -0.112,
      -0.194
    ]
  },
  "firmwareB": {
    "omegaBodyRadS": [
      0.1,
      -0.2,
      0.3
    ],
    "rawGyroDegS": [
      11.459156,
      17.188734,
      -5.729578
    ],
    "roundTripRadS": [
      0.1,
      -0.2,
      0.3
    ],
    "accelBody": [
      0.5,
      -0.3,
      9.80665
    ],
    "rawAccelG": [
      0.030591,
      1,
      -0.050986
    ]
  },
  "converterAB": {
    "rawA": [
      0.05,
      -0.1,
      0.2
    ],
    "asB": [
      5.729578,
      11.459156,
      -2.864789
    ],
    "backToA": [
      0.05,
      -0.1,
      0.2
    ],
    "R": [
      0,
      -1,
      0,
      0,
      0,
      1,
      -1,
      0,
      0
    ],
    "gyroScale": 57.29578
  },
  "timestamps": {
    "cameraT0": 1,
    "cameraT1": 1.05,
    "timeOffset": 0.018,
    "rawWindowStart": 0.982,
    "rawWindowEnd": 1.032,
    "sliceSamples": 101,
    "sliceFirstT": 0.982,
    "sliceLastT": 1.032
  },
  "attitudeFigure": {
    "t": 1.25,
    "eulerDeg": [
      28.343951,
      8.114595,
      -11.923116
    ],
    "omegaBody": [
      -0.244759,
      -0.42048,
      -0.022637
    ],
    "accelBody": [
      -1.69433,
      -2.06469,
      9.073783
    ]
  },
  "reprojectionFigure": {
    "pointWorld": [
      2.345795,
      2.022432,
      -0.123044
    ],
    "u": 160,
    "v": 120,
    "depth": 3,
    "omegaBody": [
      -0.244759,
      -0.42048,
      -0.022637
    ],
    "readoutPerRowUs": 25,
    "deskewedU": 160.016862,
    "deskewedV": 119.684634
  }
}
```
<!-- END EXAMPLES -->
