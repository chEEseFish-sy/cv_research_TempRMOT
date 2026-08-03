# TempRMOT 复现执行状态

更新时间：2026-07-22（Asia/Shanghai）  
项目目录：`/home/curtis/projects/TempRMOT`  
当前阶段：**A–I 已通过；J 已完成全量评测，HOTA 与官方值相差 +0.209**

本文记录已经实际执行并通过验证的结果。完整流程与后续门禁请参阅
[`TEMPRMOT_REPRODUCTION_GUIDE.md`](TEMPRMOT_REPRODUCTION_GUIDE.md)。

## 1. 阶段总览

| 阶段 | 状态 | 主要结果 |
|---|---|---|
| A | 已通过 | 源码清理并建立 Git 基线 |
| B | 已通过 | Python、PyTorch、CUDA 与 GPU 计算验证通过 |
| C | 已通过 | MSDA CUDA 扩展针对 PyTorch 2.8 / sm_120 编译和梯度检查通过 |
| D | 已通过 | 依赖可导入，`pip check` 通过，环境已锁定 |
| E | 已通过 | 离线加载、单 GPU 推理和 TrackEval 等兼容补丁完成 |
| F | 已通过 | RoBERTa 与官方 TempRMOT checkpoint 离线可读，哈希已记录 |
| G | 已通过 | Refer-KITTI-V2 与 KITTI 数据完整性检查错误数为 0 |
| H | 已通过 | checkpoint 严格匹配；单 query 的 3 帧、297 帧推理与 HOTA 闭环通过 |
| I | 已通过 | 4 个测试视频、862 条表达式全部推理完成，输出完整性检查通过 |
| J | 已评测 | HOTA 35.249；较官方 35.04 高 0.209，略超出预设 ±0.20 容差 0.009 |

## 2. 阶段 D：依赖与环境

- Conda 环境：`temprmot`
- Python：3.10.20
- PyTorch：2.8.0+cu128
- torchvision：0.23.0+cu128
- NumPy：2.2.6
- gdown：6.0.0
- `python -m pip check`：`No broken requirements found.`
- 项目导入：`main`、`inference`、`models`、`datasets`、
  `MultiScaleDeformableAttention` 全部成功。
- 环境锁文件：`/home/curtis/experiments/TempRMOT/requirements-lock.txt`，共 71 行。

相关提交：

- `9bb064f fix: remove unused torchtext dependency`

## 3. 阶段 E：兼容性补丁

已完成并验证：

- RoBERTa 本地路径参数化，禁止隐式联网下载。
- torchvision backbone 使用现代 `weights=` API，默认不下载 ImageNet 权重。
- 可信 checkpoint 在 PyTorch 2.8 下显式使用 `weights_only=False`。
- 推理默认单 worker、GPU 0，并支持视频、表达式和帧数筛选。
- 推理结果目录固定为 `results_{checkpoint_stem}`。
- TrackEval 图像路径参数化并兼容 NumPy 2。
- 训练日志按实际帧数记录，表达式候选列表排序。
- 新增本机 Refer-KITTI-V2 推理与评测配置。

相关提交：

- `25dd7e7 fix: make text encoder and backbone loading configurable`
- `9aa11a2 fix: load trusted checkpoints under PyTorch 2.8`
- `c1795fa fix: support single-GPU inference and arbitrary checkpoint names`
- `2b3190c fix: remove local paths and NumPy aliases from RMOT TrackEval`
- `71fde7d fix: make training logging frame-count aware`
- `3c0b1f1 chore: add local Refer-KITTI-V2 run configs`

## 4. 阶段 F：模型资产

### RoBERTa

- 目录：`/home/curtis/checkpoints/roberta-base`
- 来源：`FacebookAI/roberta-base`
- 固定 revision：`e2da8e2f811d1448a5b465c236feacd80ffbac7b`
- `model.safetensors` 大小：498,818,054 字节
- SHA-256：`5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81`
- 完全离线加载成功；词表 50,265，hidden size 768。

### TempRMOT 官方 Refer-KITTI-V2 checkpoint

- 文件：`/home/curtis/checkpoints/TempRMOT/checkpoint_rk2.pth`
- 大小：1,096,316,824 字节
- SHA-256：`4953bd2b2966f752b7d5dced94216f0ddacd04158b1c85c3fc3534a333e9581b`
- `torch.load(..., weights_only=False)` 成功。
- checkpoint 字段：`args`、`epoch`、`lr_scheduler`、`model`、`optimizer`。
- 模型状态项：969。

哈希文件：`/home/curtis/checkpoints/TempRMOT/checkpoint_rk2.pth.sha256`。

## 5. 阶段 G：Refer-KITTI-V2 数据

最终目录：`/home/curtis/datasets/refer-kitti-v2`

实际验收结果：

| 检查项 | 结果 |
|---|---:|
| KITTI 训练图像 | 8,008 |
| KITTI 图像序列 | 21 |
| V2 修正版标签 TXT | 7,162 |
| 表达 JSON | 9,758 |
| 训练列表图像 | 5,172 |
| 训练视频 | 17 |
| 测试视频 | 4 |
| 测试 seqmap | 862 |
| 校验错误 | **0** |

KITTI 官方压缩包：

- 文件名：`data_tracking_image_2.zip`
- 大小：15,813,146,295 字节
- SHA-256：`9206250fbcd0074328bf589ec81ab0b0681608db59537f024b1b9ff44e81d53c`
- ZIP CRC：全部通过。

数据校验脚本：

```bash
python tools/validate_refer_kitti_v2.py
```

脚本提交：

- `79958bf chore: add Refer-KITTI-V2 integrity validator`
- `b3fcd80 fix: align dataset validation with Refer-KITTI metadata`
- `60326f1 chore: enforce Refer-KITTI-V2 file baselines`

最终日志：`/home/curtis/experiments/TempRMOT/refer-kitti-v2-validation.log`。

日志中另有 181 条 `reference_without_box` 警告。这些记录来自官方表达 JSON：
对象在视频其他帧存在，但遮挡或离开画面的个别帧没有对应框。它们不是未知对象、
坏 JSON 或缺失标签，因此保留为可审计警告，不修改官方数据。

Google Drive 全量表达清单中另有 20 组同名文件 ID，均位于训练视频 `0007`；
逐对下载并计算内容哈希后确认每组内容完全相同，因此最终唯一 JSON 文件数为 9,758。

## 6. 当前 Git 状态与边界

兼容性代码和数据校验器已按功能拆分提交。模型、数据、实验日志和环境锁文件位于
仓库外部，不进入 Git。

`docs/` 中原有文档移动属于文档整理工作，不与代码补丁提交混合。

## 7. 阶段 H–J 实测结果

### H：模型、checkpoint 与单 query 闭环

- 完整模型离线构建成功；RoBERTa 保持 frozen。
- 模型放入 GPU 后占用约 676.40 MiB，清理后回到 0 MiB。
- checkpoint epoch 49，共 969 个状态项。
- 以 checkpoint 内保存的结构参数构建模型：`extra_track_attn=False`、`hist_len=4`、
  `sampler_steps=[60,80,90]`、`sampler_lengths=[5,5,5,5]`。
- 忽略 Transformers 版本差异产生的可重建缓冲区
  `text_encoder.embeddings.position_ids` 后，strict load 的 missing/unexpected keys 均为空。
- 视频 `0005` 第一条表达式完成 3 帧 smoke test 和 297 帧完整推理。
- 单 query 指标：HOTA 67.591、DetA 64.713、AssA 70.651。

### I：862-query 全量推理

- 推理进程退出码：0。
- seqmap、`predict.txt`、`gt.txt`：均为 862/862。
- 45 个空 `predict.txt` 表示阈值过滤后无预测目标；均为存在且合法的评测输入。
- 完整性审计：缺失 0、坏行/坏文件 0、重复 frame-ID 0、NaN/Inf 0。
- 预测行数 408,378；GT 行数 491,286。

### J：全量 TrackEval

| 指标 | 结果 |
|---|---:|
| HOTA | 35.249 |
| DetA | 23.412 |
| AssA | 53.191 |
| DetRe | 34.475 |
| DetPr | 41.474 |
| AssRe | 58.436 |
| AssPr | 82.952 |
| LocA | 90.335 |
| RHOTA | 42.798 |

官方 HOTA 为 35.04，本次绝对差为 +0.209（约 +0.60%）。流程与数据完整性均通过，
但按指南预先设定的 ±0.20 严格门槛，结果超出 0.009，因此记录为高度接近的复现结果，
不写成数值完全一致。完整报告见
[`TEMPRMOT_REPRODUCTION_RESULTS_2026-07-22.md`](TEMPRMOT_REPRODUCTION_RESULTS_2026-07-22.md)。
