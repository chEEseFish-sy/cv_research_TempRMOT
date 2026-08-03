# TempRMOT Refer-KITTI-V2 复现结果

日期：2026-07-22（Asia/Shanghai）  
项目目录：`/home/curtis/projects/TempRMOT`  
复现范围：官方 checkpoint 推理复现；不包含从头训练复现

## 1. 结论

官方 `checkpoint_rk2.pth` 已在本机 RTX 5060 Laptop GPU 上完成 Refer-KITTI-V2
全部 4 个测试视频、862 条自然语言表达式的推理和定制 TrackEval 评测。

本次 HOTA 为 **35.249**，官方报告值为 **35.04**，绝对差为 **+0.209**，
相对差约为 **+0.60%**。该结果证明官方权重推理链和评测链已完整复现；不过按复现指南
预先设定的 ±0.20 严格容差，当前结果超出 0.009，因此结论为“高度接近官方指标”，
而不是“数值完全一致”。

## 2. 运行环境

| 项目 | 实测值 |
|---|---|
| 操作系统 | Ubuntu 22.04 on WSL2 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，8,151 MiB |
| Python | 3.10.20 |
| PyTorch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| CUDA toolkit | 12.8 |
| GPU compute capability | 12.0（sm_120） |
| GCC / G++ | 11.4.0 |
| Conda 环境 | `temprmot` |

MSDA CUDA 扩展使用 `TORCH_CUDA_ARCH_LIST=12.0` 编译。为兼容 PyTorch 2.8，
`AT_DISPATCH_FLOATING_TYPES(value.type(), ...)` 已更新为
`AT_DISPATCH_FLOATING_TYPES(value.scalar_type(), ...)`。

## 3. 固定模型资产

### RoBERTa

- 路径：`/home/curtis/checkpoints/roberta-base`
- revision：`e2da8e2f811d1448a5b465c236feacd80ffbac7b`
- `model.safetensors` SHA-256：
  `5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81`
- 推理时保持 frozen，完全离线加载。

### TempRMOT checkpoint

- 路径：`/home/curtis/checkpoints/TempRMOT/checkpoint_rk2.pth`
- 大小：1,096,316,824 字节
- SHA-256：`4953bd2b2966f752b7d5dced94216f0ddacd04158b1c85c3fc3534a333e9581b`
- epoch：49
- 模型状态项：969

checkpoint 内保存的关键结构参数为：

```text
extra_track_attn=False
hist_len=4
sampler_steps=[60, 80, 90]
sampler_lengths=[5, 5, 5, 5]
```

仓库原测试脚本中的 `extra_track_attn=True`、`hist_len=8` 与该 checkpoint 不一致，
会产生 36 个缺失的 `update_attn/norm4` 状态项。最终推理使用 checkpoint 自身保存的参数。
旧版 Transformers checkpoint 中的 `text_encoder.embeddings.position_ids` 是可重建缓冲区；
忽略该缓冲区后 strict load 的 missing/unexpected keys 均为空。

## 4. 分阶段验证

### 模型与 GPU

- 完整模型离线构建成功。
- RoBERTa 参数全部 frozen。
- 模型 GPU 常驻显存约 676.40 MiB。
- checkpoint 加载后的显存峰值约 928.81 MiB。
- 全部参数为有限值，无 NaN/Inf。

### 单 query 闭环

测试对象：

```text
video: 0005
expression: automobiles-are-located-to-the-left
frames: 297
```

结果：

| 指标 | 数值 |
|---|---:|
| HOTA | 67.591 |
| DetA | 64.713 |
| AssA | 70.651 |
| Dets / GT_Dets | 734 / 742 |
| IDs / GT_IDs | 28 / 28 |

由此确认图像读取、文本编码、逐帧跟踪、结果写出和 TrackEval 路径全部闭环。

## 5. 全量推理完整性

| 检查项 | 结果 |
|---|---:|
| 测试视频 | 4 |
| seqmap 表达式 | 862 |
| 已执行表达式 | 862 |
| `predict.txt` | 862 |
| `gt.txt` | 862 |
| 缺失结果对 | 0 |
| 坏行或坏文件 | 0 |
| 空 `predict.txt` | 45 |
| 空 `gt.txt` | 0 |
| 预测总行数 | 408,378 |
| GT 总行数 | 491,286 |
| 推理退出码 | 0 |

45 个空预测文件表示对应文本查询在 object 0.6、referring 0.4、area 100 的阈值过滤后
没有保留目标。文件本身存在且格式语义合法，已作为无预测结果参与 TrackEval。

## 6. 全量 TrackEval 指标

评测范围：862 条 seqmap；仓库自带定制 `MotChallenge2DBox`；HOTA metric。

| 指标 | 本次结果 |
|---|---:|
| HOTA | **35.249** |
| DetA | 23.412 |
| AssA | 53.191 |
| DetRe | 34.475 |
| DetPr | 41.474 |
| AssRe | 58.436 |
| AssPr | 82.952 |
| LocA | 90.335 |
| RHOTA | 42.798 |
| HOTA(0) | 38.885 |
| LocA(0) | 85.992 |
| HOTALocA(0) | 33.438 |
| Dets | 408,378 |
| GT_Dets | 491,286 |
| IDs | 19,278 |
| GT_IDs | 8,043 |

TrackEval 退出码为 0，862 条序列全部完成。

## 7. 与官方指标比较

| 项目 | HOTA |
|---|---:|
| 官方报告值 | 35.04 |
| 本次复现 | 35.249 |
| 绝对差 | +0.209 |
| 相对差 | 约 +0.60% |

本次差异很小，但比预设 ±0.20 门槛多 0.009。现有证据已排除缺失 query、坏文件、
非有限坐标、权重缺失和评测进程失败。后续若要追查最后 0.009，优先核对官方发布指标
的小数精度、定制 TrackEval 版本以及论文所述 query averaging 与代码 detection-weighted
combine 的差异；不得通过改阈值来事后贴合结果。

## 8. 结果与日志路径

```text
/home/curtis/experiments/TempRMOT/rk2_official/inference.log
/home/curtis/experiments/TempRMOT/rk2_official/trackeval-full.log
/home/curtis/experiments/TempRMOT/rk2_official/results_checkpoint_rk2/
/home/curtis/experiments/TempRMOT/rk2_official/results_checkpoint_rk2/pedestrian_summary.txt
/home/curtis/experiments/TempRMOT/rk2_official/results_checkpoint_rk2/pedestrian_detailed.csv
```

`pedestrian_summary.txt` 是最简洁的最终指标证据；`pedestrian_detailed.csv` 保存逐表达式
明细，`trackeval-full.log` 保存完整运行配置和 combined 输出。

## 9. 复现边界

本报告证明的是：官方代码兼容后，使用官方 checkpoint 在完整 Refer-KITTI-V2 测试集上的
推理与评测复现。它不证明从随机初始化开始训练能够复现论文训练曲线或最终权重；训练 smoke
test 和多卡严格训练属于后续 K/L 阶段。
