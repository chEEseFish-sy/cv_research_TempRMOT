# TempRMOT 时序顺序初步验证：任务交接文档

> 更新时间：2026-07-23（Asia/Shanghai）  
> 项目目录：`/home/curtis/projects/TempRMOT`  
> 新任务目标：执行多阶段事件描述与目标轨迹“时序顺序对齐”的小规模可行性验证。  
> 本轮范围：完成 2 个基础样本、8 组受控表达的 Smoke/Pilot 验证；不训练模型、不做盲法复核、不做正式 Confirmatory 实验。

---

## 1. 新任务开始时必须先读

按顺序阅读：

1. 本文档；
2. `docs/research/TEMPRMOT_ORDER_SENSITIVITY_EXPERIMENT_PLAN.md`；
3. `docs/TEMPRMOT_REPRODUCTION_RESULTS_2026-07-22.md`；
4. `inference.py` 中 `Detector.__init__`、`Detector.detect` 和主函数的表达遍历部分；
5. `configs/local_temp_rmot_test_rk2.sh`。

完整实验计划很长，本轮只执行其中 P0–P4 和最小 P5，不进入 Confirmatory 阶段。

---

## 2. 当前真实进度

### 2.1 已完成

- TempRMOT 运行环境已配置完成。
- PyTorch 2.8、CUDA 12.8 和 MSDA CUDA 扩展已适配。
- 官方 TempRMOT Checkpoint 已成功加载。
- Refer-KITTI-V2 数据校验通过。
- 4 个官方测试视频、862 条表达的全量推理完成。
- 862 对 `predict.txt` / `gt.txt` 完整，无缺失结果对。
- TrackEval 全量评测完成，HOTA 为 35.249；官方报告值为 35.04。
- 时序顺序敏感性实验的完整协议已经写入：
  `docs/research/TEMPRMOT_ORDER_SENSITIVITY_EXPERIMENT_PLAN.md`。

### 2.2 尚未完成

- 尚未为顺序实验修改 `inference.py`。
- 尚未导出阈值过滤前的 Refer 分数。
- 尚未建立时序实验数据 overlay。
- 尚未筛选第一个有效多阶段轨迹样本。
- 尚未构造 correct/paraphrase/orderless/swapped 表达。
- 尚未运行任何顺序对照实验。
- 尚未完成 Pilot，更没有 Confirmatory 结论。

因此，新任务不得把“实验方案已经设计”误写为“已证明模型不理解顺序”。

---

## 3. 本轮研究问题

本轮只判断方向是否值得继续：

> 当目标、事件词和视频保持不变，仅改变事件先后顺序时，TempRMOT 对真实目标轨迹的响应是否出现可测差异？

本轮允许三种结果：

1. 模型对顺序交换有明显响应；
2. 模型对顺序交换响应很小；
3. 数据、文本或分数接口不足以判断。

第三种也是有效结论，不得为了得到“模型不理解顺序”而挑选案例或调整阈值。

---

## 4. 本轮明确不做的内容

- 不训练或微调 TempRMOT。
- 不修改模型结构。
- 不实现新的事件 Token、记忆模块或损失函数。
- 不做30–60样本的正式统计实验。
- 暂不做第二标注者或24小时后的盲法复核。
- 不把两个样本的结果写成论文级结论。
- 不使用 MLLM 自动决定事件真假。
- 不直接修改原 Refer-KITTI-V2 数据目录。

---

## 5. 当前固定资产与配置

### 5.1 模型与数据

```text
项目：/home/curtis/projects/TempRMOT
数据：/home/curtis/datasets/refer-kitti-v2
Checkpoint：/home/curtis/checkpoints/TempRMOT/checkpoint_rk2.pth
RoBERTa：/home/curtis/checkpoints/roberta-base
原复现输出：/home/curtis/experiments/TempRMOT/rk2_official
Conda 环境：temprmot
```

### 5.2 Checkpoint 关键事实

```text
checkpoint SHA-256：4953bd2b2966f752b7d5dced94216f0ddacd04158b1c85c3fc3534a333e9581b
hist_len=4
extra_track_attn=False
```

### 5.3 固定推理阈值

```text
object_threshold=0.6
referring_threshold=0.4
area_threshold=100
```

本轮不得为获得预期结果而调整这些阈值。阈值前原始分数用于诊断，最终输出仍保留官方阈值。

---

## 6. 仓库状态与安全边界

当前工作树不是干净状态，已有修改和文档属于用户既有工作：

```text
 D TEMPRMOT_REPRODUCTION_GUIDE.md
 M TrackEval/scripts/local_evaluate_rk2.sh
 M configs/local_temp_rmot_test_rk2.sh
 M tools/validate_refer_kitti_v2.py
?? docs/
```

新任务必须：

- 不重置、不丢弃、不覆盖无关修改；
- 不使用 `git reset --hard` 或 `git checkout --`；
- 修改前检查目标文件差异；
- 实验大文件写到 `/home/curtis/experiments/TempRMOT`，不写入 Git 仓库；
- 原始数据只读，使用 overlay 或独立表达目录；
- 每完成一个工程修改先做回归验证，再继续筛样本。

---

## 7. 为什么必须先增加审计日志

当前 `inference.py` 的顺序为：

```text
模型输出
→ object score > 0.6
→ area > 100
→ refer score > 0.4
→ tracker 更新
→ predict.txt
```

模型内部已经生成：

```text
dt_instances.scores
dt_instances.refers
dt_instances.obj_idxes
dt_instances.boxes
dt_instances.labels
```

但 `predict.txt` 只保留阈值后的最终框。如果 swapped 表达下目标消失，仅看最终文件无法判断：

- Refer 分数是否下降；
- object score 是否下降；
- 框是否偏移；
- 还是固定阈值把候选过滤掉。

因此第一项工程任务是：在任何过滤发生前写出候选审计日志，同时保证原 `predict.txt` 完全不变。

---

## 8. 审计日志最小实现要求

### 8.1 建议新增参数

在 `main.py` 参数解析器中增加等价参数：

```text
--audit_candidates
--audit_dir <path>
```

参数名可根据代码风格微调，但功能必须明确，默认关闭，不影响原推理。

### 8.2 记录位置

在 `Detector.detect()` 中，完成：

```python
dt_instances = track_instances.to(torch.device('cpu'))
```

之后、调用三个 filter 函数之前，保存原始候选。

### 8.3 每行至少包含

```text
video_id
expression_id / sample_id
variant
frame_id
query_id（obj_idxes）
x1,y1,x2,y2
object_score
refer_score
predicted_state（labels）
area
pass_object
pass_area
pass_refer
```

推荐每个表达输出一个 CSV 或 JSONL，文件与表达目录一一对应。所有 Tensor 必须先转为 CPU 标量，不要在循环中保留 GPU Tensor。

### 8.4 工程验收

对同一个已有官方表达，在关闭和开启审计日志时各运行一次：

- `predict.txt` 内容必须一致；
- `gt.txt` 内容必须一致；
- 处理帧数一致；
- 审计日志帧号完整；
- 至少人工核对一帧中 `pass_*` 与阈值逻辑一致；
- 日志写出不得改变模型、Tracker 或阈值。

如回归不一致，停止样本筛选，先修复日志实现。

---

## 9. 本轮样本筛选规则

### 9.1 样本来源

优先使用非官方测试视频做初步验证，例如：

```text
0000–0004
0006–0010
```

不要先使用 `0013`、`0019` 作为反复调试数据；它们可留给后续确认性实验。

### 9.2 最小纳入条件

每个样本必须满足：

- 同一个目标发生至少两个分离的事件阶段；
- 每个阶段原则上持续至少5帧，总跨度至少20帧；
- 事件先后顺序可以从视频事实判断；
- 目标和必要参照物的 Track ID 明确；
- 关键阶段目标大部分可见；
- 交换顺序后不再准确描述原轨迹；
- 纳入决定在查看 swapped 推理结果前完成。

### 9.3 优先寻找

```text
后方 → 前方
运动 → 停止
原车道 → 新车道
可见 → 遮挡 → 重现
接近 → 远离
直行 → 转弯
```

优先使用相对位置、停止、转弯等可观察事件，不使用“准备超车”“有意避让”等主观意图描述。

### 9.4 本轮数量

- 先建立至少5个候选；
- 从中选2个最清楚的基础样本；
- 被排除的候选保留在表格中并记录理由；
- 不因模型结果不符合预期而替换已锁定样本。

---

## 10. 每个样本的四种表达

固定生成以下四类：

| Variant | 含义 | 约束 |
|---|---|---|
| `correct` | 与真实视频一致的顺序 | 原始参照 |
| `paraphrase` | 顺序和事件不变，仅同义改写 | 测措辞稳定性 |
| `orderless` | 保留事件，去掉显式顺序连接 | 测顺序词作用 |
| `swapped` | 使用相同事件子句，交换顺序 | 核心反事实 |

两阶段统一使用：

```text
E1 → E2    correct
E2 → E1    swapped
```

三阶段统一使用：

```text
E1 → E2 → E3    correct
E3 → E2 → E1    swapped
```

`correct` 与 `swapped` 不得增加不同的颜色、类别、方位或对象信息；尽量只交换事件子句。

示例：

```text
correct:
The car that first follows behind the truck and then moves ahead of it.

paraphrase:
The car initially driving behind the truck and later traveling in front of it.

orderless:
The car that follows behind the truck and moves ahead of it.

swapped:
The car that first moves ahead of the truck and then follows behind it.
```

---

## 11. 实验数据必须使用 overlay

不得修改：

```text
/home/curtis/datasets/refer-kitti-v2/expression
```

建议创建：

```text
/home/curtis/experiments/TempRMOT/order_sensitivity_v1/
├── dataset_overlay/
│   ├── KITTI -> /home/curtis/datasets/refer-kitti-v2/KITTI
│   └── expression/
│       └── <video_id>/
│           ├── S0001__correct.json
│           ├── S0001__paraphrase.json
│           ├── S0001__orderless.json
│           └── S0001__swapped.json
├── manifests/
│   ├── candidate_samples.csv
│   ├── samples.csv
│   └── expressions.csv
├── runs/
├── analysis/
├── figures/
└── logs/
```

表达 JSON 保持官方必要结构：

```json
{
  "sentence": "...",
  "label": {
    "0": [1],
    "1": [1]
  }
}
```

实验附加元数据写入 Manifest，不要依赖自定义 JSON 字段。

---

## 12. Ground Truth 边界

- `correct` 与 `paraphrase` 应指向同一目标身份。
- `orderless` 只有在没有新增其他匹配目标时才复用相同 GT。
- 如果 `swapped` 明确对应同场景另一个目标，则写入该目标 GT，这是最强的 A 级样本。
- 如果 `swapped` 不对应任何目标，逐帧标签可为空；该表达不计算传统 HOTA，只分析目标证据和错误接受。
- 不允许为了计算 HOTA，强行把无匹配的 swapped 表达标成原目标。

---

## 13. 本轮运行顺序

严格按以下顺序执行：

### Step 1：记录起始状态

- 保存 `git status --short`；
- 记录当前 commit；
- 记录 Checkpoint 哈希；
- 确认原复现输出仍存在；
- 不清理当前工作树。

### Step 2：增加审计日志

- 修改参数解析和 `inference.py`；
- 默认关闭；
- 不改变原有推理输出。

### Step 3：官方表达回归测试

使用已经验证过的表达：

```text
video: 0005
expression: automobiles-are-located-to-the-left
frames: 297
```

比较开启/关闭审计的文件哈希或逐行差异。

### Step 4：建立候选池

- 从训练视频筛5个候选；
- 在看模型反事实结果前锁定2个样本；
- 记录目标、参照物、事件区间和排除理由。

### Step 5：构造8条表达

- 每个基础样本4条；
- 人工检查语义和顺序；
- 建立独立 JSON 与 Manifest。

### Step 6：运行8个表达

- 使用同一 Checkpoint、参数和阈值；
- 每条表达保存原始审计日志、`predict.txt`、`gt.txt` 和运行日志；
- 不在看到前几个结果后修改后续表达。

### Step 7：结果完整性检查

- 计划表达数：8；
- 完成表达数：8；
- 四变体帧数一致；
- 无坏行和 NaN/Inf；
- 原始 Refer 分数范围合理；
- 无匹配 swapped 不送入普通 HOTA 汇总。

### Step 8：最小分析

对每个基础样本输出：

```text
evidence_correct
evidence_paraphrase
evidence_orderless
evidence_swapped
order_margin = correct - swapped
paraphrase_shift = abs(correct - paraphrase)
order_specific_index = order_margin - paraphrase_shift
match_availability
四种表达的 top Track ID
```

初步目标证据使用阈值前、与 GT 框 IoU≥0.5 的候选 Refer 分数；无匹配候选时记0，同时单独报告 match availability。

### Step 9：写可行性结论

只能使用以下表述层级：

- “两个样本中观察到顺序交换后的可测响应”；
- “两个样本中未观察到超过同义改写波动的顺序响应”；
- “当前样本/接口不足以判断”。

不得写“证明 TempRMOT 不理解顺序”。

---

## 14. 本轮进度跟踪表

| ID | 任务 | 当前状态 | 完成证据 | 通过标准 |
|---|---|---|---|---|
| T0 | 官方 Checkpoint 全量复现 | Done | `docs/TEMPRMOT_REPRODUCTION_RESULTS_2026-07-22.md` | 862/862，HOTA 35.249 |
| T1 | 完整顺序实验协议 | Done | `docs/research/TEMPRMOT_ORDER_SENSITIVITY_EXPERIMENT_PLAN.md` | 计划与表格完整 |
| T2 | 记录新任务起始状态 | Not started | 起始日志 | Git、资产、输出可追溯 |
| T3 | 实现阈值前审计日志 | Not started | 代码差异、示例日志 | 默认关闭、字段完整 |
| T4 | 官方表达回归测试 | Not started | 回归报告 | 开关审计时原输出一致 |
| T5 | 建立5个候选样本 | Not started | `candidate_samples.csv` | 5个候选均有纳入/排除理由 |
| T6 | 锁定2个基础样本 | Not started | `samples.csv` | 事件、ID、帧区间明确 |
| T7 | 构造8条受控表达 | Not started | `expressions.csv`、JSON | 2×4，语义检查通过 |
| T8 | 运行8条表达 | Not started | `runs/`、日志 | 8/8退出码0 |
| T9 | 完整性与结果分析 | Not started | `analysis/results.csv` | 指标可复算、无静默排除 |
| T10 | 可行性报告 | Not started | `PILOT_FEASIBILITY_REPORT.md` | 结论不超出2样本证据 |

状态只使用：`Not started`、`In progress`、`Blocked`、`Done`、`Dropped`。

---

## 15. 今日完成定义

理想完成：

- T2–T10 全部完成；
- 得到2个基础样本、8组表达的完整可复算结果；
- 写出保守的可行性结论。

最低可接受完成：

- T2–T4 完成，证明审计接口可靠；
- T5 建立候选池；
- 如有效样本不足，明确记录阻塞原因，不强造样本。

如果今日只完成计划或候选筛选，不得把简历写成“完成初步可行性验证”。

---

## 16. 关键科学性检查

在交付前逐项回答：

- [ ] 样本是否在查看 swapped 结果前锁定？
- [ ] correct/swapped 是否使用相同事件词，仅交换顺序？
- [ ] correct 本身是否能被模型处理？若不能，是否避免归因于顺序？
- [ ] 是否保留全部候选和排除理由？
- [ ] 是否同时报告 Refer 分数和 match availability？
- [ ] 是否没有调整阈值迎合结果？
- [ ] 是否没有把帧当作独立样本扩大证据？
- [ ] 无目标 swapped 是否没有被错误计算 HOTA？
- [ ] 是否保留失败、空预测和异常案例？
- [ ] 结论是否明确限定为2个样本的流程与现象验证？

任一关键项为“否”，本轮不得标记为完成。

---

## 17. 新任务完成后必须交付

### 17.1 代码与配置

- 审计日志实现；
- 新增参数说明；
- 官方表达回归测试结果；
- 不得混入无关模型改动。

### 17.2 数据与 Manifest

- 5个候选的完整记录；
- 2个锁定基础样本；
- 8条表达及 JSON；
- 所有排除理由。

### 17.3 原始结果

- 8条表达的运行日志；
- 阈值前候选审计日志；
- `predict.txt` 和 `gt.txt`；
- 完整性检查结果。

### 17.4 分析与报告

- 每样本四变体结果表；
- 至少2组可视化或阶段曲线；
- `PILOT_FEASIBILITY_REPORT.md`；
- 下一步建议：Go、Revise 或 Stop。

---

## 18. 新任务可直接使用的开场提示词

```text
请先完整阅读：
1. docs/TEMPRMOT_HANDOFF_2026-07-19.md
2. docs/research/TEMPRMOT_ORDER_SENSITIVITY_EXPERIMENT_PLAN.md
3. docs/TEMPRMOT_REPRODUCTION_RESULTS_2026-07-22.md

然后执行“多阶段事件描述与目标轨迹的时序顺序对齐”初步可行性验证。
本轮目标是2个基础样本、每个4种受控表达，共8组推理；暂不做盲法复核、
不训练模型、不进入Confirmatory实验。

请先核对当前工作树和已有修改，保留用户现有文件；随后从交接文档 Step 1 开始。
必须先实现阈值前Refer候选审计并通过官方表达回归测试，再筛选样本。
不要修改原Refer-KITTI-V2数据，使用独立overlay和外部实验目录。
完成后交付代码差异、样本与表达Manifest、8组原始输出、结果表、可视化和
PILOT_FEASIBILITY_REPORT.md。结论必须限定为小规模可行性验证，不能预设模型
不理解顺序。
```

---

## 19. 最后提醒

本轮最重要的成果不是让某个指标下降，而是确认下面这条链路是否成立：

```text
真实多阶段轨迹
→ 受控顺序表达
→ 阈值前模型证据
→ 配对比较
→ 可解释的方向可行性判断
```

如果链路成立，即使模型表现出顺序敏感，也完成了方向可行性验证；如果链路无法成立，也应准确记录问题发生在数据、表达、模型还是分析接口。
