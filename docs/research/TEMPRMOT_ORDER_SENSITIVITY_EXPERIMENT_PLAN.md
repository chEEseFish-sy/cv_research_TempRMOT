# TempRMOT 多阶段事件顺序敏感性实验：完整操作计划与进度跟踪表

> 版本：v1.0  
> 日期：2026-07-23  
> 适用项目：`/home/curtis/projects/TempRMOT`  
> 当前基础：已使用官方 Checkpoint 完成 Refer-KITTI-V2 全量 862-query 推理与 TrackEval 评测，HOTA 35.249；尚未进行从头训练复现。  
> 本计划目标：验证 TempRMOT 是否利用多阶段事件的先后顺序，而不是训练新模型或证明预设结论。

---

## 1. 执行摘要

本实验采用“先小规模流程验证，再冻结协议并做确认性实验”的两阶段设计：

1. **Pilot 阶段**：使用训练视频构建 8–12 个基础事件样本，每个样本生成正确顺序、同义改写、无顺序和交换顺序四种表达；验证标注、推理、原始 Refer 分数导出和分析链路。
2. **Confirmatory 阶段**：在未参与 Pilot 调试的视频上构建至少 30 个、目标 40–60 个独立基础样本；运行冻结后的模型、阈值、指标和统计脚本，形成可复核结论。

实验不以 HOTA 为唯一结论。主要证据是：在控制词汇和目标不变后，正确顺序与交换顺序之间的目标证据差异，是否显著大于正确顺序与同义改写之间的自然波动。

本计划可以高置信度执行，但不保证得到“TempRMOT 不理解顺序”的正结果。科学实验允许三种结论：模型对顺序敏感、模型对顺序不敏感、当前数据不足以判断。

---

## 2. 已核实的工程事实

### 2.1 当前可直接复用的部分

- 推理入口：`inference.py`
- 本地配置：`configs/local_temp_rmot_test_rk2.sh`
- 官方 Checkpoint：`/home/curtis/checkpoints/TempRMOT/checkpoint_rk2.pth`
- 数据集：`/home/curtis/datasets/refer-kitti-v2`
- 文本编码器：`/home/curtis/checkpoints/roberta-base`
- 默认测试视频：`0005`、`0011`、`0013`、`0019`
- 当前关键阈值：object 0.6、referring 0.4、area 100
- 当前历史长度：`hist_len=4`
- 表达 JSON 的必要字段：`sentence` 和逐帧 `label`

### 2.2 当前必须补齐的接口

当前 `inference.py` 在写出 `predict.txt` 前依次执行：

1. object score 过滤；
2. area 过滤；
3. referring score 过滤；
4. 写出最终框和 Track ID。

模型内部已经计算 `dt_instances.refers`，其值来自 Refer logit 经 sigmoid 后的概率，但当前没有把阈值前的 Refer 分数写入文件。因此正式实验前必须增加只读审计输出，保存过滤前候选框、object score、refer score 和 Track Query ID。

### 2.3 现有表达的限制

对 4 个官方测试视频的 862 条表达进行关键词初筛，仅约 23 条命中跟随、停止、转弯、前后关系等时间/动作词，而且绝大多数仍是单阶段描述。现有表达不能直接作为系统的多阶段顺序测试集。

结论：本实验必须从真实视频和轨迹出发，人工构造受控的多阶段表达，不能只交换现有句子中的几个单词。

---

## 3. 研究问题、假设与结论边界

### 3.1 主要研究问题

当表达中的对象、事件词和场景保持不变，仅交换事件先后顺序时，TempRMOT 对真实目标轨迹的判断是否发生方向正确且超过同义改写噪声的变化？

### 3.2 预先声明的假设

**H1：顺序敏感假设**

```text
目标证据(正确顺序) > 目标证据(交换顺序)
```

**H2：改写稳定假设**

```text
|目标证据(正确顺序) - 目标证据(同义改写)| 较小
```

**H3：顺序特异性假设**

```text
正确-交换的差异 > 正确-同义改写的差异
```

### 3.3 允许出现的结论

| 观察 | 可支持的结论 | 不可声称的内容 |
|---|---|---|
| 正确明显高于交换，且同义改写稳定 | 模型对当前样本的事件顺序具有可测敏感性 | 模型已经具备通用事件推理能力 |
| 正确与交换近似，同义改写稳定 | 模型在当前诊断集上表现出较低顺序敏感性 | 模型完全没有时间建模能力 |
| 同义改写波动大于顺序交换 | 模型更可能受语言表面形式影响 | 一定是 RoBERTa 导致失败 |
| 样本太少或标注分歧高 | 证据不足，需要改善数据 | 负结果等同于方向无价值 |

---

## 4. 实验单位与数据拆分

### 4.1 独立实验单位

统计单位是一个“基础事件轨迹样本”，而不是单帧，也不是四条表达中的每一条。

一个基础样本由以下内容唯一确定：

```text
video_id + target_id + anchor_id(s) + event spans
```

同一基础样本的四种表达是配对观测，不能被当作四个独立样本；同一视频中的样本也具有相关性，统计时需要按视频或基础样本聚类。

同一条目标轨迹原则上只贡献一个基础事件样本。如果确实需要从同一目标构造多个事件描述，主分析只保留预先指定的一条，其余进入补充分析；否则必须按 `video_id + target_id` 聚类，避免把同一轨迹重复计算成独立证据。

### 4.2 推荐的数据拆分

为了避免一边观察结果一边修改表达，采用视频级拆分：

| 集合 | 建议来源 | 用途 | 是否允许反复查看结果 |
|---|---|---|---|
| Pilot/Discovery | 非官方测试视频，例如 `0000–0004`、`0006–0010` | 调通流程、完善规则、估计分数尺度 | 允许 |
| Confirmatory | 预先锁定且不参与调试的视频；优先从 `0013`、`0019` 或其他未用视频选择 | 形成最终诊断结论 | 协议冻结后只运行一次 |

如果后续计划投稿，不应把已经反复观察过的 `0005`、`0011` 当作最终隐藏测试。可以将其明确标记为 Discovery，并另建未参与调试的确认集。

### 4.3 样本量

| 阶段 | 最低基础样本数 | 目标样本数 | 用途 |
|---|---:|---:|---|
| Smoke test | 2 | 2 | 检查文件、日志和模型输出 |
| Pilot | 8 | 12–15 | 判断流程和现象是否存在 |
| Confirmatory | 30 | 40–60 | 估计效应和置信区间 |

30 是确认性分析的最低门槛，不代表已完成严格统计功效保证。Pilot 完成后应根据观测方差进行功效分析，再冻结最终样本量；不得根据 p 值边运行边追加样本。

样本还应尽可能覆盖至少 8 个视频。若最终只能使用 4 个官方测试视频，则按视频聚类的置信区间会很不稳定：此时必须逐视频报告结果，并将结论限定为这 4 个场景，不能声称对一般道路视频成立。

---

## 5. 样本定义与筛选规则

### 5.1 必须满足的纳入条件

每个基础样本必须同时满足：

1. 同一目标至少包含两个按时间分离的事件阶段。
2. 每个阶段原则上持续至少 5 帧；总有效跨度原则上至少 20 帧。
3. 阶段顺序可以通过画面、轨迹位置或可测运动判断。
4. 交换顺序后，描述与原目标明显不符，或者对应另一个明确目标。
5. 关键阶段中目标可见帧比例原则上不低于 70%。
6. 目标 Track ID 在整个事件过程中可追溯。
7. 如果描述涉及卡车、车辆或行人等参照物，必须记录其 Track ID。
8. 标注者在不知道模型预测的情况下完成纳入决定。

### 5.2 排除条件

- 只有颜色、类别、静态位置等单阶段属性。
- 两个动作同时发生，无法定义顺序。
- 关键阶段主要由摄像机运动造成，目标自身行为不明确。
- 需要推测驾驶意图，例如“准备超车”。
- 目标或参照物在转折阶段长时间不可见。
- 交换顺序后仍可以合理描述同一轨迹。
- 同一场景中多个目标都满足表达，但无法定义唯一或明确的目标集合。
- GT ID 明显断裂且无法人工确认身份连续性。

### 5.3 优先筛选的事件模板

按照可观察性从高到低排序：

1. 后方 → 并行 → 前方；
2. 运动 → 减速 → 停止；
3. 原车道 → 横向移动 → 新车道；
4. 可见 → 遮挡 → 再出现；
5. 接近 → 最近点 → 远离；
6. 直行 → 转弯；
7. 进入画面 → 穿过场景 → 离开。

优先使用“相对位置发生变化”这类可量化事件，谨慎使用“追赶、准备、主动避让”等带意图推断的词。

### 5.4 样本强度分级

| 级别 | 定义 | 主要用途 |
|---|---|---|
| A：双目标对照 | 正确顺序对应目标 A，交换顺序对应同场景目标 B | 最强证据，可测 Track ID 是否按顺序切换 |
| B：无匹配反事实 | 正确顺序对应目标 A，交换后没有任何有效目标 | 测原目标分数下降与错误接受 |
| C：单目标扰动 | 只有目标 A，可比较四种表达下的分数和输出 | 现象发现，证据较弱 |

Confirmatory 集应优先寻找 A 级样本。B 级样本的交换表达属于“无匹配目标”，而传统 RMOT 通常没有接受过拒答训练，因此 B 级结果主要说明模型面对无效顺序时的行为，不能单独证明模型缺少顺序理解。C 级样本可以补充数量，但不得独立支撑强结论。

---

## 6. 标注操作流程

### 6.1 第一轮：轨迹候选发现

1. 按视频浏览轨迹，不查看模型在反事实表达上的结果。
2. 记录速度、横向位置、相对前后关系、遮挡和运动方向明显变化的位置。
3. 将可能包含两阶段以上的轨迹填入候选表，状态设为 `candidate`。
4. 此时只描述可观察事实，不编写最终自然语言表达。

### 6.2 第二轮：逐帧确定事件边界

对每个候选样本记录：

- target ID；
- anchor ID；
- 事件 1 名称与起止帧；
- 事件 2 名称与起止帧；
- 可选事件 3；
- 转折帧；
- 可见帧比例；
- 其他可能满足描述的目标；
- 纳入或排除理由。

边界容许少量主观差异，但事件顺序必须明确。对于渐变动作，记录一个边界区间而不是伪造单帧精确值，例如 `transition=82–87`。

### 6.3 第三轮：盲法复核

优先方案：由第二名标注者在不知道第一轮结论和模型输出的情况下，复核全部 Confirmatory 样本。

如果暂时只有一名研究者：

1. 第一轮完成后至少间隔 24 小时；
2. 隐去第一轮的 `valid` 和表达文本；
3. 随机打乱候选顺序；
4. 独立进行第二次纳入判断和边界标注；
5. 将不一致样本标记为 `adjudication`，不得静默修改。

单人双次复核不能等价于双标注者，报告中必须如实说明。

### 6.4 质控指标

- 纳入/排除一致性：报告百分比一致率；有两名标注者时计算 Cohen's kappa。
- 阶段边界一致性：计算两个标注区间的 temporal IoU。
- 目标身份一致性：目标 ID 和参照物 ID 必须完全一致。
- Confirmatory 样本建议：纳入一致率至少 85%，阶段区间平均 temporal IoU 至少 0.70。

低于门槛时先修订标注指南，再重新标注；不得直接进入正式推理。

---

## 7. 四种受控表达的构造

每个基础样本固定构造四种表达。

### 7.1 V1：正确顺序 `correct`

```text
The car that first follows behind the truck, then moves to the left lane,
and finally travels ahead of the truck.
```

### 7.2 V2：顺序保持的同义改写 `paraphrase`

```text
The car initially driving behind the truck, later entering the left lane,
and eventually moving in front of the truck.
```

### 7.3 V3：去除显式顺序 `orderless`

```text
The car that follows the truck, moves to the left lane, and travels ahead of it.
```

### 7.4 V4：交换顺序 `swapped`

```text
The car that first travels ahead of the truck, then moves to the left lane,
and finally follows behind the truck.
```

### 7.5 构造约束

`correct` 与 `swapped` 必须：

- 使用相同对象名词和参照物；
- 使用相同事件词或严格相同事件子句；
- 仅改变事件子句的排列；
- 不新增颜色、方位或类别信息；
- 尽量保持长度和语法复杂度一致；
- 交换后确实与原轨迹冲突。

交换规则必须统一：两阶段样本使用 `E2 → E1`；三阶段样本默认使用完全反序 `E3 → E2 → E1`。如完全反序会产生语法错误或无法判定的表达，该样本应在推理前排除，而不是为某个样本事后选择最容易使模型失败的交换方式。

`paraphrase` 必须保持事件和顺序不变，只改变表面措辞。为了避免把同义词质量引入主结论，Confirmatory 阶段的改写模板也要预先冻结。

### 7.6 文本质量检查

每条表达逐项检查：

- 是否能仅凭视频判断真假；
- 代词是否有唯一指向；
- 是否包含训练数据中极少出现的生僻词；
- 四个版本是否意外改变对象数量；
- `swapped` 是否仍可能描述原目标；
- 是否存在另一个未记录的匹配目标。

LLM 可以协助生成候选措辞，但不得自动决定表达是否与视频一致；最终文本必须人工核验并记录修改者。

---

## 8. Ground Truth 与 JSON 组织

### 8.1 不修改原始数据

不得直接改写 `/home/curtis/datasets/refer-kitti-v2/expression`。创建独立实验数据根目录，并以符号链接复用图像与 KITTI 标签，只复制新建的表达 JSON。

建议结构：

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
├── runs/
├── analysis/
├── figures/
└── logs/
```

### 8.2 JSON 标签语义

- `correct`、`paraphrase` 和 `orderless`：如果表达仍指向同一个目标，三者使用相同目标身份标签。
- A 级样本的 `swapped`：标签改为真正符合反向顺序的目标 B。
- B/C 级样本的 `swapped`：没有有效目标时，逐帧标签为空列表；该变体不进入标准 HOTA 汇总，只进入自定义反事实分析。
- 轨迹级表达指向目标身份时，目标应在其可见帧中保持同一 referent 身份，不应只在事件发生的几帧标正。

### 8.3 外部 Manifest

不要把实验元数据全部塞进官方 JSON。另建 `samples.csv`，保存：

```text
sample_id,split,level,video_id,target_id,reverse_target_id,anchor_ids,
event1,event1_start,event1_end,event2,event2_start,event2_end,
event3,event3_start,event3_end,visibility,valid,review_status,notes
```

另建 `expressions.csv`：

```text
sample_id,variant,json_path,sentence,expected_target_ids,is_no_match,
author,reviewer,text_status
```

---

## 9. 推理审计接口：正式实验前必须实现

### 9.1 原始候选日志

在 `inference.py` 中，应在 object、area 和 referring 三个过滤步骤之前保存每帧候选。每行至少包含：

```text
run_id,sample_id,variant,video_id,frame_id,query_id,
x1,y1,x2,y2,object_score,refer_score,predicted_state,
pass_object,pass_area,pass_refer
```

这里的 `query_id` 对应过滤前的 `obj_idxes`。同时保留最终 `predict.txt`，不要用审计日志替换官方输出。

### 9.2 为什么必须记录过滤前分数

如果交换表达导致目标消失，可能有三种原因：

1. Refer 分数下降；
2. object score 下降或框位置变化；
3. 目标仍存在，但被固定阈值过滤。

只看 `predict.txt` 无法区分这三种情况。论文若声称“语言顺序改变了目标选择”，必须展示对应的原始证据链。

### 9.3 GT—预测匹配

分析脚本按帧将预测候选与 GT 目标框进行匹配：

- 默认 IoU 阈值固定为 0.5；
- 同一 GT 有多个候选时，取 IoU 最大者；
- 若无候选达到 0.5，则该帧记为 `unmatched`；
- 不得在看到结果后调整 IoU 阈值；
- 可在附录补充 IoU 0.3/0.7 的敏感性分析，但主分析保持 0.5。

### 9.4 目标证据定义

主分析使用阈值前的配对目标证据：

```text
frame_evidence = refer_score of the candidate matched to the GT target
```

如果没有候选与 GT 匹配：

```text
frame_evidence = 0
```

同时单独报告 match availability，避免把检测失败完全解释为语言失败。

每个基础样本的目标证据取目标有效帧上的中位数。使用中位数可以降低少数异常帧和短暂遮挡的影响。

### 9.5 可重复性检查

在正式批量运行前，对 2 个样本重复推理两次：

- 文件行数一致；
- 框、object score、refer score 在预设容差内一致；
- 相同表达的最终 TrackEval 输出一致；
- 随机种子、模型哈希、代码提交和参数写入 run manifest。

如无法完全确定性复现，应对 Confirmatory 样本运行至少 3 个固定种子，并将种子作为重复测量，而不是挑选最佳种子。

---

## 10. 运行协议

### 10.1 运行前冻结项

在 Confirmatory 推理前冻结并写入 `protocol_frozen.yaml`：

- Checkpoint 路径、大小和 SHA-256；
- RoBERTa revision 和 SHA-256；
- Git commit；
- CUDA、PyTorch、GPU 环境；
- `hist_len=4`；
- object/referring/area 阈值；
- IoU 匹配阈值；
- 四种表达模板规则；
- 纳入/排除条件；
- 样本量；
- 主指标和统计检验；
- 排除样本的允许理由。

协议冻结后，只允许修复会使所有样本同等受影响的明确软件错误。每次修复必须记录在 decision log，并从头重跑整个 Confirmatory 集。

### 10.2 Smoke test

1. 选择 2 个 Pilot 样本，每个建立 4 个 JSON。
2. 使用独立输出目录运行 8 个表达。
3. 检查每个表达均生成 `predict.txt`、`gt.txt` 和原始审计日志。
4. 检查四个变体处理的帧数完全一致。
5. 检查 `correct/paraphrase/orderless` 的 GT 标签一致。
6. 检查 B 级 `swapped` 的空 GT 没有被误送入标准 HOTA 汇总。

### 10.3 Pilot 运行

1. 使用固定 Checkpoint 和阈值运行全部 8–12 个基础样本。
2. 先生成完整原始结果，再打开分析图；不要边运行边改表达。
3. 计算指标和可视化失败案例。
4. 只根据流程缺陷修订协议，例如表达歧义、日志字段不足或匹配错误。
5. Pilot 数据不进入最终确认性显著性结论。

### 10.4 Confirmatory 运行

1. 确认样本表和表达表状态全部为 `locked`。
2. 备份 `protocol_frozen.yaml` 和文件哈希。
3. 在全新的输出目录一次性运行所有表达。
4. 运行结束后生成完整性报告：计划数、完成数、空文件数、坏行数、帧数一致性和异常退出。
5. 只有完整性检查通过后才运行统计脚本。
6. 所有被排除的运行必须保留原始文件和排除原因。

---

## 11. 指标设计

### 11.1 主指标：Order-Specific Index

对每个基础样本计算：

```text
order_margin = evidence(correct) - evidence(swapped)
paraphrase_shift = abs(evidence(correct) - evidence(paraphrase))
order_specific_index = order_margin - paraphrase_shift
```

解释：

- `order_margin > 0`：正确顺序比交换顺序更支持原目标；
- `paraphrase_shift`：模型对普通措辞变化的自然波动；
- `order_specific_index > 0`：顺序交换产生的影响大于同义改写噪声。

主结论以基础样本级 `order_specific_index` 为主，不以帧数放大样本量。

### 11.2 次要指标

| 指标 | 定义 | 意义 |
|---|---|---|
| Order Margin | correct evidence − swapped evidence | 顺序交换对原目标支持度的影响 |
| Paraphrase Stability | correct 与 paraphrase 的绝对差 | 表面改写稳定性 |
| Orderless Gap | correct evidence − orderless evidence | 显式顺序词是否提供额外信息 |
| Track Selection Change | 四种表达下 top Track ID 是否改变 | 预测身份是否响应文本 |
| A-level Pair Accuracy | 正向表达选 A、反向表达选 B 的比例 | 最直接的顺序选择能力 |
| False Acceptance | 无匹配 swapped 仍高置信度选择原目标的比例 | 缺少拒答能力的程度 |
| Match Availability | GT 帧中存在 IoU≥0.5 候选的比例 | 区分检测失败和 Refer 失败 |
| Threshold Survival | 目标候选通过 refer 0.4 的帧比例 | 与当前部署阈值的关系 |

### 11.3 HOTA 的使用边界

- `correct`、`paraphrase` 和有明确 GT 的 `orderless` 可以计算 HOTA，作为端到端辅助指标。
- A 级 `swapped` 有真实反向目标时可以计算 HOTA。
- B/C 级 `swapped` 没有目标时，不计算或汇总传统 HOTA。
- 不得用总体 HOTA 代替顺序敏感性指标。

### 11.4 正确表达本身失败时的处理

多阶段表达可能比 TempRMOT 的训练文本更长、更复杂。如果模型在 `correct` 上就无法找到真实目标，那么 `correct` 与 `swapped` 都失败不能被直接解释为“模型不理解顺序”。

因此分析分成两层：

1. **全样本分析**：所有预先锁定样本均保留，回答模型对多阶段受控表达的总体响应，禁止因结果不好而删除。
2. **Correct-recognized 条件分析**：仅作为预声明的辅助分析，考察 `correct` 能正确选择目标或达到固定证据门槛的样本中，交换顺序是否造成下降。

条件分析的门槛必须在 Confirmatory 揭盲前冻结，且结果不得替代全样本主分析。报告时还要单列 `correct` 成功率，从而区分“不会处理复杂表达”和“能处理事件但不区分顺序”。

### 11.5 分布外语言的控制

除四种表达外，每个样本记录 RoBERTa token 数、事件子句数和连接词。主报告应比较四类文本的长度分布，并说明这些多阶段表达相对 Refer-KITTI-V2 原始短表达可能属于分布外输入。

如果 `correct` 成功率过低，下一步应先建立“相同长度但不需要顺序推理”的控制表达，判断失败来自长度/句法复杂度还是事件顺序；在该混淆排除前，不进入方法创新阶段。

---

## 12. 统计分析计划

### 12.1 Pilot

Pilot 只报告：

- 每个样本的配对数值；
- 中位数和四分位数；
- 可视化；
- 流程问题。

不根据 Pilot p 值决定是否“证明假设”。

### 12.2 Confirmatory 主分析

1. 对基础样本级 `order_specific_index` 报告中位数、均值和 95% 置信区间。
2. 使用配对置换检验或 Wilcoxon signed-rank test 检验其是否大于 0。
3. 使用按视频聚类的 bootstrap 生成 95% 置信区间；建议 10,000 次重采样。
4. 主指标只有一个，不进行多重比较修正。
5. 次要指标标记为辅助/探索性；若对多个次要假设给出 p 值，使用 Benjamini–Hochberg 控制 FDR。
6. 同时报告效应量与置信区间，不能只报告 p 值。

### 12.3 如何支持“顺序不敏感”

“未达到显著”不等于“模型不敏感”。如果要支持低敏感性，应在 Pilot 后预先定义最小有意义差异，例如 Refer probability 的 0.05，并使用等效性检验或证明 95% 置信区间落在预设的小效应区间内。

最小有意义差异必须在 Confirmatory 结果揭盲前冻结，不能根据最终结果倒推。

### 12.4 分层分析

预先限定以下分层，避免事后切分：

- 两阶段 vs 三阶段；
- A/B/C 样本级别；
- 相对位置、速度变化、遮挡恢复、方向变化；
- 短跨度 vs 长跨度；
- 单目标表达 vs 多目标表达。

分层结果用于解释现象，不替代主分析。

---

## 13. 偏差控制与科学性检查

### 13.1 防止挑样本

- 在运行反事实推理前完成纳入/排除。
- 候选表保留所有被排除样本及理由。
- 不因模型“没有按预期失败”而删除样本。
- Confirmatory 样本编号和表达在运行前生成哈希。

### 13.2 防止文本混淆

- correct/swapped 尽量只交换子句。
- paraphrase 用于估计措辞噪声。
- 报告 token 长度和关键词差异。
- 不把生僻词导致的失败解释成顺序失败。

### 13.3 防止 Tracker 性能混淆

- 同时保存 object score、refer score、框和匹配可用率。
- 分别报告“候选存在但 Refer 分数不变”和“候选框本身消失”。
- 不将更低 object score 自动解释为语言理解变化；TempRMOT 是端到端模型，语言可能影响整个网络，应如实称为端到端响应。

### 13.4 防止阈值后验调优

- 主结果固定使用当前阈值：object 0.6、refer 0.4、area 100。
- 原始分数分析不依赖 refer 阈值。
- 若画阈值曲线，必须覆盖预先声明的完整范围，不能只展示最好点。

### 13.5 防止帧级伪样本膨胀

- 主统计以基础轨迹样本为单位。
- 帧级曲线只用于解释事件阶段变化。
- 不把数千帧当作数千个独立样本计算显著性。

---

## 14. 分阶段执行计划

### P0：冻结问题定义

**作用与意义：** 防止实验中途把“顺序敏感性”改写成任何能产生正结果的问题。

**操作：**

- 确认 H1–H3；
- 确认四种表达；
- 确认独立单位和主指标；
- 建立 decision log。

**输出：** `protocol_draft.md`、`decision_log.csv`。

**通过门槛：** 任何观察者能根据协议判断一个结果支持或不支持哪项假设。

### P1：实现审计日志

**作用与意义：** 将最终框结果拆解为 detection、refer 和阈值影响。

**操作：**

- 在阈值过滤前导出原始候选；
- 添加 sample/variant/run 标识；
- 保留官方 `predict.txt` 输出不变；
- 对原官方表达做回归测试。

**输出：** 审计 CSV/JSONL、字段说明、回归测试结果。

**通过门槛：** 启用日志前后同一表达的 `predict.txt` 完全一致，HOTA 不变。

### P2：建立标注工具与候选池

**作用与意义：** 从真实轨迹建立事件证据，避免纯文本臆造。

**操作：**

- 生成带 GT ID 的视频或关键帧；
- 按事件模板浏览训练视频；
- 记录全部候选与排除原因。

**输出：** `candidate_samples.csv`、候选可视化。

**通过门槛：** 至少得到 15 个 Pilot 候选，其中 8 个通过第一轮筛选。

### P3：双轮标注与表达构造

**作用与意义：** 保证事件顺序、对象身份和反事实文本真实可判定。

**操作：**

- 标记事件区间；
- 构造四个文本版本；
- 进行第二轮盲审；
- 处理分歧。

**输出：** `samples.csv`、`expressions.csv`、四类 JSON。

**通过门槛：** Pilot 至少 8 个有效样本；身份完全一致；表达质检全部通过。

### P4：Smoke test

**作用与意义：** 在批量运行前排除文件、标签和日志错误。

**操作：** 运行 2 个基础样本共 8 条表达，检查完整性和重复性。

**输出：** `smoke_test_report.md`。

**通过门槛：** 8/8 完成；无坏文件；重复运行结果一致；无 GT 泄漏错误。

### P5：Pilot 推理与分析

**作用与意义：** 判断顺序现象是否可测，并估计确认性实验规模。

**操作：**

- 一次性运行 Pilot；
- 计算所有预定义指标；
- 绘制每样本配对图和阶段曲线；
- 记录流程缺陷。

**输出：** `pilot_results.csv`、`pilot_report.md`、图表和运行日志。

**通过门槛：** 至少 8 个有效基础样本完成；主指标可计算；不存在系统性映射错误。

### P6：方向 Go/Revise/Stop 判断

**作用与意义：** 在投入大规模标注前检查问题是否真实可研究。

**Go：** 存在稳定的顺序不敏感或顺序敏感现象，且同义改写控制能够解释结果。

**Revise：** 样本有效但 Refer 分数映射、拒答定义或文本控制存在问题。

**Stop：** 无法获得至少 8 个明确多阶段样本，或双轮标注一致性长期低于门槛。

**输出：** `pilot_decision.md`，必须同时记录支持和反对继续的证据。

### P7：Confirmatory 数据构建与预注册

**作用与意义：** 将探索性发现转化为可检验结论。

**操作：**

- 在未参与 Pilot 的视频上标注；
- 进行独立复核；
- Pilot 方差驱动功效分析；
- 冻结协议、样本、表达和分析脚本。

**输出：** `protocol_frozen.yaml`、锁定 Manifest、文件哈希、功效分析。

**通过门槛：** 至少 30 个有效基础样本，目标 40–60；质控指标达标。

### P8：Confirmatory 一次性运行

**作用与意义：** 避免对隐藏集反复调参。

**操作：** 全新目录运行、完整性检查、保留所有原始输出。

**输出：** 原始审计日志、`predict.txt`、`gt.txt`、运行 Manifest、完整性报告。

**通过门槛：** 计划表达 100% 完成或所有缺失均有事先允许的技术理由。

### P9：统计与可视化

**作用与意义：** 用效应量、置信区间和配对证据回答研究问题。

**操作：**

- 执行冻结后的主分析；
- 进行预声明分层；
- 绘制配对散点、置信区间、阶段曲线和案例图；
- 记录所有异常值但不静默删除。

**输出：** `confirmatory_results.csv`、统计报告、图表。

**通过门槛：** 从原始日志到每张表和图均可由单条命令重新生成。

### P10：结论与下一步

**作用与意义：** 决定是否进入方法研发，而不是强行为 Pilot 包装模型。

**输出：** `conclusion.md`，必须包含结论强度、限制、失败样本和下一实验。

---

## 15. 进度总跟踪表

状态只使用：`Not started`、`In progress`、`Blocked`、`Done`、`Dropped`。

| ID | 阶段 | 任务 | 负责人 | 状态 | 计划开始 | 计划完成 | 实际完成 | 输出证据路径 | 阻塞/备注 |
|---|---|---|---|---|---|---|---|---|---|
| P0-1 | 协议 | 确认 H1–H3 和主指标 |  | Not started |  |  |  |  |  |
| P0-2 | 协议 | 建立 decision log |  | Not started |  |  |  |  |  |
| P1-1 | 工程 | 导出阈值前候选日志 |  | Not started |  |  |  |  |  |
| P1-2 | 工程 | 增加 sample/variant/run 标识 |  | Not started |  |  |  |  |  |
| P1-3 | 工程 | 官方表达回归测试 |  | Not started |  |  |  |  |  |
| P2-1 | 数据 | 生成带 GT ID 的候选视频 |  | Not started |  |  |  |  |  |
| P2-2 | 数据 | 建立至少 15 个 Pilot 候选 |  | Not started |  |  |  |  |  |
| P3-1 | 标注 | 第一轮事件与边界标注 |  | Not started |  |  |  |  |  |
| P3-2 | 文本 | 构造四种受控表达 |  | Not started |  |  |  |  |  |
| P3-3 | 质控 | 第二轮盲法复核 |  | Not started |  |  |  |  |  |
| P4-1 | 测试 | 2 样本 Smoke test |  | Not started |  |  |  |  |  |
| P4-2 | 测试 | 重复性与完整性检查 |  | Not started |  |  |  |  |  |
| P5-1 | Pilot | 批量运行 8–12 个样本 |  | Not started |  |  |  |  |  |
| P5-2 | Pilot | 计算指标并可视化 |  | Not started |  |  |  |  |  |
| P6-1 | 决策 | Go/Revise/Stop 评审 |  | Not started |  |  |  |  |  |
| P7-1 | 确认集 | 建立 30–60 个锁定样本 |  | Not started |  |  |  |  |  |
| P7-2 | 确认集 | 功效分析与协议冻结 |  | Not started |  |  |  |  |  |
| P8-1 | 确认集 | 一次性正式推理 |  | Not started |  |  |  |  |  |
| P8-2 | 确认集 | 完整性报告 |  | Not started |  |  |  |  |  |
| P9-1 | 统计 | 冻结主分析 |  | Not started |  |  |  |  |  |
| P9-2 | 统计 | 分层分析与图表 |  | Not started |  |  |  |  |  |
| P10-1 | 结论 | 写结论、限制和下一步 |  | Not started |  |  |  |  |  |

---

## 16. 样本筛选跟踪表

| Sample ID | Split | Level | Video | Target ID | Anchor ID | Event 1 / frames | Event 2 / frames | Event 3 / frames | Visibility | Ambiguity | Review 1 | Review 2 | Final status | Exclusion reason |
|---|---|---|---|---:|---:|---|---|---|---:|---|---|---|---|---|
| S0001 | Pilot | B |  |  |  |  |  |  |  |  | Candidate |  | Pending |  |
| S0002 | Pilot |  |  |  |  |  |  |  |  |  | Candidate |  | Pending |  |
| S0003 | Pilot |  |  |  |  |  |  |  |  |  | Candidate |  | Pending |  |

填写规则：

- `Visibility` 填 0–1；
- `Ambiguity` 填 `none/low/high`；
- `Final status` 只用 `included/excluded/adjudication`；
- 被排除样本必须填写原因，不能删除行。

---

## 17. 表达构造跟踪表

| Sample ID | Variant | Sentence frozen | Target label checked | Same event vocabulary | Grammar checked | Video consistency checked | Reviewer | Status | JSON path |
|---|---|---|---|---|---|---|---|---|---|
| S0001 | correct | No | No | N/A | No | No |  | Draft |  |
| S0001 | paraphrase | No | No | N/A | No | No |  | Draft |  |
| S0001 | orderless | No | No | Yes | No | No |  | Draft |  |
| S0001 | swapped | No | No | Yes | No | No |  | Draft |  |

`Status` 只使用 `Draft`、`Reviewed`、`Locked`。

---

## 18. 推理运行跟踪表

| Run ID | Sample | Variant | Commit | Checkpoint SHA | Seed | Frames expected | Frames logged | predict.txt | audit log | Exit code | Integrity | Runtime | Status |
|---|---|---|---|---|---:|---:|---:|---|---|---:|---|---|---|
|  | S0001 | correct |  |  | 2020 |  |  | No | No |  | Pending |  | Not started |
|  | S0001 | paraphrase |  |  | 2020 |  |  | No | No |  | Pending |  | Not started |
|  | S0001 | orderless |  |  | 2020 |  |  | No | No |  | Pending |  | Not started |
|  | S0001 | swapped |  |  | 2020 |  |  | No | No |  | Pending |  | Not started |

完整性判定必须自动检查，不应只凭日志末尾显示“完成”。

---

## 19. 结果跟踪表

| Sample | Level | Evidence correct | Evidence paraphrase | Evidence orderless | Evidence swapped | Order margin | Paraphrase shift | Order-specific index | Match availability | Correct top ID | Swapped top ID | Interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| S0001 |  |  |  |  |  |  |  |  |  |  |  |  |

所有数值应由分析脚本生成，禁止手工复制后覆盖原结果。

---

## 20. 决策与问题日志

| Date | ID | Type | Problem / decision | Evidence | Affected samples | Action | Requires rerun | Owner | Status |
|---|---|---|---|---|---|---|---|---|---|
|  | D001 | Decision |  |  |  |  |  |  | Open |

`Type` 使用 `Decision`、`Bug`、`Annotation`、`Data`、`Runtime`。所有协议冻结后的更改都必须在这里留下记录。

---

## 21. 时间安排

### 最小 Pilot：3–5 个工作日

| 天数 | 主要任务 | 当日输出 |
|---|---|---|
| Day 1 | 冻结草案、实现审计日志、2 样本 Smoke test | 审计日志与回归结果 |
| Day 2 | 浏览训练视频，建立 15 个候选 | 候选表与关键帧 |
| Day 3 | 事件边界、四种表达、第一次质控 | 8–12 个完整样本 |
| Day 4 | 第二轮盲审、Pilot 批量推理 | 锁定 Pilot 与原始结果 |
| Day 5 | 指标、图表、Go/Revise/Stop | Pilot 报告 |

如果已有可靠的可视化工具，Day 1–2 可以压缩；不应压缩第二轮复核和结果完整性检查。

### Confirmatory：约 2–3 周

- 第 1 周：扩展并双轮标注 40–60 个样本；
- 第 2 周：冻结协议、运行和统计；
- 第 3 周：补充质控、整理结论和决定方法方向。

---

## 22. 风险矩阵与处理

| 风险 | 可能性 | 影响 | 预防与处理 |
|---|---|---|---|
| 多阶段真实样本不足 | 高 | 高 | 先做候选池 Gate；不足时将贡献转向诊断集构建，或扩展到其他道路视频 |
| swapped 仍可描述原目标 | 中 | 高 | 双轮视频核验；优先相对位置模板；歧义样本排除 |
| 无目标表达无法使用 HOTA | 高 | 中 | 使用原始 Refer 分数、false acceptance 和拒答指标，不强行计算 HOTA |
| Refer 分数与 GT ID 无法直接对应 | 中 | 高 | 固定 IoU 匹配规则；保存阈值前框、query ID 和分数 |
| 语言改变检测框而非只改变 Refer Head | 高 | 中 | 同时报告 match availability、object score 和 refer score，称为端到端响应 |
| 单人标注偏差 | 中 | 中 | 24 小时后盲法复核；争取第二标注者；如实报告限制 |
| 只挑支持假设的案例 | 中 | 高 | 推理前冻结候选；保留所有排除记录；Confirmatory 一次性运行 |
| 计算时间超出预期 | 中 | 中 | Pilot 先测单表达耗时；按样本批量运行；不牺牲完整视频上下文换速度 |
| Pilot 结果不显著 | 中 | 低 | 报告效应与 CI；区分证据不足和等效性；不追逐 p 值 |

---

## 23. Go / Revise / Stop 标准

### Go：进入 Confirmatory 或方法研究

满足以下多数条件：

- 至少 8 个清晰 Pilot 样本；
- 原始 Refer 分数和 GT 匹配稳定可导出；
- correct/paraphrase 的变化通常小于 correct/swapped，或出现稳定的相反失败模式；
- 至少 3 个案例能通过阶段曲线解释；
- 结果不能仅由检测候选缺失解释。

### Revise：修订流程后重做 Pilot

- 文本改写造成过大词汇差异；
- 无目标样本占比过高；
- GT—预测匹配错误；
- 审计日志改变了官方输出；
- 标注一致性不达标。

### Stop：暂缓该方向

- 浏览足够视频后仍无法获得至少 8 个明确多阶段样本；
- 事件真假无法由视频事实稳定判断；
- 模型对所有文本变化都极不稳定，无法隔离顺序效应；
- 工程接口无法获得阈值前证据，只能依赖最终 HOTA 猜测原因。

Stop 不是实验失败；它意味着当前数据与模型接口不足以支持强结论。

---

## 24. 最终交付物清单

### Pilot 必须交付

- [ ] 实验协议草案；
- [ ] 候选与排除样本完整表；
- [ ] 至少 8 个基础样本的事件区间；
- [ ] 每样本 4 条受控表达；
- [ ] 阈值前候选审计日志；
- [ ] 官方输出回归测试；
- [ ] Pilot 原始结果和分析脚本；
- [ ] 配对图、阶段曲线和案例图；
- [ ] Go/Revise/Stop 决策报告。

### Confirmatory 必须交付

- [ ] 冻结协议和文件哈希；
- [ ] 至少 30、目标 40–60 个锁定基础样本；
- [ ] 标注一致性报告；
- [ ] 一次性完整推理结果；
- [ ] 运行完整性报告；
- [ ] 主指标、效应量、95% CI 和统计检验；
- [ ] 预声明分层结果；
- [ ] 结论、限制与所有失败案例索引；
- [ ] 从原始日志一键生成表图的说明。

---

## 25. 反向审查后的关键修正

本计划经过以下反向检查：

1. **如果模型结果不变，是否能证明它不理解顺序？** 不能只靠“不显著”；因此加入同义改写控制、最小有意义差异和等效性分析。
2. **如果预测消失，是否一定是 Refer 分数下降？** 不是；因此必须记录阈值前框、object score、refer score 和 match availability。
3. **能否直接交换现有 862 条表达？** 不能；现有表达以单阶段为主，因此先从真实轨迹标注事件。
4. **能否用帧数扩大统计样本？** 不能；基础轨迹才是独立单位，帧只用于聚合和解释。
5. **能否用 swapped 的 HOTA 作为主指标？** 多数 swapped 没有有效目标，传统 HOTA 不适用，因此采用自定义配对证据。
6. **如何防止挑选失败案例？** 推理前冻结纳入样本，保留所有排除记录，并设置独立 Confirmatory 视频。
7. **如何防止措辞差异冒充顺序理解？** correct/swapped 固定事件词，paraphrase 估计语言表面噪声。
8. **如何保证结果可重现？** 固定资产哈希、代码提交、参数和种子，并要求日志功能不改变官方输出。

在以下前提下，我对“能够按文档完整执行并得到可解释结论”的置信度约为 95%：

- 数据集中能筛得至少 8 个 Pilot 和 30 个 Confirmatory 的明确事件轨迹；
- 能在当前推理脚本中导出 `dt_instances.refers` 和过滤前候选；
- Confirmatory 前确实冻结文本、样本与统计协议。

置信度不覆盖“结果一定支持模型顺序不敏感”或“该方向一定能投稿”。这两点必须由实验决定。

---

## 26. 实际开始时的第一批动作

严格按以下顺序开始，不要先大规模看视频：

1. 复制本计划为实际 protocol，并填写负责人和日期。
2. 建立 decision log、样本表、表达表和运行表。
3. 为 `inference.py` 增加阈值前审计日志。
4. 用一个现有官方表达进行回归测试，确认 `predict.txt` 和 HOTA 不变。
5. 从训练视频中找到 2 个明确的两阶段轨迹。
6. 为这 2 个样本制作四种表达和独立 JSON。
7. 完成 8 次 Smoke test 和重复性检查。
8. 只有 Smoke test 全部通过，才扩展到 8–12 个 Pilot 样本。

这组顺序把最高风险的工程接口和数据有效性提前验证，可以避免完成大量人工标注后才发现分数无法提取或指标无法解释。
