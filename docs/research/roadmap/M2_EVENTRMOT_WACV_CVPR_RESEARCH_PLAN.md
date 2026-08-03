# M²-EventRMOT 可行性论证与 WACV/CVPR 双路线科研计划

> 版本：v1.0  
> 调研与计划日期：2026-07-21  
> 研究方向：面向有序多阶段事件的外观—运动—几何证据自适应 Referring Multi-Object Tracking  
> 基础代码：TempRMOT  
> 目标会议：WACV、CVPR

---

## 0. 结论先行

### 0.1 方向是否可行

该方向在科学问题上**可行且有真实缺口**，但必须准确表述为：

> 在保持多目标身份的前提下，将自然语言中的有序事件阶段与轨迹片段进行单调对齐，并根据不同事件阶段动态选择外观、运动和几何证据。

不建议把核心贡献表述为“增加多模态”。RGB 与语言本身已经是多模态，轨迹运动和几何关系大多又是从 RGB/检测框派生出来的异构证据，而不一定是独立传感器模态。更稳妥的术语是：

- stage-adaptive multi-evidence reasoning；
- appearance-motion-geometry evidence routing；
- ordered event-track alignment。

### 0.2 最重要的文献边界

- TempRMOT已经聚合历史Query，但没有显式事件阶段和顺序状态。
- DKGTrack已经把语言分成静态描述和运动状态，因此“属性流+运动流”本身不新。
- VMRMOT已经把历史轨迹转换成运动描述并使用MLLM，因此“用MLLM理解轨迹运动”本身不新。
- COAL已被IJCAI 2026接收，已经覆盖VLM语义注入和LLM属性反事实增强，因此“反事实组合学习”不能作为主创新。
- ReaMOT已经研究隐式、常识型语言推理，因此不能声称“首次做推理RMOT”。
- STORM已经实现生成式端到端MLLM-RMOT，因此不能把“首次将MLLM用于RMOT”作为主张。
- VideoITG、TimeLens等CVPR 2026工作已经研究指令条件的时间定位和高质量时间评测，因此事件时间标注必须严格，不能只靠自动生成。

在已核验文献中，尚未看到同时具备以下四点的直接RMOT方法：

1. 将查询显式拆成有序事件阶段；
2. 将每个阶段定位到保持同一Identity的轨迹片段；
3. 使用单调顺序约束判断完整事件是否成立；
4. 用顺序交换和部分满足轨迹进行反事实评测。

这四点构成真正可辩护的研究空白。

### 0.3 两个会议的路线选择

| 会议路线 | 核心定位 | 推荐程度 | 现实时间判断 |
|---|---|---:|---|
| WACV | 有序事件RMOT诊断基准 + 轻量阶段感知基线 | 条件推荐 | WACV 2027 Round 2在2026-08-28截止，从当前日期起约38天；只有在TempRMOT已可运行且有标注协助时才现实 |
| CVPR | 阶段自适应多证据路由 + 单调事件—轨迹对齐 + 组合/跨域评测 | 主要推荐 | 摘要注册：2026-11-07 15:59（GMT+8）；全文提交：2026-11-13 15:59（GMT+8）。从2026-07-21起约有115天，按16周研发加4天最终提交缓冲倒排 |

### 0.4 置信度边界

本计划所称“95%置信度”是指：

> 在满足算力、数据访问和标注前提的情况下，按照阶段门槛能够顺利得到一个明确的继续/停止结论，并避免把三个月全部投入到不可行方案。

它不表示对论文录用有95%把握，也不表示一定能够获得正向实验结果。科研结果和会议录用不存在可以诚实保证的95%成功率。

---

## 1. 科学问题的严格定义

## 1.1 现有RMOT形式

给定视频帧序列 `V = {I₁, I₂, …, I_T}` 和自然语言查询 `L`，传统RMOT输出所有符合查询的目标轨迹集合：

```text
T* = { trajectory_i | trajectory_i satisfies language query L }
```

现有方法通常将整句查询编码为一个全局文本表示，再与逐帧视觉特征或Track Query进行融合。该形式适合：

- “红色汽车”；
- “左边的人”；
- “正在转弯的车辆”。

但它没有明确表示：

> “先跟随卡车，随后变道超车，最后停在路边。”

中的阶段边界、完成状态和顺序约束。

## 1.2 本研究的任务定义

将查询解析为有序事件图：

```text
G_L = (E, R)
E   = { e₁, e₂, …, e_K }
```

其中 `e_k` 是事件阶段，`R` 表示 before/after 等顺序关系。对于候选轨迹 `trajectory_i`，模型需要找到一组按时间单调递增的轨迹片段：

```text
segment_index(i, 1) < segment_index(i, 2) < … < segment_index(i, K)
```

并判断每个片段是否满足对应事件阶段。

### 输出语义

- 最终仍输出目标的**完整轨迹**，与RMOT保持一致；
- 额外输出每个事件阶段对应的时间区间和完成置信度；
- 只有完整、有序地满足所有必要阶段的轨迹才被判定为referent；
- 可选阶段、重复阶段和并行事件不纳入第一版任务，避免定义失控。

## 1.3 证据流定义

### 外观证据

- 目标类别；
- 颜色、纹理、服饰；
- 局部目标特征；
- 场景语义，如道路、路边、车道和人群。

### 运动证据

- 归一化中心点和框大小序列；
- 速度、加速度和运动方向；
- 横向/纵向位移；
- 静止、启动、转向、接近和远离状态。

### 几何/关系证据

- 与其他目标的相对位置和距离；
- 前后、左右、接近、远离关系变化；
- 候选目标与场景区域的关系；
- 可选扩展：真实深度、3D位置或LiDAR几何。

第一阶段不强制使用深度或LiDAR。核心方法必须仅依靠现有RGB、检测框和轨迹即可运行，真实3D作为后续增强实验，避免被额外数据和传感器卡住。

---

## 2. 可验证的研究假设

### H1：现有RMOT模型对事件顺序不敏感

当查询从“A后B”改为“B后A”，但词汇和动作集合不变时，TempRMOT、FlexHook等模型的轨迹得分变化不足，导致顺序反事实错误。

### H2：不同阶段需要不同证据

“红色卡车”主要依赖外观；“超车”主要依赖相对运动；“停在路边”同时依赖运动和场景几何。固定拼接或固定权重融合不能为每个阶段选择最有判别力的证据。

### H3：单调事件—轨迹对齐优于全局历史聚合

对整条轨迹平均池化或用FIFO历史注意力容易稀释短暂事件；显式阶段片段和单调顺序约束应提高完整事件识别。

### H4：顺序反事实训练改善未见事件组合

使用“相同事件、错误顺序”和“只满足部分阶段”的硬负样本，可以减少关键词捷径，提高未见动作组合和顺序上的泛化。

### H5：提升来自语言—事件理解，而不是更强tracker

在固定检测框、固定轨迹和固定视觉Backbone时，阶段对齐仍应提高目标选择、Order Accuracy和Event-Track F1；否则不能证明语言理解增强。

---

## 3. 文献覆盖与剩余空间

| 文献 | 已覆盖能力 | 仍未覆盖的关键点 | 对本研究的约束 |
|---|---|---|---|
| [TransRMOT](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_Referring_Multi-Object_Tracking_CVPR_2023_paper.html) | 端到端Detect/Track Query与语言融合 | 长期事件阶段、顺序 | 可作为最基础端到端baseline |
| [TempRMOT](https://arxiv.org/abs/2406.05039) | 历史Query和时间增强 | 离散阶段、事件边界、顺序反事实 | 适合作为主代码基础 |
| [iKUN](https://openaccess.thecvf.com/content/CVPR2024/html/Du_iKUN_Speak_to_Trackers_without_Retraining_CVPR_2024_paper.html) | 可插拔轨迹语言匹配 | 事件级轨迹表示、反馈 | 证明插件路线价值，但不是主方法基础 |
| [HFF-Tracker](https://ojs.aaai.org/index.php/AAAI/article/view/33143) | 多层细粒度语言融合 | 阶段顺序和事件完成 | 不能只增加更多融合层 |
| [DKGTrack](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Language_Decoupling_with_Fine-grained_Knowledge_Guidance_for_Referring_Multi-object_Tracking_ICCV_2025_paper.html) | 静态描述与运动状态解耦 | 多阶段状态转移和单调对齐 | “外观+运动双流”不能单独声称创新 |
| [VMRMOT](https://arxiv.org/abs/2511.17681) | MLLM运动描述、视觉—运动—语言对齐 | 有序阶段、事件边界 | 不应以“MLLM理解运动”为主贡献 |
| [COAL](https://arxiv.org/abs/2605.14795) | VLM语义注入、LLM属性反事实、组合判别 | 关系/动作/顺序held-out评测 | 反事实只能作为训练支撑，不是主创新；论文已列入IJCAI 2026接收列表 |
| [ReaMOT](https://arxiv.org/abs/2505.20381) | 隐式常识和高层语义推理 | 明确事件阶段和顺序 | 不得宣称首次“reasoning RMOT” |
| [FlexHook](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Rethinking_Two-Stage_Referring-by-Tracking_in_Referring_Multi-Object_Tracking_Make_it_Strong_CVPR_2026_paper.html) | 强双阶段匹配、低训练成本 | 语言反馈、长期有序事件 | 必须纳入强baseline |
| [STORM](https://openaccess.thecvf.com/content/CVPR2026F/html/Lu_STORM_End-to-End_Referring_Multi-Object_Tracking_in_Videos_CVPRF_2026_paper.html) | 端到端生成式MLLM-RMOT、关系/交互数据 | 显式事件状态和顺序反事实 | 不得把MLLM或生成坐标当新意 |
| [DRMOT](https://arxiv.org/abs/2602.04692) | RGB-D-Language RMOT | 有序事件 | 不能仅增加深度分支 |
| [RT-RMOT](https://arxiv.org/abs/2602.22033) | RGB-Thermal-Language RMOT | 有序事件 | 不建议转向热红外，覆盖和数据成本高 |
| [VideoITG](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_VideoITG_Multimodal_Video_Understanding_with_Instructed_Temporal_Grounding_CVPR_2026_paper.html) | 指令条件的关键帧/时间定位 | 多目标身份一致性 | 可借鉴时间定位，但必须突出track identity |
| [TimeLens](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_TimeLens_Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs_CVPR_2026_paper.pdf) | 时间标注质量审计、可靠VTG评测 | 多目标身份与有序事件满足 | 数据质量标准必须达到其强调的唯一性、存在性、清晰度和准确性 |
| [TrajTok](https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_TrajTok_Learning_Trajectory_Tokens_Enhances_Video_Understanding_CVPR_2026_paper.pdf) | 高效轨迹Token用于视频理解 | RMOT有序事件 | 不能仅把轨迹做成Token；必须有事件对齐机制 |

### 文献审查后的结论

方向不是“完全无人研究”，而是多个相邻能力已经存在，但尚未被统一到身份保持的有序事件RMOT中。因此论文必须围绕一个紧凑的新问题，而不是把TempRMOT、轨迹特征、MLLM和多模态路由全部并列为多个松散模块。

---

## 4. 统一技术框架

## 4.1 总体信息流

```text
自然语言查询
    ↓
有序事件解析器
    ↓
Stage Tokens：e1 → e2 → ... → eK
    ↓
TempRMOT候选轨迹与Track Queries
    ↓
轨迹分段/状态转折候选
    ↓
┌────────────┬────────────┬──────────────┐
│ 外观证据流  │ 运动证据流  │ 几何/关系证据流 │
└────────────┴────────────┴──────────────┘
    ↓
阶段条件的证据路由器
    ↓
单调事件—轨迹对齐器
    ↓
轨迹满足得分 + 阶段区间 + 顺序一致性
```

## 4.2 模块A：有序事件解析器

### 功能

从查询中提取：

- 主体约束；
- 事件阶段；
- 每阶段动作、关系和场景条件；
- before/after顺序边。

### 实现优先级

1. MVP：人工结构标注 + 模板/规则解析；
2. 稳定版：RoBERTa/BERT上的轻量stage query生成器；
3. 扩展：离线MLLM产生结构，再蒸馏给小模型。

MLLM不进入第一版在线推理，避免API成本、幻觉和不可复现性。

## 4.3 模块B：轨迹状态与片段生成

### 输入

每条轨迹的框序列、Track Query和局部视觉特征。

### 特征

- 归一化框序列 `(x, y, width, height)`；
- 一阶速度、二阶加速度；
- 方向变化；
- 静止概率；
- 与其他轨迹的相对位置变化。

### 分段策略

- MVP：多尺度固定长度片段；
- 正式版：变化点预测器产生软阶段边界；
- 对照：整轨迹平均池化和TempRMOT FIFO历史。

## 4.4 模块C：阶段自适应证据路由

对于阶段 Token `e_k`，分别计算外观、运动和几何证据：

```text
appearance_evidence[k] = z_a[k]
motion_evidence[k]     = z_m[k]
geometry_evidence[k]   = z_g[k]
```

路由器输出可靠性权重：

```text
[weight_a[k], weight_m[k], weight_g[k]]
    = softmax(router(stage_token[k], z_a[k], z_m[k], z_g[k]))
```

融合结果为：

```text
fused_stage[k] =
    weight_a[k] × z_a[k]
  + weight_m[k] × z_m[k]
  + weight_g[k] × z_g[k]
```

### 必须避免的问题

- 不能只报告注意力可视化；
- 必须做模态/证据缺失、噪声和冲突测试；
- 必须证明不同事件阶段确实选择不同证据；
- 应加入modality dropout，防止路由器退化为始终选择外观流。

## 4.5 模块D：单调事件—轨迹对齐

建立大小为 `K × M` 的阶段—片段得分矩阵 `S`，其中 K 为事件阶段数，M 为轨迹片段数。

使用单调动态规划、soft-DTW或单调注意力寻找：

```text
segment₁ < segment₂ < … < segment_K
```

最终得分综合：

- 各阶段匹配得分；
- 阶段完整性；
- 顺序一致性；
- 原TempRMOT referring score。

MVP建议先使用可解释动态规划；只有确认任务有效后再实现可微soft-DTW或单调Transformer。

## 4.6 训练目标

建议损失：

```text
total_loss =
    rmot_loss
  + λ_stage          × stage_loss
  + λ_order          × order_loss
  + λ_counterfactual × counterfactual_loss
  + λ_reliability    × reliability_loss
```

- `rmot_loss`：保持 TempRMOT 原检测、框回归、关联和 refer 损失；
- `stage_loss`：阶段—片段匹配/时间区间监督；
- `order_loss`：正确顺序得分高于交换顺序；
- `counterfactual_loss`：完整轨迹高于部分满足和错误顺序轨迹；
- `reliability_loss`：证据缺失或加入噪声时降低对应权重。

第一版不应同时实现全部损失。执行顺序为：stage → order → router → reliability。

---

## 5. 数据与评测协议

## 5.1 事件本体第一版

只保留能够从轨迹和视频中较可靠判定的事件：

- start / stop；
- turn-left / turn-right；
- approach / move-away；
- follow；
- pass / overtake；
- enter / leave region；
- change-left / change-right；
- stand / walk / run（人物数据可用时）。

第一版不纳入：

- 需要复杂社会常识的“更有团队精神”；
- 无法从画面验证的意图和情绪；
- 没有稳定边界的抽象事件；
- 四阶段以上的长事件程序。

## 5.2 查询类型

1. 二阶段正确顺序；
2. 三阶段正确顺序；
3. 相同事件集合、交换顺序；
4. 删除一个必要阶段；
5. 加入错误阶段；
6. 只满足前缀/后缀的硬负轨迹；
7. 同义改写；
8. 未见动作—对象组合。

## 5.3 标注字段

```json
{
  "video_id": "...",
  "query": "...",
  "subject_constraints": [],
  "stages": [
    {
      "predicate": "follow",
      "arguments": ["truck"],
      "start_frame": 0,
      "end_frame": 0,
      "required_evidence": ["motion", "geometry"]
    }
  ],
  "ordering_edges": [[0, 1]],
  "target_track_ids": [],
  "hard_negative_track_ids": [],
  "counterfactual_type": null,
  "annotator_confidence": 0.0
}
```

## 5.4 数据质量规则

参考TimeLens对时间评测的质量要求，每条查询必须满足：

- Event existence：事件确实发生；
- Event uniqueness：阶段区间尽可能唯一；
- Query clarity：两位标注者能独立理解；
- Annotation accuracy：阶段边界误差在预设容忍范围内；
- Identity consistency：所有阶段属于同一目标轨迹；
- Negative validity：硬负轨迹确实不满足完整事件。

测试集必须双人独立标注并仲裁。训练集可采用自动候选加单人复核，但需要报告抽样错误率。

## 5.5 数据切分

- IID split：事件组合在训练和测试均出现；
- Composition split：保留单事件原子，但测试组合未见；
- Order split：事件集合相同、顺序不同；
- Video-disjoint split：原始视频不能跨集合；
- Template-disjoint split：查询模板和近义改写簇不能跨集合；
- Cross-domain split：可选，用另一数据源进行测试。

## 5.6 评测指标

### 标准RMOT指标

- HOTA；
- DetA；
- AssA；
- IDF1；
- ID Switch。

### 事件专用指标

- **Event-Track F1**：正确选择完整满足事件的轨迹；
- **Order Accuracy**：正确查询得分高于其顺序交换版本的比例；
- **Stage mIoU**：预测阶段区间与人工区间的时间IoU；
- **Stage Completion Accuracy**：是否正确判断每个阶段完成；
- **Partial-Negative Rejection**：拒绝只满足部分阶段轨迹的准确率；
- **Order Sensitivity Margin**：正确顺序和错误顺序查询的得分差。

### 公平性原则

- 固定检测框和候选轨迹时比较语言/事件模块；
- 单独报告完整端到端结果；
- 标明额外训练数据、LLM生成数据和预训练模型；
- 保持Backbone、输入分辨率和帧率一致；
- 报告参数量、FLOPs、显存、FPS和训练GPU小时。

## 5.7 资源预算与关键依赖

| 资源 | WACV最低配置 | CVPR建议配置 | 作用 |
|---|---|---|---|
| GPU | 1张24GB GPU | 2张24GB或1张48GB GPU | TempRMOT推理、事件头训练、消融 |
| 存储 | 200–500GB | 500GB–1TB | 视频帧、特征缓存、多个baseline输出 |
| 标注人员 | 2人，测试集双标 | 2–3人，测试集双标和仲裁 | 保证事件类别、目标ID和时间边界可靠 |
| 标注工时 | 约60–100人时 | 约200–350人时 | 查询编写、阶段边界、硬负样本和质量复核 |
| 代码基础 | TempRMOT必须可推理 | TempRMOT可训练，另有2–3个baseline可推理 | 公平比较和主模型开发 |
| MLLM/API | 非必需 | 非必需，仅可做离线辅助 | 避免主要结论依赖闭源服务 |

### 关键路径

```text
TempRMOT可运行
    ↓
Pilot证明顺序盲点
    ↓
事件标注可靠
    ↓
M0单调对齐有效
    ↓
M1可学习对齐
    ↓
M2动态证据路由
    ↓
组合/跨域/鲁棒性实验
```

其中前四项是不可跳过的关键路径。深度、3D、MLLM、SAM2和多Agent不在关键路径中。

---

## 6. WACV方案：事件理解诊断与轻量基线

## 6.1 会议定位

建议投稿WACV的Evaluations & Datasets Track，而不是用一个匆忙的复杂网络投稿Algorithms Track。WACV 2027官方征稿明确接受：

- 对现有评测的失败模式分析；
- 严格复现、审计和压力测试；
- 新评测协议；
- 新数据集和基准；
- 负面结果和应用驱动评测。

官方信息：[WACV 2027 Call for Papers](https://wacv.thecvf.com/Conferences/2027/CallForPapers)。

## 6.2 暂定标题

**Do RMOT Models Understand Ordered Events? A Diagnostic Benchmark and Stage-Aware Multi-Evidence Baseline**

## 6.3 论文故事

1. 现有RMOT在标准HOTA上不断提高；
2. 但标准查询无法判断模型是否理解事件顺序；
3. 构建小而严格的Ordered-Event RMOT诊断集；
4. 发现现有模型对顺序交换和部分满足轨迹不敏感；
5. 提出轻量阶段感知基线，在固定tracker下改善事件指标；
6. 给出数据质量、失败类型和未来研究建议。

WACV版本的主贡献是评测和实证发现，方法只需足以证明任务可学，不应为了“看起来复杂”堆模块。

## 6.4 最小研究范围

- 300–500条人工核验查询；
- 至少50条三阶段查询，其余以二阶段为主；
- 至少三种查询反事实；
- 2–3个开源baseline；
- 一个轻量stage scorer；
- 不加入在线MLLM、真实深度、音频、SAM2或多Agent。

## 6.5 WACV 2027 Round 2时间表

当前日期为2026-07-21。官方时间：

- 新论文注册：2026-08-21；
- 论文提交：2026-08-28；
- 补充材料：2026-08-30。

### 阶段W0：可提交性审计（7月21日—7月23日）

**作用与意义**

判断38天应急计划是否具备最低前提，避免在无法按时完成的条件下启动。

**具体步骤**

1. 确认TempRMOT环境能否运行单视频推理；
2. 确认Refer-KITTI/Refer-KITTI-V2数据完整；
3. 确认至少一张24GB GPU可持续使用；
4. 确认第二位标注者；
5. 确认每天可投入时间；
6. 列出FlexHook/DKGTrack可否在一周内跑通。

**输出内容**

- `readiness_report.md`；
- 环境、数据、算力和人员清单；
- baseline可运行状态表；
- 风险登记表。

**Gate W0**

TempRMOT不能在7月23日前完成推理、没有第二位标注者，或数据不完整，则停止WACV 2027应急计划，转入CVPR/WACV下一周期方案。

### 阶段W1：任务和标注协议（7月24日—7月27日）

**作用与意义**

消除“完整轨迹还是事件片段”“什么叫先后顺序”等任务歧义，保证后续标注和指标有效。

**具体步骤**

1. 冻结第一版事件本体；
2. 定义完整轨迹输出语义；
3. 定义事件阶段边界；
4. 定义正确、交换、删除和部分满足查询；
5. 编写标注示例和反例；
6. 实现指标单元测试。

**输出内容**

- `task_spec_v1.md`；
- `annotation_guideline_v1.md`；
- JSON Schema；
- Event-Track F1、Order Accuracy和Stage mIoU评测脚本；
- 20条金标准示例。

**Gate W1**

两位标注者对20条示例的目标轨迹一致率低于90%，必须修改任务定义，不得直接扩大标注。

### 阶段W2：60条Pilot和研究假设验证（7月28日—8月2日）

**作用与意义**

这是整项研究最关键的早期验证：确认真实视频中确实存在多阶段事件，并证明现有模型存在顺序盲点。

**具体步骤**

1. 选择包含明显状态变化的轨迹；
2. 双人独立标注60条查询；
3. 计算目标ID一致率、事件类别Cohen's kappa、阶段边界平均绝对误差和时间IoU；
4. 运行TempRMOT及至少一个对照模型；
5. 比较原查询、顺序交换和阶段删除查询；
6. 统计模型得分对顺序的敏感性。

**输出内容**

- `pilot_60.json`；
- 标注一致性报告；
- 基线顺序敏感性图；
- 初步错误分类；
- 一页研究假设验证报告。

**Gate W2：继续WACV的必要条件**

- 至少40条查询具有无歧义目标；
- 目标ID一致率不低于90%；
- 事件类别Cohen's kappa不低于0.70，阶段区间平均时间IoU不低于0.70；
- 至少一个强baseline的Order Accuracy低于65%，或正确/交换查询得分差非常小；
- 至少存在20条“关键词相同但顺序改变真值”的强反事实。

任一关键条件失败，则不应继续WACV 2027投稿。

### 阶段W3：诊断集扩展和质量控制（8月3日—8月10日）

**作用与意义**

将pilot扩展为可以支撑论文结论的诊断集，同时控制自动生成和模板泄漏。

**具体步骤**

1. 基于GT轨迹和运动状态自动提出候选事件；
2. 人工编写或修改查询；
3. 生成顺序交换、阶段删除和部分满足反事实；
4. 双人标注测试集；
5. 对训练候选进行抽样复核；
6. 建立视频、轨迹和模板不重叠的split。

**输出内容**

- 300–500条查询的`ordered_event_rmot_v0.9.json`；
- split文件；
- 标注统计表；
- 数据卡和局限说明；
- 50条可视化示例。

**Gate W3**

如果最终可靠查询少于250条、三阶段查询少于40条，或事件类别被一两种模板主导，则降级为内部研究集，不以新benchmark为论文主贡献。

### 阶段W4：轻量阶段感知基线（8月4日—8月14日）

该阶段可与W3的数据扩展交错进行。

**作用与意义**

证明诊断任务不是只能依赖人工规则或大型MLLM解决，并提供未来工作的可复现起点。

**具体步骤**

1. 冻结TempRMOT预测框和Track Query；
2. 实现规则/模板事件解析器；
3. 从框序列计算速度、方向、停止和相对位置；
4. 构建阶段—轨迹片段得分矩阵；
5. 使用动态规划执行单调对齐；
6. 将事件得分与原refer score融合。

**输出内容**

- 可运行的Stage-Aware Baseline；
- 配置文件和推理脚本；
- 正确/交换查询得分可视化；
- 固定tracker下的Event-Track F1和Order Accuracy。

**Gate W4**

轻量基线应满足至少一项：

- Order Accuracy比TempRMOT提高不低于8个百分点；
- Event-Track F1提高不低于5个百分点；
- Partial-Negative Rejection提高不低于8个百分点。

若三项均不满足，WACV论文缺少“任务可学”的证据，建议停止应急投稿。

### 阶段W5：完整实验和消融（8月15日—8月21日）

**作用与意义**

排除更强tracker、更多数据和更大模型造成的混淆。

**具体步骤**

1. 固定tracker比较原句、bag-of-events和ordered-events；
2. 比较整轨迹池化、固定窗口和单调对齐；
3. 去除外观、运动或关系证据；
4. 比较IID和order split；
5. 报告速度、参数、显存；
6. 完成统计置信区间和典型错误。

**输出内容**

- 主结果表；
- 4–6张消融表/图；
- 错误分析图；
- 公平比较声明；
- 论文标题、摘要和作者列表，并在8月21日前完成WACV注册。

### 阶段W6：论文和提交（8月22日—8月30日）

**作用与意义**

把技术结果压缩成一个审稿人能够快速验证的故事，并完成匿名、格式和复现检查。

**具体步骤**

1. 8月22日前冻结所有主要结果；
2. 按问题—证据—基准—发现—基线结构写作；
3. 主文8页内保留任务定义、数据质量、关键结果和局限；
4. 补充材料放完整标注协议、更多可视化和实现细节；
5. 执行匿名、引用、图表、数值一致性检查；
6. 8月28日提交论文，8月30日前提交补充材料。

**输出内容**

- WACV匿名论文PDF；
- Supplementary PDF/ZIP；
- 匿名代码快照；
- 数据卡；
- 最终复现清单。

## 6.6 WACV方案的采纳判断

### 故事完整性：8/10

问题明确：标准RMOT是否理解顺序；基准揭示失败；轻量方法证明任务可学；错误分析给出结论。作为E&D论文，故事完整。

### 科学可行性：8/10

事件顺序可以由轨迹和视频验证，轻量动态规划不依赖大模型，主要技术没有不可控环节。

### WACV 2027截止期可行性：4/10

从零开始只有约38天，最大的风险不是算法，而是环境、baseline和双人标注。只有Gate W0和W2都按时通过，才可能提高到6/10。

### 创新性：6.5/10

新意主要来自RMOT中的顺序诊断和Identity-preserving event alignment，而不是网络结构。对WACV E&D足够有潜力，对Algorithms Track偏弱。

### 可被采纳性：条件性5.5/10

这是主观审稿风险评分，不是录用概率。若满足以下条件，可提高到6.5–7/10：

- 至少三个baseline出现一致顺序盲点；
- 数据质量可靠；
- 轻量方法在事件指标上有稳定明显提升；
- 固定tracker后提升仍成立；
- 主文不夸大“首个推理RMOT”或“首个多模态RMOT”。

若只有几十条查询、只有TempRMOT一个baseline或只提高普通HOTA，论文更可能被拒。

**明确判断**：以当前尚未完成pilot和baseline的状态直接预测，WACV 2027 Round 2更接近“高风险、偏拒稿”。只有W0—W4全部按时通过后，才转为“有现实录用竞争力”。如果8月2日前不能证明稳定顺序盲点，应主动停止本轮投稿，而不是压缩实验质量。

---

## 7. CVPR方案：阶段自适应多证据事件—轨迹对齐

## 7.1 会议定位

CVPR官方主题覆盖Multimodal Learning、Video Action and Event Understanding、Video Motion and Tracking、Vision-Language-Reasoning。CVPR 2026审稿指南强调技术正确性、创新性、潜在影响和可复现性，而不要求必须刷新现有SOTA。

- [CVPR 2026 Call for Papers](https://cvpr.thecvf.com/Conferences/2026/CallForPapers)
- [CVPR 2026 Reviewer Guidelines](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines)

根据用户提供的 CVPR 2027 日程截图，本计划采用以下硬截止：

| 事项 | 北京时间（GMT+8） | 计划约束 |
|---|---:|---|
| 摘要注册 | 2026-11-07 15:59 | 标题、作者、摘要和研究主线必须在此之前稳定 |
| 全文提交 | 2026-11-13 15:59 | 主论文最终 PDF 必须在此之前完成上传与检查 |
| 会议时间 | 2027-06-14 至 2027-06-20 | Nashville, Tennessee |

从 2026-07-21 到全文截止约有 115 天。计划按 16 个完整研发周执行，并保留 2026-11-10 至 2026-11-13 作为最终检查和上传缓冲。截图未给出补充材料截止日期，因此本文不推测；待官方 Call for Papers 明确后单独加入。

## 7.2 暂定标题

**M²-EventRMOT: Stage-Adaptive Appearance-Motion-Geometry Reasoning for Ordered-Event Referring Multi-Object Tracking**

更保守且不易被“伪多模态”质疑的备选标题：

**EventRMOT: Identity-Preserving Ordered Event-Track Alignment with Stage-Adaptive Evidence Routing**

第二个标题更推荐。

## 7.3 CVPR论文故事

1. RMOT从静态属性扩展到动作和关系，但仍将整句和整轨迹粗粒度对齐；
2. 长历史不等于事件推理，模型无法验证多个阶段是否按顺序发生在同一Identity上；
3. 提出Ordered-Event RMOT任务和严格反事实评测；
4. 提出阶段自适应证据路由，为每个阶段选择外观、运动和几何证据；
5. 提出单调事件—轨迹对齐，显式维护阶段完成状态；
6. 在IID、未见组合、顺序交换、部分满足、证据缺失和跨域条件下验证；
7. 标准RMOT性能不退化，事件能力显著提高。

## 7.4 预期贡献

1. **任务/评测贡献**：Identity-preserving Ordered-Event RMOT定义和人类核验基准；
2. **方法贡献**：Stage-Adaptive Evidence Router；
3. **时序贡献**：Monotonic Event-Track Aligner与阶段完成记忆；
4. **泛化贡献**：顺序反事实、部分满足负样本和未见事件组合评测；
5. **分析贡献**：证明长历史、强tracker和更大文本模型不能替代显式事件推理。

## 7.5 16周研究计划

### 阶段C0：复现和Claim Freeze（第1周）

**作用与意义**

建立可信baseline并冻结唯一核心主张，防止项目演化成多个模块堆叠。

**具体步骤**

1. 运行TempRMOT训练或至少官方checkpoint评测；
2. 记录环境、数据版本和标准指标；
3. 尝试FlexHook和DKGTrack推理；
4. 固定核心主张为“有序事件—轨迹对齐”；
5. 将真实深度、MLLM和多Agent列入非必需扩展。

**输出内容**

- baseline复现表；
- 环境锁定文件；
- 一页Problem Statement；
- Claim/Non-Claim清单；
- 项目风险登记表。

**Gate C0**

TempRMOT标准评测无法在一周内跑通，先暂停新模型开发，继续解决复现问题。

### 阶段C1：事件本体和100条Pilot（第2–3周）

**作用与意义**

验证数据中存在足够事件，并确定哪些动作可以可靠标注。

**具体步骤**

1. 从至少两个数据源抽取候选轨迹；
2. 建立事件本体v1；
3. 双人标注100条二/三阶段查询；
4. 生成顺序交换和部分满足负样本；
5. 运行TempRMOT、FlexHook和bag-of-events baseline；
6. 分析模型是否忽略顺序。

**输出内容**

- `ontology_v1.yaml`；
- `pilot_100.json`；
- 标注一致性报告；
- 顺序敏感性基线；
- 数据来源选择报告。

**Gate C1**

- 至少70条查询无歧义；
- 至少25条三阶段事件；
- 标注一致性不低于0.75；
- baseline存在可复现顺序盲点。

若失败，优先缩小事件本体或更换数据源，而不是继续增加模型复杂度。

### 阶段C2：基准构建和泄漏控制（第4–6周）

**作用与意义**

建立能够支撑CVPR级结论的数据和评测，避免自动模板、视频重叠和近义词泄漏。

**具体步骤**

1. 自动挖掘运动状态变化候选；
2. 人工编写/校正事件查询；
3. 建立目标和硬负轨迹；
4. 双人核验测试集；
5. 构建IID、composition、order和cross-domain split；
6. 检查视频、轨迹、模板和语义簇泄漏；
7. 编写数据卡和伦理/局限说明。

**目标规模**

- 最低：800条人类核验查询；
- 目标：1,200–1,500条；
- 三阶段查询不少于200条；
- 测试集全部双人核验；
- 至少两个视频来源或一个数据源加明确跨域测试。

**输出内容**

- EventRMOT-Bench v1；
- 全部split；
- 数据质量报告；
- 泄漏审计报告；
- 基准评测代码和单元测试。

**Gate C2**

若可靠查询少于800条、事件分布极度单一或跨域数据无法获得，则CVPR故事降级，应转投WACV/ACM MM等更匹配场合。

### 阶段C3：M0——无学习/轻学习事件对齐基线（第5–7周）

**作用与意义**

证明任务信号确实来自事件和轨迹，而不是复杂网络偶然拟合；同时建立后续模型的下限。

**具体步骤**

1. 实现规则/模板事件解析；
2. 计算外观、运动和关系手工/预训练特征；
3. 多尺度轨迹切片；
4. 动态规划执行单调对齐；
5. 比较bag-of-events、无序最大池化和正确顺序；
6. 评估固定GT轨迹和预测轨迹两种情况。

**输出内容**

- M0 baseline代码；
- GT-track上限与predicted-track结果；
- 事件指标基准表；
- 数据可学习性报告。

**Gate C3**

M0应显著超过随机和bag-of-events。如果规则单调对齐已经接近饱和，则研究重点应转向基准和泛化，而不是设计大型神经网络。

### 阶段C4：M1——可学习事件编码与单调对齐（第7–9周）

**作用与意义**

建立论文的第一项核心方法贡献：让模型学习事件阶段和轨迹片段对应关系。

**具体步骤**

1. 使用文本编码器产生K个Stage Token；
2. 轨迹编码器产生多尺度片段Token；
3. 建立阶段—片段相似度矩阵；
4. 实现可微单调对齐；
5. 加入stage loss和order margin loss；
6. 保持TempRMOT检测/关联部分固定，先独立验证。

**输出内容**

- M1模型；
- 与M0、TempRMOT、无序attention的比较；
- 阶段区间可视化；
- 训练稳定性报告。

**Gate C4**

在固定tracker下至少满足：

- Order Accuracy提高8个百分点；
- Event-Track F1提高5个百分点；
- Stage mIoU有稳定正提升；
- 标准refer target F1不下降超过1个百分点。

### 阶段C5：M2——阶段自适应证据路由（第9–11周）

**作用与意义**

验证不同事件阶段需要不同证据，这是CVPR版本区别于简单时间对齐的关键方法贡献。

**具体步骤**

1. 分离外观、运动和几何/关系编码器；
2. 用Stage Token预测证据权重；
3. 加入modality/evidence dropout；
4. 加入噪声感知可靠性训练；
5. 比较concat、add、固定权重、普通cross-attention和动态路由；
6. 可选加入真实深度/3D，但只作为stretch experiment。

**输出内容**

- M2完整模型；
- 路由权重统计；
- 缺失/噪声/冲突鲁棒性表；
- 各事件阶段的证据选择矩阵；
- 参数、FLOPs和FPS报告。

**Gate C5**

动态路由必须在相同参数预算下明显优于固定融合，并在证据损坏时表现出合理的权重转移。若只提升标准HOTA或权重始终集中在外观流，则不能保留“阶段自适应证据”主张。

### 阶段C6：组合泛化与反事实训练（第11–12周）

**作用与意义**

证明模型没有只记住事件模板，并与COAL的属性反事实工作形成清晰区分。

**具体步骤**

1. 加入顺序交换负样本；
2. 加入阶段删除/插入；
3. 加入只满足前缀或后缀的轨迹；
4. 在composition/order split上训练和测试；
5. 对比无反事实、只有属性反事实和事件反事实。

**输出内容**

- 组合泛化结果；
- 不同反事实类型消融；
- 训练—测试泄漏检查；
- 与COAL差异的实证证据。

**Gate C6**

事件反事实必须在order/composition split上带来稳定收益，而不是只提高IID结果。

### 阶段C7：完整基线、公平比较和跨域（第13–14周）

**作用与意义**

形成CVPR所需的完整证据链，并排除tracker、Backbone和额外数据造成的不公平提升。

**Baseline矩阵**

- TransRMOT；
- TempRMOT；
- iKUN或FlexHook；
- DKGTrack；
- bag-of-events；
- M0、M1、M2；
- 可运行时加入ReferGPT；
- STORM若模型代码仍未开放，只做定性讨论，不伪造直接比较。

**具体实验**

1. 标准Refer-KITTI/V2；
2. EventRMOT-Bench IID；
3. composition split；
4. order split；
5. partial-negative；
6. predicted-track和GT-track；
7. 跨域测试；
8. 证据缺失和噪声；
9. 速度、显存和训练成本；
10. 置信区间和至少三次随机种子。

**输出内容**

- 冻结主结果表；
- 完整消融；
- 公平性和资源表；
- 失败案例分类；
- 可复现实验清单。

**Gate C7**

正式进入CVPR写作需要：

- Event-Track F1提高至少5点；
- Order Accuracy提高至少8点；
- order/composition split均有收益；
- 标准HOTA下降不超过1点，最好持平或提高；
- 固定tracker后主要结论仍成立；
- 至少两个数据设置支持结论。

### 阶段C8：论文写作与内部审稿（第15–16周）

**作用与意义**

把复杂实验压缩成单一因果故事，并在投稿前模拟最严格审稿。

**主文结构**

1. 问题：长历史不等于事件推理；
2. 诊断：现有模型忽略事件顺序；
3. 任务：Identity-preserving Ordered-Event RMOT；
4. 方法：Stage-Adaptive Router + Monotonic Aligner；
5. 实验：顺序、组合、跨域和鲁棒性；
6. 分析：不同阶段需要不同证据；
7. 局限：数据规模、事件本体和派生运动特征。

**内部审稿清单**

- 是否只是TempRMOT加模块；
- 是否只是把运动特征拼接进来；
- 是否把更强tracker误认为语言理解；
- 是否与DKGTrack/VMRMOT/COAL重叠；
- 是否存在自动模板泄漏；
- 是否只有HOTA提升而没有顺序指标；
- 是否使用额外数据但未公平说明；
- 是否在标题中夸大multimodal；
- 是否所有主要结论都能被一项消融直接支持。

**输出内容**

- 8页匿名主文；
- Supplementary；
- 匿名代码和配置；
- 数据卡；
- 模拟审稿意见和逐条答复；
- 最终提交检查表。

## 7.6 CVPR交稿前硬里程碑

| 日期 | 必须完成的事项 | 未完成时的处置 |
|---|---|---|
| 2026-10-26 | 核心方法与主要对照实验冻结 | 停止加入新模块，只修复影响结论的缺陷 |
| 2026-11-02 | 论文初稿、主表和主图完整 | 若故事仍不闭环，立刻缩减次要实验与扩展模块 |
| 2026-11-05 | 内部审稿完成，标题与摘要冻结 | 只保留能够被实验支持的贡献表述 |
| 2026-11-07 15:59 | 完成摘要注册 | 至少提前24小时完成系统录入与作者确认 |
| 2026-11-09 | 全部结果、表格、图和代码版本冻结 | 后续只允许复核与修正错误 |
| 2026-11-11 | 最终PDF、匿名性、引用和页数审计 | 使用逐项检查表，由另一人交叉复核 |
| 2026-11-12 | 完成一次正式上传演练 | 下载并回读平台生成文件，核验无缺页、乱码或错误版本 |
| 2026-11-13 15:59 | 全文提交 | 目标为至少提前6小时完成最终提交 |

## 7.7 CVPR方案的采纳判断

### 故事完整性：9/10

问题、诊断、任务、方法、泛化评测和分析形成闭环。身份保持使其区别于普通Video Temporal Grounding，阶段路由使其区别于TempRMOT的历史聚合。

### 科学可行性：8/10

核心方法可以从固定tracker和二维轨迹开始，不依赖额外传感器或大模型。数据和标注是最大工作量，但有明确pilot和缩减机制。

### 工程可行性：7/10

建立在TempRMOT上，M0/M1/M2逐步推进，任一高级模块失败仍有可用前一阶段。若需要从零修复TempRMOT环境或只有单张低显存GPU，降至5/10。

### 创新性：8/10

创新不在单个attention模块，而在身份保持的有序事件任务、单调对齐和阶段证据选择的组合。若没有可靠新评测或只有固定融合，则降至5/10。

### CVPR可被采纳性：条件性6.5/10

该评分表示“具备竞争性但仍然是高风险主会论文”。达到以下条件时，可能形成strong borderline至accept profile：

- 新基准规模和质量可信；
- 至少三个强RMOT baseline显示一致顺序盲点；
- 方法同时改善order、composition和partial-negative；
- 固定tracker后结论成立；
- 标准RMOT性能不明显下降；
- 路由器在证据缺失/冲突时表现出可解释且定量有效的适应；
- 不依赖闭源MLLM得到主要结果。

以下情况更可能被CVPR拒绝：

- 只有一个小数据集或几百条自动表达；
- 方法表现为TempRMOT加运动分支和attention；
- 只报告HOTA；
- 没有顺序交换硬负样本；
- 使用更强tracker或更多预训练数据造成提升；
- 把派生运动特征夸大成新传感器多模态；
- 与DKGTrack、VMRMOT、COAL的差异没有实验证明。

**明确判断**：CVPR方案在科学故事上达到主会候选水平，但当前仍只是研究计划，不能判断为“会被接收”。若C2—C7全部达到门槛，预期属于strong-borderline到有竞争力的accept profile；若只有一个阶段模块和标准HOTA小幅提升，则更可能被拒。

---

## 8. WACV与CVPR方案对照

| 维度 | WACV路线 | CVPR路线 |
|---|---|---|
| 核心问题 | 现有RMOT是否理解事件顺序 | 如何通过阶段证据路由和单调对齐解决有序事件RMOT |
| 主贡献 | 诊断基准、评测协议、轻量baseline | 新任务/基准、完整方法、组合泛化和跨域分析 |
| 推荐Track | WACV E&D | CVPR Main Technical Program |
| 数据目标 | 300–500条严格查询 | 最低800，目标1,200–1,500条 |
| 方法复杂度 | 规则/轻量stage scorer + DP | 可学习Stage Token + Router + Monotonic Aligner |
| MLLM | 不进入主流程 | 仅可做离线解析/教师，不作为主要结果依赖 |
| 真实深度/3D | 不做 | 通过Gate后作为可选增强 |
| 核心指标 | Order Accuracy、Event-Track F1、Stage mIoU | 上述全部 + composition、跨域、鲁棒性 |
| 最大风险 | 38天期限和标注规模 | 数据质量、方法被认为模块堆叠 |
| 故事风格 | 实证、评测、应用 | 新问题、新方法、强分析 |
| 当前建议 | 只有前置条件全部满足才冲WACV 2027 | 主要研究路线 |

---

## 9. 录用风险模拟

## 9.1 WACV审稿人可能给出的正面意见

- 首次系统揭示RMOT中的事件顺序盲点；
- 评测协议严格且符合E&D Track；
- 轻量baseline实用、可复现；
- 对现有高HOTA结论形成有价值修正；
- 资源需求低，适合实际部署。

## 9.2 WACV审稿人可能的拒稿理由

- 数据规模过小；
- 查询主要由模板自动生成；
- 只测试TempRMOT；
- stage scorer过于简单且提升有限；
- 与Video Temporal Grounding的区别不清楚；
- 由于时间紧张，论文和实验不完整。

## 9.3 CVPR审稿人可能给出的正面意见

- 提出具有普适意义的identity-preserving event reasoning问题；
- 顺序反事实证明现有方法的真实缺陷；
- 动态证据路由和单调对齐具有明确因果动机；
- 泛化、跨域、鲁棒性和效率实验充分；
- 方法不依赖大型闭源模型。

## 9.4 CVPR审稿人可能的拒稿理由

- 任务过于小众，影响范围有限；
- 多个模块组合但单项创新不足；
- 运动/几何仍来自RGB，不是真正新模态；
- 事件标注带有主观性；
- 与DKGTrack、VMRMOT或VideoITG差异不足；
- 主要收益来自更强轨迹表示而非语言顺序理解。

## 9.5 对应防御证据

| 可能质疑 | 必须准备的证据 |
|---|---|
| 只是增加模块 | M0→M1→M2逐步结果，证明单调对齐和动态路由分别解决不同失败 |
| 更强tracker造成提升 | 固定GT/predicted tracks和固定Backbone实验 |
| 不是真多模态 | 使用multi-evidence表述；加入证据缺失/冲突实验，不把派生运动说成独立传感器 |
| 数据太小 | 多数据源、双人测试标注、置信区间和泄漏审计 |
| 与VTG相同 | 强调输出多目标完整轨迹、Identity一致性、关联指标和部分满足轨迹 |
| 与DKGTrack相同 | DKGTrack只有静态/运动解耦；本方法需要多阶段单调对齐、完成状态和顺序反事实 |
| 与COAL相同 | COAL重点是属性反事实和稀疏语义；本方法重点是动作阶段顺序和轨迹片段 |
| MLLM是否必要 | 主模型不依赖MLLM；如果使用，仅提供离线结构教师并与规则/小模型比较 |

---

## 10. 95%可执行性复审

本计划经过以下六轮逻辑复审。

### 复审1：问题是否真实

通过TempRMOT、DKGTrack、VMRMOT、ReaMOT、STORM、VideoITG和TimeLens的机制对比，确认“历史时序”“运动语言”“隐式推理”“时间定位”均不能自动等同于身份保持的有序事件满足。

结论：通过。

### 复审2：是否只是热点组合

删除MLLM、SAM2、深度、音频和多Agent后，核心任务和方法仍成立：Stage Token + appearance/motion/geometry evidence + monotonic alignment。

结论：通过。

### 复审3：是否能从TempRMOT最小实现

MVP只需要：

1. 导出候选轨迹和Track Query；
2. 从框序列计算运动；
3. 人工/规则得到阶段Token；
4. 动态规划对齐；
5. 重打refer score。

不需要修改检测器、重新训练MLLM或引入新传感器。

结论：通过。

### 复审4：数据失败时是否有止损

100条pilot、标注一致性和事件数量均设置Gate；数据不足时可以缩小事件本体、更换数据源或转为组合评测，而不是继续盲目开发模型。

结论：通过。

### 复审5：方法失败后是否仍有贡献

- WACV路线可保留诊断基准和负面发现；
- CVPR路线中M2失败仍可保留M1单调对齐；
- 完整方法失败仍能形成事件顺序审计和baseline。

结论：通过。

### 复审6：会议故事是否匹配

- WACV E&D接受失败分析、评测协议和数据集；
- CVPR覆盖多模态学习、事件理解、tracking和vision-language reasoning，并强调创新和潜在影响。

结论：通过。

## 10.1 达到95%执行置信度所需前提

只有满足以下条件，才对“能够执行到明确科研结论”给出约95%置信度：

1. TempRMOT在一周内能够完成标准推理；
2. Refer-KITTI/V2及至少一个补充视频数据源可用；
3. 至少一张24GB GPU，推荐两张；
4. 测试集有两位独立标注者，并分别统计事件类别一致性与时间边界误差；
5. 能保证连续12–16周研究时间；
6. 每个Gate失败时愿意缩小或终止方向，而不是不断增加模块；
7. CVPR 路线严格按 2026-11-07 摘要注册和 2026-11-13 全文提交执行；若官方 Call for Papers 后续更新，以官方通知为最高优先级同步修订。

如果第1、2、4项不满足，则不能声称95%可执行，应先解决基础条件。

---

## 11. 最终决策建议

### 如果目标是WACV 2027 Round 2

仅在2026-07-23前TempRMOT已经可运行、2026-08-02前pilot证明强顺序盲点时继续。论文定位为E&D诊断工作，不做完整多模态网络。

### 如果目标是高质量三个月研究

选择CVPR路线的核心技术，但不要在标题和贡献中夸大“多模态”。优先顺序：

1. 数据pilot和顺序盲点；
2. M0单调对齐；
3. M1可学习事件对齐；
4. M2阶段证据路由；
5. 组合泛化；
6. 可选3D/深度。

### 如果中途结果不足

- 数据可靠但模型提升小：转WACV E&D或benchmark论文；
- 顺序盲点不明显：停止EventRMOT，转向语言反馈RepairHook；
- 动态路由无效但单调对齐有效：删除路由器，以更紧凑的EventRMOT投稿；
- 只有标准HOTA提升：不应声称事件推理，重新检查混淆因素；
- 真实事件数据不足：不要依赖大量LLM模板伪造，应终止或更换数据源。

最终推荐：**以CVPR版本为主研究路线，以WACV诊断版本作为可降级、可保留的安全出口。**

---

## 12. 关键参考文献

### RMOT

- [Referring Multi-Object Tracking / TransRMOT, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_Referring_Multi-Object_Tracking_CVPR_2023_paper.html)；[代码](https://github.com/wudongming97/RMOT)
- [Bootstrapping Referring Multi-Object Tracking / TempRMOT, arXiv 2024](https://arxiv.org/abs/2406.05039)；[代码](https://github.com/zyn213/TempRMOT)
- [iKUN: Speak to Trackers without Retraining, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Du_iKUN_Speak_to_Trackers_without_Retraining_CVPR_2024_paper.html)；[代码](https://github.com/dyhBUPT/iKUN)
- [HFF-Tracker, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/33143)
- [Cross-View Referring Multi-Object Tracking, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32219)；[代码](https://github.com/chen-si-jia/CRMOT)
- [Language Decoupling with Fine-grained Knowledge Guidance / DKGTrack, ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Language_Decoupling_with_Fine-grained_Knowledge_Guidance_for_Referring_Multi-object_Tracking_ICCV_2025_paper.html)；[代码](https://github.com/acyddl/DKGTrack)
- [ReferGPT, CVPR Workshops 2025](https://openaccess.thecvf.com/content/CVPR2025W/WDFM-AD/html/Chamiti_ReferGPT_Towards_Zero-Shot_Referring_Multi-Object_Tracking_CVPRW_2025_paper.html)；[代码](https://github.com/Tzoulio/ReferGPT)
- [Vision-Motion-Reference Alignment / VMRMOT, arXiv 2025](https://arxiv.org/abs/2511.17681)
- [COAL, IJCAI 2026](https://arxiv.org/abs/2605.14795)；[IJCAI接收列表](https://2026.ijcai.org/accepted-papers/)
- [ReaMOT, arXiv 2025/2026](https://arxiv.org/abs/2505.20381)；[仓库](https://github.com/chen-si-jia/ReaMOT)
- [FlexHook, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Rethinking_Two-Stage_Referring-by-Tracking_in_Referring_Multi-Object_Tracking_Make_it_Strong_CVPR_2026_paper.html)；[代码](https://github.com/buptLwz/FlexHook)
- [STORM, CVPR Findings 2026](https://openaccess.thecvf.com/content/CVPR2026F/html/Lu_STORM_End-to-End_Referring_Multi-Object_Tracking_in_Videos_CVPRF_2026_paper.html)；[数据](https://github.com/amazon-science/storm-referring-multi-object-grounding)
- [DRMOT, arXiv 2026](https://arxiv.org/abs/2602.04692)
- [RT-RMOT, arXiv 2026](https://arxiv.org/abs/2602.22033)

### 视频时间理解与轨迹表示

- [TimeChat, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Ren_TimeChat_A_Time-sensitive_Multimodal_Large_Language_Model_for_Long_Video_CVPR_2024_paper.pdf)；[代码](https://github.com/RenShuhuai-Andy/TimeChat)
- [VTG-LLM, AAAI 2025](https://arxiv.org/abs/2405.13382)；[代码](https://github.com/gyxxyg/VTG-LLM)
- [Seq2Time, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Deng_Seq2Time_Sequential_Knowledge_Transfer_for_Video_LLM_Temporal_Grounding_CVPR_2025_paper.pdf)
- [VideoITG, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_VideoITG_Multimodal_Video_Understanding_with_Instructed_Temporal_Grounding_CVPR_2026_paper.html)
- [TimeLens, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhang_TimeLens_Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs_CVPR_2026_paper.pdf)
- [TrajTok, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_TrajTok_Learning_Trajectory_Tokens_Enhances_Video_Understanding_CVPR_2026_paper.pdf)
- [MeMOTR, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_MeMOTR_Long-Term_Memory-Augmented_Transformer_for_Multi-Object_Tracking_ICCV_2023_paper.html)；[代码](https://github.com/MCG-NJU/MeMOTR)
- [MOTIP, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html)；[代码](https://github.com/MCG-NJU/MOTIP)

### 会议官方信息

- [WACV 2027 Call for Papers](https://wacv.thecvf.com/Conferences/2027/CallForPapers)
- [WACV 2027 Dates](https://wacv.thecvf.com/Conferences/2027/Dates)
- [WACV 2027 Author Guidelines](https://wacv.thecvf.com/Conferences/2027/AuthorGuides)
- [CVPR 2026 Call for Papers](https://cvpr.thecvf.com/Conferences/2026/CallForPapers)
- [CVPR 2026 Reviewer Guidelines](https://cvpr.thecvf.com/Conferences/2026/ReviewerGuidelines)
- [CVPR 2026 Author Guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines)
