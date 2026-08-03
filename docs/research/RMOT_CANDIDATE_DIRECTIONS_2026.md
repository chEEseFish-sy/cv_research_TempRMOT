# RMOT 候选研究方向图谱（2026）

> 更新日期：2026-07-23  
> 文档用途：方向调研，不做最终选题、排序或会议承诺。  
> 研究基础：TempRMOT 官方检查点的 Refer-KITTI-V2 全量推理与评测已经完成，862 个文本查询上的 HOTA 为 35.249；当前尚未完成从头训练复现。

## 1. 为什么需要重新整理方向

现有项目已经越过了“模型是否能跑通”的最低门槛。下一阶段不宜只围绕 TempRMOT 增加一个注意力模块，而应先明确 RMOT 的具体失败模式，再判断多模态大模型、轨迹 Token、长视频记忆或基础模型是否提供不可替代的能力。

2025—2026 年直接影响选题边界的新进展主要有：

- [FlexHook（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Rethinking_Two-Stage_Referring-by-Tracking_in_Referring_Multi-Object_Tracking_Make_it_Strong_CVPR_2026_paper.html)强化了低成本、双阶段 Referring-by-Tracking，说明“冻结 Tracker，再挂一个普通语言模块”本身已不够新。
- [STORM（CVPR 2026 Findings）](https://openaccess.thecvf.com/content/CVPR2026F/html/Lu_STORM_End-to-End_Referring_Multi-Object_Tracking_in_Videos_CVPRF_2026_paper.html)把 8B 多模态大模型直接用于端到端 RMOT，说明“把 MLLM 用到 RMOT”本身也不再构成创新。
- [COAL（arXiv 2026）](https://arxiv.org/abs/2605.14795)已经引入反事实属性与显式视觉语义，压缩了普通属性增强和负样本增强的空间。
- [OmniPT（AAAI 2026）](https://ojs.aaai.org/index.php/AAAI/article/view/37406)尝试用统一视觉语言模型处理 MOT、RMOT、跨视角跟踪等任务，单纯“任务统一”需要更强的问题定义。
- [TrajTok（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_TrajTok_Learning_Trajectory_Tokens_Enhances_Video_Understanding_CVPR_2026_paper.pdf)、[TimeLens（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_TimeLens_Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs_CVPR_2026_paper.html)和长视频记忆工作，使“轨迹离散表示、时序定位、结构化记忆”成为可迁移到 RMOT 的前沿技术。
- [GMT（CVPR 2026）](https://openaccess.thecvf.com/content/CVPR2026/html/Zhen_GMT_Effective_Global_Framework_for_Multi-Camera_Multi-Target_Tracking_CVPR_2026_paper.html)已经在多摄像头 MOT 中进行全局轨迹—目标关联，因此跨视角 RMOT 必须说明语言如何改变全局身份维护，而不能只复制全局轨迹库。

## 2. 候选方向总览

下表只说明研究问题与趋势来源，不代表优先级。

| 编号 | 候选方向 | RMOT 核心问题 | 主要趋势接口 |
|---|---|---|---|
| D1 | 有序多阶段事件 RMOT | 当前描述多为静态属性或局部动作，难以验证“先—再—最后” | Video Temporal Grounding、事件 Token、反事实顺序 |
| D2 | 组合泛化与泄漏受控评测 | 模型可能记住属性组合，而非理解新组合 | Open-Vocabulary、Compositional Generalization |
| D3 | 语言反馈式关联纠错 | 双阶段语言模块通常读取轨迹，却不回写和修复 Tracker | MOT 关联建模、约束推断、反馈学习 |
| D4 | 不确定性触发的 MLLM 验证 | 全程调用 MLLM 慢且易幻觉，简单样本不需要大模型 | 动态推理、Verifier、蒸馏、Agentic Vision |
| D5 | Track/Trajectory Token RMOT | 像素与逐帧框难以支撑长视频语义推理 | Object Token、Trajectory Token、Video-LLM |
| D6 | 状态转折驱动的流式轨迹记忆 | 历史聚合可能只是“看得更长”，并未保存事件结构 | Streaming Video、结构化长时记忆 |
| D7 | 开放概念、可提示的基础模型 RMOT | 固定类别和封闭词表限制零样本场景 | Grounding DINO、SAM2/SAM3、开放词汇 |
| D8 | 跨视角 Global Identity Query | 局部 Track ID 无法自然成为跨摄像头共享身份 | Multi-camera MOT、全局身份记忆 |
| D9 | 可靠与可校准 RMOT | 语言扰动、遮挡与视觉—语言冲突会造成持续错误 | Robustness、Calibration、可信多模态学习 |
| D10 | 缺失模态鲁棒的多传感器 RMOT | RGB 在夜间和恶劣天气失效，但额外传感器也可能缺失 | RGB-D、RGB-T、动态路由、MoE |

## 3. D1：有序多阶段事件 RMOT

### 研究问题

现有 RMOT 能否真正区分“先跟随卡车，随后变道超车，最后停在路边”和相同事件片段的错误排列？如果只聚合更长历史，模型可能知道各动作出现过，却不知道动作由谁、在什么阶段、按什么顺序发生。

### 可形成的方法

1. 将自然语言解析为有顺序的事件单元，例如主体、关系对象、动作、阶段连接词和时间约束。
2. 将一条轨迹划分为候选片段，为每个片段生成阶段表示。
3. 使用单调对齐或区间集合预测，使事件阶段与轨迹片段建立显式对应。
4. 加入顺序反事实：交换事件顺序、删除中间阶段、替换关系对象，要求正确轨迹得分下降。
5. 最终输出目标轨迹，同时输出阶段—时间区间，令“事件理解”能够被单独评测。

### 创新成立的边界

- 不能只把 TempRMOT 的历史长度增大；必须对事件阶段、顺序和时间区间提供显式监督或可检验输出。
- 不能只报告 HOTA；至少要报告顺序敏感准确率、阶段定位质量、反事实轨迹区分率。
- 应与 [DKGTrack](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Language_Decoupling_with_Fine-grained_Knowledge_Guidance_for_Referring_Multi-object_Tracking_ICCV_2025_paper.html)的语言解耦、[TimeLens](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_TimeLens_Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs_CVPR_2026_paper.html)的时序定位和 [STORM](https://openaccess.thecvf.com/content/CVPR2026F/html/Lu_STORM_End-to-End_Referring_Multi-Object_Tracking_in_Videos_CVPRF_2026_paper.html)的端到端 MLLM 路线区分。

### 最小验证与主要风险

- 最小验证：先从 Refer-KITTI-V2 中人工筛选或补写少量双阶段/三阶段表达，构造顺序交换测试集；验证 TempRMOT 对顺序是否不敏感。
- 风险：现有视频的事件复杂度不足，数据标注可能成为主要贡献与主要成本。

## 4. D2：组合泛化与泄漏受控 RMOT

### 研究问题

模型在训练中见过“红色汽车”和“左转卡车”后，能否在测试时定位“红色且左转的卡车”？更严格地说，它能否泛化到未见的属性—关系—动作组合，而不是依赖对象类别或场景泄漏？

### 可形成的方法

1. 为表达建立属性、类别、空间关系、交互关系、动作和事件顺序标签。
2. 设计组合隔离划分：测试组合在训练集中从未出现，同时控制同一视频、同一目标和近义表达泄漏。
3. 引入因子化语言表示或可组合原型，使不同语义因子分别对轨迹证据进行匹配。
4. 使用难负样本和组合反事实，避免模型只靠类别或显著属性完成任务。
5. 同时报告 IID、未见组合和反事实一致性，区分跟踪能力与语言泛化能力。

### 创新成立的边界

- 仅做新的 train/test split 可能被视为“缺少实验”；需要证明主流方法的系统性失败，并提供针对组合推断的机制。
- [COAL](https://arxiv.org/abs/2605.14795)已经使用属性反事实，因此本方向需要把重点放在严格的组合隔离、关系/动作/顺序组合，而非普通属性替换。
- 必须审计大模型生成数据是否泄漏答案或引入模板偏差。

### 最小验证与主要风险

- 最小验证：在 Refer-KITTI-V2 元数据上构建小规模 compositional split，对 TransRMOT、TempRMOT 和 iKUN 类方法做零训练诊断。
- 风险：表达标签不完整；若方法只改善一个人工小集，论文说服力有限。

## 5. D3：语言反馈式关联纠错

### 研究问题

iKUN 和 FlexHook 证明了冻结或复用 MOT Tracker 的低成本路线可行，但语言模块通常是在已有轨迹上做选择。如果 Tracker 已发生 ID Switch、误合并或遮挡后错误重连，语言证据能否反向修改关联结果？

### 可形成的方法

1. Tracker 生成候选轨迹和关联图，而非立即冻结唯一关联。
2. 语言模块把描述转成属性、关系、动作和身份持续性证据。
3. 将证据写回为 must-link、cannot-link、延迟决策或轨迹拆分/重连约束。
4. 在滑动窗口中重新求解局部关联，仅修复高风险位置。
5. 保留原 Tracker 的检测与运动能力，重点评测语言是否减少特定 ID Switch。

### 创新成立的边界

- 不能只是用语言重排最终轨迹；必须让语言改变关联边或身份状态。
- 应与 [iKUN](https://openaccess.thecvf.com/content/CVPR2024/html/Du_iKUN_Speak_to_Trackers_without_Retraining_CVPR_2024_paper.html)、[FlexHook](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Rethinking_Two-Stage_Referring-by-Tracking_in_Referring_Multi-Object_Tracking_Make_it_Strong_CVPR_2026_paper.html)和 [MOTIP](https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.pdf)比较信息流差异。
- 必须排除“换了更强 Tracker 所以 HOTA 上升”的混淆，最好冻结检测结果并进行受控关联评测。

### 最小验证与主要风险

- 最小验证：人工或程序化注入轨迹断裂、身份交换和误合并，测试语言约束能否恢复正确身份。
- 风险：自然语言未必包含足够信息来修复每一种关联错误；需要允许系统拒绝修复。

## 6. D4：不确定性触发的 MLLM 验证与轻量蒸馏

### 研究问题

MLLM 擅长关系、常识和复杂指令，但全视频逐帧调用成本高、定位不稳且可能幻觉。是否可以只在遮挡、身份竞争、视觉—语言冲突或低置信度时调用它，并让验证结果影响关联或目标选择？

### 可形成的方法

1. 轻量 RMOT 主模型持续运行，并估计关联熵、候选间隔、遮挡程度和跨模态一致性。
2. 路由器只选择高风险时间窗与少量候选轨迹。
3. MLLM 作为 Verifier 比较证据、提出假设或判定约束，不负责全程输出坐标。
4. 可靠性模块校准或拒绝不确定的 MLLM 结论。
5. 将高质量验证结果蒸馏回轻量模型，降低推理调用率。

### 创新成立的边界

- “用了 MLLM”不再新；[STORM](https://openaccess.thecvf.com/content/CVPR2026F/html/Lu_STORM_End-to-End_Referring_Multi-Object_Tracking_in_Videos_CVPRF_2026_paper.html)、[ReferGPT](https://openaccess.thecvf.com/content/CVPR2025W/WDFM-AD/html/Chamiti_ReferGPT_Towards_Zero-Shot_Referring_Multi-Object_Tracking_CVPRW_2025_paper.html)和 [OmniPT](https://ojs.aaai.org/index.php/AAAI/article/view/37406)已经覆盖多种 MLLM 路线。
- 关键创新应是“何时调用、看什么证据、如何可靠地回写、节省多少成本”。
- 多 Agent 只有在角色分工可消融、优于单次提示且成本可控时才成立，否则属于热点堆叠。

### 最小验证与主要风险

- 最小验证：用 TempRMOT 置信度筛出错误密集窗口，测试一个离线 Verifier 能否提高错误识别率，再决定是否接入闭环。
- 风险：调用延迟与闭源模型依赖；Verifier 可能“解释得像对，但无法稳定改变跟踪结果”。

## 7. D5：Track/Trajectory Token 驱动的 RMOT

### 研究问题

逐帧图像 Token 数量大，单个框又丢失运动与身份上下文。能否把一条轨迹或轨迹片段压缩为保持身份、外观、运动和事件状态的 Token，作为 Video-LLM 与 RMOT 解码器之间的接口？

### 可形成的方法

1. 从轨迹的外观、位置、速度、遮挡和时间跨度构造连续 Track Token。
2. 在状态转折处切分轨迹，得到可组合的 Trajectory Segment Token。
3. 用文本 Token 与轨迹 Token 做粗到细匹配，再回到原视频完成精确框级输出。
4. 通过轨迹重建、身份一致性和文本—轨迹对比目标保持可追踪信息。
5. 在长视频上比较 Token 数量、速度、性能与事件推理能力。

### 创新成立的边界

- [TrajTok](https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_TrajTok_Learning_Trajectory_Tokens_Enhances_Video_Understanding_CVPR_2026_paper.pdf)已提出可微轨迹 Tokenizer；本方向必须体现 RMOT 的身份持续、框级定位和在线更新需求。
- 不能只把轨迹特征换个名字叫 Token；需要证明它是有效的信息瓶颈，并支持长序列或复杂语言能力。
- 与 D1 的关系：D5 是表示接口，D1 是事件推理问题；可组合，但任何单篇工作需要明确主次。

### 最小验证与主要风险

- 最小验证：冻结现有 Tracker，用轨迹级编码器替换逐帧跨模态匹配，测准确率、Token 数和吞吐量。
- 风险：压缩后丢失精确局部证据；数据规模不足以训练通用 Tokenizer。

## 8. D6：状态转折驱动的流式轨迹记忆

### 研究问题

TempRMOT 使用历史 Query 增强当前帧，但已复现检查点的实际历史长度为 4。新的记忆工作不应只追求更长 FIFO 队列，而应保存“何时发生了身份可靠更新、动作改变、遮挡、交互开始或结束”。

### 可形成的方法

1. 建立短期运动记忆、长期身份原型和稀疏事件记忆。
2. 用状态转折检测决定写入，而不是每帧等权写入。
3. 读取时结合当前文本阶段、候选身份和记忆可靠性。
4. 对矛盾、过时和低质量记忆进行衰减或删除。
5. 评测长遮挡恢复、干扰目标竞争、事件阶段切换和固定显存下的性能。

### 创新成立的边界

- 必须超越 [TempRMOT](https://arxiv.org/abs/2406.05039)、[MeMOTR](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_MeMOTR_Long-Term_Memory-Augmented_Transformer_for_Multi-Object_Tracking_ICCV_2023_paper.html)以及长短期记忆 RMOT 的普通历史聚合。
- 单纯将 `hist_len=4` 压缩为更少历史，科研叙事偏弱；更有价值的问题是“写什么、何时写、何时信”。
- 应报告按遮挡长度、记忆年龄和状态转折分桶的结果，而非只有总体 HOTA。

### 最小验证与主要风险

- 最小验证：离线分析 TempRMOT 错误与遮挡长度、动作转折、历史 Query 相似度的关系，确认记忆确实是瓶颈。
- 风险：Refer-KITTI-V2 视频长度和遮挡复杂度可能不足，需要额外长视频数据或合成压力测试。

## 9. D7：开放概念、可提示的基础模型 RMOT

### 研究问题

Grounding DINO、SAM2 和 SAM3 已能按开放文本概念检测、分割或追踪对象。RMOT 的剩余难点是：如何把开放概念响应转为稳定的多目标身份，并在概念相似、遮挡和关系表达下避免身份漂移？

### 可形成的方法

1. 使用开放词汇检测/分割基础模型产生候选实例与掩码轨迹。
2. 建立文本概念、实例身份和跨帧记忆三者之间的可校准匹配。
3. 将关系、动作或否定描述转成对候选轨迹的约束，而非只做名词检测。
4. 在开放类别、同类多实例和未见组合上评测身份稳定性。
5. 研究基础模型置信度与 RMOT 拒答/降级策略。

### 创新成立的边界

- 直接串联 Grounding DINO + SAM2 + Tracker 属于系统组合，不足以成为方法创新。
- [SAM3](https://arxiv.org/abs/2511.16719)已面向可提示概念的实例检测、分割和跟踪；新工作必须聚焦其未解决的复杂指代、身份持续或时序事件。
- 应公平报告外部数据、参数规模和推理成本，不能把更大预训练模型带来的增益误写为 RMOT 机制创新。

### 最小验证与主要风险

- 最小验证：将同一 RMOT 表达送入开放概念模型，量化召回、同类身份混淆与关系/动作失败类型。
- 风险：基础模型快速迭代导致方法被覆盖；硬件和数据许可也可能限制复现。

## 10. D8：跨视角 Global Identity Query / Memory

### 研究问题

CRMOT 已定义跨视角指代跟踪，但局部摄像头内 Track Query 与跨摄像头全局身份仍是不同层级。能否维护一个跨视角共享、语言条件化、可持续更新的 Global Identity Query？

### 可形成的方法

1. 每个摄像头维护 Local Track Query，负责局部检测与短时关联。
2. 全局记忆为每个真实身份维护 Global Identity Query，聚合跨视角外观、时间、拓扑和语言证据。
3. 通过局部—全局双向匹配，将新轨迹接入已有身份或创建新身份。
4. 语言不仅筛选轨迹，还参与跨视角身份冲突消解。
5. 加入记忆可靠性与证据来源，避免错误跨视角匹配长期污染全局身份。

### 创新成立的边界

- 必须与 [CRMOT](https://ojs.aaai.org/index.php/AAAI/article/view/32219)的跨视角建模以及 [GMT](https://openaccess.thecvf.com/content/CVPR2026/html/Zhen_GMT_Effective_Global_Framework_for_Multi-Camera_Multi-Target_Tracking_CVPR_2026_paper.html)的全局轨迹关联明确区分。
- 只增加一个全局特征池不够；应定义身份状态、更新规则、跨视角冲突与语言作用点。
- 数据规模可能不足，需谨慎区分“方法问题”和“标注稀疏问题”。

### 最小验证与主要风险

- 最小验证：基于 CRTrack 统计跨视角切换、同类目标冲突和语言可消歧样本，建立 Global ID oracle 上限。
- 风险：数据和工程复杂度高；跨摄像头时间同步、拓扑和检测误差会同时影响结论。

## 11. D9：可靠、可校准与可自恢复 RMOT

### 研究问题

语言中一个属性、关系或顺序词的扰动可能改变目标；历史记忆又可能让错误持续传播。RMOT 能否检测视觉—语言冲突、量化自身不确定性，并在证据不足时拒绝或恢复？

### 可形成的方法

1. 建立语言扰动、视觉遮挡、模态冲突、描述歧义和历史污染测试协议。
2. 分别校准目标存在、目标选择、关联和指代匹配置信度。
3. 用一致性检查或反事实查询识别脆弱预测。
4. 当冲突出现时触发记忆回滚、候选保留、局部重关联或请求更明确描述。
5. 报告 Expected Calibration Error、风险—覆盖率、攻击成功率、恢复时间与 HOTA。

### 创新成立的边界

- [See No Evil](https://arxiv.org/abs/2509.02028)已经显示 RMOT 的语言—视觉关联存在对抗脆弱性和记忆持续效应；仅增加噪声测试已不够。
- 方法需要提供可校准、可拒答或可恢复机制，且不应通过普遍降低匹配置信度换取表面鲁棒性。
- 这是“可信 RMOT”问题，不应被包装成普通数据增强。

### 最小验证与主要风险

- 最小验证：对已复现 TempRMOT 建立词语替换、顺序交换、视觉遮挡与历史 Query 污染测试，绘制置信度—错误关系。
- 风险：鲁棒性指标多且容易分散主线；攻击设定必须符合真实失效而非只为击败模型。

## 12. D10：缺失模态鲁棒的多传感器 RMOT

### 研究问题

RGB-D 与 RGB-T 能在深度歧义、夜间和恶劣天气中提供额外证据，但传感器可能噪声严重、时间不同步或临时缺失。模型能否根据文本需求和传感器可靠性动态选择证据？

### 可形成的方法

1. 为 RGB、深度或热红外建立模态专属编码与轻量 Adapter。
2. 使用文本条件路由判断当前指代表达需要颜色、几何距离、热目标还是运动信息。
3. 显式估计每个模态的可靠性，在缺失或退化时调整融合权重。
4. 用模态丢弃、噪声和不同步训练提高降级能力。
5. 同时评测正常、单模态缺失、夜间/遮挡和跨域条件。

### 创新成立的边界

- [DRMOT](https://arxiv.org/abs/2602.04692)和 [RT-RMOT](https://arxiv.org/abs/2602.22033)已分别公开 RGB-D-language 与 RGB-thermal-language RMOT 预印本，因此“增加一种传感器”本身不再足够。
- 研究重点应是缺失模态、可靠性校准、按文本需求路由或跨传感器泛化。
- 新数据集若只是规模扩充而没有新的失效设定，创新性可能主要停留在资源贡献。

### 最小验证与主要风险

- 最小验证：先阅读并复核 DRMOT/RT-RMOT 的数据可得性，再决定能否进行受控模态丢失实验。
- 风险：当前 TempRMOT 基础为 RGB 单摄像头，迁移成本高；公开数据和代码成熟度尚需持续核验。

## 13. 方向之间的关系与重复风险

这些方向并非互斥，但一篇论文应只有一个首要研究问题。

| 主方向 | 可作为支撑技术 | 不宜同时作为并列主贡献 |
|---|---|---|
| D1 多阶段事件 | D5 轨迹 Token、D6 结构化记忆 | 再叠加完整 Agent 系统和跨视角任务 |
| D2 组合泛化 | D7 开放概念、反事实训练 | 把“新划分、数据增强、更强模型”都写成独立创新 |
| D3 关联纠错 | D4 条件验证、D6 可靠记忆 | 同时重做检测器、Tracker 和 MLLM 主干 |
| D4 条件 MLLM | D3 回写约束、蒸馏 | 仅用多个 Agent 增加流程长度 |
| D5 轨迹 Token | D1 事件对齐、D6 长期记忆 | 仅做特征换名或 Token 数对比 |
| D6 结构化记忆 | D1 阶段状态、D9 记忆恢复 | 仅扩大历史窗口 |
| D7 开放概念 | D2 未见组合、D9 校准 | 直接串联系统组件 |
| D8 Global Identity | D6 全局记忆、D3 冲突修复 | 同时承担新数据集和通用 Video-LLM |
| D9 可信 RMOT | D3 自恢复、D4 Verifier | 只做攻击、不提供机制 |
| D10 多传感器 | D4 动态路由、D9 可靠性 | 只增加传感器或更大 Backbone |

## 14. 后续调研应产出的共同证据

在最终选题前，不论选择哪个方向，都应先得到以下证据：

1. **失败证据**：至少两个公开基线在该问题上出现稳定、可复现的失败，而非单个案例。
2. **非 Tracker 混淆证据**：语言理解增益不能只来自更强检测器或更高基础 MOT 性能。
3. **独立任务指标**：除 HOTA 外，必须有与研究问题直接对应的指标。
4. **公平性记录**：预训练数据、额外标注、模型参数、推理成本和闭源 API 调用需单列。
5. **泄漏审计**：涉及组合、MLLM 生成或新划分时，应审查视频、目标、模板和语义组合泄漏。
6. **最小机制消融**：证明核心能力来自所提出机制，而非模块数量、更多数据或更强 Backbone。

## 15. 当前阶段不宜直接写成方向的内容

- “在 TempRMOT 上增加一个注意力模块”。
- “把 CLIP 换成更强文本编码器”。
- “把 Grounding DINO、SAM2 和 Tracker 串联起来”。
- “将历史长度从 4 增大到 8 或压缩到 2”。
- “使用 MLLM/多个 Agent，但不定义它们不可替代的决策”。
- “只提升 HOTA，不评测事件顺序、组合泛化、关联修复或可靠性”。

这些内容可以成为实现手段或消融项，但不足以单独构成清晰的研究问题。
