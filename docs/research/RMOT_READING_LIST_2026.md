# RMOT 方向调研论文阅读清单（2026）

> 更新日期：2026-07-23  
> 目标：用有限阅读量建立 RMOT 主线、掌握 2025—2026 年直接竞争工作，并连接轨迹 Token、长视频时序推理、基础模型、Agentic Verification 和跨摄像头跟踪。  
> 状态说明：主会、Findings、Workshop 和 arXiv 预印本分开标注；“代码未确认”表示截至本文更新时未核实到完整官方训练/评测代码。

## 1. 建议阅读方式

清单共 24 篇，不要求一次读完：

- **第一层：主线复核，5 篇。** 你已经接触过，用半天到一天建立统一比较表即可。
- **第二层：新增必读，9 篇。** 直接决定当前 RMOT 选题是否与最新工作重叠。
- **第三层：按方向选读，10 篇。** 确定感兴趣的问题后再精读，不需要全部同时展开。

精读每篇论文时建议固定记录六项：任务输入输出、信息流、语言作用点、时间建模方式、额外数据/算力、作者没有验证的失败模式。

## 2. 第一层：RMOT 主线复核（5 篇）

### 1. Referring Multi-Object Tracking — TransRMOT

- **年份与发表位置：** CVPR 2023
- **论文：** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_Referring_Multi-Object_Tracking_CVPR_2023_paper.html)
- **代码：** [wudongming97/RMOT](https://github.com/wudongming97/RMOT)
- **核心创新：** 正式提出 RMOT 任务与 Refer-KITTI 数据集；用语言条件化的端到端 Transformer 同时完成多目标检测、关联和指代筛选。
- **阅读原因：** 它定义了后续工作的基本任务和指标。重读时重点画清 Detect Query、Track Query、语言特征和 Refer Head 的信息流，后续所有新方向都要说明改变了哪一层。

### 2. Bootstrapping Referring Multi-Object Tracking — TempRMOT

- **年份与发表位置：** arXiv 2024；正式发表位置未确认
- **论文：** [arXiv](https://arxiv.org/abs/2406.05039)
- **代码：** [zyn213/TempRMOT](https://github.com/zyn213/TempRMOT)
- **核心创新：** 用半自动标注扩展 Refer-KITTI-V2，并通过历史 Query 增强当前目标表示，强化长期时间信息。
- **阅读原因：** 这是当前项目代码基础。精读重点不是再次理解整体结构，而是确认历史 Query 的写入、读取、训练监督和推理状态；已复现官方检查点的实际 `hist_len` 为 4，这会影响“长时记忆”选题边界。

### 3. iKUN: Speak to Trackers without Retraining

- **年份与发表位置：** CVPR 2024
- **论文：** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024/html/Du_iKUN_Speak_to_Trackers_without_Retraining_CVPR_2024_paper.html)
- **代码：** [dyhBUPT/iKUN](https://github.com/dyhBUPT/iKUN)
- **核心创新：** 将现成 MOT 与可插拔语言模块连接，不必重新训练 Tracker；结合知识统一模块、神经 Kalman Filter 和置信度校准，并提出 Refer-Dance。
- **阅读原因：** 它是低成本 RMOT 的基本参照。阅读时重点判断语言模块是否只筛选已有轨迹，以及 Tracker 出错后语言是否能够回写关联。

### 4. HFF-Tracker: Hierarchical Feature Fusion Tracker for RMOT

- **年份与发表位置：** AAAI 2025
- **论文：** [AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/33143)
- **代码：** 未确认
- **核心创新：** 在视觉编码、Transformer Decoder 和预测头三个层级融合语言，并使用自适应回看加强历史信息。
- **阅读原因：** 它是“多层语言融合”方向的直接上限检查。任何新的融合模块都必须证明不是 HFF-Tracker 的局部替代或更多层堆叠。

### 5. Cross-View Referring Multi-Object Tracking — CRMOT

- **年份与发表位置：** AAAI 2025
- **论文：** [AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/32219)
- **代码与数据：** [chen-si-jia/CRMOT](https://github.com/chen-si-jia/CRMOT)
- **核心创新：** 将 RMOT 扩展到跨摄像头场景，提出 CRTrack 数据集与跨视角关联框架。
- **阅读原因：** 它定义了 Global Identity Query 方向必须超越的起点。精读局部身份、跨视角特征、语言表达和关联监督之间的关系。

## 3. 第二层：新增必读（9 篇）

### 6. Language Decoupling with Fine-grained Knowledge Guidance for Referring Multi-object Tracking — DKGTrack

- **年份与发表位置：** ICCV 2025
- **论文：** [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Language_Decoupling_with_Fine-grained_Knowledge_Guidance_for_Referring_Multi-object_Tracking_ICCV_2025_paper.html)
- **代码：** [acyddl/DKGTrack](https://github.com/acyddl/DKGTrack)
- **核心创新：** 将表达解耦为局部静态描述与运动状态；Static Semantic Enhancement 对齐局部外观，Motion Perception Alignment 对齐动作与目标 Query。
- **阅读原因：** 它直接占据“细粒度语言拆解 + 运动语义”位置。研究多阶段事件时，需要明确阶段顺序推理与普通静态/运动解耦的差异。

### 7. Rethinking Two-Stage Referring-by-Tracking in RMOT: Make it Strong and Flexible — FlexHook

- **年份与发表位置：** CVPR 2026
- **论文：** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Rethinking_Two-Stage_Referring-by-Tracking_in_Referring_Multi-Object_Tracking_Make_it_Strong_CVPR_2026_paper.html)
- **代码：** [buptLwz/FlexHook](https://github.com/buptLwz/FlexHook)
- **核心创新：** 用 Conditioning Hook 将语言条件注入已有 Tracker 表示，并以 Pairwise Correspondence Decoder 对轨迹与文本进行细粒度匹配，在较低训练成本下强化双阶段路线。
- **阅读原因：** 它显著抬高了“可插拔 RMOT”的新颖性门槛。若考虑低成本方案，必须从关联反馈、动态调用、可靠性或新评测问题切入，而非再挂一个普通 Adapter。

### 8. ReferGPT: Towards Zero-Shot Referring Multi-Object Tracking

- **年份与发表位置：** CVPR 2025 Workshop（WDFM-AD），不是 CVPR 主会
- **论文：** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025W/WDFM-AD/html/Chamiti_ReferGPT_Towards_Zero-Shot_Referring_Multi-Object_Tracking_CVPRW_2025_paper.html)
- **代码：** [Tzoulio/ReferGPT](https://github.com/Tzoulio/ReferGPT)
- **核心创新：** 采用零样本双阶段流程：先跟踪，再由 MLLM 生成带三维线索的轨迹描述，使用 CLIP 与模糊匹配完成文本—轨迹对应。
- **阅读原因：** 它是“MLLM 不训练直接接 RMOT”的典型方案，也暴露离线成本高、描述可能幻觉、Tracker 错误难以修复的问题。

### 9. STORM: End-to-End Referring Multi-Object Tracking in Videos

- **年份与发表位置：** CVPR 2026 Findings，不是 CVPR 主会
- **论文：** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026F/html/Lu_STORM_End-to-End_Referring_Multi-Object_Tracking_in_Videos_CVPRF_2026_paper.html)
- **数据仓库：** [amazon-science/storm-referring-multi-object-grounding](https://github.com/amazon-science/storm-referring-multi-object-grounding)；完整训练代码未确认
- **核心创新：** 以 8B 多模态大模型直接进行端到端 RMOT，使用任务组合学习，并提出更具挑战的 STORM-Bench。
- **阅读原因：** 它证明“MLLM-based RMOT”已经成为直接竞争路线。阅读时重点审查其坐标/身份输出机制、训练数据、速度和对复杂事件的真实评测，而不是只看参数规模。

### 10. Counterfactual and Observation-enhanced Alignment for RMOT — COAL

- **年份与发表位置：** arXiv 2026；正式发表位置未确认
- **论文：** [arXiv](https://arxiv.org/abs/2605.14795)
- **代码：** 未确认
- **核心创新：** 使用视觉语言模型注入显式观察语义，并通过大语言模型构造属性反事实，再进行层次化跨模态对齐。
- **阅读原因：** 它直接影响组合泛化和反事实训练方向。需要辨别其反事实是否只覆盖属性替换，以及是否真正建立未见组合的隔离评测。

### 11. Reasoning-based Multi-Object Tracking — ReaMOT

- **年份与发表位置：** arXiv 2025；正式发表位置未确认
- **论文：** [arXiv](https://arxiv.org/abs/2505.20381)
- **项目仓库：** [chen-si-jia/ReaMOT](https://github.com/chen-si-jia/reamot)；完整数据/代码开放状态需按仓库最新说明复核
- **核心创新：** 提出需要推理的多目标跟踪基准，并给出 LVLM 与 SAM2 结合的训练免费方案 ReaTrack。
- **阅读原因：** 它把“复杂指令 + 推理 + 跟踪”具体化。多阶段事件方向需要借鉴其表达设计，同时检查它是否评测严格的事件顺序和轨迹片段对齐。

### 12. OmniPT: Unifying Multi-Task Visual Tracking through a Large Vision-Language Model

- **年份与发表位置：** AAAI 2026
- **论文：** [AAAI Proceedings](https://ojs.aaai.org/index.php/AAAI/article/view/37406)
- **代码：** 未确认
- **核心创新：** 用统一视觉语言模型覆盖 MOT、RMOT、CRMOT、语义跟踪与描述等任务，并使用包含强化学习的分阶段训练流程。
- **阅读原因：** 它代表“统一跟踪基础模型”路线。阅读时判断统一训练带来的能力来自共享表示、额外数据还是模型规模，并明确单任务研究还能解决哪些结构性问题。

### 13. ViewSAM: Weakly-Supervised Cross-View Referring Multi-Object Tracking

- **年份与发表位置：** arXiv 2026；正式发表位置未确认
- **论文：** [arXiv](https://arxiv.org/abs/2605.02638)
- **代码：** 未确认
- **核心创新：** 使用 SAM3 生成伪轨迹，通过跨视角重新提示和 SAM2 的视角感知条件进行弱监督跨视角 RMOT。
- **阅读原因：** 它同时覆盖跨视角、弱监督和基础模型。Global Identity Query 方向必须确认它是否已经维护显式全局身份，以及弱监督伪标签的错误如何跨视角传播。

### 14. See No Evil: Adversarial Attacks Against Linguistic-Visual Association in RMOT Systems

- **年份与发表位置：** 2025，WiML @ NeurIPS 2025 / arXiv；不是 NeurIPS 主会
- **论文：** [arXiv](https://arxiv.org/abs/2509.02028) · [OpenReview](https://openreview.net/forum?id=76vQobPpTN)
- **代码：** 未确认
- **核心创新：** 系统研究 RMOT 的语言—视觉关联攻击，并观察到 FIFO 历史记忆可能让攻击影响持续传播。
- **阅读原因：** 它为“可信 RMOT、记忆污染和自恢复”提供直接问题证据，也提醒研究者不能把更长历史默认视为更鲁棒。

## 4. 第三层：按候选方向选读（10 篇）

### 15. Multiple Object Tracking as ID Prediction — MOTIP

- **年份与发表位置：** CVPR 2025
- **论文：** [CVF 论文 PDF](https://openaccess.thecvf.com/content/CVPR2025/papers/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.pdf)
- **核心创新：** 将多目标跟踪重新表述为基于历史轨迹上下文的身份预测，直接学习关联身份，而不只依赖手工匹配代价。
- **阅读原因：** 对 D3 语言反馈式关联纠错最重要。它提供“语言约束应写回哪里”的现代关联建模接口。

### 16. MeMOTR: Long-Term Memory-Augmented Transformer for Multi-Object Tracking

- **年份与发表位置：** ICCV 2023
- **论文：** [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_MeMOTR_Long-Term_Memory-Augmented_Transformer_for_Multi-Object_Tracking_ICCV_2023_paper.html)
- **核心创新：** 为 Track Query 建立长期记忆与记忆读写机制，提高长时身份保持。
- **阅读原因：** 它是 D6 结构化轨迹记忆的基础对照。需要理解普通长时记忆已经解决什么，避免把 FIFO 扩容当成新贡献。

### 17. TrajTok: Learning Trajectory Tokens Enhances Video Understanding

- **年份与发表位置：** CVPR 2026
- **论文：** [CVF 论文 PDF](https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_TrajTok_Learning_Trajectory_Tokens_Enhances_Video_Understanding_CVPR_2026_paper.pdf)
- **核心创新：** 提出端到端可微的轨迹 Tokenizer，以较少 Token 保留对象运动和时间结构，并将其接入 Video-LLM 进行长视频理解。
- **阅读原因：** 它是 D5 的最直接相邻工作。应重点分析轨迹从何而来、Token 是否保留身份、训练监督是什么，以及如何适配在线框级 RMOT。

### 18. VideoITG: Multimodal Video Understanding with Instructed Temporal Grounding

- **年份与发表位置：** CVPR 2026
- **论文：** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_VideoITG_Multimodal_Video_Understanding_with_Instructed_Temporal_Grounding_CVPR_2026_paper.html)
- **核心创新：** 用指令条件的时间定位选择与问题相关的帧段，并构建大规模时序定位标注，作为可插拔模块服务视频理解。
- **阅读原因：** D1 可借鉴“先定位事件时间，再做对象级推理”的分解方式；同时要判断帧级时序定位如何下降到具体轨迹身份。

### 19. TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs

- **年份与发表位置：** CVPR 2026
- **论文：** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_TimeLens_Rethinking_Video_Temporal_Grounding_with_Multimodal_LLMs_CVPR_2026_paper.html)
- **核心创新：** 诊断现有时序定位数据质量，重新标注 TimeLens-Bench，并研究带可验证奖励的训练方法。
- **阅读原因：** 它提醒多阶段事件论文的数据与评测本身可能比模块更关键。D1 应借鉴其标注审计、边界质量与可验证奖励设计。

### 20. SEAL: Semantic Attention Learning for Long Video Representation

- **年份与发表位置：** CVPR 2025
- **论文：** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_SEAL_Semantic_Attention_Learning_for_Long_Video_Representation_CVPR_2025_paper.html)
- **核心创新：** 围绕场景、对象和动作实体学习语义注意力，并兼顾相关性与多样性选择长视频信息。
- **阅读原因：** 对 D6 有启发：长期记忆不必保存每一帧，可围绕对象、动作和状态转折稀疏写入。

### 21. Think Then Verify: A Hypothesis-Verification Multi-Agent Framework for Long Video Understanding — VideoHV-Agent

- **年份与发表位置：** CVPR 2026
- **论文：** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Think_Then_Verify_A_Hypothesis-Verification_Multi-Agent_Framework_for_Long_Video_CVPR_2026_paper.html)
- **核心创新：** 将长视频推理拆分为提出假设、判断、检索验证和回答等角色，通过证据验证降低直接作答错误。
- **阅读原因：** 它为 D4 的 Verifier 提供流程参考。阅读重点是验证是否带来不可替代增益、调用成本多大，以及哪些部分可以压缩成单模型而不是盲目多 Agent 化。

### 22. GMT: Effective Global Framework for Multi-Camera Multi-Target Tracking

- **年份与发表位置：** CVPR 2026
- **论文：** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhen_GMT_Effective_Global_Framework_for_Multi-Camera_Multi-Target_Tracking_CVPR_2026_paper.html)
- **代码与数据：** [FoxCanned/GMT](https://github.com/FoxCanned/GMT)
- **核心创新：** 在全局层面进行轨迹—目标关联，并使用跨摄像头全局轨迹与记忆，而非只做局部轨迹后处理。
- **阅读原因：** 它是 D8 Global Identity Query 的强相邻工作。需要明确语言条件化的全局身份状态相比普通 MTMCT 全局轨迹库增加了什么能力。

### 23. SAM 3: Segment Anything with Concepts

- **年份与发表位置：** arXiv 2025；正式发表位置未确认
- **论文：** [arXiv](https://arxiv.org/abs/2511.16719) · [Meta Research](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/)
- **代码：** [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- **核心创新：** 将开放文本概念提示扩展到实例检测、分割和视频跟踪，使模型能定位并跟踪符合概念的多个实例。
- **阅读原因：** 它直接改变 D7 的问题边界。开放概念定位本身已不够，RMOT 研究应转向复杂指代、身份持续、关系/事件和校准。

### 24. WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning

- **年份与发表位置：** CVPR 2026
- **论文：** [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Yeo_WorldMM_Dynamic_Multimodal_Memory_Agent_for_Long_Video_Reasoning_CVPR_2026_paper.html)
- **核心创新：** 构建情景、语义和视觉等多种动态记忆，并按问题进行多尺度检索与推理。
- **阅读原因：** 它为 D4/D6 提供多模态记忆组织方式，但 RMOT 必须将抽象记忆落到在线身份、轨迹证据和可测的关联恢复上。

## 5. 只需快速了解的基础与前沿补充

以下工作暂不计入 24 篇精读量，遇到相应方向再查看：

- [Grounding DINO（ECCV 2024）](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06319.pdf)及[代码](https://github.com/IDEA-Research/GroundingDINO)：开放词汇检测基础，适合 D7 的候选生成。
- [SAM 2（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/file/45c1f6a8cbf2da59ebf2c802b4f742cd-Paper-Conference.pdf)及[代码](https://github.com/facebookresearch/sam2)：带流式记忆的可提示视频分割基础。
- [V-STaR（CVPR 2026 Findings）](https://openaccess.thecvf.com/content/CVPR2026F/html/Cheng_V-STaR_Benchmarking_Video-LLMs_on_Video_Spatio-Temporal_Reasoning_CVPRF_2026_paper.html)：评测 Video-LLM 的 when/where/before/what 时空推理，适合 D1 的指标设计参考。
- [TimeLens2（arXiv 2026-07）](https://arxiv.org/abs/2607.17423)：很新的多区间时序定位预印本，支持可变数量的时间区间输出；前沿性强，但成熟度需继续观察。
- [DRMOT（arXiv 2026）](https://arxiv.org/abs/2602.04692)与 [RT-RMOT（arXiv 2026）](https://arxiv.org/abs/2602.22033)：分别探索 RGB-D-language 与 RGB-thermal-language RMOT，是 D10 多传感器方向的直接重叠检查。
- [Enhancing Referring Multi-Object Tracking via Language Guidance and Temporal Representation（Pattern Recognition 2026）](https://www.sciencedirect.com/science/article/pii/S0031320326007880)：同时建模语言引导与长短期时间表示；选择 D6 前需核对其记忆设计。
- [STCTracker（Image and Vision Computing 2026）](https://www.sciencedirect.com/science/article/pii/S0262885626001800)：DynaSORT 加双指代模块的两阶段方案；选择 D3 前需核对其时间一致性与关联接口。
- [Vision-Motion-Reference Multi-Object Tracking（arXiv 2025）](https://arxiv.org/abs/2511.17681)：显式对齐视觉、运动和指代信息；选择 D1 或 D5 前应检查其运动描述覆盖范围。
- [AerialMind（AAAI 2026）](https://ojs.aaai.org/index.php/AAAI/article/view/37270)：面向无人机 RMOT 的数据集与方法，适合了解场景扩展和 MLLM 辅助标注，但不是当前单摄像头道路场景的首轮精读重点。

## 6. 方向—论文映射

| 候选方向 | 首先精读 | 随后补充 |
|---|---|---|
| D1 多阶段事件 | DKGTrack、ReaMOT、TimeLens、VideoITG | V-STaR、TrajTok、STORM |
| D2 组合泛化 | COAL、DKGTrack、ReferGPT | SAM3、STORM、See No Evil |
| D3 关联纠错 | iKUN、FlexHook、MOTIP | TempRMOT、See No Evil |
| D4 条件 MLLM | STORM、ReferGPT、VideoHV-Agent | OmniPT、WorldMM |
| D5 轨迹 Token | TrajTok、MOTIP | TempRMOT、VideoITG |
| D6 结构化记忆 | TempRMOT、MeMOTR、SEAL | WorldMM、See No Evil |
| D7 开放概念 | SAM3、ReferGPT、ViewSAM | Grounding DINO、SAM2 |
| D8 Global Identity | CRMOT、GMT、ViewSAM | OmniPT、MeMOTR |
| D9 可信 RMOT | See No Evil、COAL | VideoHV-Agent、TempRMOT |
| D10 多传感器 | DRMOT、RT-RMOT | STORM、OmniPT |

## 7. 两周阅读节奏示例

这只是阅读安排，不代表选题顺序。

### 第 1—2 天：统一基线

快速复核 TransRMOT、TempRMOT、iKUN、HFF-Tracker、CRMOT；输出一张统一的信息流与监督表。

### 第 3—6 天：排除直接重叠

精读 DKGTrack、FlexHook、ReferGPT、STORM、COAL、ReaMOT、OmniPT、ViewSAM、See No Evil；每篇只写一页笔记，重点记录“已经覆盖什么”和“没有测什么”。

### 第 7 天：形成问题清单

不选方法，先写 3—5 个可复现实验问题，例如“交换事件顺序后模型分数是否变化”“语言是否能修复固定检测结果上的 ID Switch”。

### 第 8—12 天：按兴趣分支阅读

从第三层选 4—6 篇：事件方向读 TimeLens/VideoITG/TrajTok；纠错方向读 MOTIP；记忆方向读 MeMOTR/SEAL/WorldMM；跨视角读 GMT；开放概念读 SAM3。

### 第 13—14 天：只做纸面研究设计

为每个保留问题写：失败原因、不可替代能力、最小机制、独立指标、数据需求、最小实验和终止条件。完成后再进入最终选题或代码实验。

## 8. 阅读时的审稿人检查项

1. 性能提升是否只是来自更强的 MOT、Backbone、额外预训练数据或更大模型？
2. “长时序”是否只表示更长历史窗口，而没有阶段、顺序或状态转折监督？
3. “组合泛化”是否存在同一视频、同一目标、近义模板或属性组合泄漏？
4. MLLM 是否提供传统模型无法提供的验证能力，还是只生成更长文本？
5. 双阶段语言模块能否修改 Tracker 的错误关联，还是只能在错误轨迹中做选择？
6. 除 HOTA 外，论文是否有与其核心主张直接对应的指标？
7. 代码、数据和检查点是否真正开放；若没有，结果是否能被独立验证？

只要持续用这七项做笔记，阅读清单就会自然收敛为研究问题，而不是模型名称的堆积。
