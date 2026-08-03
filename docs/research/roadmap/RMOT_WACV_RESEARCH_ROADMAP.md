# RMOT 最新进展与 WACV 研究方向决策文档

> 调研截止日期：2026-07-19  
> 研究范围：Referring Multi-Object Tracking（RMOT）、多目标跟踪、视觉语言理解、长视频理解、MLLM、跨视角身份建模与 Agentic Vision  
> 目标：在三个月研究周期内选择一个具有明确问题、可验证假设和止损标准的 WACV 研究方向

---

## 0. 执行摘要

截至 2026 年 7 月 19 日，本调研的核心结论如下。

1. 如果按完整三个月周期推进，首选方向是 **多阶段有序事件 RMOT（EventRMOT）**。它最符合现有 TempRMOT 基础，也能避开 FlexHook、COAL、ReaMOT 和 STORM 已覆盖的主要区域。
2. 如果追求最低风险和 WACV 适配度，首选 **组合泛化审计基准 + 小型反事实基线（CompRMOT）**。
3. 跨视角 Global Identity Query 的创新上限很高，但 OmniPT 和 ViewSAM 已显著抬高比较门槛，且 CRTrack 数据规模较小，属于高风险方向。
4. “给 TempRMOT 接入一个 MLLM”“把历史 Query 改名为 Track Token”“增加多个 Agent”都不足以单独构成方法创新。
5. FlexHook 已显著压缩普通可插拔语言匹配模块的空间；插件路线剩余的主要机会是语言反馈纠错、不确定性触发、组合校准和流式事件记忆。

### 投稿时间约束

WACV 2027 第二轮注册截止为 2026 年 8 月 21 日，投稿截止为 2026 年 8 月 28 日，距调研日约 40 天。完整三个月方案无法赶上这一轮。

- 40 天应急投稿：只建议考虑 CompRMOT，并尽快验证是否存在稳定的组合泛化差距。
- 正常三个月研究：建议选择 EventRMOT，目标调整为 WACV 2028 或其他时间合适的会议。
- Global Identity Query：只有在已有跨视角代码、数据和较充足算力时才建议启动。

WACV 2027 的 Evaluations & Datasets Track 明确接受基准审计、失败模式分析和新评测协议，因此 CompRMOT 与该 track 高度匹配：[WACV 2027 Call for Papers](https://wacv.thecvf.com/Conferences/2027/CallForPapers)。

---

## 1. 基础概念

### 1.1 RMOT

Referring Multi-Object Tracking 的输入是视频和一条自然语言表达，输出是视频中所有符合表达的目标轨迹，并保持目标身份在时间上的一致性。

### 1.2 Object Query 与 Track Query

- **Object Query**：Transformer 中用于寻找潜在目标的向量。
- **Track Query**：携带上一帧目标身份和状态、继续在下一帧寻找同一目标的向量。

### 1.3 三类主要方法

- **端到端 RMOT**：检测、身份关联和语言匹配在同一网络中联合训练。
- **双阶段 Referring-by-Tracking（RBT）**：先用普通 MOT 跟踪所有物体，再根据语言筛选轨迹。
- **MLLM/LVLM RMOT**：利用多模态大模型理解图像、视频和自然语言，并生成目标描述、定位结果或轨迹。

---

## 2. 2025—2026 年关键新方法

## 2.1 FlexHook

论文：[Rethinking Two-Stage Referring-by-Tracking in Referring Multi-Object Tracking: Make it Strong Again](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Rethinking_Two-Stage_Referring-by-Tracking_in_Referring_Multi-Object_Tracking_Make_it_Strong_CVPR_2026_paper.html)，CVPR 2026。  
代码：[buptLwz/FlexHook](https://github.com/buptLwz/FlexHook)

### 任务和路线

- 输入：视频、自然语言描述、现成 MOT 产生的全部轨迹。
- 输出：符合语言描述的轨迹集合。
- 类型：双阶段 RBT。

### 架构和流程

1. 普通 MOT 产生所有轨迹。
2. Conditioning Hook（C-Hook）直接从 tracker Backbone 的多尺度特征图中采样轨迹区域特征，避免 iKUN 式裁图再编码。
3. 通过语言条件参考点采样与表达相关的视觉线索。
4. 对短轨迹片段执行 Temporal Integration。
5. Pairwise Correspondence Decoder（PCD）学习轨迹和文本是否匹配，替代静态 CLIP 相似度。
6. 训练时模拟轨迹碎片、定位噪声和 ID Switch，提高匹配器对 tracker 错误的鲁棒性。

### 与已有工作的关系

- 相比 iKUN：从“裁图特征 + CLIP 相似度”升级为可学习的轨迹—文本对应关系。
- 相比 TransRMOT、TempRMOT、HFF-Tracker：不需要重训整个 tracker，训练和部署成本低。
- 相比 TempRMOT：没有更强的长期历史机制，但语言匹配模块更灵活。
- 与 CRMOT 基本正交，不解决跨摄像头身份。

### 关键实验

- Refer-KITTI：最高 53.83 HOTA。
- Refer-KITTI-V2：最高 42.53 HOTA。
- Refer-Dance：32.17 HOTA。
- LaMOT：56.77 HOTA，高于 LaMOTer 的 48.45。
- Refer-KITTI-V2 上冻结编码器训练约 0.77 小时；论文报告 TempRMOT 对应训练约 51.68 小时，但两者还涉及不同的 detector/tracker 训练成本，不能把这一数字直接理解为完整系统总成本。

### 局限

- 语言模块位于 tracker 之后，无法恢复 tracker 完全漏检的目标。
- 虽然训练中模拟 ID Switch，但没有把语言判断反馈给 tracker 的关联矩阵或状态。
- Temporal Integration 是短片段建模，不是事件级长期推理。
- 没有未见属性、关系、动作和顺序组合的系统切分。

### 对研究方向的影响

- 普通“设计一个更强轨迹—文本匹配器”的创新空间已经明显缩小。
- EventRMOT 可以使用 FlexHook 作为强双阶段基线。
- 组合泛化方向必须超越标准 PCD，并加入严格 held-out split。
- 语言反馈纠错仍然是明确缺口。

## 2.2 ReferGPT

论文：[ReferGPT: Towards Zero-Shot Referring Multi-Object Tracking](https://openaccess.thecvf.com/content/CVPR2025W/WDFM-AD/html/Chamiti_ReferGPT_Towards_Zero-Shot_Referring_Multi-Object_Tracking_CVPRW_2025_paper.html)，CVPR Workshops 2025。  
代码：[Tzoulio/ReferGPT](https://github.com/Tzoulio/ReferGPT)

### 任务和路线

- 输入：视频和自然语言查询。
- 输出：符合描述的轨迹。
- 类型：双阶段、MLLM、zero-shot。

这里的 zero-shot 表示没有使用 RMOT 标注训练，并不等于已经通过未见语言组合测试。

### 架构和流程

1. 使用 3D CasA 或 2D QT-3DT 获得检测。
2. PC3T 完成普通多目标跟踪。
3. GPT-4o-mini 查看目标裁图及位置、运动等 3D 信息，为每条轨迹生成文字描述。
4. 用 CLIP 文本相似度和模糊字符串匹配比较生成描述与用户查询。
5. 通过轨迹多数投票和聚类选择最终目标。

其核心可以概括为：

> 先跟踪 → 为每条轨迹生成说明书 → 比较说明书和查询。

### 关键实验

- Refer-KITTI：3D 版本 49.46 HOTA。
- Refer-KITTI-V2：30.12 HOTA。
- Refer-KITTI+：43.44 HOTA。
- 固定阈值在线版本约 36.61 HOTA，多数投票加聚类达到 49.46，说明最佳配置明显依赖离线轨迹级处理。
- Qwen2-VL-2B、Phi-4-MM-5B、GPT-4o-mini 的 HOTA 分别约为 19.97、36.56、49.46，对 MLLM 能力高度敏感。

### 局限

- 每个目标反复调用 MLLM，速度和内存成本高。
- 如果 MLLM 生成的描述没有涉及查询所需属性，后续匹配仍然失败。
- 依赖外部 detector/tracker，语言不能修正已经发生的错误关联。
- 最佳版本包含离线聚类。
- GPT-4o-mini 是闭源 API，版本稳定性和可复现性有限。
- 没有系统评估组合泛化和事件顺序。

### 对研究方向的影响

- “MLLM 为轨迹生成 caption，再做文本匹配”已经被覆盖。
- 有价值的空间是结构化事件输出、条件调用、置信度校准和反馈纠错。
- 研究中必须严格区分 open vocabulary、zero-shot 和 compositional generalization。

## 2.3 STORM

论文：[STORM: End-to-End Referring Multi-Object Tracking in Videos](https://openaccess.thecvf.com/content/CVPR2026F/html/Lu_STORM_End-to-End_Referring_Multi-Object_Tracking_in_Videos_CVPRF_2026_paper.html)，CVPR Findings 2026。  
数据：[amazon-science/storm-referring-multi-object-grounding](https://github.com/amazon-science/storm-referring-multi-object-grounding)

### 任务和路线

- 输入：视频和自然语言描述。
- 输出：直接用文本序列生成目标编号、帧编号、坐标和身份。
- 类型：8B 级端到端 MLLM，不依赖外部 detector/tracker。

### 架构

- NaViT 类视觉编码器提取视频特征。
- 两层 MLP 将视觉特征投影到 LLaMA 语言空间。
- LLM 自回归生成目标和 bounding box。
- 长视频切分为多个 clip，前一个 clip 的 box 作为下一个 clip 的提示。

### Task-Composition Learning

STORM 将 RMOT 能力分解为图像目标定位、单目标/语言单目标跟踪和少量 RMOT 微调。这里的 composition 是“任务能力组合”，不是“未见语言概念组合”。

### 数据和结果

公开 STORM-Bench 仓库包含：

- 6,723 个有 referring annotation 的 clip；
- 90,617 帧；
- 29,933 条轨迹；
- 30,700 条 referring expression；
- 80 类日常物体，并包含属性、空间关系和物体交互。

STORM 在 STORM-Bench 上报告 66.7 HOTA、78.3 IDF1；适配的 VisionLLMv2 和微调 LaMOT 约为 45.3、46.7 HOTA。

该结果不能与 FlexHook 的 Refer-KITTI 结果直接比较，因为两者基准不同。

### 局限

- 8B 模型和大规模预训练数据，不是低成本路线。
- 自回归坐标生成存在速度、格式稳定性和精确定位风险。
- 长视频依赖 clip 切分和提示拼接，不是持续可检索的轨迹记忆。
- 没有专门的事件阶段顺序交换测试。
- 公开仓库目前主要包含 STORM-Bench 数据，完整模型训练代码未确认。

### 对研究方向的影响

仅仅“将 TempRMOT 接入 Video-LLM”已经难以形成充分创新。新方法必须明确解决 STORM 没有解决的有序事件、小模型推理、语言反馈纠错、严格组合泛化或不确定性触发问题。

## 2.4 其他直接影响选题的新工作

| 方法 | 核心贡献 | 对选题边界的影响 |
|---|---|---|
| [DKGTrack](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Language_Decoupling_with_Fine-grained_Knowledge_Guidance_for_Referring_Multi-object_Tracking_ICCV_2025_paper.html)，ICCV 2025 | 将语言分成静态局部描述和运动状态，分别用于定位和运动对齐；HOTA 约 52.08/35.26。[代码](https://github.com/acyddl/DKGTrack) | “把语言拆成属性和运动”已经做过；事件方向必须显式建模多个阶段及顺序。 |
| [VMRMOT](https://arxiv.org/abs/2511.17681)，arXiv 2025 | 将轨迹转成位置、方向、速度趋势等运动文本，再由 Qwen2.5-Omni 提取运动语义；HOTA 约 53.00/35.21。 | “MLLM 理解运动历史”已覆盖，但仍没有事件阶段状态机。 |
| [COAL](https://arxiv.org/abs/2605.14795)，arXiv 2026 | VLM 注入目标描述，LLM 生成颜色等反事实负样本，约 53.38/43.46 HOTA。 | 已覆盖属性反事实增强；新组合泛化工作必须加入关系、动作、顺序和严格 held-out split。 |
| [ReaMOT/ReaTrack](https://arxiv.org/abs/2505.20381)，arXiv 2025/2026 更新 | 1,156 条推理指令、869 个场景，研究隐式语言条件和常识推理。 | “复杂推理 RMOT 完全无人研究”已不成立；但隐式推理不等于有序多阶段事件。当前[仓库](https://github.com/chen-si-jia/ReaMOT)仍未释放完整代码和数据。 |
| [AerialMind](https://ojs.aaai.org/index.php/AAAI/article/view/37270)，AAAI 2026 | UAV RMOT，93 个视频、约 24.6K 表达；HETrack 强化跨模态表示和小目标感知。[数据](https://github.com/shawnliang420/AerialMind) | 单纯构造更复杂场景或更多表达不再新，可作为跨域/事件数据来源。 |
| [OmniPT](https://ojs.aaai.org/index.php/AAAI/article/view/37406)，AAAI 2026 | LVLM 统一行人 MOT、RMOT、CRMOT 和语义问答；CRTrack 上 CVRIDF1/CVRMA 约 62.13/42.39。 | 跨视角大模型路线已有强结果，Global Query 必须证明稳定性、效率或记忆机制上的独特价值。 |
| [ViewSAM](https://arxiv.org/abs/2605.02638)，arXiv 2026 | SAM3 产生跨视角伪轨迹，SAM2 加入 view-aware 条件，只用类别级弱监督。 | “SAM2 + 跨视角”已经出现，Global Query 不能只是 SAM/ReID 特征融合。 |
| [DRMOT](https://arxiv.org/abs/2602.04692)、[RT-RMOT](https://arxiv.org/abs/2602.22033)、[ORMOT](https://arxiv.org/abs/2603.05384)，arXiv 2026 | 扩展到 RGB-D、RGB-Thermal 和全景 RMOT。 | 说明新传感器/新场景数据集已成为拥挤方向；仅换模态不足以构成强创新。 |

---

## 3. RMOT 技术路线

| 技术范式 | 代表方法 | 已解决问题 | 主要重复和局限 |
|---|---|---|---|
| 端到端 Query-based | TransRMOT、TempRMOT、HFF、DKGTrack、SKTrack、CDRMT、COAL | 检测、身份和语言联合优化 | 大量工作都在增加融合模块，容易把更强检测误认为语言理解。 |
| 双阶段 RBT | iKUN、MEX、ReferGPT、FlexHook | 低成本接入现成 MOT，可灵活替换 tracker | tracker 错误先发生，语言通常只有筛选权。 |
| 长期时序建模 | TempRMOT、DKGTrack、VMRMOT、STORM | 遮挡、运动描述和历史稳定性 | 多数只是 FIFO、短窗口或 clip 拼接，没有事件阶段和顺序约束。 |
| 细粒度语言融合 | HFF、DKGTrack、SKTrack、CGATracker、CDRMT、COAL | 属性、空间和运动对齐 | 最拥挤路线；增加注意力或多层融合很难再成立。 |
| 跨视角 RMOT | CRTracker、OmniPT、ViewSAM | 多摄像头全局一致身份和视角互补 | 数据规模小，强依赖 ReID/伪标签，尚无明确持久 Global Identity Query 范式。 |
| 开放词汇/zero-shot | ReferGPT、LaMOTer、VideoGrounding-DINO | 未训练类别和灵活查询 | zero-shot 不等于未见组合；开放检测不自动解决身份。 |
| MLLM-based | ReferGPT、VMRMOT、STORM、ReaTrack、OmniPT | 常识、生成式理解和任务统一 | 幻觉、速度、定位精度、数据需求和公平比较问题突出。 |

### 3.1 统一基准中的大致性能位置

以下 HOTA 来自 FlexHook 和相关论文报告，仅用于判断方法所处位置。不同 tracker、backbone、训练数据和实现会造成不公平比较。

| 方法 | Refer-KITTI | Refer-KITTI-V2 | 说明 |
|---|---:|---:|---|
| TransRMOT | 46.56 | 31.00 | 端到端基础 |
| TempRMOT | 52.21 | 35.04 | 历史 Query |
| HFF-Tracker | 52.41 | 36.18 | 多层细粒度融合 |
| DKGTrack | 52.08 | 35.26 | 静态/运动语言解耦 |
| iKUN | 48.84 | 10.32 | 双阶段插件 |
| ReferGPT | 49.46 | 30.12 | zero-shot、3D 版本 |
| FlexHook | **53.83** | 42.53 | 强双阶段 |
| COAL | 53.38 | **43.46** | arXiv，属性反事实增强 |

STORM 的 66.7 HOTA 来自 STORM-Bench，不能加入该表直接排序。

---

## 4. 已解决和未解决的问题

等级定义：

1. 尚未研究；
2. 已有研究但不充分；
3. 已较充分解决；
4. 只是缺少实验，单独不足以构成方法创新。

| 问题 | 判断 | 证据和结论 |
|---|---|---|
| 真正理解有先后顺序的多阶段事件 | **2** | ReaMOT研究隐式推理，VMRMOT研究运动趋势，STORM-Bench包含关系/交互，但均没有系统的阶段 Token、单调顺序对齐和顺序交换测试。 |
| 历史聚合是否只是“看得更长” | **2** | TempRMOT、DKGTrack、VMRMOT和STORM扩大了时间范围，但没有判断“阶段一完成后才能进入阶段二”。 |
| 未见语言组合是否被系统评估 | **2；如果只补 split 则为 4** | COAL已有属性反事实，ReferGPT已有zero-shot，LaMOT有开放类别，但缺少属性×关系×动作×顺序的泄漏控制切分。 |
| CRMOT 是否维护共享 Global Identity Query | **科学问题为2，精确机制为1** | CRTracker使用跨视角ReID和显式关联；OmniPT使用LVLM统一生成；ViewSAM使用view-aware条件。公开架构中未见持续可读写并反向约束局部tracker的共享Query。 |
| 双阶段方法是否用语言反馈修正 tracker | **1，限已核验直接RMOT文献** | iKUN、ReferGPT和FlexHook均先完成跟踪再筛选。FlexHook模拟ID Switch，但没有写回关联矩阵。 |
| iKUN/FlexHook后插件路线是否仍有空间 | **2** | 通用匹配器已经拥挤，但反馈纠错、不确定性触发、组合校准、流式内存仍有空间。 |
| MLLM的幻觉、速度和定位问题是否解决 | **2** | ReferGPT明确存在速度和描述缺失问题；STORM使用8B模型和大规模数据；ReaTrack不是实时系统。 |
| 仅更换Grounding DINO/SAM2是否构成创新 | **4** | 它们解决开放候选检测和mask传播，但不直接解决多目标身份、事件顺序和语言反馈。 |

---

## 5. 热门技术与 RMOT 的结合判断

## 5.1 MLLM 拆解多阶段语言：成立

问题链：

> 整句 embedding 混合多个动作  
> → 模型无法区分阶段完成情况  
> → MLLM 将表达转换为结构化事件 Token 和顺序边  
> → 轨迹记忆检测状态转折并执行单调对齐  
> → 验证正确顺序是否比仅包含相同动作更能识别目标。

必要实验：顺序交换、阶段删除、重复阶段、部分满足的干扰轨迹。

MLLM只承担结构解析，因此具有明确、不可替代的能力，也可以被小型 parser 蒸馏。

## 5.2 Track/Object Token 输入 Video-LLM：有条件成立

问题链：

> 长视频含大量冗余帧  
> → Video-LLM消耗大量patch token  
> → Object/Track Token保留目标中心表示和身份  
> → 每条轨迹压缩成外观、运动、时间和语言匹配Token  
> → 在相同token budget下比较帧Token和Track Token。

如果只提升速度而没有证明身份、事件顺序或长时语义保留，则只是工程优化。

[MOTIP](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html)已证明object-level特征可用于ID prediction；[TrajTok](https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_TrajTok_Learning_Trajectory_Tokens_Enhances_Video_Understanding_CVPR_2026_paper.pdf)已证明轨迹Token可提高视频理解。因此“把轨迹输入LLM”本身不再新。

## 5.3 长期轨迹记忆：成立，但必须保存转折

建议记忆：

- 遮挡前的可靠身份；
- 动作开始和结束；
- 车道或相对关系改变；
- 事件阶段完成状态；
- 高不确定性片段。

如果只是把 TempRMOT 的 FIFO 从 4 帧增加到 16 帧，仍然只是看得更长。

## 5.4 组合泛化：成立

至少构造四种 disjoint split：

- 未见属性—类别组合；
- 未见主体—关系—客体组合；
- 未见动作—对象组合；
- 相同事件集合、不同顺序。

同时检查原始视频、轨迹、模板和近义改写是否跨 train/test 泄漏。

## 5.5 语言反馈修正关联：成立

问题链：

> Tracker产生多个相近关联假设  
> → 视觉/运动相似导致ID竞争  
> → 语言提供颜色、角色、动作或关系约束  
> → 输出must-link/cannot-link或修改关联cost  
> → 在固定DetA下验证AssA、IDSW和repair precision。

语言无法区分语义完全相同的目标，因此不能声称普遍修复ID Switch。

## 5.6 Local Query + Global Identity Query：成立但高风险

问题链：

> 局部相机Query在遮挡/出视野后丢失身份  
> → Global Query保存跨相机外观、语言、时间和路径状态  
> → 各相机从全局bank读取身份先验，并写回高置信更新  
> → 评估跨域CVRIDF1、跨视角ID Switch和摄像头缺失鲁棒性。

如果Global Query只是跨视角ReID向量换名，则不成立。

## 5.7 只在低置信度时调用MLLM：成立

必须比较：

- 全调用；
- 从不调用；
- 随机调用；
- oracle调用；
- 学习触发。

核心结果应是性能—调用率—延迟曲线，以及错误MLLM约束造成的负收益。

## 5.8 多Agent：当前属于热点堆叠

[VideoHV-Agent](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Think_Then_Verify_A_Hypothesis-Verification_Multi-Agent_Framework_for_Long_Video_CVPR_2026_paper.html)证明“假设—检索—验证”适合长视频问答，但直接搬到RMOT存在多次调用、高延迟、输出难监督和单Agent可替代等问题。

除非定义清晰的纠错动作空间、证据链和调用预算，否则不建议优先。

## 5.9 Grounding DINO、SAM2、LoRA、Adapter、MoE和蒸馏

这些技术适合作为支撑：

- [Grounding DINO](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06319.pdf)：开放词汇候选检测；
- [SAM2](https://proceedings.iclr.cc/paper_files/paper/2025/file/45c1f6a8cbf2da59ebf2c802b4f742cd-Paper-Conference.pdf)：mask传播和遮挡恢复；
- LoRA/Adapter：低成本适配语言或MLLM；
- 蒸馏：把离线MLLM判断压缩到小模型；
- MoE/动态推理：按歧义程度分配计算。

它们不应单独作为核心贡献。

---

## 6. 候选研究方向

## 6.1 C1：CompRMOT——泄漏受控的组合泛化审计

### 研究问题和假设

现有高HOTA模型可能依赖单属性、语言模板和场景共现捷径，在未见关系、动作和顺序组合上显著退化。

### 创新点

- 属性、关系、动作、顺序四类组合切分；
- 顺序反事实和最小对比测试；
- 因子化FlexHook/TempRMOT基线。

与COAL的关键差异是：COAL主要在标准split上进行属性反事实增强，本方向直接评估真正未见组合，并扩展到关系、动作和顺序。

### 最近工作

COAL、DKGTrack、HFF-Tracker、FlexHook、ReferGPT。

### Baseline和数据

- Baseline：TransRMOT、TempRMOT、FlexHook。
- 数据：Refer-KITTI-V2、STORM-Bench，辅以人工复核。

### 信息流程

语言解析成因子 → 构建disjoint split → 生成反事实负查询 → 多因子得分 → 组合校准。

### MVP

先不修改模型，测试三个开源基线的IID—组合差距；随后为FlexHook PCD加入属性、关系、动作和顺序分支。

### 必要消融

- 随机split和组合split；
- 视频、轨迹和模板去重；
- 各类反事实；
- 因子头和整体句子头；
- LLM生成表达和人工表达。

### 资源和时间

- 1–2张24GB GPU；
- 1,000–2,000条查询人工检查；
- 8–10周。

### 风险、质疑和保留贡献

- 风险：现有数据缺少真实多阶段动作。
- 审稿质疑：是否只是重新切分数据；自动表达是否可靠。
- 失败后保留：公开评测协议、泄漏分析和排行榜。

## 6.2 C2：EventRMOT——有序事件Token与阶段轨迹记忆

### 研究问题和假设

轨迹是否满足完整的事件序列，比单帧属性或平均运动更能确定语言所指目标。

### 创新点

- 事件阶段Token；
- 状态转折轨迹记忆；
- 单调顺序对齐；
- 顺序反事实评测。

### 最近工作

TempRMOT、DKGTrack、VMRMOT、ReaMOT、STORM。

### Baseline和数据

- Baseline：TempRMOT、FlexHook、VMRMOT。
- 数据：Refer-KITTI-V2、STORM-Bench，以及AerialMind/LaMOT中的可挖掘轨迹。

### 信息流程

表达 → 事件图 → 轨迹状态变化点 → 阶段片段 → 单调动态规划或CTC对齐 → 完整事件满足得分。

### MVP

冻结TempRMOT detector/tracker，只在历史Query后增加三阶段event scorer。

### 必要消融

- 无顺序；
- 平均池化；
- 固定窗口；
- 无状态转折；
- 规则parser、小型语言模型和MLLM；
- 交换事件顺序。

### 资源和时间

- 2张24GB GPU较稳妥；
- 500–1,500条事件查询；
- 10–12周。

### 风险、质疑和保留贡献

- 风险：现有数据缺少真实多阶段行为。
- 审稿质疑：是否只是temporal grounding拼到RMOT；事件是否人工构造。
- 失败后保留：Ordered-Event RMOT probe和阶段标注。

## 6.3 C3：RepairHook——语言反馈式轨迹关联修复

### 研究问题和假设

在语义可区分的ID竞争中，语言约束可以修改tracker的关联，而不只是事后筛选轨迹。

### 创新点

把iKUN/FlexHook的单向筛选变为：

> 语言匹配 → must-link/cannot-link → 关联重算 → 写回tracker状态。

### 最近工作

iKUN、FlexHook、ReferGPT、MOTIP、MeMOTR。

### Baseline和数据

- ByteTrack、StrongSORT、MeMOTR + FlexHook；
- Refer-KITTI-V2、Refer-Dance。

### MVP

先做离线tracklet stitching，不立即修改端到端tracker。

### 必要消融

- 仅视觉、仅语言；
- 不同触发器；
- oracle修复；
- 错误语言反馈；
- 固定DetA下比较AssA/IDSW。

### 资源和时间

- 1–2张24GB GPU；
- 可自动注入ID Switch；
- 8–10周。

### 风险、质疑和保留贡献

- 风险：很多ID Switch发生在语言不可区分的同质目标之间。
- 审稿质疑：提升是否来自普通ReID或轨迹后处理。
- 保留贡献：语言可修复/不可修复ID Switch诊断基准。

## 6.4 C4：Gated-MLLM——不确定性触发的MLLM验证器

- 假设：只在遮挡或ID竞争时调用MLLM，可保留复杂推理能力并降低成本。
- 创新：结构化pairwise constraint、置信度校准和调用率—精度曲线。
- 最近工作：ReferGPT、ReaTrack、FlexHook、VMRMOT、VideoHV-Agent。
- MVP：对两个候选轨迹的关键帧调用本地7B MLLM或API，再蒸馏到小模型。
- 消融：全调用、随机、置信度、oracle；有无运动摘要；不同MLLM。
- 风险：MLLM在细粒度颜色和身份上可能不如CLIP/ReID。
- 审稿质疑：是否只是昂贵后处理。
- 保留贡献：MLLM在RMOT关联中的能力和成本边界。

## 6.5 C5：GlobalID-RMOT——跨视角共享身份Query

### 研究问题和假设

持久的全局身份Query能够减少跨相机身份碎片，并利用语言保持视角不变语义。

### 创新点

- Local Track Query—Global Identity Query双层表示；
- view-aware、时间路径和语言联合读写门控；
- Global Query向局部tracker反向提供身份先验。

### 最近工作

CRTracker、OmniPT、ViewSAM、MeMOTR、MOTIP。

### Baseline和数据

- CRTrack；
- CRTracker、OmniPT、ViewSAM、普通跨相机ReID bank。

### 信息流程

局部Query → 全局检索 → 多模态门控融合 → Global ID分配/更新 → 局部关联约束。

### MVP

冻结单相机tracker，先实现tracklet-level全局Query bank。

### 必要消融

- 无语言；
- 无时间；
- 无view condition；
- ReID prototype；
- 固定/可学习Query；
- 异步摄像头。

### 资源和时间

- 3–4张24GB GPU更现实；
- 工程复杂，至少12周。

### 风险、质疑和保留贡献

- 风险：CRTrack只有13个场景，容易过拟合；OmniPT代码未确认。
- 审稿质疑：是否只是learned ReID memory。
- 保留贡献：跨视角Query读写协议和错误分析。

## 6.6 C6：TrackToken-RMOT

- 假设：对象中心轨迹Token比均匀帧Token更高效地保留身份和事件。
- 最近工作：TrajTok、MOTIP、MovieChat、STORM、VMRMOT。
- MVP：现成MOT → 每条轨迹固定K个Token → 小型temporal transformer或Video-LLM。
- 必要实验：相同FLOPs、相同token budget、错误轨迹鲁棒性、身份/顺序保留。
- 风险：TrajTok和STORM已覆盖大量技术叙事。
- 保留贡献：RMOT object-token效率基准。

## 6.7 C7：AgentRMOT

- 假设：跟踪、语言解析、验证和纠错的角色分工能够减少错误累积。
- 最近工作：VideoHV-Agent、ReferGPT、ReaTrack、STORM、OmniPT。
- MVP：固定Tracker Agent，增加Parser、Verifier和Repair角色。
- 必要实验：单Agent同预算、无CoT、无验证、随机Agent、延迟和费用。
- 风险：最高；三个月完成概率最低。
- 审稿质疑：角色名称是否只是prompt；多Agent是否不可替代。
- 当前结论：热点堆叠，不建议优先。

---

## 7. 统一评分

正向指标越高越好；覆盖风险、难度、资源和堆叠风险越高越差。

| 方向 | 热度 | 新颖性 | 学术价值 | RMOT匹配 | 覆盖风险↓ | WACV匹配 | 难度↓ | 数据算力↓ | 三月可行 | 故事完整 | 堆叠风险↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 CompRMOT | 9 | 7 | 9 | 10 | 6 | 10 | 5 | 4 | 9 | 9 | 3 |
| C2 EventRMOT | 9 | 9 | 9 | 10 | 4 | 9 | 7 | 6 | 7 | 10 | 3 |
| C3 RepairHook | 8 | 8 | 8 | 9 | 4 | 9 | 6 | 4 | 8 | 8 | 4 |
| C4 Gated-MLLM | 10 | 7 | 7 | 8 | 7 | 8 | 7 | 7 | 6 | 7 | 8 |
| C5 GlobalID | 9 | 8 | 9 | 9 | 6 | 9 | 9 | 8 | 4 | 9 | 5 |
| C6 TrackToken | 10 | 6 | 8 | 8 | 8 | 8 | 8 | 8 | 5 | 7 | 8 |
| C7 AgentRMOT | 10 | 6 | 7 | 6 | 8 | 7 | 9 | 9 | 3 | 6 | 10 |

### 评分解释

- **C1**：资源低、WACV E&D高度匹配、故事清楚；COAL和ReaMOT使大类覆盖风险不低。
- **C2**：有序阶段机制仍较空，故事最强；主要风险来自数据标注和进度。
- **C3**：实现和算力友好，直接填补双阶段单向流程缺口；语言对同质目标可能没有身份信息。
- **C4**：热度高，但容易被认为依靠更大模型，成本、幻觉和复现性均是风险。
- **C5**：价值和上限高，但CRTrack小、OmniPT/ViewSAM强、工程链长。
- **C6**：轨迹Token热门，但TrajTok、VMRMOT和STORM已经造成较高覆盖风险。
- **C7**：缺少多Agent不可替代性的证据，几乎所有负面风险都高。

---

## 8. 最终推荐

## 8.1 A：最低风险——CompRMOT

### 暂定标题

**Do RMOT Models Compose? A Leakage-Controlled Benchmark and Counterfactual Baseline for Compositional Referring Multi-Object Tracking**

### 一句话问题

现有RMOT的高HOTA究竟来自真正的语言组合能力，还是训练集中的单词、模板和场景捷径？

### 核心贡献

1. 属性、关系、动作和顺序四类泄漏受控切分；
2. 顺序交换和最小反事实测试；
3. 小型因子化FlexHook/TempRMOT基线。

### 流程

数据审计 → 因子解析 → 视频/轨迹/模板去重 → 组合切分 → 基线压力测试 → 因子化反事实训练。

### 与最近工作的差异

- COAL在标准split上进行属性反事实训练；本方向评估真正未见组合并扩展到关系、动作和顺序。
- ReaMOT关注隐式常识推理，不是训练—测试组合泄漏审计。

### 三个月计划

- 第1–2周：切分规范、泄漏分析、100–200条试标；
- 第3–4周：复现三种基线，完成IID/组合差距；
- 第5–7周：扩展并人工检查数据；
- 第8–9周：因子化基线；
- 第10周：消融和统计检验；
- 第11–12周：论文和补充材料。

### 继续标准

- 至少三类模型在组合split上比IID下降不低于10 HOTA或15个语义F1点；
- 因子化方法恢复不低于3 HOTA或5个F1点；
- 人工标注一致性κ不低于0.8。

### 放弃标准

- 去除泄漏后基线下降不足3点；
- 数据不能可靠确定关系、动作或顺序；
- 简单CLIP/规则与设计方法几乎相同。

## 8.2 B：创新性与可行性最平衡——EventRMOT

### 暂定标题

**EventRMOT: Ordered Event Tokens and Phase-Aware Trajectory Memory for Referring Multi-Object Tracking**

### 一句话问题

模型能否根据“先A、随后B、最后C”的完整事件顺序选择轨迹，而不是只匹配其中一个动作？

### 核心贡献

1. 将自然语言转换为有序事件图；
2. 通过轨迹状态转折形成事件片段记忆；
3. 单调阶段对齐和顺序反事实基准。

### 流程

查询 → 事件Token/顺序边 → TempRMOT轨迹Query → 状态转折检测 → 片段Token → 单调对齐 → 轨迹满足性得分。

### 与最近工作的差异

- TempRMOT：聚合历史特征，没有事件状态；
- DKGTrack：只分静态和运动；
- VMRMOT：生成运动描述，没有多阶段顺序；
- ReaMOT：隐式语义推理；
- STORM：生成式时空推理，但无显式顺序对齐评测。

### 三个月计划

- 第1–2周：定义任务语义和事件指标；
- 第3–4周：建立300–500条事件查询试验集；
- 第5–6周：完成TempRMOT/FlexHook顺序反事实baseline；
- 第7–8周：实现事件Token和变化点记忆；
- 第9周：实现单调对齐；
- 第10周：消融和错误分析；
- 第11–12周：扩大测试和写作。

### 继续标准

- 原模型对查询顺序交换不敏感；
- 新方法事件满足F1提高不低于5点、事件子集HOTA提高不低于2点；
- 去除顺序约束后性能明确下降；
- 普通单阶段查询无明显退化。

### 放弃标准

- 数据中真实三阶段轨迹过少；
- 阶段边界一致性低于κ=0.7；
- 简单规则或通用MLLM离线打分已经达到相同性能；
- 改善全部来自detector而非顺序判断。

## 8.3 C：高风险高上限——GlobalID-RMOT

### 暂定标题

**GlobalID-RMOT: Language-Conditioned Global Identity Queries for Cross-View Referring Tracking**

### 一句话问题

跨摄像头共享的持久身份Query能否比ReID后匹配更可靠地保持语言所指目标的全局身份？

### 核心贡献

1. Local Track Query—Global Identity Query双层身份表示；
2. view-aware、时间路径和语言联合读写门控；
3. 全局身份先验反向约束局部tracker。

### 流程

各相机局部跟踪 → 查询全局bank → 多模态门控融合 → 分配/更新Global ID → 向局部Query回传身份先验。

### 与最近工作的差异

- CRTracker：跨视角ReID和显式匹配；
- ViewSAM：view-aware条件和伪监督；
- OmniPT：LVLM统一生成；
- 本方向的核心必须是持久可读写的身份状态，以及它对局部关联的反向作用。

### 三个月计划

- 第1–2周：复现CRTrack/CRTracker；
- 第3–4周：冻结tracker，建立tracklet-level memory；
- 第5–7周：可学习Global Query及时间/view门控；
- 第8周：反向局部约束；
- 第9–10周：跨域、摄像头缺失和异步实验；
- 第11–12周：消融和写作。

### 继续标准

- 相比强ReID bank，CVRIDF1提高不低于4点；
- 跨视角ID Switch降低不低于20%；
- 去除语言或Global Query后明显退化；
- 跨域提升仍然成立。

### 放弃标准

- ReID prototype与Global Query差距不超过1点；
- 去掉语言几乎不影响结果；
- 提升只出现在训练场景；
- 无法建立与OmniPT/ViewSAM的公平比较。

---

## 9. EventRMOT 第一周行动计划

若拥有完整三个月，默认推进B：EventRMOT。

### 第1天：固定科学问题

明确以下内容：

- 查询描述的是“目标整个轨迹是否满足完整事件”，而不是只输出事件发生期间的box；
- 什么叫阶段完成；
- 哪些顺序改变会改变答案；
- 普通HOTA为什么不足。

### 第2天：设计最小评测

至少包括：

1. 正确顺序；
2. 相同事件、顺序交换；
3. 删除中间阶段；
4. 只满足部分阶段的干扰轨迹。

指标：HOTA、Event-Track F1、Order Accuracy、Stage Completion Accuracy。

### 第3天：审查现有数据

从Refer-KITTI-V2、STORM-Bench、LaMOT/AerialMind中抽查至少100条轨迹，统计：

- 二阶段事件数量；
- 三阶段事件数量；
- 可人工确认的状态转折；
- 是否存在相似干扰目标。

### 第4天：建立标注规范

建议每条表达保存：

```text
subject constraints
stage_1 predicate
stage_2 predicate
stage_3 predicate
ordering edges
target track IDs
stage time spans
hard-negative track IDs
```

由两人独立标注50条并计算一致性。

### 第5天：测试原模型是否理解顺序

比较TempRMOT/FlexHook对以下查询的响应：

- 原查询；
- 顺序交换；
- 同义改写；
- 只保留最后动作。

如果模型分数对顺序交换高度不敏感，研究假设获得初步支持。

### 第6天：实现最简单baseline

- 基于速度、方向和位置变化检测状态转折；
- 使用规则或小语言模型解析三阶段；
- 使用动态规划执行单调匹配。

如果该baseline已经明显超过TempRMOT，说明任务中存在可学习信号。

### 第7天：Go/No-Go

继续EventRMOT的必要条件：

- 至少200条可靠二阶段、50条可靠三阶段样本；
- 顺序交换确实改变语义真值；
- 现有模型Order Accuracy接近随机或明显低于普通查询准确率；
- 阶段标注一致性κ不低于0.7。

否则应立即转向CompRMOT，不应在数据不足的事件方向继续消耗时间。

---

## 10. 关键参考文献和代码

## 10.1 直接RMOT工作

| 论文 | 年份/发表位置 | 代码状态 |
|---|---|---|
| [Referring Multi-Object Tracking / TransRMOT](https://openaccess.thecvf.com/content/CVPR2023/html/Wu_Referring_Multi-Object_Tracking_CVPR_2023_paper.html) | CVPR 2023 | [GitHub](https://github.com/wudongming97/RMOT) |
| [Bootstrapping Referring Multi-Object Tracking / TempRMOT](https://arxiv.org/abs/2406.05039) | arXiv 2024，正式发表未确认 | [GitHub](https://github.com/zyn213/TempRMOT) |
| [iKUN: Speak to Trackers without Retraining](https://openaccess.thecvf.com/content/CVPR2024/html/Du_iKUN_Speak_to_Trackers_without_Retraining_CVPR_2024_paper.html) | CVPR 2024 | [GitHub](https://github.com/dyhBUPT/iKUN) |
| [HFF-Tracker](https://ojs.aaai.org/index.php/AAAI/article/view/33143) | AAAI 2025 | 未确认 |
| [Cross-View Referring Multi-Object Tracking](https://ojs.aaai.org/index.php/AAAI/article/view/32219) | AAAI 2025 | [GitHub](https://github.com/chen-si-jia/CRMOT) |
| [DKGTrack](https://openaccess.thecvf.com/content/ICCV2025/html/Li_Language_Decoupling_with_Fine-grained_Knowledge_Guidance_for_Referring_Multi-object_Tracking_ICCV_2025_paper.html) | ICCV 2025 | [GitHub](https://github.com/acyddl/DKGTrack) |
| [ReferGPT](https://openaccess.thecvf.com/content/CVPR2025W/WDFM-AD/html/Chamiti_ReferGPT_Towards_Zero-Shot_Referring_Multi-Object_Tracking_CVPRW_2025_paper.html) | CVPR Workshops 2025 | [GitHub](https://github.com/Tzoulio/ReferGPT) |
| [MEX](https://arxiv.org/abs/2502.13875) | arXiv 2025 | 未确认 |
| [LaMOT](https://arxiv.org/abs/2406.08324) | ICRA 2025 | [GitHub](https://github.com/Nathan-Li123/LaMOT) |
| [SKTrack](https://doi.org/10.1109/TMM.2025.3557710) | IEEE TMM 2025 | 未确认 |
| [CGATracker](https://doi.org/10.1109/TCSVT.2025.3576341) | IEEE TCSVT 2025 | 未确认 |
| [Cognitive Disentanglement for RMOT](https://www.sciencedirect.com/science/article/pii/S1566253525004221) | Information Fusion 2025 | 未确认 |
| [VMRMOT](https://arxiv.org/abs/2511.17681) | arXiv 2025 | 未确认 |
| [AerialMind](https://ojs.aaai.org/index.php/AAAI/article/view/37270) | AAAI 2026 | [数据](https://github.com/shawnliang420/AerialMind)，模型代码未确认 |
| [FlexHook](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Rethinking_Two-Stage_Referring-by-Tracking_in_Referring_Multi-Object_Tracking_Make_it_Strong_CVPR_2026_paper.html) | CVPR 2026 | [GitHub](https://github.com/buptLwz/FlexHook) |
| [STORM](https://openaccess.thecvf.com/content/CVPR2026F/html/Lu_STORM_End-to-End_Referring_Multi-Object_Tracking_in_Videos_CVPRF_2026_paper.html) | CVPR Findings 2026 | [数据](https://github.com/amazon-science/storm-referring-multi-object-grounding)，完整模型代码未确认 |
| [OmniPT](https://ojs.aaai.org/index.php/AAAI/article/view/37406) | AAAI 2026 | 未确认 |
| [ReaMOT](https://arxiv.org/abs/2505.20381) | arXiv 2025，2026更新 | [占位仓库](https://github.com/chen-si-jia/ReaMOT)，代码/数据未释放 |
| [ViewSAM](https://arxiv.org/abs/2605.02638) | arXiv 2026 | 未确认 |
| [COAL](https://arxiv.org/abs/2605.14795) | arXiv 2026 | 未确认 |

## 10.2 关键相邻工作

| 论文 | 发表位置 | 代码状态 |
|---|---|---|
| [Grounding DINO](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06319.pdf) | ECCV 2024 | [GitHub](https://github.com/IDEA-Research/GroundingDINO) |
| [VideoGrounding-DINO](https://openaccess.thecvf.com/content/CVPR2024/html/Wasim_VideoGrounding-DINO_Towards_Open-Vocabulary_Spatio-Temporal_Video_Grounding_CVPR_2024_paper.html) | CVPR 2024 | 未确认 |
| [SAM 2](https://proceedings.iclr.cc/paper_files/paper/2025/file/45c1f6a8cbf2da59ebf2c802b4f742cd-Paper-Conference.pdf) | ICLR 2025 | [GitHub](https://github.com/facebookresearch/sam2) |
| [MeMOTR](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_MeMOTR_Long-Term_Memory-Augmented_Transformer_for_Multi-Object_Tracking_ICCV_2023_paper.html) | ICCV 2023 | [GitHub](https://github.com/MCG-NJU/MeMOTR) |
| [MOTIP](https://openaccess.thecvf.com/content/CVPR2025/html/Gao_Multiple_Object_Tracking_as_ID_Prediction_CVPR_2025_paper.html) | CVPR 2025 | [GitHub](https://github.com/MCG-NJU/MOTIP) |
| [MovieChat](https://openaccess.thecvf.com/content/CVPR2024/html/Song_MovieChat_From_Dense_Token_to_Sparse_Memory_for_Long_Video_CVPR_2024_paper.html) | CVPR 2024 | [项目页](https://rese1f.github.io/MovieChat/) |
| [TimeChat](https://openaccess.thecvf.com/content/CVPR2024/papers/Ren_TimeChat_A_Time-sensitive_Multimodal_Large_Language_Model_for_Long_Video_CVPR_2024_paper.pdf) | CVPR 2024 | [GitHub](https://github.com/RenShuhuai-Andy/TimeChat) |
| [VTG-LLM](https://arxiv.org/abs/2405.13382) | AAAI 2025 | [GitHub](https://github.com/gyxxyg/VTG-LLM) |
| [TrajTok](https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_TrajTok_Learning_Trajectory_Tokens_Enhances_Video_Understanding_CVPR_2026_paper.pdf) | CVPR 2026 | 官方代码链接未确认 |
| [Think, Then Verify / VideoHV-Agent](https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Think_Then_Verify_A_Hypothesis-Verification_Multi-Agent_Framework_for_Long_Video_CVPR_2026_paper.html) | CVPR 2026 | [GitHub](https://github.com/Haorane/VideoHV-Agent) |

---

## 11. 最终决策

- **正常三个月研究：选择 EventRMOT。**
- **急投 WACV 2027：选择 CompRMOT，并在第一周验证组合泛化差距。**
- **有跨视角资源且能够承担高失败风险：选择 GlobalID-RMOT。**
- **不建议优先：普通融合模块、简单MLLM外挂、Track Token换名、多Agent堆叠。**

