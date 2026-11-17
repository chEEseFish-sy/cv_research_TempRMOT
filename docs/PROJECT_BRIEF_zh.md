# TempRMOT 合作说明

更新日期：2026-11-17（北京时间）。供新加入的同学了解方向、进展和协作入口。

## 项目目标

基于 TempRMOT 做指代表达多目标跟踪（RMOT）：输入视频和自然语言，例如“左侧的汽车”，持续输出符合描述的目标框和身份 ID。

研究扩展是在时序推理之后、预测头之前加入每条轨迹独立的 TTT 快速权重记忆，让模型选择 TRUST（相信当前观测）、REASSOCIATE（接管候选并保留旧 ID）、HOLD（保留身份等待）或 TERMINATE（结束轨迹）。目标是改善遮挡、重现和身份漂移场景的连续性，避免错误接管和误删。

这是基于上游开源项目的研究扩展，原论文、作者与引用见根目录 README。

## 已完成与主要结果

已完成基线复现、TTT 原型、GT teacher/动作监督、全量联合训练、冻结基座 pilot，以及训练和逐帧推理诊断。**目前尚未证明 TTT 稳定优于基线。**

最近完成的全量实验 `joint_recovery_handoff_17plus4_20261110_01` 使用 17 个训练视频、60 轮；以下三组均完整评测官方测试集 862 条表达，固定末轮权重：

| 评测组 | HOTA % | IDF1 % | IDSW |
|---|---:|---:|---:|
| 原 TempRMOT 基线 | 35.8945 | 37.5726 | 2439 |
| 联合末轮，推理关闭 TTT | 32.8652 | 34.0459 | 2699 |
| 同一联合末轮，推理开启 TTT | 27.6715 | 27.3642 | 9788 |

已定位问题：候选身份选错；正确接管受到相反分类监督而低分；候选冲突强制回退 HOLD；正常轨迹被错误接管或终止。

11 月 16 日冻结基座 pilot 中，监督修复版在开发集 120 条表达上的 HOTA 为 79.2452%，基线为 80.2706%；IDSW 为 214 对 86。检测有所改善，但身份连续性仍下降；后续候选分配版本还出现大量 REASSOCIATE 的动作塌缩。**开发集120与官方测试集862口径不同，分数不能直接横向比较。**

数字来自 11 月 17 日核验的本地报告，本次交接没有重跑训练或评测。

## 目前推进的工作

11 月 17 日方案主要统一动作与候选的监督/执行分数，改为先读旧记忆再决定写入，并同预算比较完整接管、比例融合和旧训练身份流程。

保存的状态显示：unified/simple 在 13:09 等待空闲 GPU；另一个独立消融实验已通过 GPU smoke 和断点恢复检查，进入 baseline120 评测。**这是保存的进度快照，未重新查询远程实时状态；新方案尚无性能结论。**

接下来先完成同预算开发集评测，检查正确找回、正常轨迹误接管、误终止、重复目标与 HOLD 覆盖，再讨论多 seed 和官方测试泛化。

## 代码版本边界

- `main`：默认入口，包含本说明和较早基础实现。
- `ttt-memory`：已提交的 TTT 原型，代码快照 `a21e536`；未包含全部 11 月改动，分支 README 的实验结果属于历史记录。
- `reflife-mvp`：早期 RefLife 探索，供回溯。
- 本地还有未提交的 teacher、模型和审计修改；11 月 16–17 日最新实现位于独立实验源码目录，**尚未完整同步到公开仓库，直接 clone 不能复现最新实验**。

合作开始时先同步实际要改的隔离源码、配置、固定表达清单和 checkpoint，记录 commit/hash；不要混用旧模型与新日志或覆盖历史实验。完整数据、权重和逐帧审计产物另行交接。

## 入门与分工

```bash
git clone https://github.com/chEEseFish-sy/cv_research_TempRMOT.git
cd cv_research_TempRMOT
git switch ttt-memory
git switch -c collab/your-name-topic
```

先看 `models/transrmot_pro.py`（主流程）、`models/spatial_temporal_reason.py`（时序推理）、`models/ttt_memory.py`（快速权重）、`models/track_ttt_action.py`（动作），再看 `tools/run_ttt_pilot.py`、`tools/evaluate_ttt_pilot.py` 和 `tests/test_ttt_*.py`。

环境参考 `docs/TEMPRMOT_REPRODUCTION_GUIDE.md`、`datasets/README.md` 与配置脚本：Linux/WSL、Python、PyTorch/CUDA 和 Deformable Attention 扩展。复现指南是历史记录，机器路径、版本与已修问题需按当前源码核对；新机器另行准备数据、RoBERTa 和 checkpoint，仓库不含这些大文件。

依赖及资产准备好后，可先验证所 checkout 的原型版本：

```bash
conda activate temprmot
PYTHONPATH=. python -m unittest discover -s tests -p 'test_ttt_*.py' -v
```

- 项目负责人：整理最新可协作源码与资产，维护基线及训练/评测口径。
- 新加入同学：先复现一个“正常轨迹被错误接管”的逐帧案例，核对候选、监督和最终输出，再负责一个独立消融。
- 共同约定：独立分支、固定数据划分/预算/阈值、保留负结果，用 HOTA、IDF1、IDSW 和逐帧证据判断；暂不从60轮全量训练入手。

公开仓库可直接 clone/fork；向原仓库 push 仍需单独授予协作者权限。
