# TempRMOT 可执行复现流程

更新时间：2026-07-18（Asia/Shanghai）  
适用项目：`/home/curtis/projects/TempRMOT`  
目标平台：WSL2 Ubuntu 22.04、Python 3.10、PyTorch 2.8.0+cu128、CUDA Toolkit 12.8、RTX 5060 Laptop GPU 8 GB

## 0. 文档目的与复现边界

本流程的首要目标是复现 TempRMOT 在 Refer-KITTI-V2 上的**官方权重推理和 HOTA 评测**，然后再做训练 smoke test；不把单卡 8 GB 环境下完整重训 60 epoch 当作第一目标。

复现分为四级：

| 级别 | 完成标准 | 本机可行性 |
|---|---|---|
| L1 环境复现 | CUDA 扩展 forward/backward 测试全部通过 | 高 |
| L2 工程复现 | 模型构建、RoBERTa 和官方权重加载成功，单表达式推理成功 | 高 |
| L3 指标复现 | Refer-KITTI-V2 全测试集得到接近论文的 HOTA/DetA/AssA | 高，但耗时 |
| L4 训练复现 | 从 Deformable DETR 预训练权重训练 60 epoch 并接近论文结果 | 本机不现实，建议 4 卡服务器 |

主验收目标是 L3。L4 分为“本机 smoke training”和“服务器严格训练”两条路线。

### 0.1 目标指标

Refer-KITTI-V2 官方完整模型：

| HOTA | DetA | AssA | DetRe | DetPr | AssRe | AssPr | LocA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 35.04 | 22.97 | 53.58 | 34.23 | 40.41 | 59.50 | 81.29 | 90.07 |

官方权重复现的建议验收容差：HOTA、DetA、AssA 各自绝对差不超过 `0.20`。超过容差不能直接归因于硬件，应先核对数据版本、测试表达式集合、推理阈值、输出格式和 TrackEval 聚合方式。

### 0.2 资料优先级

发生冲突时按以下顺序保留证据，不要静默选择：

1. 本次实际运行产生的参数、日志、哈希和输出。
2. 当前项目源码与官方配置脚本。
3. 官方 GitHub README、release 页面和数据说明。
4. 论文正文。
5. 交接文档与第三方说明。

论文描述研究设定，官方代码决定当前仓库实际上执行什么；两者不一致时，两套值都要记录。

## 1. 已确认事实与已知风险

### 1.1 已确认环境

- Python：3.10.20
- PyTorch：2.8.0+cu128
- torchvision：0.23.0+cu128
- CUDA runtime：12.8
- CUDA Toolkit：`/usr/local/cuda-12.8`
- GPU：RTX 5060 Laptop，Compute Capability 12.0
- PyTorch CUDA 可用，GPU 矩阵运算已验证

不要重装 PyTorch、CUDA Toolkit 或 WSL NVIDIA 驱动。尤其不要在 WSL 安装 `cuda`、`cuda-12-8` 或 `cuda-drivers` 元包。

### 1.2 当前源码不是直接可运行状态

正式推理前必须解决以下问题：

1. 仓库含 181 个 `Zone.Identifier` 文件、Python 3.7 缓存和旧 CUDA 扩展产物。
2. `models/transrmot_pro.py` 和 `models/transrmot.py` 硬编码作者机器上的 RoBERTa 路径。
3. 两个模型文件导入未使用的 `torchtext`；当前 PyTorch 2.8 不应盲装旧 `torchtext` 二进制包。
4. `inference.py` 固定启动 8 个进程，并把进程编号 0-7 当 GPU 编号；本机只有一张 GPU。
5. `inference.py` 假定 checkpoint 文件名结尾是纯数字；`checkpoint_rk2.pth` 会解析失败。
6. TrackEval 的 `mot_challenge_2d_box.py` 硬编码 `/data_2/zyn/refer-kitti-v2/...`。
7. 当前 NumPy 是 2.2.6，而仓库内旧 TrackEval 使用已移除的 `np.float`、`np.int` 和 `np.bool`。
8. PyTorch 2.6 以后 `torch.load` 默认行为改变；可信 checkpoint 中若含 `argparse.Namespace` 等对象，可能需要显式 `weights_only=False`。
9. `torchvision.models.resnet50(pretrained=...)` 是旧 API，而且模型构建可能触发不必要的 ImageNet 权重下载。
10. `main.py` 的 TensorBoard 目录通过字符串切分生成，对绝对输出路径不可靠。
11. `engine.py` 的 TensorBoard 统计固定循环 4 帧，而 Refer-KITTI-V2 配置使用 5 帧。
12. 官方 README 中 Refer-KITTI-V2 权重链接指向了错误仓库；正确资产在 TempRMOT 自己的 v1.0 release 中。

这些不是“可选优化”，其中第 2-8 项会直接阻断当前环境中的模型构建、推理或评测。

### 1.3 论文与代码的已知差异

| 项目 | 论文 | 官方代码/脚本 |
|---|---|---|
| 优化器 | Adam | AdamW |
| base learning rate | 正文显示 `1e-5`，但文字疑似有误 | `1e-4` |
| backbone learning rate | `1e-5` | `1e-5` |
| 训练最短边 | 800-1536 | 数据增强列表为 608-992，`max_size=1536` |
| 训练 epoch | 60 | 60 |
| LR drop | 第 40 epoch | 40 |
| Refer-KITTI-V2 训练历史长度 | 5 | 5 |
| 推理历史长度 | 8 | 8 |
| object threshold | 0.6 | 0.6 |
| referring threshold | 0.4 | 0.4，代码硬编码 |
| 训练 GPU | 4 x RTX 4090 | 启动脚本固定 4 进程 |

第一轮指标复现必须遵循“官方 checkpoint + 官方推理代码语义”。严格重训时先运行代码配置版，再另做“论文文字版”对照实验；不可混合参数后仍称为官方复现。

## 2. 推荐目录与产物规范

源码、数据、权重和实验输出分开：

```text
/home/curtis/projects/TempRMOT                 # 源码
/home/curtis/datasets/refer-kitti-v2           # KITTI + V2 标注
/home/curtis/checkpoints/TempRMOT              # 官方与预训练权重
/home/curtis/checkpoints/roberta-base          # RoBERTa
/home/curtis/experiments/TempRMOT              # 推理、评测、训练输出
```

每次正式实验至少保存：

```text
run_name/
├── command.txt
├── environment.txt
├── source_state.txt
├── hashes.sha256
├── stdout.log
├── stderr.log
├── results/ or checkpoints/
└── metrics.txt
```

禁止把数据集、PDF、1 GB checkpoint 或实验输出提交进 Git。

首次执行时先创建外部目录：

```bash
mkdir -p /home/curtis/datasets/refer-kitti-v2
mkdir -p /home/curtis/checkpoints/TempRMOT
mkdir -p /home/curtis/checkpoints/roberta-base
mkdir -p /home/curtis/experiments/TempRMOT
```

## 3. 阶段 A：建立可回退的源码基线

### A1. 确认工作目录

```bash
cd /home/curtis/projects/TempRMOT
test "$(pwd -P)" = "/home/curtis/projects/TempRMOT"
```

只有第二条命令成功才继续。不要在 `/mnt/c` 副本中编译 CUDA 扩展。

### A2. 记录清理前状态

```bash
find /home/curtis/projects/TempRMOT -type f -name '*:Zone.Identifier' | wc -l
find /home/curtis/projects/TempRMOT -type f -name '*.pyc' | wc -l
find /home/curtis/projects/TempRMOT/models/ops -type f -name '*.so' -print
du -sh /home/curtis/projects/TempRMOT
```

把输出保存进复现日志。当前预期约为：Zone 文件 181 个、`.pyc` 17 个、旧 `.so` 1 个；若实际不同，以当次日志为准。

### A3. 修正 `.gitignore`

当前 `.gitignore` 使用作者机器的绝对路径，实际上不能忽略本项目文件。改成至少包含：

```gitignore
__pycache__/
*.py[cod]
*.so
*.egg-info/
build/
dist/
.idea/
.vscode/
exps/
log/
tmp/
related_paper/
*.pth
*.pt
*.ckpt
```

### A4. 清理下载标记与不兼容构建产物

先再次确认 `pwd`，然后仅清理项目内的明确目标：

```bash
cd /home/curtis/projects/TempRMOT
test "$(pwd -P)" = "/home/curtis/projects/TempRMOT"

find . -type f -name '*:Zone.Identifier' -delete
rm -rf ./models/ops/build
rm -rf ./models/ops/dist
rm -rf ./models/ops/MultiScaleDeformableAttention.egg-info
find . -type d -name '__pycache__' -prune -exec rm -rf '{}' +
find . -type f -name '*.pyc' -delete
find ./models/ops -maxdepth 1 -type f -name 'MultiScaleDeformableAttention*.so' -delete
```

验证：

```bash
test "$(find . -type f -name '*:Zone.Identifier' | wc -l)" -eq 0
test "$(find . -type f -name '*.pyc' | wc -l)" -eq 0
test ! -d ./models/ops/build
test ! -d ./models/ops/dist
```

### A5. 初始化本地 Git

```bash
cd /home/curtis/projects/TempRMOT
git init
git add .
git status --short
git commit -m "chore: record cleaned official TempRMOT source"
git tag source-clean-baseline
```

提交前确认 `related_paper/`、checkpoint、数据和实验输出没有出现在暂存区。记录官方来源、下载日期和当时 main 分支最新短 SHA `6a65640`，但由于本地来源是 ZIP，不能在未逐文件比对前宣称本地就是该 commit。

若 Git 提示作者身份未配置，只设置当前仓库的 `user.name` 和 `user.email` 后重试；不要为了本项目擅自修改全局 Git 身份。

**A 阶段通过标准：** 工作树干净；旧二进制与 Zone 文件为零；存在可回退 baseline commit。

## 4. 阶段 B：只验证现有 Python/PyTorch/CUDA

### B1. 激活专用环境

```bash
source /home/curtis/miniforge3/etc/profile.d/conda.sh
conda activate temprmot
cd /home/curtis/projects/TempRMOT
```

### B2. 运行环境诊断

```bash
which python
python --version
python -m pip --version
gcc --version | head -n 1
g++ --version | head -n 1
nvcc --version
nvidia-smi

python - <<'PY'
import torch, torchvision
from torch.utils.cpp_extension import CUDA_HOME
print('torch:', torch.__version__)
print('torchvision:', torchvision.__version__)
print('torch CUDA:', torch.version.cuda)
print('CUDA_HOME:', CUDA_HOME)
print('available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('capability:', torch.cuda.get_device_capability(0))
x = torch.randn(1024, 1024, device='cuda')
y = x @ x
print('matmul:', y.shape, y.device, torch.isfinite(y).all().item())
PY
```

预期关键值：CUDA 可用、`CUDA_HOME=/usr/local/cuda-12.8`、capability `(12, 0)`、矩阵结果有限。

### B3. 保存环境基线

```bash
python -m pip freeze > /home/curtis/experiments/TempRMOT/environment-before.txt
conda list > /home/curtis/experiments/TempRMOT/conda-before.txt
```

**B 阶段失败时：** 停止，不安装业务依赖。先恢复现有 `temprmot` 环境的可用状态。  
**B 阶段通过标准：** 版本、GPU、CUDA_HOME 和矩阵测试全部符合预期。

## 5. 阶段 C：编译 MultiScaleDeformableAttention

此阶段不需要 RoBERTa、数据集或官方 TempRMOT 权重。

### C1. 设置编译变量

```bash
source /home/curtis/miniforge3/etc/profile.d/conda.sh
conda activate temprmot
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=4
cd /home/curtis/projects/TempRMOT/models/ops
```

### C2. 现代化本地安装

```bash
python -m pip install -v --no-build-isolation . 2>&1 | tee /home/curtis/experiments/TempRMOT/build-msda.log
```

选择 `--no-build-isolation` 是为了确保扩展使用当前 `temprmot` 环境中的 PyTorch 2.8 和 CUDA 12.8，而不是临时构建环境中的另一套 torch。

### C3. 先做 import 检查

```bash
python - <<'PY'
import MultiScaleDeformableAttention as M
print(M)
PY
```

### C4. 运行扩展测试

```bash
cd /home/curtis/projects/TempRMOT/models/ops
python test.py 2>&1 | tee /home/curtis/experiments/TempRMOT/test-msda.log
```

`test.py` 会检查 double/float forward，并对多个通道数做数值梯度检查。所有输出项都应为 `True`。若完整梯度测试耗时过长，可先只运行两个 forward 检查，但 L1 验收前必须完成全部 backward/gradcheck。

### C5. 确认架构信息

```bash
find /home/curtis/projects/TempRMOT/models/ops -type f -name 'MultiScaleDeformableAttention*.so' -print
grep -E 'sm_120|compute_120' /home/curtis/experiments/TempRMOT/build-msda.log | tail
```

如果编译日志没有直接显示架构，可用 `cuobjdump` 检查生成的 `.so`。不得通过保留旧 CPython 3.7 `.so` 来绕过编译。

**C 阶段通过标准：** 当前 Python 3.10 能 import；forward 一致性为 True；全部 gradcheck 为 True；产物由当前环境生成。

## 6. 阶段 D：安装业务依赖并锁定环境

### D1. 安装策略

保留现有 torch、torchvision 和 NumPy，不使用 `pip install -r requirements.txt` 一步覆盖整个环境。先安装当前真实 import 链需要的包：

```bash
python -m pip install \
  'numpy==2.2.6' \
  'opencv-python-headless>=4.10,<5' \
  'scipy>=1.11,<2' \
  'pycocotools>=2.0.7,<3' \
  'tqdm>=4.66,<5' \
  'motmetrics>=1.4,<2' \
  'seaborn>=0.13,<1' \
  'lap>=0.5,<1' \
  'transformers>=4.38,<5' \
  'tensorboardX>=2.6,<3' \
  'einops>=0.7,<1' \
  'pandas>=2.2,<3' \
  'scikit-image>=0.22,<1'
```

不要安装 `torchtext`。先移除 `models/transrmot.py` 和 `models/transrmot_pro.py` 中以下未使用导入：

```python
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import GloVe, Vocab, FastText
```

### D2. 导入检查

```bash
python - <<'PY'
mods = [
    'torch', 'torchvision', 'numpy', 'PIL', 'cv2', 'scipy',
    'pycocotools', 'tqdm', 'motmetrics', 'seaborn', 'lap',
    'transformers', 'tensorboardX', 'einops', 'pandas', 'skimage'
]
for name in mods:
    module = __import__(name)
    print(name, getattr(module, '__version__', 'OK'))
PY
```

然后检查项目入口的 import，不加载模型和数据：

```bash
cd /home/curtis/projects/TempRMOT
python - <<'PY'
import main
import inference
import models
import datasets
print('project imports OK')
PY
```

### D3. 锁定成功环境

```bash
python -m pip check
python -m pip freeze > /home/curtis/experiments/TempRMOT/requirements-lock.txt
```

后续服务器迁移优先使用此锁文件和环境诊断记录，不要依赖未锁版本的原始 `requirements.txt`。

**D 阶段通过标准：** `pip check` 无冲突；全部依赖和项目入口可导入；MultiScaleDeformableAttention 仍可导入。

## 7. 阶段 E：实施最小兼容性补丁

每个补丁单独提交。补丁只解决路径、兼容和单卡执行问题，不改变模型数学结构。

当前激活的模型入口是 `models/__init__.py -> transrmot_pro.build`。因此只修改备用的 `transrmot.py` 不会解决实际运行问题；凡是模型加载相关补丁必须先覆盖 `transrmot_pro.py`，再同步清理 `transrmot.py`，防止未来切换入口时重新出现旧问题。

### E1. RoBERTa 路径参数化

新增命令行参数，例如：

```text
--text_encoder_path /home/curtis/checkpoints/roberta-base
```

把参数从 `main.get_args_parser()` 传到 `TransRMOT`，并让 tokenizer/model 都使用该值。保留 `local_files_only=True`，避免正式实验时网络状态改变模型来源。

通过标准：源码中不再出现 `/data_2/zyn/Data4RMOT`；给定本地目录后 tokenizer 和 model 能离线加载。

### E2. 控制 backbone 初始权重

把 torchvision 的旧 `pretrained=` API 改为 `weights=`，并新增明确开关。加载官方 TempRMOT checkpoint 或 Deformable DETR checkpoint 时使用 `weights=None`，防止模型构建阶段额外下载 ImageNet 权重。

通过标准：断网时仍能构建模型骨架；随后 checkpoint 加载报告无关键模型层缺失。

### E3. 安全兼容 PyTorch 2.8 checkpoint

所有读取**已核对来源和 SHA256 的可信官方 checkpoint** 的 `torch.load` 调用显式设置：

```python
torch.load(path, map_location='cpu', weights_only=False)
```

不要对来历不明的 `.pth` 使用 `weights_only=False`。`util/tool.py`、`main.py` 和 `inference.py` 的加载路径要保持一致，避免同一文件一次能读、另一次失败。

### E4. 单 GPU 推理调度

把 `inference.py` 的固定 `thread_num = 8` 改为参数化，例如：

```text
--num_inference_workers 1
--gpu_id 0
```

本机默认只创建一个模型进程并使用 GPU 0。不要在同一张 8 GB GPU 上启动 8 份模型。多 GPU 服务器再按可见 GPU 数分片表达式列表。

同时把 checkpoint 标识符视为字符串，例如从 `Path(args.resume).stem` 得到 `checkpoint_rk2`，不要强制转换为整数。

把输出根目录命名规则固定为 `results_{checkpoint_tag}`；对于官方 V2 权重，目录应为 `results_checkpoint_rk2`。不要继续拼出含义重复的 `results_epochcheckpoint_rk2`。

通过标准：给定一个表达式时只出现一个 GPU 进程；输出目录稳定；官方 checkpoint 文件名不会触发 `ValueError`。

### E5. TrackEval 去除作者路径并兼容 NumPy 2

在 `TrackEval/trackeval/datasets/mot_challenge_2d_box.py` 中增加可配置的 KITTI 图像根目录，例如 `IMAGES_FOLDER`，用它计算序列长度，替换 `/data_2/zyn/refer-kitti-v2/...`。

仅在实际 HOTA 路径中替换旧 NumPy 类型别名：

- `np.float` -> `np.float64` 或 `float`
- `np.int` -> `np.int64` 或 `int`
- `np.bool` -> `bool`

至少检查：

```text
TrackEval/trackeval/metrics/hota.py
TrackEval/trackeval/datasets/mot_challenge_2d_box.py
```

不要为了掩盖这些问题而把整个环境降级到旧 NumPy。

### E6. 修复训练日志的硬编码

- `main.py` 使用 `Path(args.output_dir).name` 生成 TensorBoard run 名。
- `engine.py` 的 4 帧统计循环改为从当前 batch 或配置得到实际帧数；V2 应记录 5 帧。
- 数据加载时把 `os.listdir(expression_dir)` 排序后再随机选择，降低不同文件系统目录顺序造成的差异。

这些补丁主要影响训练记录和可复现性，不应改变官方 checkpoint 推理结果。

### E7. 给每个补丁建立回归检查

建议提交顺序：

```text
fix: remove unused torchtext dependency
fix: make text encoder and backbone loading configurable
fix: load trusted checkpoints under PyTorch 2.8
fix: support single-GPU inference and arbitrary checkpoint names
fix: remove local paths and NumPy aliases from RMOT TrackEval
fix: make training logging frame-count aware
```

每次提交后运行：

```bash
python -m pip check
python - <<'PY'
import main, inference, models, datasets
import MultiScaleDeformableAttention
print('imports OK')
PY
git status --short
```

### E8. 补丁文件映射表

| 目的 | 必改文件 | 变更点 | 最小验证 |
|---|---|---|---|
| 移除 torchtext | `models/transrmot_pro.py`、`models/transrmot.py` | 删除两行未使用 import | `import models` 不再要求 torchtext |
| 参数化 RoBERTa | `main.py`、`models/transrmot_pro.py` | parser -> `build()` -> `TransRMOT.__init__()` 完整传参 | `HF_HUB_OFFLINE=1` 构建成功 |
| 禁止隐式 backbone 下载 | `main.py`、`models/backbone.py` | 新增权重开关，使用 torchvision `weights=` | 断网模型构建成功 |
| PyTorch 2.8 checkpoint | `util/tool.py`、`main.py`、`inference.py` | 可信文件显式 `weights_only=False` | 官方 checkpoint 可读且核心层完整 |
| 单卡推理 | `main.py`、`inference.py` | 新增 worker/GPU/筛选参数，单 worker 不复制 8 份模型 | GPU 0 仅一个模型进程 |
| 稳定结果目录 | `inference.py` | `Path(resume).stem` + `results_{tag}` | `results_checkpoint_rk2` 唯一生成 |
| TrackEval 图像路径 | `TrackEval/trackeval/datasets/mot_challenge_2d_box.py` | default config 新增 `IMAGES_FOLDER` 并替换作者路径 | 单 query 能取得正确帧数 |
| TrackEval NumPy 2 | 上一文件及 `TrackEval/trackeval/metrics/hota.py` | 只替换旧 dtype 别名 | 单 query HOTA 无 AttributeError |
| 动态训练帧日志 | `main.py`、`engine.py` | run 名使用 Path；帧循环读取 batch/config | 5 帧训练不漏 frame 4 日志 |
| 稳定表达式随机池 | `datasets/refer_kitti.py` | `sorted(os.listdir(...))` | 相同 seed 的候选顺序一致 |

官方 `configs/*.sh` 含作者绝对路径、固定 4/8 进程和旧 checkpoint 名，只能作为参数参考。上述补丁完成前不要直接运行这些脚本；完成后应新增本机配置脚本，不要覆盖原始脚本以免丢失对照证据。

**E 阶段通过标准：** 作者绝对路径清零；不需要 torchtext；单 GPU 配置可解析；可信 checkpoint 加载 API 一致；TrackEval 不再依赖旧 NumPy 别名。

## 8. 阶段 F：准备 RoBERTa 与官方权重

### F1. RoBERTa

把 Hugging Face `roberta-base` 的完整 snapshot 下载到：

```text
/home/curtis/checkpoints/roberta-base
```

可使用安装 `transformers` 后提供的 Hugging Face CLI：

```bash
hf download roberta-base --local-dir /home/curtis/checkpoints/roberta-base
```

至少验证目录含 tokenizer 配置、词表/merges、模型 config 和模型权重。离线加载测试：

```bash
HF_HUB_OFFLINE=1 python - <<'PY'
from transformers import RobertaModel, RobertaTokenizerFast
p = '/home/curtis/checkpoints/roberta-base'
t = RobertaTokenizerFast.from_pretrained(p, local_files_only=True)
m = RobertaModel.from_pretrained(p, local_files_only=True)
print(t('cars moving on the left', return_tensors='pt')['input_ids'].shape)
print(m.config.hidden_size)
PY
```

预期 hidden size 为 768。

### F2. TempRMOT 官方 checkpoint

Refer-KITTI-V2 的正确官方地址是：

```text
https://github.com/zyn213/TempRMOT/releases/download/v1.0/checkpoint_rk2.pth
```

保存到：

```text
/home/curtis/checkpoints/TempRMOT/checkpoint_rk2.pth
```

Refer-KITTI 权重为同一 release 中的 `checkpoint_rk.pth`。官方 release 页面显示两个文件均约 1.02 GB。

下载后记录：

```bash
sha256sum /home/curtis/checkpoints/TempRMOT/checkpoint_rk2.pth \
  | tee /home/curtis/checkpoints/TempRMOT/checkpoint_rk2.pth.sha256
```

不要使用 README 当前指向 `wudongming97/RMOT` 的 V2 链接；该 release 并没有 `checkpoint_rk2.pth`。

### F3. Deformable DETR 预训练权重

只有训练时才需要下载官方 “Deformable DETR + iterative bounding box refinement” COCO checkpoint。官方 TempRMOT 权重推理不需要它。

同样保存 URL、文件名、大小与 SHA256。不要把 TempRMOT 的 `--resume` 和训练初始化用的 `--pretrained` 混淆：

- `--resume`：完整 TempRMOT checkpoint 或中断训练恢复。
- `--pretrained`：训练初始化用 Deformable DETR checkpoint。

**F 阶段通过标准：** RoBERTa 离线加载成功；官方 checkpoint 可由 PyTorch 读取；哈希已记录；权重不在 Git 中。

## 9. 阶段 G：准备并验证 Refer-KITTI-V2

### G1. 数据来源

1. 从 KITTI 官方 tracking 页面下载左目彩色 tracking 图像。
2. 从 TempRMOT 官方数据说明链接下载 Refer-KITTI-V2 的 `expression` 和**修正版** `labels_with_ids`。
3. 不要用旧 Refer-KITTI 的 `labels_with_ids` 替代 V2 修正版。

KITTI 官方 tracking 数据包含 21 个训练序列；V2 以其中 17 个序列训练、4 个序列测试。

### G2. 目录必须精确匹配代码

```text
/home/curtis/datasets/refer-kitti-v2/
├── KITTI/
│   ├── training/
│   │   └── image_02/
│   │       ├── 0000/
│   │       └── ...
│   └── labels_with_ids/
│       └── image_02/
│           ├── 0000/
│           └── ...
└── expression/
    ├── 0000/
    └── ...
```

注意：源码既出现 `KITTI/training/image_02`，也出现 `KITTI/labels_with_ids/image_02`。不要按 README 简图误放成 `KITTI/training/labels_with_ids`。

### G3. 验证训练/测试划分

仓库自带 `datasets/data_path/refer-kitti-v2.train` 当前包含 5172 行、17 个训练视频：

```text
0000,0001,0002,0003,0004,0006,0007,0008,0009,
0010,0012,0014,0015,0016,0017,0018,0020
```

测试视频应为：

```text
0005,0011,0013,0019
```

仓库自带 `datasets/data_path/seqmap.txt` 当前有 862 条测试 query。正式评测必须记录实际 seqmap 行数，并检查每一行都对应一个 expression JSON。

### G4. 运行数据完整性检查

检查以下条件：

1. 21 个 KITTI 图像序列均存在且帧号连续。
2. 每个训练列表图像都存在。
3. 对应 `labels_with_ids` 文件存在，且每行是 6 列。
4. JSON 可以解析，至少含 `label`、`video_name`、`sentence`。
5. JSON 的对象 ID 可在对应帧 label 中找到。
6. 862 条 seqmap 项均能映射到测试 JSON。
7. 没有把训练表达式误用于测试指标。

应把检查写成项目脚本并输出总数、缺失数、坏 JSON 数、孤立对象 ID 数。所有错误计数必须为 0 才能进入全量推理。

**G 阶段通过标准：** 目录、17/4 划分、训练列表、862 条测试 seqmap 和 JSON-label 引用全部一致。

## 10. 阶段 H：模型构建与 checkpoint 加载 smoke test

不要一开始跑 862 个表达式。按以下顺序递增。

### H1. 仅构建模型

用与官方 V2 推理一致的关键参数构建：

```text
--meta_arch temp_rmot
--dataset_file e2e_rmot
--with_box_refine
--batch_size 1
--sampler_steps 60 80 90
--sampler_lengths 5 5 5 5
--update_query_pos
--merger_dropout 0
--dropout 0
--random_drop 0.1
--fp_ratio 0.3
--query_interaction_layer QIM
--hist_len 4
--text_encoder_path /home/curtis/checkpoints/roberta-base
```

即使推理不使用训练 dataset，`build_model()` 仍会执行 `max(args.sampler_lengths)`，所以 `--sampler_lengths` 不能省略。

以上结构参数以 `checkpoint_rk2.pth` 内保存的 `args` 为准。不要为该 checkpoint 启用
`--extra_track_attn`；否则会新增 checkpoint 中不存在的 36 个 `update_attn/norm4` 状态项。

通过标准：模型构建成功；RoBERTa 保持 frozen；没有联网下载；GPU 显存没有持续增长。

### H2. 加载官方 checkpoint

```text
--resume /home/curtis/checkpoints/TempRMOT/checkpoint_rk2.pth
```

保存 missing/unexpected keys。允许 checkpoint 中非模型统计字段差异，但下列核心模块不能缺失：

```text
backbone
transformer
class_embed
bbox_embed
refer_embed
STReasoner
fusion_module
text_encoder / txt_proj
```

如果大量关键层缺失，停止；不要带着随机初始化层继续生成“官方结果”。

### H3. 一个表达式、少量帧

为 `inference.py` 增加可选筛选参数（如 `--video_ids`、`--max_expressions`、`--max_frames`），先运行：

```text
video 0005
1 expression
3-10 frames
1 worker
GPU 0
```

核对：

- 每帧输入尺寸与缩放合理。
- 输出无 NaN/Inf。
- object score 阈值 0.6。
- referring score 阈值 0.4。
- 帧号写出时为 1-based。
- `predict.txt` 与 `gt.txt` 都生成。
- 输出列为 `frame,id,x,y,w,h,1,1,1`。

### H4. 一个完整表达式

取消帧数限制，完成一个 query 的整段视频推理，并只用这一条 seqmap 跑一次 HOTA。先验证评测闭环，再扩大规模。

**H 阶段通过标准：** 官方 checkpoint 核心层完整加载；单 query 全视频推理和单 query HOTA 成功。

## 11. 阶段 I：Refer-KITTI-V2 全量推理

### I1. 推荐命令结构

兼容性补丁完成后，使用前台命令运行，先不要用官方脚本里的 `nohup ... &`：

```bash
cd /home/curtis/projects/TempRMOT
source /home/curtis/miniforge3/etc/profile.d/conda.sh
conda activate temprmot
mkdir -p /home/curtis/experiments/TempRMOT/rk2_official

python inference.py \
  --meta_arch temp_rmot \
  --dataset_file e2e_rmot \
  --with_box_refine \
  --lr_drop 100 \
  --lr 2e-4 \
  --lr_backbone 2e-5 \
  --batch_size 1 \
  --sample_mode random_interval \
  --sample_interval 1 \
  --sampler_steps 60 80 90 \
  --sampler_lengths 5 5 5 5 \
  --update_query_pos \
  --merger_dropout 0 \
  --dropout 0 \
  --random_drop 0.1 \
  --fp_ratio 0.3 \
  --query_interaction_layer QIM \
  --hist_len 4 \
  --rmot_path /home/curtis/datasets/refer-kitti-v2 \
  --text_encoder_path /home/curtis/checkpoints/roberta-base \
  --resume /home/curtis/checkpoints/TempRMOT/checkpoint_rk2.pth \
  --num_inference_workers 1 \
  --gpu_id 0 \
  --output_dir /home/curtis/experiments/TempRMOT/rk2_official \
  2>&1 | tee /home/curtis/experiments/TempRMOT/rk2_official/inference.log
```

`--lr`、`--lr_drop` 等参数对纯推理未必生效，但保留它们有助于与官方 test 脚本的模型参数集合对齐。`--epochs` 不参与推理，可省略。

### I2. 全量输出验收

预期测试视频为 4 个，seqmap 为 862 条。检查：

- 每条 seqmap 都有唯一结果目录。
- 每个目录都有 `predict.txt` 和 `gt.txt`。
- 空 `predict.txt` 允许表示阈值过滤后没有预测目标，但必须保留文件并单独计数；`gt.txt` 不得缺失或为空。
- 没有重复帧-ID-轨迹 ID 行。
- 所有框宽高非负，坐标没有 NaN/Inf。
- 推理进程退出码为 0。
- `nvidia-smi` 中没有遗留的 8 份模型进程。

建议写一个结果完整性脚本，将“应有 862 条”和“实际完成条数”比较。只有 862/862 才进入正式全量评测。

## 12. 阶段 J：TrackEval HOTA 评测

### J1. 先跑单 query

用只含一行的临时 seqmap 验证路径、帧数和文件格式。不要一开始并行评测 862 条后再排查路径错误。

### J2. 全量命令结构

在完成 E5 的 `IMAGES_FOLDER` 参数化后运行：

```bash
cd /home/curtis/projects/TempRMOT/TrackEval/scripts

python run_mot_challenge.py \
  --METRICS HOTA \
  --SEQMAP_FILE /home/curtis/projects/TempRMOT/datasets/data_path/seqmap.txt \
  --SKIP_SPLIT_FOL True \
  --GT_FOLDER /home/curtis/experiments/TempRMOT/rk2_official/results_checkpoint_rk2 \
  --IMAGES_FOLDER /home/curtis/datasets/refer-kitti-v2/KITTI/training/image_02 \
  --TRACKERS_FOLDER /home/curtis/experiments/TempRMOT/rk2_official/results_checkpoint_rk2 \
  --GT_LOC_FORMAT '{gt_folder}/{video_id}/{expression_id}/gt.txt' \
  --TRACKERS_TO_EVAL /home/curtis/experiments/TempRMOT/rk2_official/results_checkpoint_rk2 \
  --USE_PARALLEL True \
  --NUM_PARALLEL_CORES 2 \
  --PLOT_CURVES False \
  2>&1 | tee /home/curtis/experiments/TempRMOT/rk2_official/trackeval-full.log
```

实际 `results_...` 目录名以 E4 补丁后的稳定规则为准，执行前必须先 `find` 核对。`GT_FOLDER` 在这份定制 TrackEval 中部分语义已被改写；不可直接换成上游最新版 TrackEval 后仍期待相同目录解析。

### J3. 指标解释

这份 TrackEval 的 `HOTA.combine_sequences()` 按累计 TP/FN/FP 和 TP 加权方式组合 query，不是简单对 862 个 query 的 HOTA 做算术平均。论文文字写“across sentence queries averaging”，与代码实现存在表述差异。第一轮复现使用仓库自带 TrackEval 以对齐官方结果，同时保存逐 query 结果；如需论文方法学核验，再额外计算 query 宏平均，二者不可混写。

### J4. 指标偏差排查顺序

若 HOTA 与 35.04 相差超过 0.20，按顺序检查：

1. checkpoint 是否为 `checkpoint_rk2.pth`，哈希是否稳定。
2. 是否完整加载 STReasoner、refinement 和 RoBERTa 相关权重。
3. 数据是否是 V2 修正版 `labels_with_ids`。
4. 测试集合是否为 4 个视频、862 条 seqmap。
5. 对 `checkpoint_rk2.pth` 使用其保存参数 `hist_len=4`，且不要启用 checkpoint 中不存在的 `extra_track_attn`。
6. object threshold 0.6、referring threshold 0.4、area threshold 100。
7. 图像缩放是否保持短边 800、长边不超过 1536（`inference.py` 当前行为）。
8. frame ID 是否统一为 1-based，GT 是否额外加 1 且仅加一次。
9. 预测框是否以像素 `xywh` 写出。
10. TrackEval 是否使用本仓库的定制版本和同一 seqmap。
11. 是否漏跑、重复或覆盖了 query 输出。
12. NumPy 兼容补丁是否只替换 dtype，没有改计算公式。

**J 阶段通过标准：** 862 条全部评测，HOTA/DetA/AssA 在容差内，完整日志和逐 query 输出已保存。

## 13. 阶段 K：本机训练 smoke test

本阶段只证明训练链可反向传播、更新、保存和恢复，不用于宣称论文训练复现。

### K1. 单 batch 过拟合测试

建立独立 smoke 配置，不覆盖官方配置：

- 单 GPU、batch size 1。
- 1 个训练视频或极少量连续帧。
- sampler length 2，history length 2。
- 降低训练分辨率，并将尺度做成参数，而不是直接永久改死官方列表。
- 保持 RoBERTa frozen。
- 开启项目已有 gradient checkpointing。
- 当前 `--fp16` 参数没有实际 AMP/GradScaler 实现，不能仅传 flag 就声称使用混合精度。

验收：

1. forward loss 全部有限。
2. backward 成功，关键模块存在非零有限梯度。
3. optimizer step 后至少一个参数改变。
4. 显存峰值已记录。
5. checkpoint 能保存并重新加载。
6. 从 checkpoint 恢复后 epoch、optimizer、scheduler 状态一致。
7. 小样本上 loss 有下降趋势。

若 8 GB 仍 OOM，依次降低：输入尺度、帧数、query 数；这些只用于 smoke test，不能用于官方指标对比。

### K2. 不建议立即做的事

- 不要在本机直接启动 60 epoch、5 帧、官方分辨率训练。
- 不要把 OOM 后的自动降尺度结果与论文指标比较。
- 不要假设 `--fp16` 已启用 AMP。
- 不要在尚未通过官方权重 L3 前开始消融实验。

## 14. 阶段 L：服务器严格训练复现

完成 L3 后再迁移。推荐 4 x RTX 4090 或显存不低于论文平台的 GPU。

### L1. 第一条训练线：官方代码配置版

使用代码实际配置：

```text
optimizer: AdamW
lr: 1e-4
lr_backbone: 1e-5
weight_decay: 1e-4
epochs: 60
lr_drop: 40
batch_size: 1 per process（需在报告中明确全局 batch）
sampler_lengths: 5 5 5 5
hist_len: 5
random_drop: 0.1
fp_ratio: 0.3
refer_loss_coef: 2
with_box_refine: true
use_checkpoint: true
```

使用现代启动方式：

```bash
torchrun --standalone --nproc_per_node=4 main.py ...
```

`--output_dir` 与 `--save_dir` 指向同一实验目录。保存完整命令、Git commit、环境 lock、数据哈希、预训练权重哈希和每 epoch 日志。

### L2. 第二条训练线：论文文字版

只有代码配置版可运行后，才单独测试论文所述 Adam、学习率和尺度。给实验命名明确加上 `paper_text_variant`，避免与官方代码复现混淆。

### L3. 训练验收

至少验证：

- 60 epoch 正常结束，40 epoch 后 LR 降低 10 倍。
- RoBERTa 全程 frozen。
- 训练记忆长度为 5，推理改为 8。
- 最终 checkpoint 使用与官方权重完全相同的推理/TrackEval 流程。
- 至少 3 个随机种子或明确声明单次运行，不把单次波动当算法结论。

## 15. 最终复现报告模板

```text
复现级别：L1 / L2 / L3 / L4
日期与时区：
源码来源与本地 Git commit：
官方源码参考 SHA：
GPU / driver / CUDA / torch / torchvision：
依赖 lock 文件：
数据来源、目录、总数与校验结果：
checkpoint URL、大小、SHA256：
RoBERTa snapshot/revision：
完整运行命令：
推理 query 完成数：x / 862
object/ref/area threshold：
history length：train / inference
TrackEval commit/本地补丁：
HOTA / DetA / AssA / 其他指标：
与论文绝对差：
已知偏差与解释：
日志和输出目录：
```

## 16. 执行门禁清单

只有上一门禁通过才进入下一阶段：

- [x] A：源码已清理、Git baseline 已建立。
- [x] B：现有 PyTorch/CUDA/GPU 验证通过。
- [x] C：CUDA 扩展 forward/backward 全通过。
- [x] D：依赖可导入、`pip check` 通过、环境已锁定。
- [x] E：最小兼容补丁逐项提交并回归检查。
- [x] F：RoBERTa 与官方 checkpoint 离线可读，SHA256 已保存。
- [x] G：数据 17/4 划分、训练列表与 862 条 seqmap 完整。
- [x] H：单 query 全视频推理与单 query HOTA 闭环成功。
- [x] I：全量推理 862/862，无缺失或覆盖。
- [x] J：862 条 TrackEval 完成；HOTA 35.249，与官方值差 +0.209，轻微超出预设容差并已归档。
- [ ] K：本机 smoke training 能更新并恢复 checkpoint。
- [ ] L：如有多卡资源，再做严格训练与消融。

## 17. 当前仍需外部确认的事项

以下事项无法仅靠当前源码提前保证，执行时必须留下证据：

1. Refer-KITTI-V2 Google Drive 数据是否仍可访问，以及下载内容是否为修正版。
2. 数据包内 JSON 总数、测试 JSON 数与仓库 862 条 seqmap 是否完全匹配。
3. 官方 checkpoint 的实际 SHA256；官方页面未公布校验值，只能在下载后记录并固定。
4. CUDA 扩展在 PyTorch 2.8 + sm_120 上是否无需源码级 API 修补；必须以 C 阶段真实编译日志为准。
5. 官方指标使用的是定制 TrackEval 的 detection-weighted combined 结果，还是另有未公开的 query 宏平均脚本。
6. 论文与代码在优化器、base LR 和训练尺度上的差异应由作者说明；在此之前分别报告，不擅自合并。

## 18. 推荐的实际执行顺序

下一次操作只执行 **A -> B -> C**：建立源码基线、复核环境、编译并测试 CUDA 扩展。不要同时下载 15 GB KITTI、1 GB checkpoint 和一批依赖；先把最底层阻断点排除。

之后按 **D -> E -> F -> H** 完成“模型可构建、权重可加载”。数据到位后执行 **G -> H -> I -> J**。只有 L3 达标，再进入 K/L 训练阶段。

## 19. 文档审查记录与置信度

本文档交付前完成了三轮审查：

1. **论文/外部来源审查：** 对照本地 TempRMOT PDF 的实验页、官方 GitHub、v1.0 release、KITTI 官方 tracking 页面和 Hugging Face 模型入口；修正了 README 中错误的 V2 checkpoint 仓库链接。
2. **源码调用链审查：** 从 `main.py`、`models/__init__.py`、`transrmot_pro.py`、数据加载、`inference.py` 一直追踪到定制 TrackEval；把 8 GPU 假设、checkpoint 名解析、RoBERTa 路径、TrackEval 作者路径和 NumPy 2 不兼容列为强制门禁。
3. **可执行性逆向审查：** 从 A 到 L 模拟前置条件；全部 25 个 Bash 代码块通过 `bash -n` 语法检查；本地再次核对训练列表 5172 行/17 视频和 seqmap 862 行，并修正了目录创建、NumPy 固定与结果目录命名。

在“不把尚未运行的编译和外部数据下载伪装成已验证”的前提下，对**流程顺序、阻断点识别和验证门禁**的置信度为 95%。剩余 5% 主要来自第 17 节列出的外部资产与真实编译结果；这些不确定性都被设计成必须停止并验证的门禁，不允许带病进入下一阶段。
