# siMLPe 项目结构解析

## 顶层概览
- `exps/`：包含针对 Human3.6M 与 AMASS/3DPW 数据集的实验配置、训练与测试脚本。
- `lib/`：项目的共享库代码，涵盖数据集封装、骨骼模型与数学工具函数。
- `body_models/`：预处理骨架模板（如 `smpl_skeleton.npz`），用于角度到关节坐标的转换。
- `checkpoints/`：模型权重的存放目录（需自行准备）。
- `data/`：数据集根目录，按 README 所示组织源数据。
- `next_steps.md`：项目后续计划/笔记。

以下按目录给出所有 Python 文件的作用与其主要函数/类说明。

## `lib` 目录

### `lib/datasets`
- `h36m.py`
  - `class H36MDataset`：Human3.6M 训练集封装。
    - `__init__`：读取配置、初始化关节点、加载序列并构建索引。
    - `_get_h36m_files`：读取 `.txt` 动作文件，将指数映射转换为 3D 关节坐标，只保留指定关节。
    - `_collect_all`：统一采样率、展平帧，生成滑动窗口索引。
    - `__getitem__`：按索引返回输入/目标序列，并执行数据增强与归一化。
- `h36m_eval.py`
  - `class H36MEval`：Human3.6M 测试集封装。
    - `_get_h36m_files`：读取测试动作列表并调用 `_preprocess`。
    - `_preprocess`：指数映射 → 旋转矩阵 → 3D 坐标 → 子采样。
    - `__getitem__`：返回评估用的输入序列与预测目标。
- `amass.py`
  - `class AMASSDataset`：AMASS 训练集 Dataset。
    - `_get_amass_names`：收集训练/测试文件名。
    - `_load_skeleton`：加载 SMPL 骨架基准。
    - `_load_all`：逐文件载入姿态，统一帧率并转为 3D 坐标。
    - `_preprocess`：随机截取窗口，拼接输入与目标片段。
    - `__getitem__`：应用随机反转增强并输出 (input, target)。
- `amass_eval.py`
  - `class AMASSEval`：AMASS 测试集 Dataset。
    - `_get_amass_names`：读取测试拆分。
    - `_load_skeleton`：同上。
    - `_collect_all`：按固定步长生成评估索引。
    - `__getitem__`：输出评估样本。
- `pw3d_eval.py`
  - `class PW3DEval`：3DPW 测试数据封装。
    - `_get_pw3d_names`：列出所有测试序列。
    - `_load_skeleton`：加载 SMPL 骨架并裁剪到 22 个关节。
    - `_collect_all`：遍历序列，转换角度为关节位置并生成滑动窗口。
    - `__getitem__`：提供输入/目标序列。
- `__init__.py`：空文件，占位以标记包。

### `lib/utils`
- `angle_to_joint.py`
  - `ang2joint(p3d0, pose, parent)`：基于罗德里格斯公式将轴角表示转换为关节位置。
  - `rodrigues(r)`：批量实现罗德里格斯旋转。
  - `with_zeros(x)`：在 3×4 仿射矩阵后附加齐次行。
- `logger.py`
  - `get_logger(file_path, name)`：构建带文件输出的 logger。
  - `print_and_log_info(logger, string)`：快捷写日志。
- `pyt_utils.py`
  - `extant_file(x)`：argparse 文件存在性校验。
  - `link_file(src, target)`：创建符号链接（已有目标先移除）。
  - `ensure_dir(path)`：递归创建目录。
  - `_dbg_interactive(var, value)`：IPython 交互调试入口。
- `misc.py`
  - `_some_variables_cmu()` / `_some_variables()`：返回不同骨架设置下的父节点、偏移和指数映射索引。
  - `fkl_torch(rotmat, parent, offset, rotInd, expmapInd)`：前向运动学，旋转矩阵 → 关节坐标。
  - `rotmat2euler_torch`, `rotmat2quat_torch`, `expmap2quat_torch`, `expmap2rotmat_torch`：姿态表示互转工具。
  - `rotmat2xyz_torch`, `rotmat2xyz_torch_cmu`：将旋转矩阵批量转换成 3D 坐标。
  - `find_indices_256`, `find_indices_srnn`：复现 SRNN 采样策略的帧索引生成器。
- `h36m_human_model.py`
  - `class H36MHuman`：Human3.6M 前向运动学模块。
    - `_some_variables`：初始化父节点和偏移。
    - `forward(rotmat)`：从旋转矩阵恢复 3D 关节位置。
- `__init__.py`：空文件。

## `exps` 目录

### `exps/baseline_h36m`
- `config.py`：Human3.6M 实验配置。
  - 定义随机种子、目录、日志路径。
  - `C.motion` 子配置描述输入/目标序列长度、维度以及模型结构参数。
- `mlp.py`：MLP 主干网络组件。
  - `class LN / LN_v2`：分别在空间维、时间维执行 LayerNorm。
  - `Spatial_FC` / `Temporal_FC`：切换空间或时间维度上的全连接。
  - `MLPblock`：单层残差 MLP 块，可选归一化与空间/时间 FC。
  - `TransMLP`：堆叠多个 `MLPblock`。
  - `build_mlps(args)`：根据配置返回 `TransMLP` 实例。
  - `_get_activation_fn`, `_get_norm_fn`：未使用的辅助构造器。
- `model.py`
  - `class siMLPe`：预测网络主体。
    - `__init__`：复制配置，构建输入/输出线性层与堆叠 MLP。
    - `reset_parameters`：对输出层做 Xavier 初始化。
    - `forward(motion_input)`：根据设定在时间/空间维度进行 FC、MLP、再还原形状。
- `train.py`
  - 命令行解析器允许调整网络结构/正则。
  - `get_dct_matrix(N)`：生成 DCT/IDCT 矩阵用于导数域处理。
  - `update_lr_multistep`：分段式学习率调整。
  - `gen_velocity`：计算相邻帧速度，供相对损失使用。
  - `train_step`：单次迭代，包含 DCT 预处理、前向传播、MPJPE 与速度损失计算、反向传播及日志记录。
  - 主流程：初始化数据集与 DataLoader、构建模型与优化器、按迭代训练并定期保存 checkpoint、调用 `test` 进行评估。
- `test.py`
  - `get_dct_matrix`：同训练脚本。
  - `regress_pred(model, pbar, ...)`：自回归生成未来帧，输出 MPJPE。
  - `test(config, model, dataloader)`：整合评估流程，返回不同预测步长上的误差。
  - CLI 入口：加载权重、构建 `H36MEval` 数据集并打印评估结果。

### `exps/baseline_amass`
- `config.py`：AMASS/3DPW 实验参数（目录、输入输出长度、损失配置等）。
- `mlp.py`：与 `baseline_h36m/mlp.py` 相同的模块化实现。
- `model.py`：`class siMLPe`，逻辑与 H36M 版本一致，但引用 AMASS 参数。
- `train.py`
  - 与 H36M 训练脚本相似，针对 AMASS 配置调用 `AMASSDataset`。
  - `train_step`：处理导数域输入、IDCT 还原、MPJPE 及速度损失、TensorBoard 日志。
  - 主循环：迭代 DataLoader、保存 checkpoint。
- `test.py`
  - `regress_pred`：自回归生成并累计 MPJPE。
  - `test(model, dataloader)`：输出各步长误差。
  - CLI：加载模型、以 `AMASSEval` 构建 DataLoader 后评估。
- `test_3dpw.py`
  - 结构与 `test.py` 相同，但使用 `PW3DEval` 数据集与 3DPW 的输入/输出长度。
  - 加载的模型类名为 `BackToMLP`（需在模型文件提供，当前代码引用 `model.py` 中的实现）。

## 使用建议
- 训练前检查 `config.py` 中的数据路径是否与本机匹配。
- AMASS 与 H36M 共用的工具位于 `lib/utils`，若需扩展新数据集，可参考现有 Dataset 架构。
- 所有核心预测逻辑集中在 `exps/*/model.py` 与 `exps/*/mlp.py`，适合进一步改造模型结构。

