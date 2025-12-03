逐步目标：
1. 校验 H36M 数据是否符合 `data/h36m/S*/动作.txt` 结构，并运行 `python train.py --exp-name logs.txt` 的 dry-run（少量迭代）确认加载正常。
2. 使用仓库自带的 `checkpoints/h36m_model_40000.pth` 执行 `python test.py --model-pth ../../checkpoints/h36m_model_40000.pth`，记录评估指标作为基准。
3. 根据需要调整 `train.py` 中的超参数（如 `--num`, `--layer-norm-axis`）在 CPU 上长时间训练，保存最新模型到 `exps/baseline_h36m/log/snapshot/`。
4. 设计真实视频→3D骨架提取流程（例如调用 VideoPose3D、VIBE 等工具），将结果整理成与 H36M 相同的 22 关节顺序和尺度。
5. 编写新的 `Dataset` 适配器或脚本把提取到的骨架序列转换为模型输入张量，并在 `test.py` 基础上实现预测与可视化。
6. 如果未来需要更丰富的动作泛化，再下载 AMASS 数据，参考 `exps/baseline_amass` 训练流程扩展模型至全身 SMPL 关节。
