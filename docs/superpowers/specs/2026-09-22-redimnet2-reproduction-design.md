# ReDimNet2 VoxCeleb2 复现设计

## 目标

在当前工作区提供一套可重复运行的 VoxCeleb2 + WeSpeaker + ReDimNet2 复现入口，先以 B3 作为默认实验，再支持 B6 扩展。工程本身不复制上游 WeSpeaker 源码，而是固定上游 commit 并在首次运行时拉取。

## 方案

- `scripts/bootstrap_wespeaker.sh` 固定拉取 `wenet-e2e/wespeaker` 的已验证 commit。
- `configs/` 保存与当前 WeSpeaker ReDimNet2 实现匹配的 B3/B6 PTN 和 LM 配置。
- `scripts/run_experiment.py` 统一执行 bootstrap、数据准备、训练和评测，并把本工程路径转换为上游 VoxCeleb recipe 参数。
- `src/redim_repro/data_validation.py` 只使用 Python 标准库检查 Kaldi manifest、说话人映射和 VoxCeleb1 trial 引用，避免数据准备完成后才在训练中失败。
- `tests/` 覆盖 manifest 校验和上游命令构造，不依赖 GPU 或音频数据。

## 训练协议

- PTN：2 秒片段、MUSAN/RIRS 增强、0.9/1.1 speed perturbation、SphereFace2-C margin 0 到 0.2、120 epochs。
- LM：6 秒片段、关闭 speed perturbation 和在线增强、SphereFace2-C margin 0.3、5 epochs。
- 输入：72-bin TFMel，25 ms window、10 ms hop、512 FFT、20--7600 Hz、pre-emphasis 和 signal normalization；这是当前 WeSpeaker 官方 recipe 的 executable configuration。
- 评测：VoxCeleb1 cleaned O/E/H，cosine score，输出 EER/minDCF；AS-Norm 单独作为可选比较，不混入默认 ReDimNet2 结果。

## 不在本次范围内

- 不自动下载需要授权的 VoxCeleb 数据到工作区。
- 不在无 GPU/无数据的环境中伪造训练结果。
- 不把论文正文的 80-bin frontend 与官方 72-bin checkpoint 混用；严格 80-bin 版本作为后续独立实验。
