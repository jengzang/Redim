# ReDimNet2 + WeSpeaker VoxCeleb2 复现

这个目录用 VoxCeleb2-dev 训练 WeSpeaker 的 ReDimNet2。服务器工作流直接读取已有的 `dev.spk2utt` / `test.spk2utt` 索引：dev 用于训练，test 自动生成固定验证 pairs，并在每个 epoch 计算 EER/minDCF。模型主体和训练超参数来自固定版本的上游 WeSpeaker；本项目负责索引转换、DDP 启动和逐 epoch 评测。

## 1. 环境

需要 Python 3.9+、Git、ffmpeg、unzip、wget，以及与本机 CUDA 匹配的 PyTorch/torchaudio。先安装 PyTorch，再执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

bash scripts/bootstrap_wespeaker.sh
python -m pip install -r .deps/wespeaker/requirements.txt
```

bootstrap 会固定到 WeSpeaker commit `9fecd6cb4f47475d01761d87c826298dff4ef18c`。如果希望把依赖放在别处，可以设置 `WESPEAKER_DIR=/path/to/wespeaker`。

## 2. 服务器 VoxCeleb2（推荐）

你的服务器数据入口是：

```text
/public/share/smiip/lili345/corpus/voxceleb2/
├── dev/
├── test/
├── dev.spk2utt
└── test.spk2utt
```

两个 `*.spk2utt` 都是两列索引，每行形如：

```text
vox2_id04627 test/audio/id04627/9iu-egicBNg/00022.wav
```

第一列是说话人标签，第二列是相对 corpus root 的音频路径。它们不是传统 Kaldi 的“一个 speaker 后跟多个 utterance”的单行格式，但完全可以作为权威索引：本项目不递归扫描百万音频，而是从这两个文件生成标准 `wav.scp`、`utt2spk`、`spk2utt` 和 `raw.list`。

`test.spk2utt` 还会生成固定的平衡验证 trial 文件。默认每个 test speaker 抽取 300 个同人 target 和 300 个跨人 nontarget pair，种子固定为 42；全程都使用同一份 trial，因而每个 epoch 的 EER 可以比较。它是用于选模型的 validation set；若要报告严格最终结果，应另保留从未参与选 epoch 的评测集。

先准备一次索引。`--check-files` 会逐条检查服务器音频是否存在，网络文件系统上可能需要几分钟：

```bash
python scripts/run_experiment.py prepare-server-vox2 \
  --corpus /public/share/smiip/lili345/corpus/voxceleb2 \
  --data /path/to/redim_vox2_prepared \
  --data-type raw --pairs-per-speaker 300 --seed 42 --check-files

python scripts/run_experiment.py validate-server-vox2 \
  --data /path/to/redim_vox2_prepared --check-files
```

上述命令产生：

```text
/path/to/redim_vox2_prepared/
├── vox2_dev/{wav.scp,utt2spk,spk2utt,raw.list}
└── vox2_test/{wav.scp,utt2spk,spk2utt,raw.list,trials/vox2_test_fixed.{kaldi,json}}
```

如果 raw 文件在共享盘上的随机读取速度不够，再生成训练 shards；这一步仍只读取索引中的 dev 音频，不扫描目录：

```bash
export WESPEAKER_DIR=/opt/wespeaker
python scripts/run_experiment.py bootstrap
python scripts/run_experiment.py prepare-server-vox2 \
  --skip-bootstrap --corpus /public/share/smiip/lili345/corpus/voxceleb2 \
  --data /path/to/redim_vox2_prepared --data-type shard --shard-workers 16
```

PTN 仍需要 MUSAN/RIRS 增强 LMDB；本项目不会臆测它们在服务器上的位置。若它们不在 prepared root 的 `musan/lmdb`、`rirs/lmdb`，训练时显式传入路径：

```bash
python scripts/run_experiment.py train \
  --skip-bootstrap --server-vox2 --model b3 --phase ptn \
  --data /path/to/redim_vox2_prepared --data-type raw --gpus 0,1 \
  --musan-lmdb /path/to/musan/lmdb --rirs-lmdb /path/to/rirs/lmdb \
  --batch-size 128 --num-workers 16 --prefetch-factor 8 --amp --pin-memory
```

每张卡的 `batch-size` 为 128；两张卡的全局 batch 为 256。训练会在 `exp/redimnet2_b3/ptn/epoch_eval/metrics.jsonl` 逐行写入 EER/minDCF，`best.json` 记录最优 epoch，实验根目录的 `best_model.pt` 指向最低 EER（同 EER 时最低 minDCF）的 checkpoint。该链接刻意不放在 `models/` 下，以免被上游 checkpoint averaging 误识别。B6 只需改为 `--model b6` 并按显存下调 batch size。

默认每个 epoch 都评测完整固定的 Vox2-test trial。若测试集完整前向成为瓶颈，PTN 可用 `--eval-interval 5` 改为每 5 个 epoch；最后一个 epoch 无论如何都会评测。LM 只有 5 个 epoch，建议始终保持每个 epoch 评测。

PTN 完成后开始 LM；LM 不做 MUSAN/RIRS 增强：

```bash
python scripts/run_experiment.py train \
  --skip-bootstrap --server-vox2 --model b3 --phase lm \
  --data /path/to/redim_vox2_prepared --data-type raw --gpus 0,1 \
  --batch-size 64 --num-workers 16 --prefetch-factor 8 --amp --pin-memory
```

多机时所有节点运行相同命令，`--gpus` 表示每台机器的本地卡号：

```bash
python scripts/run_experiment.py train \
  --skip-bootstrap --server-vox2 --model b6 --phase ptn \
  --data /path/to/redim_vox2_prepared --data-type shard --gpus 0,1 \
  --num-nodes 2 --host-node-addr 10.0.0.10:29400 --job-id redim-b6-vox2-001 \
  --musan-lmdb /path/to/musan/lmdb --rirs-lmdb /path/to/rirs/lmdb
```

## 3. 上游 VoxCeleb1 recipe（可选旧流程）

VoxCeleb 数据有授权限制，请使用已经获授权的下载或本地归档。以下是上游的官方 VoxCeleb1 O/E/H 流程；它不读取你的服务器 `dev.spk2utt/test.spk2utt`，也不用于逐 epoch 的 Vox2-test 验证。

```text
data/
├── vox2_dev/{wav.scp,utt2spk,spk2utt,shard.list}
├── vox1/{wav.scp,utt2spk,spk2utt,shard.list,trials/*.kaldi}
├── musan/{wav.scp,lmdb}
└── rirs/{wav.scp,lmdb}
```

运行上游的下载、解压、转码和 manifest 生成：

```bash
python scripts/run_experiment.py prepare --data /data/wespeaker_data
```

如果原始数据已经在服务器上解压完成，可以只运行最后的 manifest 阶段，避免重新下载或解压：

```bash
python scripts/run_experiment.py prepare \
  --data /data/wespeaker_data --stage 4 --stop-stage 4
```

如果 `vox2_dev/`、`vox1/`、`musan/lmdb`、`rirs/lmdb` 和 `shard.list` 已经由上游 recipe 准备完成，训练时使用 `--skip-conversion`，不会重复生成这些文件。准备完成后先检查：

```bash
python scripts/run_experiment.py validate --data /data/wespeaker_data
python scripts/check_data.py --data-root /data/wespeaker_data --json
```

完整数据应接近 VoxCeleb2-dev 1,092,009 条语音、5,994 个说话人，以及 VoxCeleb1 37,611 条 trial。校验器还会检查 Vox1 trial 中的每个 utterance 是否出现在 `wav.scp`。

## 4. 上游 VoxCeleb1 训练和评测

推荐先跑 B3，再跑 B6。PTN 是 2 秒片段、增强和 speed perturbation；LM 是 6 秒片段、关闭 speed perturbation 的 5 epoch large-margin fine-tuning。

```bash
# B3 PTN：数据转换、shard/LMDB 构建和训练
python scripts/run_experiment.py train \
  --model b3 --phase ptn --gpus 0 \
  --data /data/wespeaker_data

# B3 LM：自动把 PTN avg_model.pt 暂存为 LM 需要的 model_0.pt
python scripts/run_experiment.py train \
  --model b3 --phase lm --gpus 0 \
  --data /data/wespeaker_data

# B3 VoxCeleb1 cleaned O/E/H 评测
python scripts/run_experiment.py evaluate \
  --model b3 --phase lm --gpus 0 \
  --data /data/wespeaker_data
```

B6 只需把 `--model b3` 改为 `--model b6`，并按显存调整 `--gpus`：

```bash
python scripts/run_experiment.py train --model b6 --phase ptn --gpus 0,1 --data /data/wespeaker_data
python scripts/run_experiment.py train --model b6 --phase lm --gpus 0,1 --data /data/wespeaker_data
python scripts/run_experiment.py evaluate --model b6 --phase lm --gpus 0,1 --data /data/wespeaker_data
```

默认实验输出在 `exp/redimnet2_b3/` 或 `exp/redimnet2_b6/`。本项目的 `evaluate` 默认先提取 embedding，再用不做均值中心化的 raw cosine 计算 EER/minDCF，以对齐 ReDimNet2 主表；如需 WeSpeaker 的均值中心化 + top-300 AS-Norm 对照，增加 `--asnorm`：

```bash
python scripts/run_experiment.py evaluate \
  --model b3 --phase lm --gpus 0 \
  --data /data/wespeaker_data --asnorm
```

### 旧流程服务器运行

这一段仅适用于已准备好上游 `vox2_dev + vox1` 数据布局的实验；不要把它用于本 README 第 2 节的服务器索引数据。

服务器上通常先固定一次 WeSpeaker，之后用 `--skip-bootstrap` 运行，避免每次任务触发 Git 网络操作：

```bash
export WESPEAKER_DIR=/opt/wespeaker
python scripts/run_experiment.py bootstrap

python scripts/run_experiment.py train \
  --skip-bootstrap --skip-conversion \
  --model b3 --phase ptn --gpus 0,1 \
  --batch-size 128 --num-workers 16 --prefetch-factor 8 \
  --amp --pin-memory \
  --data /data/wespeaker_data
```

`batch-size` 是每张 GPU 的 batch size。上游会按 `节点数 × GPU 数 × batch-size / 64` 缩放学习率；改变 GPU 数或 batch size 后应把它当作新的实验协议记录。显存不足时优先降低 `--batch-size`，并保留 `--amp`，不要直接修改仓库里的复现配置。

多机训练时在每台机器执行同一条命令，并指定首节点地址、节点数和唯一 job id；`--gpus` 是每台机器的本地 GPU 列表：

```bash
python scripts/run_experiment.py train \
  --skip-bootstrap --skip-conversion \
  --model b6 --phase ptn --gpus 0,1 \
  --num-nodes 2 --host-node-addr 10.0.0.10:29400 \
  --job-id redim-b6-ptn-001 \
  --data /data/wespeaker_data
```

`--batch-size`、`--num-workers`、`--prefetch-factor`、`--amp`、`--pin-memory` 会生成 `exp/.../runtime_config.yaml`，不会覆盖 `configs/` 中的基准 recipe。

## 5. 与 ResNet34 recipe 的关系

相邻的 ResNet34 `configs/server_voxceleb2.yaml` 虽然填写了 `dev/audio` 与 `test/audio`，但 `train_wav_scp` 为空，实际训练会递归扫描 dev 目录；`test.spk2utt` 也没有被它读取。当前项目特意不沿用这一点：两份服务器索引是训练和验证数据的唯一枚举来源，避免扫描成本和“目录内容变了但实验样本没记录”的问题。

当前项目参考了 WeSpeaker 的 `resnet.yaml`/`resnet_lm.yaml` 的工程入口，但没有把 ResNet34 的模型协议强行套到 ReDimNet2 上：

| 项目 | WeSpeaker ResNet34 | 当前 ReDimNet2 | 处理方式 |
|---|---|---|---|
| 输入 | 80-bin 外部 fbank | 72-bin 内部 TFMel | 保留 ReDimNet2 官方 executable recipe |
| 长度 | PTN 200 帧，LM 600 帧 | PTN 200 帧，LM 600 帧 | 对齐片段时长 |
| 模型/池化 | ResNet34 + TSTP，256 维 | ReDimNet2 B3/B6 + ASTP，192 维 | 保留目标模型 |
| 分类头 | ArcMargin | SphereFace2-C | 保留论文/上游实现 |
| PTN | 150 epoch，batch 128/GPU | 120 epoch，B3 256 或 B6 512/GPU | 按 ReDimNet2 recipe |
| LM | 5 epoch，margin 0.5 | 5 epoch，margin 0.3 | 按 SphereFace2 推荐 LM margin |
| 数据增强 | speed perturb + aug 0.6 | PTN 开启，LM 关闭 | 与 ReDimNet2 两阶段协议一致 |
| 权重衰减 | `1e-4` | `2e-5` | 不混用两套 recipe |

因此，80-bin fbank、ArcMargin、150 epoch 等只应作为单独的 ResNet34 对照实验；不能直接用于当前 ReDimNet2 checkpoint。论文目标数字也会受到 GPU 数、全局 batch、随机种子和数据清洗影响。

## 6. 复现目标与注意事项

论文报告的 ReDimNet2 EER（%）大致为：

| 模型 | Vox1-O | Vox1-E | Vox1-H |
|---|---:|---:|---:|
| B3 | 0.42 | 0.66 | 1.22 |
| B6 | 0.29 | 0.52 | 0.99 |

这些数字依赖数据清洗、随机种子、batch size、GPU 数量和训练时长，不应把 B3 的结果与 B6 的目标直接比较。

当前配置采用官方 WeSpeaker 可执行 recipe：72-bin TFMel、25 ms window、10 ms hop、512 FFT、20--7600 Hz、pre-emphasis、signal normalization、ASTP 和 192 维 embedding。论文正文提到过 80-bin frontend；如果要严格复现该文字配置，应另建 80-bin 实验，不能直接加载当前 72-bin checkpoint。

## 7. 目录

```text
configs/                         B3/B6 的 PTN/LM YAML
scripts/bootstrap_wespeaker.sh   固定上游 commit
scripts/run_experiment.py        索引转换、训练与评测入口
scripts/train_with_epoch_eval.py 本地 DDP 训练入口，rank 0 逐 epoch 验证
scripts/check_data.py             数据校验 CLI
src/redim_repro/                  索引、校验、评测与命令构造
tests/                            无 GPU、无数据单元测试
reference/                        学长提供的两篇论文
```
