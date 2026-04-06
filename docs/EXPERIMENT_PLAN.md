# Dataset Pruning 复现实验计划

本文档用于对齐论文设定、组织分支与数据、以及统计 **3 次重复实验的均值与方差**（按 dataset × pruning rate × 评估模型 记录）。

---

## 1. Git 分支策略（每种方法一个分支）

在 `main` 上为每种 baseline / 方法单独开分支，便于并行实验、互不干扰；合并回 `main` 前再统一 review。

| 方法 | 建议分支名 | 说明 |
|------|------------|------|
| Random pruning | `exp/random-pruning` | 随机保留期望数量的训练样本（本阶段优先） |
| Herding | `exp/baseline-herding` | 每类聚类中心最近邻式选取 |
| Forgetting | `exp/baseline-forgetting` | 易被遗忘的样本 |
| GraNd | `exp/baseline-grand` | 较大 loss 梯度范数 |
| EL2N | `exp/baseline-el2n` | 较大 error vector 范数（预测概率 − one-hot） |
| Influence-score | `exp/baseline-influence-score` | 影响函数范数（不考虑 group effect） |
| Data Pruning（论文方法） | `exp/method-data-pruning` | 基于 Influence Function + 凸性假设等；与 cardinality 约束对齐时用 Eq. 5 类设定 |

**工作流建议**

1. 从 `main` 检出对应方法分支：`git checkout -b exp/<name> main`（若分支已存在则直接 `git checkout exp/<name>`）。
2. 仅在该分支上实现该方法的 **选子集逻辑** 与 **可复现实验脚本**（随机种子、日志路径）。
3. 表格汇总在仓库外或 `results/`（JSON/CSV）中维护，避免与代码分支强耦合。

---

## 2. 论文中的训练与评估设定（需与代码对齐）

以下与论文描述一致，**剪枝前预训练、剪枝后重训练** 应使用相同超参（若当前 `main.py` 中 batch size、epoch 等与下文不一致，各方法分支上应统一到下文）。

| 项 | 设定 |
|----|------|
| Epochs | 200 |
| Batch size | 256 |
| 学习率 | 0.1，**cosine annealing** |
| 优化器 | SGD，momentum 0.9，weight decay `5e-4` |
| 数据增强 | Random crop + random horizontal flip（CIFAR 系常用 padding=4） |
| 剪枝后评估 | **新随机初始化** 的网络仅在剪枝子集上重训练，再报 **test accuracy** |

**多架构泛化（填表用）**

- 在剪枝子集上分别重训练并测试：**~1.25M 参数量模型**（需与仓库中具体定义一致，如 MobileNet/ShuffleNet 等）、**ResNet18**（~11.69M）、**ResNet50**（~25.56M）。

**随机 pruning 论文段落对应实验**

- 使用 **随机初始化的 ResNet50** 在 **CIFAR-10 / CIFAR-100 / TinyImageNet** 上做剪枝子集选择时的对照设定（与其它 baseline 可比）；子集选定后 **重训练仍为随机初始化 ResNet50**（除非表格另有“跨架构”小节）。

---

## 3. 重复次数与统计量

对每个 **(dataset, pruning rate, 评估模型)** 组合：

- 固定 **3 个独立随机种子**（例如 `seed ∈ {0, 1, 2}` 或预生成种子列表）。
- 记录每次 run 的 **test accuracy**（及可选 train loss 曲线终点）。
- 表格中报告：**均值 ± 标准差** 或 **均值 + 方差**（与论文表格格式一致即可）。

建议在日志/CSV 中至少包含：`method`, `dataset`, `prune_rate`, `target_model`, `seed`, `test_acc`, `timestamp`。

---

## 4. 数据集准备与目录

### 4.1 CIFAR-10 / CIFAR-100

- **方式 A（推荐）**：由 `torchvision.datasets.CIFAR10` / `CIFAR100` 指定 `root=./data` 且 `download=True`，首次运行自动下载到 `./data`。
- **方式 B**：手动下载官方 tarball 放到 `root` 下，保持 torchvision 期望的目录结构（与 `download=True` 生成结构一致）。

归一化统计量与论文中 CIFAR 常用设定一致即可（当前代码使用 `(0.4914, 0.4822, 0.4465)` / `(0.2023, 0.1994, 0.2010)`，与 ResNet-on-CIFAR 惯例一致）。

### 4.2 Tiny ImageNet

Torchvision **不内置** Tiny ImageNet，需自行准备 **224×224 或 64×64** 版本（与模型输入一致即可；若用 CIFAR 式 32×32 网络，需说明是否 resize 或采用专用脚本）。

**常见做法**

1. 从 [Stanford CS231n Tiny ImageNet](http://cs231n.stanford.edu/tiny-imagenet-200.zip) 下载并解压。
2. 期望的目录结构示例（按你实现的 `Dataset` 类调整）：

```text
data/tiny-imagenet-200/
  train/<class_id>/<images>.JPEG
  val/<class_id>/<images>.JPEG   # 或 val/images/ + val/val_annotations.txt
  test/images/
  wnids.txt, words.txt, ...
```

3. 在代码中实现 `TinyImageNet` 或使用已有第三方 loader，**训练集 100k / 200 类** 与论文一致；验证集划分与论文对齐（若论文使用 val 作 test，需在计划中写死同一划分）。

---

## 5. Pruning rate 与 cardinality

- **Random pruning**：从全训练集中均匀随机选取 **固定基数** \(m\)（或保留比例 \(p = m/N\)），与 Herding 等 **按基数保证子集大小** 的 baseline 一致。
- 在 `exp/random-pruning` 分支中实现：`numpy.random.Generator` + 每 seed 可复现；若需 **每类平衡**，在计划中注明是否与论文一致（随机 baseline 常见为全局随机或分层随机，需与论文表格脚注一致）。

---

## 6. 当前仓库与计划的差距（实施 checklist）

当前 `main.py` 仍以 Influence + CVXPY 选点为主，且存在 **batch size 128、剪枝后仅少量 epoch** 等与论文不一致之处。各方法分支上建议逐项勾选：

- [ ] 训练 200 epoch、batch 256、cosine LR、SGD(0.9, 5e-4)、增强一致。
- [ ] 剪枝后 **全新初始化** 模型在子集上完整重训练 200 epoch。
- [ ] CIFAR-10 / CIFAR-100 / TinyImageNet 三数据集脚本入口统一（`--dataset` 或配置文件）。
- [ ] ResNet18 / ResNet50 / 小模型 三条训练命令与相同子集索引文件（可选）或可复现随机流。
- [ ] 三次 seed 循环 + 结果聚合脚本（均值、方差）。

---

## 7. 建议的命令备忘（占位）

在对应分支实现 CLI 后，可统一为类似形式（具体参数以代码为准）：

```bash
# 示例：random pruning，CIFAR-10，保留 50%，seed=0，ResNet50 重训练
python train_pruned_subset.py --method random --dataset cifar10 \
  --keep-ratio 0.5 --seed 0 --model resnet50 --epochs 200 --batch-size 256
```

---

## 8. 参考文献编号（来自用户摘录，便于写论文时对照）

CIFAR / TinyImageNet、ResNet、Influence Function、GraNd/EL2N、Herding、Forgetting、Simulated Annealing 等见原论文引用表；本计划不展开引用条目。

---

**文档版本**：与分支 `exp/random-pruning` 一并创建；后续若在其它方法分支上更改协议，请在对应分支更新本文件或增加 `docs/EXPERIMENT_PLAN_<method>.md` 说明差异。
