# 合成正常数据生成流程(UNSW-NB15 / NSL-KDD)

两个比赛共用同一套流程，只有超参数按数据集分别设置。

## 1. 任务回顾

- 输入：带标签的 `train.csv`（UNSW-NB15.csv、NSL-KDD.csv）。
- 输出：只含正常样本的合成数据，UNSW 70,000 行，NSL-KDD 40,000 行。列为 `id` 加上原始特征，列顺序和 train.csv 一致。
- 评分：主办方用我们的数据训练 ECOD 和 IForest（10 个种子），在隐藏测试集上算 AUPRC，最终分 = (IForest 平均 AUPRC + ECOD AUPRC) / 2。

我们提交的数据本身就是被评价的对象。数据决定了两个检测器眼中的“正常”是什么样子。

## 2. 流程总览

```
train.csv ──▶ ① 数据准备 ──▶ ② 训练 TabDiff ──▶ ③ 生成候选池（提交量的 5 到 50 倍）
                                                         │
                                                         ▼
                          ④ 优化选择：挑出对固定检测器最有利的那部分样本
                                                         │
                                                         ▼
                                    ⑤ 本地评测 ──▶ ⑥ 提交
```

TabDiff 只负责生成逼真的正常样本，训练好之后就固定不动。
提分主要来自第 ④ 步：从候选池里挑哪些样本放进提交文件。

## 3. 目录结构

所有命令都在 `TabDiff/` 目录下运行，Python 环境为 conda 的 `tabdiff`（Python 3.10）。

```
5344/
├── UNSW-NB15.csv, NSL-KDD.csv     原始训练数据
├── readme.md                      比赛说明
├── PIPELINE.md                    本文档
├── kaggle_upload/                 上传到 Kaggle 的代码包和 notebook
├── output/                        Kaggle 训练输出（模型 ckpt、日志、提交文件）
└── TabDiff/
    ├── main.py, tabdiff/          TabDiff 官方代码（有少量修改，见第 9 节）
    ├── data/<name>/               处理后的数据和本地验证集
    ├── comp/                      我们写的脚本
    ├── submissions/
    │   ├── pool/                  候选池
    │   ├── candidates/            优化后得到的候选提交文件
    │   └── kaggle/                Kaggle 训练直接产出的提交文件
    ├── experiments/               实验结果和优化历史
    └── logs/                      运行日志
```

## 4. 各步骤与命令

### ① 数据准备

```bash
python comp/prepare_data.py
python process_dataset.py --dataname unsw
python process_dataset.py --dataname nslkdd
```

`prepare_data.py` 做的事：

- 只保留 `is_anomaly == 0` 的正常样本。
- 删除正常样本里的常数列（NSL-KDD 有 3 列），导出时再补回。
- 正常样本按 90/10 切分。90% 用于训练 TabDiff，10% 作为 `holdout.csv` 留给本地评测，TabDiff 从不接触这部分。
- 记录整数列、原始列顺序、提交行数等信息，写入 `comp_meta.json`。

### ② 训练 TabDiff（Kaggle GPU）

使用 `kaggle_upload/tabdiff_kaggle.ipynb`，按论文默认配置训练 8000 个 epoch。在两块 T4 上，NSL-KDD 约 2.1 小时，UNSW 约 3.9 小时。
本地等价命令：

```bash
python main.py --dataname unsw --mode train --no_wandb --exp_name baseline --device cuda:0
```

训练好的模型在 `output/ckpt/<name>/baseline/best_ema_model_*.pt`。

### ③ 生成候选池

```bash
python main.py --dataname unsw --mode sample --no_wandb --device mps \
    --ckpt_path ../output/ckpt/unsw/baseline/best_ema_model_1.2968_5195.pt \
    --pool_factor 5 --seed 2 --submission_path submissions/pool/unsw_e8000_pool5x.csv
```

- `--pool_factor 5` 表示生成提交量 5 倍的样本，并已转换为提交格式（整数列取整、常数列补回）。
- 不同的 `--seed` 会生成不同的候选池，多个池子可以合并使用。
- 大规模候选池（50 倍）用 `kaggle_upload/tabdiff_pools_kaggle.ipynb` 在 Kaggle 上生成，输出为 parquet 格式。

### ④ 优化选择

```bash
python comp/optimize_selection.py --dataname unsw \
    --pool submissions/pool/unsw_e8000_pool5x.csv submissions/pool/unsw_e8000_pool5x_s3.csv \
    --k 40 --linear --no_source --proxy_seeds 5 --evals 900 \
    --save submissions/candidates/unsw_opt2_k40lin.csv
```

做法：

1. 用真实正常样本和攻击样本训练一个梯度提升分类器，给每条候选样本算出攻击概率 p。
2. 用数值特征（分位数变换后）、类别特征（one-hot）和 p 对候选池做 KMeans 聚类，分成 K 个簇。
3. 每条样本的权重 = exp(所在簇的权重 + 特征的线性组合)，后者在开启 `--linear` 时才有。按权重做 Gumbel top-k，抽出提交所需的行数。随机数是固定的，所以同一组参数总是选出同一批样本。
4. 用 CMA-ES 搜索这些权重，目标函数直接是比赛的计分方式（为了提速，IForest 只用 `--proxy_seeds` 个种子）。
5. 防过拟合：holdout 正常样本和攻击样本各对半切分。优化器只看优化集，最终保留验证集上分数最高的那组权重。

主要参数：

| 参数 | 含义 |
|---|---|
| `--k` | 簇的数量 |
| `--linear` | 在簇权重之外，再加上随特征线性变化的权重 |
| `--split random \| clusters` | 验证集按随机方式切分，或按攻击簇切分（偶数簇用于优化，奇数簇用于验证） |
| `--proxy_seeds` | 优化过程中 IForest 用几个种子 |
| `--evals` | CMA-ES 的评估次数 |
| `--init_theta` | 从已保存的权重继续优化（热启动） |
| `--no_source` | 不把“来自哪个候选池”作为聚类特征（用于和旧实验保持一致） |

每次运行会把权重、簇标签和优化历史保存在 `experiments/optsel_*`。

### ⑤ 本地评测

```bash
python comp/evaluate.py --dataname unsw --real --submission submissions/candidates/unsw_opt2_k40lin.csv
```

输出两部分：

- 总分：攻击样本按 train.csv 的攻击比例抽样 10 次（固定种子，每次运行都相同），取平均。同时给出 IForest、ECOD 各自的分数。
- 按攻击簇的 AUROC：看方案在每类攻击上的表现，以及最差的那一类。

### ⑥ 提交

直接提交 `submissions/candidates/` 下的 csv 文件。文件格式已经检查过：行数、列名和顺序、id 唯一、没有缺失值或无穷大。

## 5. 本地验证集

所有方案都在同一份固定的验证数据上比较。

| 组成 | 位置 | UNSW | NSL-KDD |
|---|---|---|---|
| 正常样本 | `data/<name>/holdout.csv`（TabDiff 训练时未见过） | 3,432 | 1,948 |
| 攻击样本 | 原始 csv 里 `is_anomaly=1` 的全部行 | 15,000 | 10,000 |
| 攻击簇 | `data/<name>/anomaly_clusters.csv`（KMeans 分成 8 类，当作伪攻击类型） | 8 类 | 8 类 |

注意事项：

- ECOD（pyod 实现）计算 ECDF 时会把待测数据也算进去，所以同一条样本的分数会受同批其他样本影响。评测时每个抽样子集都单独作为一批打分，和主办方整批打分的方式一致。
- 完整本地评测会用到优化时见过的攻击样本，所以分数偏乐观。选方案时以优化过程中的验证集分数为主。
- 本地分数和线上分数的排序一直一致。UNSW 上线上分约等于本地分乘以 0.9。

## 6. 两个数据集的超参数

| | UNSW | NSL-KDD |
|---|---|---|
| 测试集特点 | 攻击类型和训练集相同，难点是部分攻击和一部分正常流量几乎一样 | 有训练集中没有的新攻击，需要泛化能力 |
| 验证切分 | `random` | `clusters`（检验对没见过的攻击类型是否有效） |
| 选择强度 | 充分优化：K=40，`--linear`，900 次以上评估 | 目前最好的是不做优化，即均匀抽取（所有权重为 0），仍在实验 |

## 7. 结果记录

### UNSW

| 方案 | 本地 | 线上 |
|---|---|---|
| Starter Kit baseline | 0.4569 | — |
| TabDiff 8000 epoch，均匀抽取 | 0.4754 | 0.4322 |
| TabDiff 200 epoch | 0.4816 | 0.4265 |
| 去掉 sttl=254 的样本（手工规则，不进入最终流程） | 0.5558 | 0.4999 |
| 分类器筛选，保留攻击概率最低的 50% | 0.5679 | 0.5073 |
| 优化选择，K=20 | 0.6224 | 0.5527 |
| **优化选择，K=40 + 线性权重，70 万候选池** | **0.6760** | **0.6140** |

### NSL-KDD

| 方案 | 本地 | 线上 |
|---|---|---|
| Starter Kit baseline | 0.9145 | 0.9099 |
| TabDiff 4 epoch | 0.9419 | 0.9107 |
| **TabDiff 8000 epoch，均匀抽取** | **0.9606** | **0.9299** |
| 优化选择，混合候选池，K=80，随机切分 | 0.9766 | 0.9241 |

NSL-KDD 上本地提升没有迁移到线上，原因分析见下一节。

## 8. 主要发现

1. TabDiff 的生成质量已经很高：各特征的分布误差（KS 统计量）基本在 0.02 以下，生成数据和真实数据训练出的检测器分数相当。继续改进生成模型的空间不大。
2. 有监督分类器在 UNSW 上的 AUPRC 可以达到 0.96，而无监督检测器只有 0.5 左右。瓶颈在于 ECOD 和 IForest 能否利用这些信息，数据本身是有区分度的。
3. UNSW 里 99.9% 的攻击 sttl 为 254 或 62，而 35% 的正常流量也是这两个值。这部分正常流量会“掩护”攻击，是 UNSW 的主要难点。
4. 优化选择的效果来自大幅收窄正常范围。UNSW 最好的那版，生成数据实际只集中在大约 3 类流量上，原候选池的有效簇数是 30.6。
5. NSL-KDD 上，针对已知攻击优化得越多，对没见过的攻击类型效果越差（按攻击簇切分时，验证分数从 0.849 先升到 0.866，最终降到 0.797）。这和隐藏测试集含有新攻击一致。
6. 试过但没有效果的方向：给样本加噪声、SMOTE 插值、补厚特征的尾部、把无关特征设为常数或压平、减少样本多样性、把加噪声池和 SMOTE 池混入候选池（UNSW 上）。

## 9. 对 TabDiff 官方代码的修改

- `main.py`：新增 `--device`（支持 MPS）、`--steps`、`--mode sample`、`--submission_path`、`--seed`、`--pool_factor`。
- `tabdiff/main.py`：DataLoader 改为 `num_workers=0`，加入 `sample` 模式。
- `tabdiff/trainer.py`：去掉新版 PyTorch 不支持的 `verbose` 参数。保存最佳模型的阈值从固定的 4000 epoch 改为总 epoch 的一半。画图失败时不再中断训练。

## 10. 其他脚本（探索阶段，最终流程不依赖）

| 脚本 | 用途 |
|---|---|
| `comp/score.py` | 早期的单批次本地评分 |
| `comp/experiments.py` | 加噪声、SMOTE、加权重采样等对照实验 |
| `comp/lco.py` | 分类器加权，加上“留一类攻击”验证 |
| `comp/filter_pool.py` | 按攻击概率筛选，比较不同保留比例 |
| `comp/strat.py` | 按 sttl 分层筛选（UNSW 专用，不进入最终流程） |
| `comp/shape.py` | 尾部补厚、常数列、压平等“迎合检测器”的实验 |
| `comp/make_pool.py` | 生成加噪声池和 SMOTE 池 |

## 11. 下一步

- 目标：线上 UNSW 0.76，NSL-KDD 0.96。
- 扩大候选池：在 Kaggle 上用现有模型生成 50 倍的候选池（UNSW 350 万行、NSL-KDD 200 万行），验证候选池越大分数是否越高。
- 如果候选池大小是瓶颈：用优化后的选择策略挑出真实正常样本中有用的部分，只用这部分重新训练 TabDiff，再生成候选池并重复第 ④ 步。
- NSL-KDD：在按攻击簇切分的验证下寻找能涨分的设置，并限制正常范围被收窄的程度。
