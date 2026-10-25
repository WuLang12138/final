# SpaSRD：论文主模型

本目录只提供论文采用的 SpaSRD-D（APPNP）和 SpaSRD-S（ACM 表示分离）主模型。
四类数据分别运行；全部采用 KNN8 空间图。五个随机种子为 `0, 1, 2, 3, 42`。
无需先运行训练：本地已准备好 80 份可加载权重、模型实际使用的预处理输入和对应结果。

## 使用

在现有 `sidmgf` Python 环境中，打开 `tutorials` 下所需数据的 Notebook，依次运行。
所有路径从代码位置定位，不依赖盘符、用户目录或外部 ST 工程。
运行日志和结果会直接显示在单元格输出中：包括是否跳过训练、逐 seed 的 ARI/NMI、
预测均值与标准差、与原末轮结果的一致性，以及单独标明的历史归档指标。
绘图 Notebook 会直接展开轻量预览，可点击样本名称折叠；不必另开结果目录，高清原图照常保存。

1. `00_论文结果核验与汇总.ipynb`：核验当前模型、新预测和新训练记录，重算主模型汇总表；不读取历史归档补齐。
2. `01_DLPFC_论文实验.ipynb`：12 张切片、每片五 seed。
3. `02_HBC_论文实验.ipynb`：`paper` 为共同训练设置，`refined` 为细化方案；以 Notebook 当前的 PROFILE 设置为准。
4. `03_STARmap_论文实验.ipynb`：五 seed。
5. `04_osmFISH_论文实验.ipynb`：五 seed。
6. `05_统一参数与论文绘图.ipynb`：汇总配置与结果，生成模型框架图、主模型空间分区、UMAP、混淆热图和 seed 箱线图。

命令行同样可用：

```powershell
python process/run.py run --dataset DLPFC
python process/run.py run --dataset HBC --profile refined
python process/run.py run --dataset STARmap --inference-only
python process/run.py run --dataset HBC --recover-best
python process/run.py run --dataset HBC --model-checkpoint best --inference-only
python process/run.py verify
python process/run.py summary
python process/run.py figures --checkpoint current
```

默认优先使用 CUDA，没有 CUDA 则使用 CPU。Notebook 使用当前内核的 Python 在独立进程运行任务。
可以加 `--sample 151507 --seed 0` 只运行单个实验。
`--inference-only` 会在权重缺失时直接报错，保证不启动训练。
`--model-checkpoint auto`（默认）优先加载已完成训练的峰值权重；没有时加载末轮权重。
指定 `best` 时缺失峰值权重会报错，不会拿末轮冒充峰值。
`--recover-best` 仅对没有完整峰值权重的 seed 按原配置重跑补存，已完成的 seed 直接跳过。
补存训练开关 `--recover-best` 不能与 `--inference-only` 同用。

## 目录与保存位置

```text
config/paper.json              唯一实验配置：输入、模型、参数、seed、输出路径
data/<数据>/<样本>/             已完成预处理的表达矩阵和 KNN8 图
model/                        与历史权重兼容的编码器、解码器和损失
src/workflow.py                加载、训练、预测、核验、汇总
src/figures.py                 主模型绘图
process/run.py                唯一命令行入口
tutorials/                    6 个 Notebook
weights/<数据>/<样本>/<方案>/seed_<seed>.pt
weights/<数据>/<样本>/<方案>/seed_<seed>_best.pt
results/archived/<数据>/<样本>/<方案>/seed_<seed>/
results/predictions/<数据>/<样本>/<方案>/seed_<seed>/
results/tables/                逐 seed、逐样本、逐数据集汇总
results/figures/<结果类型>/      图件及绘图源数据
results/provenance.json        导入权重的来源、校验值
results/current_selection.json 各数据最近成功运行的方案与权重类型
results/tables/current/        当前预测重新计算的指标与来源清单
results/figures/current/<数据>/<方案>/ 当前结果图件，方案之间不混用
```

例如 DLPFC 151507 的 seed 0 模型为
`weights/DLPFC/151507/paper/seed_0.pt`，新预测保存到
`results/predictions/DLPFC/151507/paper/seed_0/`。

## 已有模型时的行为

每次运行数据 Notebook，先校验输入文件、配置、种子与模型记录是否一致；
有匹配权重就跳过训练，严格加载模型参数，关闭训练随机失活，重新提取表示、聚类并计算 ARI/NMI。
默认优先使用完整峰值模型；仅有旧末轮模型时照常预测并明确提示缺少峰值权重。
配置不匹配时停止，不会静默重训或把不同配置的结果混用。
只有对应权重缺失或显式启用补存峰值时才训练。每轮完成优化更新后关闭随机失活，提取表示并聚类；
ARI 严格超过此前最高值时，立即克隆并保存该轮完整模型参数（包括缓冲区）、ARI/NMI、轮次与配置。
相同 ARI 保留最早轮次。中断训练每 10 轮保存恢复点，恢复点也携带当时的峰值参数和结果。
训练结束分别保存完整峰值权重与末轮权重；未完成训练的峰值文件不会被自动当作完整实验加载。
新训练记录放在 `results/predictions/.../training_seed_<seed>/`，不覆盖论文归档。

四个数据 Notebook 均提供 `TRAIN_FOR_BEST = False`，默认不补跑旧实验。
需要为旧实验补齐峰值模型时改为 `True`。补跑记录保存到 `peak_recovery_seed_<seed>/`，
新末轮权重保存为 `seed_<seed>_retrained_final.pt`，保留原 `seed_<seed>.pt`。
峰值权重保存为 `seed_<seed>_best.pt`，加载后的预测保存到该 seed 预测目录下的 `best/`。
补跑完成后再次运行会自动跳过训练。补跑结果可能与历史峰值有差别，二者分别保留和对照。

本地模型兼容文件保留原初始化顺序及参数名；少量初始化辅助层和兼容接口用于重现旧训练随机数顺序，
不代表主库还提供其他候选编码器的实验入口。公开运行入口只允许 APPNP 与 ACM 分离方案。

## 两类结果必须区分

**最初导入的 `seed_<seed>.pt` 是第 500 轮权重，不是最高 ARI 轮次的权重。**
历史程序保存了峰值轮次的表示、聚类标签和完整训练曲线，但没有保存该轮模型 state_dict。
因此本库同时保存：

- `results/archived/.../peak.npz`：当时真实保存的峰值表示和预测，直接用于复核原论文峰值指标与绘图。
- `results/archived/.../final.npz`：原实验末轮表示和预测。
- `results/predictions/.../prediction.npz`：实际加载末轮权重后重新生成的表示和预测。
- `results/predictions/.../best/prediction.npz`：补存峰值权重后，实际加载峰值模型重新生成的表示和预测。

`00` 和 `05` 现在默认读取当前实验新预测，不再默认使用历史归档。
每次数据 Notebook 成功运行后，会更新 `results/current_selection.json`，记录实际完成的方案和权重类型。
单纯编辑 PROFILE 而不运行不会切换汇总；按当前记录读取完整五 seed，不按 ARI 高低自动择优。
目前选择为 DLPFC/STARmap/osmFISH 的 `paper`，以及 HBC 的 `refined`，共 75 个实验。
HBC 的另一套 `paper` 结果仍保留，可显式指定查看，不会与 refined 的五 seed 混在一起。

当前核验会检查权重 SHA、配置与轮次、预测文件中的点位顺序及坐标、从标签重算 ARI/NMI，
以及峰值权重是否对应完整新训练记录的最佳轮次。缺失或不匹配即报错，不拿历史结果填补。
图件和表格共享同一套读取规则，并写出逐文件来源及校验值。
历史结果仅在显式使用 `--source archived`（核验/汇总）或 `--checkpoint peak/final`（绘图）时读取。

四个数据 Notebook 不重新运行库外的 GCN、外部方法或消融实验；完整跨方法比较仍由
`sidmgf_paper_support` 提供，当前主模型表格和图件不依赖那些旧结果。

峰值由参考标签 ARI 选择，属于标签辅助的回顾性评价，不能表述为独立测试集上的模型选择。
聚类数采用既定组织区域数；参考标签用于评价，不进入训练损失。不同硬件和依赖版本可能产生数值差异。
不会通过重命名权重文件把末轮权重当作峰值权重。

汇总 DLPFC 时先计算每张切片五 seed 均值，再计算 12 张切片的均值、中位数和切片间样本标准差。
其他数据报告五 seed 均值和样本标准差。HBC `paper` 与 `refined` 分开汇总。

## 参数和输入

APPNP 的 DLPFC 参数按供体共享：Br5292 为 K=15、α=0.04；Br5595 为 K=20、α=0.2；
Br8100 为 K=4、α=0.0075。STARmap 与 osmFISH 为 K=10、α=0.05。
完整的结构损失权重、优化参数、潜在维度等集中在 `config/paper.json`。
KNN8 指每个点选择 8 个近邻后取无向并集，因此最终节点度不必恰好为 8。
结构监督图 A 无自环，传播矩阵 P 为 A 加自环后的行归一化。

`data` 中是原训练实际读取的固定预处理输入，保留原 feature/spot 顺序和预处理来源记录。
从这些输入可直接训练或预测，无需重新执行可能改变特征筛选的预处理。
本包目前不提供任意外部原始数据的自动转换流程。

## 库外的补充实验

完整基线对照、九种外部方法的独立入口、模型消融、历史全图件与原始导入记录，
统一存放在同级 `sidmgf_paper_support`，具体位置见其 README。
主库不依赖该目录运行；完整论文跨方法图表仍通过补充实验目录生成。
既有历史选参档案和 ST 根目录中的原实验结果不因本次整理而删除。

## 环境与发布

`requirements.txt` 记录当前验证环境的主要依赖版本，建议 Python 3.10。
PyTorch 的 CPU/CUDA 构建需按设备安装；当前环境使用 CUDA 12.8 构建。
本地附带的数据和权重不直接进入普通 Git 提交；发布时可放入版本化数据/模型附件，保持上述相对路径。
现有来源和许可材料保留在库外补充包；本次整理不擅自为上游代码或数据授予新许可证。
