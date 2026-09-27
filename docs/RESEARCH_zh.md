# ASTRA-Tangent：任务驱动的 Tucker 切空间预算分配

调研与实现日期：2026-09-28。本文是研究设计与实现说明，不是已经获得实验支持的论文结论。

## 1. 建议的论文问题

**在共享 Tucker 参数族中，什么时候值得离开当前多线性子空间，以及有限更新预算应该分给哪个模态？**

现有 ASTRA-Core 从预训练权重构造 Tucker 分解，仅学习核心。旧 ASTRA-Mode 对核心乘以小方阵，等价于在固定因子的列空间内重新组合；即使方阵可学习，也不能产生原列空间之外的方向。因此，旧版并未实现子空间移动。

新版保留预训练张量的残差，学习正交模态因子，并周期性比较核心方向与模态水平切方向的任务梯度收益。当前实现固定 Tucker 秩，动态选择开放的水平切空间块。选择更多模态意味着开放更大的**局部优化方向集合**，不是提高张量秩或永久切换到更高维秩流形。

工作名称：**ASTRA-Tangent: Budgeted Horizontal Subspace Adaptation for Tensorized Fine-Tuning**。名称与创新定位均是暂定的；不能把组合本身写成已证实的新颖性。

## 2. 相关工作及重合边界

下面均链接到论文、正式论文集或作者页面。年份采用首发/正式发表信息，不能把网站抓取时间当成发表时间。

| 工作 | 已有贡献与本项目必须面对的重合 | 本版可检验的差异 |
|---|---|---|
| [AdaLoRA，ICLR 2023](https://arxiv.org/abs/2303.10512) | 按重要性分配低秩参数预算；自适应预算不是新概念 | 本版选择固定秩下的模态切空间更新，而非裁剪奇异值 |
| [FLoRA，ICLR 2025](https://arxiv.org/abs/2405.14739) | 用低秩核心及因子表示高维参数增量 | Tucker PEFT 本身不能作为创新；重点转向更新规则 |
| [LoRETTA，2024 预印本](https://arxiv.org/abs/2402.11417) | tensor-train PEFT 与极低参数预算 | 必须比较张量化方法，不能只比较 LoRA |
| [TensLoRA，2025 预印本](https://arxiv.org/abs/2509.19391)，[作者确认 ICASSP 2026 接收](https://ax-le.github.io/news/) | 联合组织跨层/投影更新为高阶张量，支持模态压缩率；与原项目尤其接近 | family-wise tensorization 和模态秩分配均不足以单独支撑新意 |
| [Geometry-aware training of factorized layers in tensor Tucker format，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/ea48cb23a02ec3b895b0b0124307b0ec-Paper-Conference.pdf) | Tucker 几何训练、动态模态秩、理论分析；还讨论 LoRA-like Tucker 微调 | “Tucker + 流形 + 自适应秩”已有直接先例；本版研究稀疏开放模态方向 |
| [Riemannian Preconditioned LoRA，2024 预印本](https://arxiv.org/abs/2402.02347) | 小矩阵预条件改善 LoRA 优化 | 核心 Gram 预条件不是本版原创贡献 |
| [LORO，ICLR 2025](https://openreview.net/pdf?id=i0zzO7Hslk) | 低秩语言模型预训练中的几何更新，避免构造完整梯度 | 流形优化的可行性已有依据，不能声称首次使用 |
| [LoRAM，ICLR 2025 Workshop，作者机构存档](https://doi.org/10.60882/cispa.32838377) | 固定秩与有界秩的流形参数化，以及 Riemannian Adam | 不是同名的模型裁剪 LoRAM，也不是 ICLR 主会论文 |
| [StelLA，2025 预印本](https://arxiv.org/abs/2510.01938) | 三因子表示及 Stiefel 输入/输出子空间学习 | 因子正交化和子空间学习已存在；必须对比始终更新所有因子的策略 |
| [Riemannion，2025 预印本](https://arxiv.org/abs/2507.12142) | 固定秩流形上的 Muon 类更新与梯度初始化 | 本版没有实现 Muon，不应借其名义包装标准水平梯度步骤 |
| [GeLoRA，EMNLP Findings 2025](https://aclanthology.org/2025.findings-emnlp.1372.pdf) | 根据隐藏表示的内在维数配置 LoRA 秩 | 数据几何与参数流形几何需要区分 |
| [RAFFT，AISTATS 2026](https://proceedings.mlr.press/v300/zhou26a.html) | 联邦场景中的 Riemannian LoRA 与动态秩 | 自适应秩加流形并非空白；其聚合问题与本项目不同 |
| [TuckA，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39444) | Tucker 组织多专家并进行批级路由 | 本版路由的是优化方向，不是样本/专家 |
| [ReFTA，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Zheng_ReFTA_Breaking_the_Weight_Reconstruction_Bottleneck_in_Tensorized_Parameter-Efficient_Fine-Tuning_CVPR_2026_paper.html) | 处理张量 PEFT 中权重重建的开销 | 本版仍重建单层权重，参数少不意味着实际训练更快 |
| [GLT-PEFT，2026 预印本](https://arxiv.org/abs/2605.16769) | 医学影像场景中结合 Tucker、Lie 变换与门控 | “几何 + Tucker + gate”的宽泛叙述也不能作为新颖性依据 |

检索覆盖上述直接邻域，并非穷尽所有截至该日的论文。较可信的候选贡献是：**用 Tucker 正交切空间分解构造可解释的模态选择指标，并在跨层共享 PEFT 中验证固定秩下的更新预算分配。** 如果自适应策略不优于同预算随机/轮换策略，或所有收益都来自预条件，这一贡献就不成立。

## 3. 参数化与几何

对一个权重族，记预训练张量为 $W_0$，其 Tucker 近似为

$$X_0=C_0\times_1U_1^0\cdots\times_dU_d^0,\qquad (U_k^0)^TU_k^0=I.$$

使用带固定锚点的表示：

$$W=W_0+\alpha\left[X(C,U)-X_0\right],\quad
X(C,U)=C\times_1U_1\cdots\times_dU_d,\quad C=C_0+\Delta C.$$

初始化 $\Delta C=0,U_k=U_k^0$，因此不改变原模型。采用非零谱核心还避免了从全零核心出发时所有因子梯度同时消失。残差 $W_0-X_0$ 始终保留。

因子位于 Stiefel 流形。我们仅沿 $U_k^TD_k=0$ 的水平部分移动，避免把已能由核心表达的子空间内旋转重复算作“逃逸收益”。这与 Tucker 商几何的常用水平表示一致。核心可能退化，因此软件严格保证的是正交因子和固定的秩上界；没有强制每个核心展开始终满行秩，不能无条件宣称整个训练轨迹均位于光滑固定秩张量流形。

尤其要区分：$X$ 的模态秩上界为 $r_k$，但实际增量 $X-X_0$ 的模态秩可达 $\min(n_k,2r_k)$。不能将其宣传成严格秩 $r_k$ 的增量，也不能直接把 Tucker 秩与 LoRA 矩阵秩等同。

## 4. 探测指标与选择规则

反向传播给出核心梯度 $g_C$ 与因子梯度 $g_k$。在探测步骤，对每个非满秩模态计算

$$H_k=g_k-U_k(U_k^Tg_k),\quad M_k=C_{(k)}C_{(k)}^T,$$
$$\lambda_k=\mu\max(\operatorname{tr}(M_k)/r_k,\epsilon),\quad
D_k=H_k(M_k+\lambda_k I)^{-1},$$
$$e_0=\|g_C\|_F^2,\quad e_k=\langle H_k,D_k\rangle\ge0.$$

实现用小型线性方程求解，不显式求逆，不构造 $n_k\times n_k$ 投影矩阵，也不要求存储完整权重族的梯度。$n_k=r_k$ 时水平空间为空，跳过该模态。

族级逃逸比率定义为

$$\rho=\frac{\sum_ke_k}{e_0+\sum_ke_k}.$$

若 $\rho<\tau$，仅更新核心；否则按

$$s_k=e_k/[r_k(n_k-r_k)]^p$$

降序贪心选取，直到捕获至少 $\gamma$ 的模态收益或达到最多 $K$ 个模态。代价是水平自由度代理，**不是 FLOPs 或延迟的测量值**。达到数量上限时可能无法达到捕获目标，日志会记录实际捕获比例。

每 $P$ 个优化器步骤探测一次，期间沿上次选中的模态更新；先进行 $T_w$ 步核心预热。非探测步骤将未选中因子从 autograd 路径中 detach，但保留 Parameter 注册，防止统计与检查点遗漏。梯度累积的所有 microbatch 共用同一选择；计数单位是 optimizer step。

这是当前 minibatch 的一阶、带阻尼收益估计，不是验证集收益，不检测法向空间中的秩增长需求，也不是损失下降的保证。没有 EMA、置信区间、双批次确认或全局跨族预算；这些可以作为后续研究方向，不应当写成已实现功能。

## 5. 实际更新与可证明范围

选中因子使用 $U_k^+=\operatorname{polar}(U_k-\eta_U D_k)$，并限制每次因子步的 Frobenius 范数。所有方向在核心更新前计算。极分解是标准 Stiefel 回缩，未选中因子不会受到动量或 weight decay 的隐式改变。

核心增量与分类头使用 AdamW，因子使用无动量的预条件水平梯度。这样无需处理因子动量运输。**这不是完整的、规范不变的 Riemannian Adam**：方向与评分对正交坐标变换等变，但核心的逐坐标 AdamW 不满足相同不变性。测试仅证明其覆盖的性质，不把局部几何性质扩张成整条训练轨迹的结论。

可作为论文理论起点的理想化命题：在所有 $C_{(k)}$ 满行秩、因子正交、无阻尼、没有 AdamW/裁剪且使用当前精确梯度时，核心切空间与各水平模态切空间正交。若 $G=\nabla_X\mathcal L$，则

$$\|P_{T_X\mathcal M}G\|_F^2=e_0+\sum_ke_k,$$
$$\|P_{T_X\mathcal M}G-P_{T_S}G\|_F^2=\sum_{k\notin S}e_k.$$

证明要点：核心变化始终位于所有 $U_k$ 的列空间，模态 $k$ 的水平变化在该模态与 $U_k$ 正交；不同模态块之间也至少有一个正交模态。链式法则给出的 $g_k$ 右乘核心 Gram 逆恰好恢复对应投影块，二者配对即其环境空间范数平方。

因此，若核心能量占比已很高，冻结模态在理想切梯度意义下有定量依据。对于实际带阻尼实现，$e_k$ 是收缩后的收益代理，不能继续当成精确投影误差；周期性复用还引入过时信息。进一步的随机收敛、预算最优性与泛化结论均未证明。贪心单位成本排序也不是任意预算下的最优背包算法。

## 6. 实现与使用

默认 `--tuning-mode manifold`。旧 `additive`、`multiplicative`、`both` 保留。

```bash
python -m astra_core --model-path ./local_models/roberta-large \
  --dataset-path ./local_datasets/glue_sst2 --glue-task sst2 \
  --target-families q k v o --attn-ranks 4 4 16 32 \
  --tuning-mode manifold --manifold-policy adaptive \
  --manifold-probe-interval 10 --manifold-warmup-steps 20 \
  --manifold-max-active-modes 2 --manifold-escape-threshold 0.15 \
  --learning-rate 1e-3 --run-name tangent_sst2
```

| 设置 | 默认值 | 作用 |
|---|---:|---|
| `manifold-policy` | adaptive | adaptive / all / core / random |
| `manifold-capture-fraction` | 0.8 | 尝试捕获的模态收益比例 |
| `manifold-cost-power` | 1 | 自由度代理的惩罚指数，0 为按收益排序 |
| `manifold-damping` | 0.001 | 核心 Gram 相对阻尼 |
| `manifold-factor-lr-scale` | 1 | 因子相对核心学习率 |
| `manifold-max-factor-step` | 0.05 | 单模态步长上限 |
| `gradient-accumulation-steps` | 1 | 梯度累积次数 |

`all` 在预热后更新所有可移动模态；`core` 永不更新因子；`random` 每次按自身当前状态计算 adaptive 的选择数量，再随机选择相同数量的可移动模态。后者只是**当前状态下数量匹配**的对照，不保证跨两条训练轨迹总开销完全相等。

`examples/sweep_tangent_sst2.json` 提供基线、选择策略、探测间隔、成本消融和三个种子的配置。`manifold_history.json` 记录探测时的能量、选择、捕获比例、正交误差和步长；不是逐步完整日志，非探测日志中的能量只覆盖实际计算过梯度的方向，不能将其直接平均成全程效率指标。

`experiment_summary.json` 记录训练及 epoch 评估耗时、CUDA 峰值；不包含分解与初始评估耗时。`parameter_counts.json` 区分适配器参数、因子参数、水平自由度和 buffer 字节数。注册的因子全部计入参数预算，不能把当步活动数量当作总参数量。

最终 `trainable_state.pt` 与 `adapter_basis.pt` 一起保存学习状态及初始分解。用 `astra_core.checkpoint.load_adapter_checkpoint` 在相同的、未打补丁的 CPU 基座上恢复，再移到设备。最终产物用于推理，不是 Trainer 断点续训快照；优化器自身支持 controller state 序列化，已测试连续更新一致性。

当前支持单进程 CPU/单 GPU、FP32 参数及 AMP 计算、梯度累积。多 GPU、FSDP、量化、显式将因子转为 FP16/BF16、动态图编译和 decoder-only 模型尚未验证。仍采用逐层稠密权重重建；保留一份稠密 Tucker 锚点，显存节省必须测量。

## 7. 投稿前的实验决策

1. **机制验证**：构造仅需核心、仅需一个模态旋转、多模态旋转的目标，检验选中率、误差、阈值变化、间隔变化。现有单元测试已覆盖其中的子空间外目标。
2. **受控比较**：同初始化/秩/核心优化器/预热/学习率搜索，比较 core、all、adaptive、random；再补固定轮换、无 Gram 预条件、始终探测对照。后三项目前没有专门实现。
3. **PEFT 对照**：LoRA、AdaLoRA、PiSSA、TensLoRA/FLoRA、StelLA 至少选择可复现的代表。需要使用官方实现或严格核对实现；当前仓库没有代替这些方法的冒名基线。
4. **预算公平性**：分别匹配总适配器存储参数、实际 GPU 时间和峰值显存。移动因子的版本通常比旧核心版本有更多参数，不可将这部分容量收益归因于选择器。
5. **实际任务**：先 RoBERTa-base/large 的 SST-2、MRPC、RTE、CoLA；每个方法至少三个种子，报告均值及标准差、完整学习率搜索范围。GLUE 验证集用于选择最佳 epoch 时必须承认选择偏差，正式最终评价使用独立测试/官方评测。
6. **泛化范围**：若只在一个小 GLUE 任务成立，证据不足。扩展至少一个更大模型或任务族，扩展代码后再报告，不把当前 encoder 支持写成 LLM 指令微调支持。
7. **效率分析**：分解一次性开销与稳态训练分别报告，观察探测步尖峰；同时报告总参数与实际更新比例。若无法比 all 更省时间，只能主张优化/正则化效果，不能主张计算加速。

继续投入的判据：自适应选择在多个任务中，以相同实际时间或适配器预算超过随机/固定策略，并能用模态轨迹解释收益。若只有对旧 core 的提升，最多证明学习子空间有用，尚不能证明新选择机制有用。

发表判断：这是一个可运行、可证伪的研究起点，尚不具备足以声称达到任何会议录用标准的实验证据。真正需要完成的是上述对照和对“选择收益”的归因，而不是再叠加秩增长、复杂门控或更多名称。
