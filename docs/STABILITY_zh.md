# jyr：锚点计算与模态选择的稳定性

## 两类问题分别处理

旧实现虽然在实数中等价，但 `W0 + (X-X0)` 与 `(W0-X0)+X` 在有限精度、
CPU/GPU 重构及混合精度转换下并不等价。另一个独立问题是，密度贪心规则可能
在模态数上限内优先选择很便宜、但能量贡献很小的模态，未能达到声明的覆盖率；
离散决策又会把轨迹差异反馈到后续梯度。

本分支保留旧实现作为对照，并提供三个独立开关：

| 功能 | 新实现 | 兼容实现 |
|---|---|---|
| 权重表示 | `--weight-layout delta` | `two_matrix`、`residual`、`cpu_anchor` |
| 几何计算 | `--manifold-numerical-mode stable` | `legacy` |
| 模态选择 | `--manifold-policy stable` | `adaptive`、`all`、`core`、`random` |

命令行默认使用三个新实现。Python 底层构造器保留旧默认值，以免无参数调用的
已有程序静默改变算法；通过 Python API 使用新实现时应显式设置这三个选项。
旧 sweep JSON 未指定的新字段会继承 CLI 默认值。复现实验必须完整记录三个开关。

## 直接计算增量

不再计算两个大 Tucker 张量之差。保存冻结的 `C0,U0` 和独立的 `delta_core`。
在每次模态乘法中维护基准张量 `B` 与增量 `D`：

```text
D_next = U D + (U-U0) B
B_next = U0 B
```

层/阶段索引的向量收缩同样应用这个递推。最后只重构增量矩阵，执行
`base_linear(x) + alpha * linear(x, delta_weight)`。
初始增量严格为零；不依赖 CPU 与 GPU 对同一大张量的重构舍入恰好一致。
增量重构禁用 autocast，采用 FP32（FP64 输入则保留 FP64）。
加性、乘性、联合及 manifold 模式均支持；冻结/未激活模态的历史变化仍参与前向。

此实现不保存每层稠密的 `X0`，只额外保存每个共享族的初始紧凑因子。
代价是递推和独立线性分支增加计算。最终激活仍受 AMP 的舍入限制，不能声称任意
微小更新都能改变 BF16 输出，也不承诺与旧训练轨迹逐位一致。

`cpu_anchor` 用固定 CPU 锚点保留旧 `two_matrix` 的算术顺序。每次前向上传锚点，
显存下降但增加主存占用和传输；`host_anchor_bytes` 单独报告这部分内存。
它适合作为旧训练行为的兼容对照，不是消除相减误差的方案。

## 稳定评分与更新

旧评分本来就不显式含学习率。新实现进一步在平方之前用同一族全部可用梯度的
最大绝对值 `s` 归一化，在 FP64 中做投影、core Gram 和小矩阵求解：

```text
e0 = ||grad_C / s||²
ek = <Hk/s, (Hk/s) (C_(k) C_(k)^T + lambda I)^(-1)>
```

真实方向再乘回 `s`，因此没有偷偷归一化实际优化步长。所有评分共享同一比例，
随后归一化为总能量份额。历史记录的 `energy_units=gradient_scale_squared`
表示记录的是除以 `s²` 的能量，不能与旧日志的绝对 `e_k` 直接比较。
这些量是几何信号代理，不是混合 AdamW 优化器实际损失下降的保证。

缩放不变性仅指同一状态下共同缩放梯度后的选择；改变训练学习率仍会改变后续
参数和梯度。输入梯度已经下溢为零的信息无法恢复。

更新采用小 Gram 的增量 polar：
`A=UᵀS+SᵀU+SᵀS`，稳定计算 `(I+A)^(-1/2)-I` 后应用修正。
零步长严格不动。FP32 因子已有的正交误差低于 `1e-5` 时不反复投影；
超过容差才完整纠正。这样避免原 SVD 在极小步长下制造比意图更新更大的移动。
范数裁剪先缩放后求范数，避免巨大方向的平方溢出。日志同时记录意图与实际移动。

## 有限状态的覆盖选择器

每个共享族各自维护状态，只在完整探测时更新：

1. 对当次能量份额做 EMA，默认 `score_ema=0.9`。
2. 逃逸门使用 `tau ± escape_hysteresis`，默认 `0.15 ± 0.03`；
   `tau=0` 明确关闭门控。
3. 最多四个模态，枚举不超过 `max_active_modes` 的所有候选集合。
   能达到覆盖目标时选代理成本最低者；达不到时选覆盖最高者，并记录不可达。
   成本仍是既有的 `[r(n-r)]^cost_power` 代理规则，不是测量 FLOPs。
4. 现有集合在覆盖容差内且成本不高于候选集合时保留。默认覆盖容差 `0.03`。
5. 新集合连续出现 `confirm_probes=3` 次才切换。近似并列用 `decision_tolerance=1e-3`
   和固定索引顺序裁决。平滑与确认会延迟响应，属于明确的稳定性/适应性权衡。

新策略应查看 `smoothed_energy_shares`、`pending_modes`、`gate_open` 与覆盖可达性。
兼容日志中的 `s_k` / `score_order` 仍是旧密度指标，仅供对照，不表示新规则的选择顺序。

全部因子信号为零时立即清空激活与待切换状态；方形因子和零预算不激活。
能量整体极大或极小而相对结构相同时，决策不变。几何/输入超出有限值范围时应显式
报错或跳过非有限梯度，不应悄悄给出随机选择。

旧 `random` 对照仍匹配旧 `adaptive` 的候选数量。它不自动成为新 `stable` 策略的
等成本随机对照；论文比较时需重新匹配实际更新成本。

## 保存、恢复与验证

优化器 checkpoint 包含 EMA、门控、激活集合、待切换集合及确认计数；恢复后继续
同一轨迹。推理 adapter 的 `training_config.json` 新增 `weight_layout`。
加载器先读该字段，再查相邻 `experiment_config.json`；都缺失时警告并按历史
`two_matrix` 解释。无元数据的 residual 文件需显式传入 `weight_layout="residual"`。

```bash
python -m unittest discover -s tests
python scripts/replay_mode_selection.py path/to/manifold_history.json
python scripts/stability_cola.py --project /path/to/project --output /path/to/new/results \
  --variant stable --seed 42 --stop-epochs 20 --schedule-epochs 40
```

重放固定旧日志只检验选择器，不能推断重新训练后的分数。CoLA 对照保留原 40 epoch
学习率调度，在第 20 epoch 停止，以便与历史前 20 epoch 比较。
`baseline` 为 CPU 锚点＋旧几何＋旧选择器；`numeric` 为增量＋新几何＋旧选择器；
`stable` 为增量＋新几何＋新选择器。结果应同时报告 MCC、切换次数、能量覆盖、
实际成本和内存，不能仅凭更少切换宣称多种子结果更稳定。
