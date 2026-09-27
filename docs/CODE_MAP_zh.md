# ASTRA-Core 代码导读

## v0.2：ASTRA-Tangent

新版默认 `--tuning-mode manifold`，原三种模式保留。算法推导、相关论文与实验计划见 [RESEARCH_zh.md](RESEARCH_zh.md)。

- `astra_core/manifold.py`：水平梯度、核心 Gram 预条件、极分解回缩、模态预算选择、混合优化器及控制器状态恢复。
- `astra_core/modeling.py`：manifold 模式将 Tucker 因子注册为参数；根据探测计划 detach 非活动因子，维持固定初始锚点。
- `astra_core/experiment.py`：优化器接入 Trainer、梯度累积、几何日志与资源统计。
- `astra_core/checkpoint.py`：利用保存的初始分解在原基座上恢复适配器。
- `tests/test_manifold.py`：几何性质、冻结/恢复、合成任务和微型 BERT 端到端验证。
- `scripts/smoke_train.py`：离线构造微型模型/数据，验证实际 CLI、四种策略 sweep 和最终检查点重载。
- `examples/sweep_tangent_sst2.json`：新版消融与多种子实验入口。

以下是原 ASTRA-Core/Mode/Hybrid 的导读；其中“固定因子”只适用于旧模式，不适用于新版 manifold 模式。

## 原版结构

这份文档对应当前仓库的 SFT 参考实现。和蓝图的关系如下：

- `additive` 模式对应 `ASTRA-Core`
- `multiplicative` 模式对应 `ASTRA-Mode`
- `both` 模式对应 `ASTRA-Hybrid`

当前仓库只覆盖 encoder-side GLUE 分类训练，不包含 DPO、PPO、GRPO、ASTRA-G、ASTRA-Muon。

## 入口文件

1. `astra_core/cli.py`
   负责命令行参数解析，并把参数交给实验执行层。
2. `astra_core/experiment.py`
   负责单次实验和 sweep 的总流程，包括数据加载、分解缓存、模型 patch、训练、评估和结果落盘。
3. `astra_core/modeling.py`
   负责 family 张量化、Tucker/HOOI 分解缓存、共享状态构造，以及把线性层替换成 ASTRA 适配层。

## 训练流程

1. 根据 `--target-families` 选择要处理的 attention / FFN family。
2. 把同一 family 的矩阵组织成高阶张量。
3. 对 family 张量做 Tucker/HOOI 分解，得到：
   - `base_core`
   - contracted factor matrices
   - remaining mode factors
4. 根据 `--tuning-mode` 构造可训练部分：
   - `additive`: 训练 `delta_core`
   - `multiplicative`: 训练 mode transform coefficients
   - `both`: 同时训练两者
5. 用 `TuckerAdaptedLinear` 回写到原模型线性层。
6. 训练结束后保存 trainable state、曲线、配置和参数统计。

## 关键类和函数

- `MultiplicativeModeAdapter`
  为每个 Tucker mode 学一个方阵变换。
- `TuckerFamilySharedState`
  保存 family 级共享状态，包括 `base_core`、可训练增量和 mode transforms。
- `TuckerAdaptedLinear`
  在前向中重建该层的有效权重。
- `build_attention_tucker_basis_cache`
  为 attention family 构造分解缓存。
- `build_ffn_tucker_basis_cache`
  为 FFN family 构造分解缓存。
- `get_or_create_decomposition_cache_entry`
  读取或创建缓存文件。
- `run_single_experiment`
  执行一次完整训练。

## 参数统计口径

`experiment.py` 中的 `compute_parameter_statistics()` 会输出：

- `adapter_params`
  只统计 `astra_core_shared_state_*` 下的可训练参数。
- `classifier_trainable_params`
  统计分类头可训练参数。
- `other_trainable_params`
  统计剩余可训练参数，理想情况下应接近 0。
- `total_trainable_params`
  全部可训练参数总数。
- `all_params`
  模型总参数量。
- `trainable_ratio`
  可训练参数占比。

## 输出文件

一次实验通常会产生：

- `trainable_state.pt`
- `training_config.json`
- `parameter_counts.json`
- `experiment_config.json`
- `experiment_summary.json`
- `train_loss_history.csv`
- `eval_history.csv`
- 训练曲线 PNG

## 常用命令

```bash
python -m astra_core \
  --model-path /data/models/roberta-large \
  --dataset-path /data/datasets/glue_sst2 \
  --glue-task sst2 \
  --target-families q k v o \
  --attn-ranks 4 4 16 32 \
  --ffn-ranks 2 4 64 64 \
  --tuning-mode additive \
  --multiplicative-num-bases 50 \
  --learning-rate 1e-3 \
  --num-train-epochs 3 \
  --run-name astra_core_sst2
```

后台运行示例：

```bash
nohup python -m astra_core \
  --model-path /data/models/roberta-large \
  --dataset-path /data/datasets/glue_sst2 \
  --glue-task sst2 \
  --target-families q k v o \
  --attn-ranks 4 4 16 32 \
  --ffn-ranks 2 4 64 64 \
  --tuning-mode both \
  --multiplicative-num-bases 50 \
  --learning-rate 1e-3 \
  --num-train-epochs 3 \
  --run-name astra_hybrid_sst2 \
  > astra_hybrid_sst2.log 2>&1 &
```
