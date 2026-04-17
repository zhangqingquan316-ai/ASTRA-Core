# TensLoRA-XS 代码导读

这份文档主要服务于“读懂代码”和“对齐论文操作”，对应的公开仓库根目录见上一级 `README.md`。

## 推荐阅读顺序

1. `tenslora_xs/cli.py`
   只负责命令行参数解析。
2. `tenslora_xs/experiment.py`
   负责完整实验流程：加载模型、读取 decomposition cache、训练、评估、保存结果。
3. `tenslora_xs/modeling.py`
   负责最核心的张量化、Tucker 分解缓存、乘性/加性调节和 patch 逻辑。

## 这份代码和 HOSVD 版最大的不同

- HOSVD 版更接近：
  - 固定 basis
  - 训练一个小的 `delta_core`
- PM 版在此基础上加入了：
  - 对 Tucker core 每个 mode 的乘性线性变换
  - 每个 mode 的变换不是直接学习一个满矩阵，而是学习固定 basis 的线性组合系数

最值得重点看的函数：

- `MultiplicativeModeAdapter`
- `TuckerFamilySharedState`
- `TuckerAdaptedLinear.forward`
- `build_attention_tucker_basis_cache`
- `build_ffn_tucker_basis_cache`
- `get_or_create_decomposition_cache_entry`
- `run_single_experiment`

## 和论文操作的对应关系

- 先对一个 family 的原始权重做 Tucker/HOOI 分解，得到：
  - `base_core`
  - 各 mode 的 factor matrices
- 训练阶段有三种模式：
  - `additive`：只训练 `delta_core`
  - `multiplicative`：只训练每个 mode 的变换系数
  - `both`：两者都训练
- 每个被替换的线性层共享同一个 family 的状态：
  - 同一个 `base_core`
  - 同一组可训练参数
  - 根据自己所在层或阶段选择不同的 contracted factor row
- 最终有效权重不是“直接覆盖原权重”，而是：
  - 先重建一个调节后的 Tucker 权重
  - 再和预先保存的 residual weight 拼回完整线性层

## 参数量统计怎么看

- `adapter_params`
  - 只统计 `tucker_shared_state_*` 里的可训练参数
  - 对应这套算法真正新增并参与训练的 adapter 参数量
  - 在 `additive` 模式下，主要是 `delta_core`
  - 在 `multiplicative` 模式下，主要是每个 mode 的 basis 组合系数
  - 在 `both` 模式下，两部分都会被统计进去
- `classifier_trainable_params`
  - 分类头的可训练参数量，不算进 adapter 本体
- `other_trainable_params`
  - 理论上通常应当接近 0
  - 如果这里明显大于 0，说明还有别的参数被意外解冻了
- `all_params`
  - 整个模型总参数量，包括冻结参数

## 参数量最后会不会保存

会保存，而且保存了两份，便于之后做实验核对：

- `trainable_state.pt`
  - 真正的可训练参数权重
- `training_config.json`
  - 完整实验配置，其中包含 `parameter_stats`
- `parameter_counts.json`
  - 单独导出的参数量统计文件，适合快速查看

## 服务器运行命令示例

单次实验：

```bash
python -m tenslora_xs \
  --model-path /data/models/roberta-large \
  --dataset-path /data/datasets/glue_sst2 \
  --glue-task sst2 \
  --target-families q k v o \
  --attn-ranks 4 4 16 32 \
  --ffn-ranks 2 4 64 64 \
  --attn-alpha 1.0 \
  --ffn-alpha 1.0 \
  --tuning-mode additive \
  --multiplicative-num-bases 50 \
  --learning-rate 1e-3 \
  --per-device-train-batch-size 32 \
  --per-device-eval-batch-size 32 \
  --num-train-epochs 3 \
  --run-name pm_additive_sst2
```

后台运行：

```bash
nohup python -m tenslora_xs \
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
  --run-name pm_both_sst2 \
  > pm_both_sst2.log 2>&1 &
```
