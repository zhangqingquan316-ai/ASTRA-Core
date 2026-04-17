import argparse
import csv
import gc
import json
import math
import os
import zlib
from copy import deepcopy
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None

import numpy as np
import tensorly as tl
import torch
import torch.nn as nn
from datasets import load_from_disk
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

tl.set_backend("pytorch")

ATTENTION_FAMILY_SUFFIXES = {
    "q": [".attention.self.query", ".self_attn.q_proj", ".q_proj"],
    "k": [".attention.self.key", ".self_attn.k_proj", ".k_proj"],
    "v": [".attention.self.value", ".self_attn.v_proj", ".v_proj"],
    "o": [
        ".attention.output.dense",
        ".self_attn.o_proj",
        ".self_attn.out_proj",
        ".encoder_attn.o_proj",
        ".encoder_attn.out_proj",
    ],
}

MODULE_TO_FAMILY = {
    "query": "q",
    "q_proj": "q",
    "q": "q",
    "key": "k",
    "k_proj": "k",
    "k": "k",
    "value": "v",
    "v_proj": "v",
    "v": "v",
    "attention.output.dense": "o",
    "o_proj": "o",
    "out_proj": "o",
    "o": "o",
    "ffn": "ffn",
}

EXCLUDED_MODULE_KEYWORDS = (
    "classifier",
    "lm_head",
    "qa_outputs",
    "pooler",
    "predictions",
    "seq_relationship",
)

TASK_CONFIGS = {
    "sst2": {
        "text_keys": ("sentence", None),
        "num_labels": 2,
        "metric": "accuracy",
        "is_regression": False,
    },
    "mrpc": {
        "text_keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "metric": "accuracy",
        "is_regression": False,
    },
    "cola": {
        "text_keys": ("sentence", None),
        "num_labels": 2,
        "metric": "matthews",
        "is_regression": False,
    },
    "qnli": {
        "text_keys": ("question", "sentence"),
        "num_labels": 2,
        "metric": "accuracy",
        "is_regression": False,
    },
    "rte": {
        "text_keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "metric": "accuracy",
        "is_regression": False,
    },
    "stsb": {
        "text_keys": ("sentence1", "sentence2"),
        "num_labels": 1,
        "metric": "pearson",
        "is_regression": True,
    },
}

SWEEP_KEY_ALIASES = {
    "attn-ranks": "attn_ranks",
    "ffn-ranks": "ffn_ranks",
    "attn-alpha": "attn_alpha",
    "ffn-alpha": "ffn_alpha",
    "tuning-mode": "tuning_mode",
    "multiplicative-num-bases": "multiplicative_num_bases",
    "num-train-epochs": "num_train_epochs",
    "learning-rate": "learning_rate",
    "per-device-train-batch-size": "per_device_train_batch_size",
    "per-device-eval-batch-size": "per_device_eval_batch_size",
    "weight-decay": "weight_decay",
    "target-families": "target_families",
    "target-modules": "target_modules",
    "max-length": "max_length",
}


def get_base_model(model):
    prefix = getattr(model, "base_model_prefix", None)
    if prefix and hasattr(model, prefix):
        return getattr(model, prefix), prefix
    raise ValueError("Could not resolve the base encoder model from AutoModelForSequenceClassification.")


def get_encoder_layers(model):
    base_model, base_prefix = get_base_model(model)
    encoder = getattr(base_model, "encoder", None)
    layer_stack = getattr(encoder, "layer", None)
    if layer_stack is None:
        raise ValueError("This script expects the base model to expose encoder.layer.")
    return list(layer_stack), base_prefix


def get_attention_geometry(model):
    num_heads = getattr(model.config, "num_attention_heads", None)
    hidden_size = getattr(model.config, "hidden_size", None)
    if num_heads is None or hidden_size is None:
        raise ValueError(
            "Model config must define num_attention_heads and hidden_size "
            "to build a head-aware 4D attention tensor."
        )
    if hidden_size % num_heads != 0:
        raise ValueError(
            f"hidden_size={hidden_size} is not divisible by num_attention_heads={num_heads}."
        )
    return num_heads, hidden_size // num_heads, hidden_size


def get_ffn_geometry(model):
    hidden_size = getattr(model.config, "hidden_size", None)
    intermediate_size = getattr(model.config, "intermediate_size", None)
    if hidden_size is None or intermediate_size is None:
        raise ValueError(
            "Model config must define hidden_size and intermediate_size to build the FFN tensor."
        )
    return intermediate_size, hidden_size


def tensor_to_matrix(tensor, matrix_shape, reshape_kind):
    if reshape_kind == "direct":
        return tensor.contiguous().reshape(matrix_shape)
    if reshape_kind == "transpose":
        d_out, d_in = matrix_shape
        return tensor.contiguous().reshape(d_in, d_out).t().contiguous()
    raise ValueError(f"Unsupported reshape kind: {reshape_kind}")


def matrix_to_attention_tensor(weight, family_name, num_heads):
    d_out, d_in = weight.shape

    if family_name in {"q", "k", "v"}:
        if d_out % num_heads != 0:
            raise ValueError(
                f"{family_name} matrix with shape {(d_out, d_in)} cannot be split into "
                f"{num_heads} attention heads along the output dimension."
            )
        head_dim = d_out // num_heads
        tensor = weight.contiguous().reshape(num_heads, head_dim, d_in)
        reshape_kind = "direct"
    elif family_name == "o":
        if d_in % num_heads != 0:
            raise ValueError(
                f"{family_name} matrix with shape {(d_out, d_in)} cannot be split into "
                f"{num_heads} attention heads along the input dimension."
            )
        head_dim = d_in // num_heads
        tensor = weight.t().contiguous().reshape(num_heads, head_dim, d_out)
        reshape_kind = "transpose"
    else:
        raise ValueError(f"Unsupported attention family: {family_name}")

    return tensor, {
        "matrix_shape": (d_out, d_in),
        "tensor_shape": tuple(tensor.shape),
        "reshape_kind": reshape_kind,
    }


def matrix_to_canonical_ffn_tensor(weight, canonical_shape=None):
    direct_shape = tuple(weight.shape)
    transpose_shape = (weight.shape[1], weight.shape[0])

    if canonical_shape is None:
        canonical_shape = (max(direct_shape), min(direct_shape))

    if direct_shape == canonical_shape:
        return weight.contiguous(), canonical_shape, "direct"
    if transpose_shape == canonical_shape:
        return weight.t().contiguous(), canonical_shape, "transpose"

    raise ValueError(
        f"FFN matrix with shape {direct_shape} cannot be aligned to canonical shape {canonical_shape}."
    )


def build_fixed_basis_matrices(rank, num_bases, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    basis = torch.randn(num_bases, rank, rank, generator=generator, dtype=torch.float32)
    return (basis / math.sqrt(max(rank, 1))).contiguous()


def build_transform_seed(base_seed, family_name, mode_index):
    seed_payload = f"{family_name}:{mode_index}:{int(base_seed)}".encode("utf-8")
    return zlib.crc32(seed_payload) & 0xFFFFFFFF


class MultiplicativeModeAdapter(nn.Module):
    def __init__(self, rank, num_bases, seed):
        super().__init__()
        self.rank = int(rank)
        self.num_bases = int(num_bases)

        if self.rank <= 0:
            raise ValueError(f"Factor rank must be positive, got {self.rank}.")
        if self.num_bases <= 0:
            raise ValueError(
                "Multiplicative tuning requires a positive number of basis matrices for each Tucker-core mode transform. "
                f"Got {self.num_bases}."
            )

        self.register_buffer(
            "basis_matrices",
            build_fixed_basis_matrices(self.rank, self.num_bases, seed),
        )
        self.register_buffer("identity", torch.eye(self.rank, dtype=torch.float32))
        self.coefficients = nn.Parameter(torch.zeros(self.num_bases, dtype=torch.float32))

    def get_transform(self, dtype):
        scaled_basis = self.coefficients.to(dtype).view(-1, 1, 1) * self.basis_matrices.to(dtype)
        return self.identity.to(dtype) + scaled_basis.sum(dim=0)

class TuckerFamilySharedState(nn.Module):
    def __init__(
        self,
        family_name,
        base_core,
        factor_matrices,
        remaining_mode_factors,
        tuning_mode,
        multiplicative_num_bases,
        base_seed,
    ):
        super().__init__()
        self.family_name = family_name
        self.tuning_mode = tuning_mode
        self.has_additive = tuning_mode in {"additive", "both"}
        self.has_multiplicative = tuning_mode in {"multiplicative", "both"}
        self.num_contracted_modes = len(factor_matrices)
        self.num_remaining_modes = len(remaining_mode_factors)

        self.register_buffer("base_core", base_core.contiguous())
        if self.has_additive:
            self.delta_core = nn.Parameter(torch.zeros_like(base_core).contiguous())
        else:
            self.register_buffer("delta_core_placeholder", torch.zeros_like(base_core).contiguous())

        for idx, factor in enumerate(factor_matrices):
            self.register_buffer(f"contracted_factor_matrix_{idx}", factor.contiguous())
        for idx, factor in enumerate(remaining_mode_factors):
            self.register_buffer(f"remaining_mode_factor_{idx}", factor.contiguous())

        mode_ranks = [factor.shape[1] for factor in factor_matrices] + [
            factor.shape[1] for factor in remaining_mode_factors
        ]
        if self.has_multiplicative:
            self.mode_transform_adapters = nn.ModuleList(
                [
                    MultiplicativeModeAdapter(
                        rank=rank,
                        num_bases=multiplicative_num_bases,
                        seed=build_transform_seed(base_seed, family_name, idx),
                    )
                    for idx, rank in enumerate(mode_ranks)
                ]
            )
        else:
            self.mode_transform_adapters = nn.ModuleList()

    def get_updated_core(self, dtype):
        base_core = self.base_core.to(dtype)
        if self.has_additive:
            return base_core + self.delta_core.to(dtype)
        return base_core

    def get_base_core(self, dtype):
        return self.base_core.to(dtype)

    def get_delta_core(self, dtype):
        if self.has_additive:
            return self.delta_core.to(dtype)
        return torch.zeros_like(self.base_core, dtype=dtype)

    def get_contracted_factor_row(self, mode_idx, row_idx, dtype):
        factor = getattr(self, f"contracted_factor_matrix_{mode_idx}")
        return factor.to(dtype)[int(row_idx)]

    def get_remaining_factors(self, dtype):
        factors = []
        for idx in range(self.num_remaining_modes):
            factor = getattr(self, f"remaining_mode_factor_{idx}")
            factors.append(factor.to(dtype))
        return factors

    def apply_mode_transforms(self, core_tensor, dtype):
        if not self.has_multiplicative:
            return core_tensor

        transforms = [adapter.get_transform(dtype) for adapter in self.mode_transform_adapters]
        return tl.tenalg.multi_mode_dot(
            core_tensor,
            transforms,
            modes=list(range(self.num_contracted_modes + self.num_remaining_modes)),
        )


class TuckerAdaptedLinear(nn.Module):
    def __init__(
        self,
        base_layer,
        family_name,
        shared_state,
        contracted_indices,
        matrix_shape,
        reshape_kind,
        alpha,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.family_name = family_name
        self.matrix_shape = tuple(matrix_shape)
        self.reshape_kind = reshape_kind
        self.alpha = float(alpha)
        self.contracted_indices = tuple(int(index) for index in contracted_indices)

        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        object.__setattr__(self, "shared_state", shared_state)

        with torch.no_grad():
            base_dtype = self.base_layer.weight.dtype
            base_device = self.base_layer.weight.device
            prior_tensor = self._compose_tensor(
                self.shared_state.get_base_core(torch.float32),
                current_dtype=torch.float32,
                apply_mode_transforms=False,
            )
            prior_weight = tensor_to_matrix(prior_tensor, self.matrix_shape, self.reshape_kind)
            residual_weight = self.base_layer.weight.detach().to(torch.float32) - prior_weight

        self.register_buffer(
            "prior_tucker_weight",
            prior_weight.to(device=base_device, dtype=base_dtype).contiguous(),
        )
        self.register_buffer(
            "residual_weight",
            residual_weight.to(device=base_device, dtype=base_dtype).contiguous(),
        )

    def _compose_tensor(self, core_tensor, current_dtype, apply_mode_transforms):
        if apply_mode_transforms:
            core_tensor = self.shared_state.apply_mode_transforms(core_tensor, current_dtype)

        reduced_core = core_tensor
        for idx, factor_index in enumerate(self.contracted_indices):
            factor_row = self.shared_state.get_contracted_factor_row(
                mode_idx=idx,
                row_idx=factor_index,
                dtype=current_dtype,
            )
            reduced_core = torch.tensordot(factor_row, reduced_core, dims=([0], [0]))

        remaining_mode_factors = self.shared_state.get_remaining_factors(dtype=current_dtype)
        return tl.tenalg.multi_mode_dot(
            reduced_core,
            remaining_mode_factors,
            modes=list(range(self.shared_state.num_remaining_modes)),
        )

    def forward(self, x):
        current_dtype = x.dtype

        if self.shared_state.tuning_mode == "additive":
            adapted_tensor = self._compose_tensor(
                self.shared_state.get_updated_core(current_dtype),
                current_dtype=current_dtype,
                apply_mode_transforms=False,
            )
        elif self.shared_state.tuning_mode == "multiplicative":
            adapted_tensor = self._compose_tensor(
                self.shared_state.get_base_core(current_dtype),
                current_dtype=current_dtype,
                apply_mode_transforms=True,
            )
        else:
            adapted_tensor = self._compose_tensor(
                self.shared_state.get_updated_core(current_dtype),
                current_dtype=current_dtype,
                apply_mode_transforms=True,
            )

        adapted_weight = tensor_to_matrix(adapted_tensor, self.matrix_shape, self.reshape_kind)
        prior_weight = self.prior_tucker_weight.to(current_dtype)
        tuned_tucker_weight = prior_weight + self.alpha * (adapted_weight - prior_weight)
        effective_weight = self.residual_weight.to(current_dtype) + tuned_tucker_weight

        output = torch.matmul(x, effective_weight.t())
        if self.base_layer.bias is not None:
            output = output + self.base_layer.bias.to(current_dtype)
        return output


def matches_attention_family(name, module, family_name):
    if family_name not in ATTENTION_FAMILY_SUFFIXES:
        raise ValueError(f"Unsupported attention family: {family_name}")
    if not hasattr(module, "weight") or module.weight is None:
        return False
    if len(module.weight.shape) != 2:
        return False

    lowered_name = name.lower()
    if any(keyword in lowered_name for keyword in EXCLUDED_MODULE_KEYWORDS):
        return False
    return any(lowered_name.endswith(suffix) for suffix in ATTENTION_FAMILY_SUFFIXES[family_name])


def build_attention_tucker_basis_cache(model, family_name, num_heads, ranks):
    family_items = []

    for name, module in model.named_modules():
        if matches_attention_family(name, module, family_name):
            tensor, metadata = matrix_to_attention_tensor(
                module.weight.detach().cpu(),
                family_name,
                num_heads,
            )
            family_items.append((name, tensor, metadata))

    if not family_items:
        raise ValueError(f"Could not find any layers for attention family '{family_name}'.")

    tensor_shapes = {item[2]["tensor_shape"] for item in family_items}
    if len(tensor_shapes) != 1:
        raise ValueError(
            f"Attention family '{family_name}' has mismatched tensor shapes: {sorted(tensor_shapes)}"
        )

    layer_paths = [item[0] for item in family_items]
    family_tensors = [item[1] for item in family_items]
    family_metadata = family_items[0][2]

    full_tensor_shape = (len(family_tensors),) + family_metadata["tensor_shape"]
    for rank_value, dim_size in zip(ranks, full_tensor_shape):
        if rank_value > dim_size:
            raise ValueError(
                f"Attention ranks {ranks} are incompatible with family '{family_name}' tensor shape "
                f"{full_tensor_shape}."
            )

    print(
        f"[{family_name}] Tucker/HOOI tensor shape: {full_tensor_shape} "
        f"(layer, head, head_dim, hidden)"
    )

    family_tensor = torch.stack(family_tensors, dim=0).to(torch.float32).numpy()
    tl.set_backend("numpy")
    core, factors = tl.decomposition.tucker(family_tensor, rank=ranks, init="svd")
    tl.set_backend("pytorch")
    del family_tensor

    module_specs = [
        {
            "path": path,
            "contracted_indices": (layer_idx,),
            "matrix_shape": family_metadata["matrix_shape"],
            "reshape_kind": family_metadata["reshape_kind"],
        }
        for layer_idx, path in enumerate(layer_paths)
    ]

    return {
        "family_name": family_name,
        "method": "tucker_hooi_svd_init",
        "ranks": tuple(ranks),
        "base_core": torch.from_numpy(core).contiguous(),
        "factor_matrices": [torch.from_numpy(factors[0]).contiguous()],
        "remaining_mode_factors": [torch.from_numpy(factor).contiguous() for factor in factors[1:]],
        "module_specs": module_specs,
        "tensor_shape": full_tensor_shape,
    }


def build_ffn_tucker_basis_cache(model, ranks):
    encoder_layers, base_prefix = get_encoder_layers(model)
    canonical_shape = None
    stage_names = None
    stage_tensors = None
    module_specs = []

    for layer_idx, layer in enumerate(encoder_layers):
        current_stage_modules = []
        for local_name, module in layer.named_modules():
            if not local_name or not isinstance(module, nn.Linear):
                continue
            lowered_name = local_name.lower()
            if "attention" in lowered_name:
                continue
            current_stage_modules.append((local_name, module))

        if not current_stage_modules:
            raise ValueError(f"Layer {layer_idx} has no FFN linear modules.")

        current_stage_names = [name for name, _ in current_stage_modules]
        if stage_names is None:
            stage_names = current_stage_names
            stage_tensors = [[] for _ in stage_names]
        elif current_stage_names != stage_names:
            raise ValueError(
                f"FFN linear-module layout is inconsistent across layers. "
                f"Layer 0: {stage_names}, layer {layer_idx}: {current_stage_names}"
            )

        for stage_idx, (local_name, module) in enumerate(current_stage_modules):
            tensor, canonical_shape, reshape_kind = matrix_to_canonical_ffn_tensor(
                module.weight.detach().cpu(),
                canonical_shape=canonical_shape,
            )
            stage_tensors[stage_idx].append(tensor)
            module_specs.append(
                {
                    "path": f"{base_prefix}.encoder.layer.{layer_idx}.{local_name}",
                    "contracted_indices": (stage_idx, layer_idx),
                    "matrix_shape": tuple(module.weight.shape),
                    "reshape_kind": reshape_kind,
                }
            )

    stage_count = len(stage_names)
    num_layers = len(encoder_layers)
    full_tensor_shape = (stage_count, num_layers) + canonical_shape
    for rank_value, dim_size in zip(ranks, full_tensor_shape):
        if rank_value > dim_size:
            raise ValueError(
                f"FFN ranks {ranks} are incompatible with FFN tensor shape {full_tensor_shape}."
            )

    print(
        f"[ffn] Tucker/HOOI tensor shape: {full_tensor_shape} "
        f"(ffn_stage, layer, m, n); stage names: {stage_names}"
    )

    family_tensor = torch.stack(
        [torch.stack(stage_tensor_list, dim=0) for stage_tensor_list in stage_tensors],
        dim=0,
    ).to(torch.float32).numpy()

    tl.set_backend("numpy")
    core, factors = tl.decomposition.tucker(family_tensor, rank=ranks, init="svd")
    tl.set_backend("pytorch")
    del family_tensor

    return {
        "family_name": "ffn",
        "method": "tucker_hooi_svd_init",
        "ranks": tuple(ranks),
        "base_core": torch.from_numpy(core).contiguous(),
        "factor_matrices": [
            torch.from_numpy(factors[0]).contiguous(),
            torch.from_numpy(factors[1]).contiguous(),
        ],
        "remaining_mode_factors": [torch.from_numpy(factor).contiguous() for factor in factors[2:]],
        "module_specs": module_specs,
        "tensor_shape": full_tensor_shape,
        "stage_names": stage_names,
        "canonical_shape": canonical_shape,
    }


def build_runtime_family_from_decomposition_cache(
    cache_entry,
    alpha,
    tuning_mode,
    multiplicative_num_bases,
    base_seed,
):
    return {
        "family_name": cache_entry["family_name"],
        "base_core": cache_entry["base_core"].clone().contiguous(),
        "factor_matrices": [factor.clone().contiguous() for factor in cache_entry["factor_matrices"]],
        "remaining_mode_factors": [
            factor.clone().contiguous() for factor in cache_entry["remaining_mode_factors"]
        ],
        "module_specs": cache_entry["module_specs"],
        "tensor_shape": cache_entry["tensor_shape"],
        "ranks": tuple(cache_entry["ranks"]),
        "tuning_mode": tuning_mode,
        "multiplicative_num_bases": int(multiplicative_num_bases),
        "base_seed": int(base_seed),
        "alpha": alpha,
    }


def resolve_target_families(args):
    if args.target_modules:
        return infer_families_from_modules(args.target_modules)
    return list(args.target_families)


def build_decomposition_cache_file(cache_root, family_name, ranks):
    rank_part = "-".join(str(x) for x in ranks)
    return cache_root / f"{family_name}_r{rank_part}.pt"


def build_decomposition_cache_key(model_path, family_name, ranks):
    return (str(Path(model_path).resolve()), family_name, tuple(ranks))


def load_cached_decomposition_entry(cache_file, resolved_model_path, family_name, ranks):
    if not cache_file.exists():
        return None

    try:
        cache_entry = torch.load(cache_file, map_location="cpu")
    except Exception as exc:
        print(f"Warning: failed to load cache file {cache_file}. Recomputing it. Error: {exc}")
        return None

    if cache_entry.get("model_path") != resolved_model_path:
        print(f"Warning: cache file {cache_file} belongs to a different model path. Recomputing it.")
        return None

    if cache_entry.get("family_name") != family_name:
        print(f"Warning: cache file {cache_file} has mismatched family metadata. Recomputing it.")
        return None

    if tuple(cache_entry.get("ranks", ())) != tuple(ranks):
        print(f"Warning: cache file {cache_file} has mismatched rank metadata. Recomputing it.")
        return None

    required_keys = ("base_core", "factor_matrices", "remaining_mode_factors", "module_specs", "tensor_shape")
    if any(key not in cache_entry for key in required_keys):
        print(
            f"Warning: cache file {cache_file} is missing fields required by the current script. "
            "Recomputing it."
        )
        return None

    return cache_entry


def get_or_create_decomposition_cache_entry(
    model,
    model_path,
    cache_root,
    family_name,
    ranks,
    num_heads,
    shared_cache_entries,
):
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    resolved_model_path = str(Path(model_path).resolve())
    ranks = tuple(ranks)
    cache_key = build_decomposition_cache_key(model_path, family_name, ranks)

    if shared_cache_entries is not None and cache_key in shared_cache_entries:
        print(f"Reusing in-memory Tucker/HOOI cache for family '{family_name}' with ranks {ranks}")
        return shared_cache_entries[cache_key]

    cache_file = build_decomposition_cache_file(cache_root, family_name, ranks)
    cache_entry = load_cached_decomposition_entry(
        cache_file=cache_file,
        resolved_model_path=resolved_model_path,
        family_name=family_name,
        ranks=ranks,
    )

    if cache_entry is None:
        print(f"Cache miss for family '{family_name}' with ranks {ranks}. Computing Tucker/HOOI now...")
        if family_name == "ffn":
            cache_entry = build_ffn_tucker_basis_cache(model, ranks)
        else:
            cache_entry = build_attention_tucker_basis_cache(
                model=model,
                family_name=family_name,
                num_heads=num_heads,
                ranks=ranks,
            )

        cache_entry["cache_file"] = str(cache_file.resolve())
        cache_entry["model_path"] = resolved_model_path
        torch.save(cache_entry, cache_file)
        print(f"Saved Tucker/HOOI decomposition cache to {cache_file}")
    else:
        print(f"Loading Tucker/HOOI decomposition cache for family '{family_name}' from {cache_file}")

    if shared_cache_entries is not None:
        shared_cache_entries[cache_key] = cache_entry

    return cache_entry


def validate_local_inputs(model_path, dataset_path):
    model_dir = Path(model_path)
    dataset_dir = Path(dataset_path)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Local model directory does not exist: {model_dir.resolve()}. "
            "Please point --model-path to the real Hugging Face model folder."
        )
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Local dataset directory does not exist: {dataset_dir.resolve()}. "
            "Please point --dataset-path to the real datasets.save_to_disk folder."
        )


def get_task_config(task_name):
    if task_name not in TASK_CONFIGS:
        raise ValueError(
            f"Unsupported GLUE task '{task_name}'. Supported tasks: {sorted(TASK_CONFIGS)}"
        )
    return TASK_CONFIGS[task_name]


def get_primary_eval_metric_key(task_name):
    return f"eval_{get_task_config(task_name)['metric']}"


def get_prefixed_metric_key(task_name, prefix):
    return f"{prefix}_{get_task_config(task_name)['metric']}"


def infer_families_from_modules(target_modules):
    families = []
    for module_name in target_modules:
        family = MODULE_TO_FAMILY.get(module_name.lower())
        if family is None:
            raise ValueError(
                f"Unsupported target module '{module_name}'. "
                f"Supported values include: {sorted(MODULE_TO_FAMILY)}"
            )
        if family not in families:
            families.append(family)
    return families


def compute_binary_accuracy(predictions, labels):
    return float((predictions == labels).mean())


def compute_matthews_correlation(predictions, labels):
    predictions = predictions.astype(np.int64)
    labels = labels.astype(np.int64)

    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())

    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator == 0:
        return 0.0

    return float((tp * tn - fp * fn) / np.sqrt(denominator))


def compute_pearson_correlation(predictions, labels):
    predictions = predictions.astype(np.float64)
    labels = labels.astype(np.float64)
    pred_std = predictions.std()
    label_std = labels.std()
    if pred_std == 0 or label_std == 0:
        return 0.0
    return float(np.corrcoef(predictions, labels)[0, 1])


def build_compute_metrics(task_name):
    task_config = get_task_config(task_name)
    metric_name = task_config["metric"]
    is_regression = task_config["is_regression"]

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.squeeze(logits) if is_regression else np.argmax(logits, axis=-1)

        if metric_name == "accuracy":
            return {"accuracy": compute_binary_accuracy(predictions, labels)}
        if metric_name == "matthews":
            return {"matthews": compute_matthews_correlation(predictions, labels)}
        if metric_name == "pearson":
            return {"pearson": compute_pearson_correlation(predictions, labels)}
        raise ValueError(f"Unsupported metric '{metric_name}' for task '{task_name}'.")

    return compute_metrics


def get_trainable_state(model):
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def load_trainable_state(model, state_dict):
    if state_dict is None:
        return
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in state_dict.items():
            if name in named_params:
                param = named_params[name]
                param.copy_(tensor.to(device=param.device, dtype=param.dtype))


class BestModelByMetricCallback(TrainerCallback):
    def __init__(self, metric_name):
        self.metric_name = metric_name
        self.best_metric = float("-inf")
        self.best_epoch = None
        self.best_state = None

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if model is None or metrics is None or self.metric_name not in metrics:
            return control

        current_metric = float(metrics[self.metric_name])
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.best_epoch = float(state.epoch) if state.epoch is not None else None
            self.best_state = get_trainable_state(model)
            print(
                f"New best model found: {self.metric_name}={self.best_metric:.6f} "
                f"at epoch={self.best_epoch}"
            )
        return control


def sanitize_experiment_name(name):
    cleaned = []
    for ch in str(name):
        if ch.isalnum() or ch in "-_.":
            cleaned.append(ch)
        else:
            cleaned.append("_")
    sanitized = "".join(cleaned).strip("._")
    return sanitized or "experiment"


def build_default_experiment_name(spec, index):
    parts = [f"exp_{index:02d}"]
    if "tuning_mode" in spec:
        parts.append(spec["tuning_mode"])
    if "attn_ranks" in spec:
        parts.append("r" + "-".join(str(x) for x in spec["attn_ranks"]))
    if "attn_alpha" in spec:
        parts.append(f"a{spec['attn_alpha']}")
    if "multiplicative_num_bases" in spec and spec.get("tuning_mode") in {"multiplicative", "both"}:
        parts.append(f"mb{spec['multiplicative_num_bases']}")
    if "num_train_epochs" in spec:
        parts.append(f"e{spec['num_train_epochs']}")
    return sanitize_experiment_name("_".join(parts))


def normalize_sweep_spec(spec):
    normalized = {}
    for key, value in spec.items():
        normalized[SWEEP_KEY_ALIASES.get(key, key)] = value
    return normalized


def get_model_tag(model_path):
    name = Path(model_path).name or "model"
    return sanitize_experiment_name(name.replace("-", "_"))


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv_rows(path, rows, preferred_fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = []
    for field in preferred_fields or []:
        if field not in fieldnames:
            fieldnames.append(field)

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def get_metric_display_name(task_name):
    metric_name = get_task_config(task_name)["metric"]
    display_names = {
        "accuracy": "Accuracy",
        "matthews": "Matthews Correlation",
        "pearson": "Pearson Correlation",
    }
    return display_names.get(metric_name, metric_name)


def print_training_configuration(args, target_families, experiment_name):
    print(f"Experiment name: {experiment_name}")
    print("Current training configuration:")
    print(f"  target_families = {target_families}")
    print(f"  tuning_mode = {args.tuning_mode}")
    print(f"  attn_ranks = {tuple(args.attn_ranks)}")
    print(f"  ffn_ranks = {tuple(args.ffn_ranks)}")
    print(f"  attn_alpha = {args.attn_alpha}")
    print(f"  ffn_alpha = {args.ffn_alpha}")
    print(f"  multiplicative_num_bases = {args.multiplicative_num_bases}")
    print(f"  num_train_epochs = {args.num_train_epochs}")
    print(f"  learning_rate = {args.learning_rate}")
    print(f"  per_device_train_batch_size = {args.per_device_train_batch_size}")
    print(f"  per_device_eval_batch_size = {args.per_device_eval_batch_size}")
    print(f"  weight_decay = {args.weight_decay}")
    print(f"  seed = {args.seed}")


def save_trainable_checkpoint(model, output_dir, args, target_families, experiment_name, resolved_dataset_path):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    trainable_state = get_trainable_state(model)
    torch.save(trainable_state, output_path / "trainable_state.pt")

    metadata = {
        "experiment_name": experiment_name,
        "model_path": args.model_path,
        "dataset_path": resolved_dataset_path,
        "glue_task": args.glue_task,
        "target_families": target_families,
        "target_modules": args.target_modules,
        "attn_ranks": list(args.attn_ranks),
        "ffn_ranks": list(args.ffn_ranks),
        "attn_alpha": args.attn_alpha,
        "ffn_alpha": args.ffn_alpha,
        "tuning_mode": args.tuning_mode,
        "multiplicative_num_bases": args.multiplicative_num_bases,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "decomposition_method": "tucker_hooi_svd_init",
        "decomposition_cache_dir": getattr(args, "_resolved_decomposition_cache_dir", None),
    }
    save_json(output_path / "training_config.json", metadata)


def extract_history_records(log_history, primary_eval_metric_key):
    train_history = []
    eval_history = []

    for record in log_history:
        if "loss" in record and "eval_loss" not in record:
            train_row = {
                "step": float(record.get("step", len(train_history) + 1)),
                "loss": float(record["loss"]),
            }
            if record.get("epoch") is not None:
                train_row["epoch"] = float(record["epoch"])
            train_history.append(train_row)

        if "eval_loss" in record or primary_eval_metric_key in record:
            if record.get("epoch") is None:
                continue
            eval_row = {"epoch": float(record["epoch"])}
            if "eval_loss" in record:
                eval_row["eval_loss"] = float(record["eval_loss"])
            if primary_eval_metric_key in record:
                eval_row[primary_eval_metric_key] = float(record[primary_eval_metric_key])
            eval_history.append(eval_row)

    return train_history, eval_history


def prepend_initial_history(train_history, eval_history, initial_train_metrics, initial_eval_metrics, task_name):
    train_history_with_initial = list(train_history)
    eval_history_with_initial = list(eval_history)

    initial_train_loss = initial_train_metrics.get("train_init_loss")
    if initial_train_loss is not None:
        train_history_with_initial.insert(
            0,
            {
                "step": 0.0,
                "epoch": 0.0,
                "loss": float(initial_train_loss),
            },
        )

    initial_eval_metric_key = get_prefixed_metric_key(task_name, "eval_init")
    initial_eval_row = {"epoch": 0.0}
    if "eval_init_loss" in initial_eval_metrics:
        initial_eval_row["eval_loss"] = float(initial_eval_metrics["eval_init_loss"])
    if initial_eval_metric_key in initial_eval_metrics:
        initial_eval_row[get_primary_eval_metric_key(task_name)] = float(
            initial_eval_metrics[initial_eval_metric_key]
        )
    if len(initial_eval_row) > 1:
        eval_history_with_initial.insert(0, initial_eval_row)

    return train_history_with_initial, eval_history_with_initial


def save_training_plots(train_history, eval_history, output_dir, task_name, primary_eval_metric_key, experiment_name):
    if plt is None:
        print("Warning: matplotlib is not installed, skipping plot generation.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_rows = [row for row in train_history if "loss" in row]
    if train_rows:
        plt.figure(figsize=(8, 5))
        plt.plot(
            [row["step"] for row in train_rows],
            [row["loss"] for row in train_rows],
            marker="o",
            linewidth=1.8,
        )
        plt.xlabel("Training Step")
        plt.ylabel("Loss")
        plt.title(f"Training Loss Curve ({experiment_name})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_path / "train_loss_curve.png", dpi=200)
        plt.close()

    eval_rows = [row for row in eval_history if primary_eval_metric_key in row]
    if eval_rows:
        metric_display_name = get_metric_display_name(task_name)
        plt.figure(figsize=(8, 5))
        plt.plot(
            [row["epoch"] for row in eval_rows],
            [row[primary_eval_metric_key] for row in eval_rows],
            marker="o",
            linewidth=1.8,
        )
        plt.xlabel("Epoch")
        plt.ylabel(metric_display_name)
        plt.title(f"Validation {metric_display_name} Curve ({experiment_name})")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(output_path / "eval_metric_curve.png", dpi=200)
        plt.close()


def save_history_artifacts(
    output_dir,
    train_history,
    eval_history,
    initial_train_metrics,
    initial_eval_metrics,
    final_metrics,
    task_name,
    primary_eval_metric_key,
    experiment_name,
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    save_json(
        output_path / "history.json",
        {
            "train_history": train_history,
            "eval_history": eval_history,
            "initial_train_metrics": initial_train_metrics,
            "initial_eval_metrics": initial_eval_metrics,
            "final_metrics": final_metrics,
        },
    )

    write_csv_rows(
        output_path / "train_loss_history.csv",
        train_history,
        preferred_fields=["step", "epoch", "loss"],
    )
    write_csv_rows(
        output_path / "eval_history.csv",
        eval_history,
        preferred_fields=["epoch", "eval_loss", primary_eval_metric_key],
    )
    save_training_plots(
        train_history=train_history,
        eval_history=eval_history,
        output_dir=output_path,
        task_name=task_name,
        primary_eval_metric_key=primary_eval_metric_key,
        experiment_name=experiment_name,
    )


def save_sweep_comparison_plot(experiment_results, output_dir, task_name, primary_eval_metric_key):
    if plt is None:
        print("Warning: matplotlib is not installed, skipping sweep comparison plot.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metric_slug = get_task_config(task_name)["metric"]
    metric_display_name = get_metric_display_name(task_name)

    plt.figure(figsize=(10, 6))
    line_count = 0
    for result in experiment_results:
        rows = [row for row in result["eval_history"] if primary_eval_metric_key in row]
        if not rows:
            continue
        plt.plot(
            [row["epoch"] for row in rows],
            [row[primary_eval_metric_key] for row in rows],
            marker="o",
            linewidth=1.8,
            label=result["experiment_name"],
        )
        line_count += 1

    if line_count == 0:
        plt.close()
        return

    plt.xlabel("Epoch")
    plt.ylabel(metric_display_name)
    plt.title(f"{task_name.upper()} {metric_display_name} Comparison Across Experiments")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / f"sweep_{metric_slug}_comparison.png", dpi=220)
    plt.close()


def create_sweep_summary_rows(experiment_results, task_name):
    primary_eval_metric_key = get_primary_eval_metric_key(task_name)
    initial_eval_metric_key = get_prefixed_metric_key(task_name, "eval_init")

    rows = []
    for result in experiment_results:
        config = result["config"]
        initial_train_metrics = result["initial_train_metrics"]
        initial_eval_metrics = result["initial_eval_metrics"]
        final_metrics = result["final_metrics"]

        rows.append(
            {
                "experiment_name": result["experiment_name"],
                "attn_ranks": "-".join(str(x) for x in config["attn_ranks"]),
                "ffn_ranks": "-".join(str(x) for x in config["ffn_ranks"]),
                "attn_alpha": config["attn_alpha"],
                "ffn_alpha": config["ffn_alpha"],
                "tuning_mode": config.get("tuning_mode"),
                "multiplicative_num_bases": config.get("multiplicative_num_bases"),
                "num_train_epochs": config["num_train_epochs"],
                "learning_rate": config["learning_rate"],
                "train_batch_size": config["per_device_train_batch_size"],
                "eval_batch_size": config["per_device_eval_batch_size"],
                "best_epoch": result["best_epoch"],
                f"best_{primary_eval_metric_key}": result["best_metric"],
                "initial_train_loss": initial_train_metrics.get("train_init_loss"),
                "initial_eval_loss": initial_eval_metrics.get("eval_init_loss"),
                f"initial_{primary_eval_metric_key}": initial_eval_metrics.get(initial_eval_metric_key),
                "final_eval_loss": final_metrics.get("eval_loss"),
                primary_eval_metric_key: final_metrics.get(primary_eval_metric_key),
                "experiment_dir": result["experiment_dir"],
            }
        )
    return rows


def prepare_experiment_dirs(args, sweep_root=None):
    if sweep_root is not None:
        experiment_root = Path(sweep_root) / args.run_name
        results_dir = experiment_root / "results"
        final_dir = experiment_root / "final"
        return experiment_root, results_dir, final_dir

    if getattr(args, "single_output_mode", True):
        results_dir = Path(f"./tucker_lora_{args.glue_task}_results")
        final_dir = Path(f"./tucker_lora_{args.glue_task}_final")
        experiment_root = final_dir
    else:
        experiment_root = Path("./tucker_lora_runs") / args.glue_task / args.run_name
        results_dir = experiment_root / "results"
        final_dir = experiment_root / "final"

    return experiment_root, results_dir, final_dir


def cleanup_experiment_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train family-wise Tucker adapters with Tucker/HOOI decomposition caches on local GLUE tasks."
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("TUCKER_MODEL_PATH", "./local_models/roberta-large"),
        help="Local Hugging Face model directory.",
    )
    parser.add_argument(
        "--dataset-path",
        default=os.getenv("TUCKER_DATASET_PATH"),
        help="Local datasets.save_to_disk() directory. Defaults to ./local_datasets/glue_<task>.",
    )
    parser.add_argument(
        "--glue-task",
        choices=sorted(TASK_CONFIGS),
        default="sst2",
        help="Which GLUE task to run.",
    )
    parser.add_argument(
        "--target-families",
        nargs="+",
        choices=["q", "k", "v", "o", "ffn"],
        default=["q", "k", "v", "o", "ffn"],
        help="Which families to adapt. Default: q k v o ffn",
    )
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help="Backward-compatible aliases such as query key value attention.output.dense ffn.",
    )
    parser.add_argument(
        "--attn-ranks",
        "--ranks",
        dest="attn_ranks",
        type=int,
        nargs=4,
        metavar=("R_LAYER", "R_HEAD", "R_HEAD_DIM", "R_HIDDEN"),
        default=(4, 4, 16, 32),
        help="Attention Tucker ranks for [layer, head, head_dim, hidden].",
    )
    parser.add_argument(
        "--ffn-ranks",
        type=int,
        nargs=4,
        metavar=("R_FFN_STAGE", "R_LAYER", "R_M", "R_N"),
        default=(2, 4, 64, 64),
        help="FFN Tucker ranks for [ffn_stage, layer, m, n].",
    )
    parser.add_argument(
        "--attn-alpha",
        type=float,
        default=1.0,
        help="Attention adapter scaling coefficient.",
    )
    parser.add_argument(
        "--ffn-alpha",
        type=float,
        default=1.0,
        help="FFN adapter scaling coefficient.",
    )
    parser.add_argument(
        "--tuning-mode",
        choices=["additive", "multiplicative", "both"],
        default="additive",
        help="Which Tucker adapter parameterization to train: additive core update, multiplicative core-mode transform, or both.",
    )
    parser.add_argument(
        "--multiplicative-num-bases",
        "--factor-tuning-params",
        dest="multiplicative_num_bases",
        type=int,
        default=50,
        help="Number of TinyLoRA-style fixed basis matrices for each Tucker-core mode transform used in multiplicative tuning.",
    )
    parser.add_argument("--max-length", type=int, default=128, help="Tokenizer max length.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=32,
        help="Train batch size per device.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=32,
        help="Eval batch size per device.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=3.0,
        help="Number of training epochs.",
    )
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional name for a single experiment run.",
    )
    parser.add_argument(
        "--sweep-config",
        default=None,
        help="Path to a JSON file that lists multiple experiment settings to run sequentially.",
    )
    parser.add_argument(
        "--sweep-output-dir",
        default=None,
        help="Optional directory for storing sweep experiment folders and the final comparison plot.",
    )
    parser.add_argument(
        "--decomposition-cache-dir",
        "--hosvd-cache-dir",
        dest="decomposition_cache_dir",
        default=None,
        help="Directory for storing Tucker/HOOI decomposition caches. '--hosvd-cache-dir' is kept as a backward-compatible alias.",
    )
    return parser.parse_args()


def patch_model_with_tucker(model, decomposition_results):
    family_name = decomposition_results["family_name"]
    shared_state_name = f"tucker_shared_state_{family_name}"
    shared_state = TuckerFamilySharedState(
        family_name=family_name,
        base_core=decomposition_results["base_core"],
        factor_matrices=decomposition_results["factor_matrices"],
        remaining_mode_factors=decomposition_results["remaining_mode_factors"],
        tuning_mode=decomposition_results["tuning_mode"],
        multiplicative_num_bases=decomposition_results["multiplicative_num_bases"],
        base_seed=decomposition_results["base_seed"],
    )
    model.add_module(shared_state_name, shared_state)
    shared_state = getattr(model, shared_state_name)

    for module_spec in decomposition_results["module_specs"]:
        parent_path = ".".join(module_spec["path"].split(".")[:-1])
        module_name = module_spec["path"].split(".")[-1]
        parent_module = model.get_submodule(parent_path)
        base_linear = getattr(parent_module, module_name)

        tucker_layer = TuckerAdaptedLinear(
            base_layer=base_linear,
            family_name=family_name,
            shared_state=shared_state,
            contracted_indices=module_spec["contracted_indices"],
            matrix_shape=module_spec["matrix_shape"],
            reshape_kind=module_spec["reshape_kind"],
            alpha=decomposition_results["alpha"],
        )
        setattr(parent_module, module_name, tucker_layer)

    return model


def unfreeze_classifier_head(model):
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
        return
    if hasattr(model, "score"):
        for param in model.score.parameters():
            param.requires_grad = True
        return
    print("Warning: no classifier head named 'classifier' or 'score' was found.")


def run_single_experiment(
    args,
    experiment_index,
    total_experiments,
    sweep_root=None,
    decomposition_cache_entries=None,
):
    set_seed(args.seed)

    experiment_root, results_dir, final_dir = prepare_experiment_dirs(args, sweep_root=sweep_root)
    experiment_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    final_dir.mkdir(parents=True, exist_ok=True)

    resolved_dataset_path = args.dataset_path or f"./local_datasets/glue_{args.glue_task}"
    validate_local_inputs(args.model_path, resolved_dataset_path)

    target_families = resolve_target_families(args)
    task_config = get_task_config(args.glue_task)
    primary_eval_metric_key = get_primary_eval_metric_key(args.glue_task)

    if args.tuning_mode in {"multiplicative", "both"} and args.multiplicative_num_bases <= 0:
        raise ValueError(
            "When tuning_mode is 'multiplicative' or 'both', multiplicative_num_bases must be positive. "
            f"Got {args.multiplicative_num_bases}."
        )

    print("=" * 80)
    print(f"Running experiment {experiment_index}/{total_experiments}: {args.run_name}")
    print(f"GLUE task: {args.glue_task}")
    print(f"Experiment directory: {experiment_root}")
    print_training_configuration(args, target_families, args.run_name)

    print("Loading local model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    model_kwargs = {
        "num_labels": task_config["num_labels"],
        "low_cpu_mem_usage": False,
        "local_files_only": True,
    }
    if task_config["is_regression"]:
        model_kwargs["problem_type"] = "regression"

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path, **model_kwargs)

    num_heads, head_dim, hidden_size = get_attention_geometry(model)
    ffn_m, ffn_n = get_ffn_geometry(model)
    print(
        f"Attention geometry: num_heads={num_heads}, head_dim={head_dim}, hidden_size={hidden_size}"
    )
    print(f"FFN canonical geometry target: m={ffn_m}, n={ffn_n}")

    for param in model.parameters():
        param.requires_grad = False

    if decomposition_cache_entries is None:
        decomposition_cache_entries = {}
    if not getattr(args, "_resolved_decomposition_cache_dir", None):
        raise ValueError("Decomposition cache directory was not prepared.")

    decomposition_results = []
    decomposition_cache_root = Path(args._resolved_decomposition_cache_dir)
    print("Preparing Tucker/HOOI decompositions for requested families...")
    for family_name in target_families:
        requested_ranks = tuple(args.ffn_ranks) if family_name == "ffn" else tuple(args.attn_ranks)
        alpha = args.ffn_alpha if family_name == "ffn" else args.attn_alpha
        cache_entry = get_or_create_decomposition_cache_entry(
            model=model,
            model_path=args.model_path,
            cache_root=decomposition_cache_root,
            family_name=family_name,
            ranks=requested_ranks,
            num_heads=num_heads,
            shared_cache_entries=decomposition_cache_entries,
        )
        print(
            f"[{family_name}] decomposition method = {cache_entry['method']}, "
            f"cached ranks = {tuple(cache_entry['ranks'])}"
        )
        result = build_runtime_family_from_decomposition_cache(
            cache_entry=cache_entry,
            alpha=alpha,
            tuning_mode=args.tuning_mode,
            multiplicative_num_bases=args.multiplicative_num_bases,
            base_seed=args.seed,
        )
        decomposition_results.append(result)

    print("Patching model with Tucker adapters...")
    for result in decomposition_results:
        model = patch_model_with_tucker(model, result)

    unfreeze_classifier_head(model)

    adapter_params = sum(
        param.numel()
        for name, param in model.named_parameters()
        if name.startswith("tucker_shared_state_")
    )
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(
        f"Adapter params: {adapter_params:,d} || "
        f"Total trainable params: {total_trainable_params:,d} || "
        f"All params: {all_params:,d} || "
        f"Percentage: {100 * total_trainable_params / all_params:.4f}%"
    )

    print("Loading and preparing local dataset...")
    dataset = load_from_disk(resolved_dataset_path)
    if "train" not in dataset or "validation" not in dataset:
        raise KeyError(
            f"Dataset must contain at least train and validation splits. "
            f"Found: {list(dataset.keys())}"
        )

    sentence1_key, sentence2_key = task_config["text_keys"]
    train_columns = dataset["train"].column_names
    missing_columns = [key for key in (sentence1_key, sentence2_key) if key and key not in train_columns]
    if missing_columns:
        raise KeyError(
            f"Task '{args.glue_task}' expects columns {missing_columns}, "
            f"but train split has columns: {train_columns}"
        )

    def tokenize_function(examples):
        if sentence2_key is None:
            return tokenizer(
                examples[sentence1_key],
                truncation=True,
                max_length=args.max_length,
            )
        return tokenizer(
            examples[sentence1_key],
            examples[sentence2_key],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    compute_metrics = build_compute_metrics(args.glue_task)

    bf16_supported = (
        torch.cuda.is_available()
        and hasattr(torch.cuda, "is_bf16_supported")
        and torch.cuda.is_bf16_supported()
    )
    use_bf16 = bf16_supported
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
        output_dir=str(results_dir),
        overwrite_output_dir=True,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        evaluation_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        bf16=use_bf16,
        fp16=use_fp16,
        save_safetensors=False,
        report_to="none",
        seed=args.seed,
    )

    best_callback = BestModelByMetricCallback(metric_name=primary_eval_metric_key)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[best_callback],
    )

    print("Computing initial metrics before training...")
    initial_train_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["train"],
        metric_key_prefix="train_init",
    )
    initial_eval_metrics = trainer.evaluate(
        eval_dataset=tokenized_datasets["validation"],
        metric_key_prefix="eval_init",
    )
    trainer.state.log_history.clear()

    print("Starting training...")
    trainer.train()
    plot_log_history = list(trainer.state.log_history)

    if best_callback.best_state is not None:
        print(
            f"Restoring best model from epoch={best_callback.best_epoch} "
            f"with {primary_eval_metric_key}={best_callback.best_metric:.6f}"
        )
        load_trainable_state(model, best_callback.best_state)
    else:
        print("Warning: best model state was not captured. Using final training state.")

    print("Evaluating best model...")
    final_metrics = trainer.evaluate()
    print(final_metrics)

    raw_train_history, raw_eval_history = extract_history_records(
        plot_log_history,
        primary_eval_metric_key=primary_eval_metric_key,
    )
    train_history, eval_history = prepend_initial_history(
        raw_train_history,
        raw_eval_history,
        initial_train_metrics=initial_train_metrics,
        initial_eval_metrics=initial_eval_metrics,
        task_name=args.glue_task,
    )

    save_trainable_checkpoint(
        model=model,
        output_dir=final_dir,
        args=args,
        target_families=target_families,
        experiment_name=args.run_name,
        resolved_dataset_path=resolved_dataset_path,
    )
    save_history_artifacts(
        output_dir=experiment_root,
        train_history=train_history,
        eval_history=eval_history,
        initial_train_metrics=initial_train_metrics,
        initial_eval_metrics=initial_eval_metrics,
        final_metrics=final_metrics,
        task_name=args.glue_task,
        primary_eval_metric_key=primary_eval_metric_key,
        experiment_name=args.run_name,
    )

    experiment_config = {
        "experiment_name": args.run_name,
        "model_path": args.model_path,
        "dataset_path": resolved_dataset_path,
        "glue_task": args.glue_task,
        "target_families": target_families,
        "target_modules": args.target_modules,
        "attn_ranks": list(args.attn_ranks),
        "ffn_ranks": list(args.ffn_ranks),
        "attn_alpha": args.attn_alpha,
        "ffn_alpha": args.ffn_alpha,
        "tuning_mode": args.tuning_mode,
        "multiplicative_num_bases": args.multiplicative_num_bases,
        "max_length": args.max_length,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "decomposition_method": "tucker_hooi_svd_init",
        "decomposition_cache_dir": getattr(args, "_resolved_decomposition_cache_dir", None),
    }

    summary_payload = {
        "experiment_name": args.run_name,
        "best_epoch": best_callback.best_epoch,
        "best_metric": best_callback.best_metric,
        "primary_eval_metric_key": primary_eval_metric_key,
        "initial_train_metrics": initial_train_metrics,
        "initial_eval_metrics": initial_eval_metrics,
        "final_metrics": final_metrics,
        "experiment_dir": str(experiment_root.resolve()),
        "final_dir": str(final_dir.resolve()),
    }

    save_json(experiment_root / "experiment_config.json", experiment_config)
    save_json(experiment_root / "experiment_summary.json", summary_payload)

    result = {
        "experiment_name": args.run_name,
        "experiment_dir": str(experiment_root.resolve()),
        "final_dir": str(final_dir.resolve()),
        "config": experiment_config,
        "initial_train_metrics": initial_train_metrics,
        "initial_eval_metrics": initial_eval_metrics,
        "final_metrics": final_metrics,
        "best_epoch": best_callback.best_epoch,
        "best_metric": best_callback.best_metric,
        "eval_history": eval_history,
    }

    del trainer
    del model
    del tokenized_datasets
    del dataset
    cleanup_experiment_memory()

    print(f"Finished experiment: {args.run_name}")
    return result


def load_sweep_experiments(args):
    base_args = deepcopy(vars(args))

    if args.sweep_config is None:
        single_args = deepcopy(base_args)
        single_args["run_name"] = sanitize_experiment_name(args.run_name or f"{args.glue_task}_run")
        single_args["single_output_mode"] = args.run_name is None
        return [argparse.Namespace(**single_args)], False

    sweep_path = Path(args.sweep_config)
    if not sweep_path.exists():
        raise FileNotFoundError(f"Sweep config file not found: {sweep_path.resolve()}")

    experiments = json.loads(sweep_path.read_text(encoding="utf-8"))
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Sweep config must be a non-empty JSON list.")

    protected_keys = {"sweep_config", "sweep_output_dir", "run_name", "single_output_mode"}

    experiment_args_list = []
    seen_names = set()
    for index, raw_spec in enumerate(experiments, start=1):
        if not isinstance(raw_spec, dict):
            raise ValueError(f"Sweep experiment at index {index} must be a JSON object.")

        spec = normalize_sweep_spec(raw_spec)
        merged = deepcopy(base_args)
        for key, value in spec.items():
            if key == "name":
                continue
            if key in protected_keys:
                raise ValueError(f"Field '{key}' cannot be overridden inside sweep config.")
            if key not in merged:
                raise ValueError(f"Unknown sweep field '{key}' in experiment {index}.")
            merged[key] = value

        experiment_name = sanitize_experiment_name(spec.get("name") or build_default_experiment_name(spec, index))
        if experiment_name in seen_names:
            raise ValueError(f"Duplicate experiment name '{experiment_name}' in sweep config.")
        seen_names.add(experiment_name)

        merged["run_name"] = experiment_name
        merged["single_output_mode"] = False
        experiment_args_list.append(argparse.Namespace(**merged))

    task_names = {exp.glue_task for exp in experiment_args_list}
    if len(task_names) != 1:
        raise ValueError("All experiments in one sweep must use the same GLUE task.")

    return experiment_args_list, True


def main():
    args = parse_args()
    experiment_args_list, is_sweep = load_sweep_experiments(args)

    if is_sweep:
        if args.sweep_output_dir:
            sweep_root = Path(args.sweep_output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            sweep_root = Path("./tucker_lora_sweeps") / f"{experiment_args_list[0].glue_task}_{timestamp}"
        sweep_root.mkdir(parents=True, exist_ok=True)
        print(f"Sweep output directory: {sweep_root}")
    else:
        sweep_root = None

    if args.decomposition_cache_dir:
        decomposition_cache_root = Path(args.decomposition_cache_dir)
    else:
        decomposition_cache_root = Path("./tucker_hooi_cache") / get_model_tag(experiment_args_list[0].model_path)
    decomposition_cache_root.mkdir(parents=True, exist_ok=True)
    print(f"Tucker/HOOI decomposition cache directory: {decomposition_cache_root}")

    for experiment_args in experiment_args_list:
        experiment_args._resolved_decomposition_cache_dir = str(decomposition_cache_root.resolve())

    decomposition_cache_entries = {}
    experiment_results = []
    total_experiments = len(experiment_args_list)
    for index, experiment_args in enumerate(experiment_args_list, start=1):
        experiment_results.append(
            run_single_experiment(
                args=experiment_args,
                experiment_index=index,
                total_experiments=total_experiments,
                sweep_root=sweep_root,
                decomposition_cache_entries=decomposition_cache_entries,
            )
        )

    if is_sweep:
        task_name = experiment_args_list[0].glue_task
        primary_eval_metric_key = get_primary_eval_metric_key(task_name)
        summary_rows = create_sweep_summary_rows(experiment_results, task_name)

        write_csv_rows(
            sweep_root / "sweep_summary.csv",
            summary_rows,
            preferred_fields=[
                "experiment_name",
                "attn_ranks",
                "attn_alpha",
                "num_train_epochs",
                "best_epoch",
                f"best_{primary_eval_metric_key}",
                primary_eval_metric_key,
            ],
        )
        save_json(sweep_root / "sweep_summary.json", summary_rows)
        save_sweep_comparison_plot(
            experiment_results=experiment_results,
            output_dir=sweep_root,
            task_name=task_name,
            primary_eval_metric_key=primary_eval_metric_key,
        )
        print(f"Sweep completed. Summary saved to {sweep_root}")
    else:
        print("Single experiment completed.")


if __name__ == "__main__":
    main()
