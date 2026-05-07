"""Tensor reshaping, Tucker/HOOI decomposition caches, and ASTRA layer patching."""

import math
from pathlib import Path
import zlib

import numpy as np
import tensorly as tl
import torch
import torch.nn as nn

from .constants import ATTENTION_FAMILY_SUFFIXES, EXCLUDED_MODULE_KEYWORDS

tl.set_backend("pytorch")


def get_base_model(model):
    """Resolve the encoder model hidden under AutoModelForSequenceClassification."""
    prefix = getattr(model, "base_model_prefix", None)
    if prefix and hasattr(model, prefix):
        return getattr(model, prefix), prefix
    raise ValueError("Could not resolve the base encoder model from AutoModelForSequenceClassification.")


def get_encoder_layers(model):
    """Return the transformer blocks and the base-model prefix."""
    base_model, base_prefix = get_base_model(model)
    encoder = getattr(base_model, "encoder", None)
    layer_stack = getattr(encoder, "layer", None)
    if layer_stack is None:
        raise ValueError("This script expects the base model to expose encoder.layer.")
    return list(layer_stack), base_prefix


def get_attention_geometry(model):
    """Read the attention geometry needed to tensorize Q/K/V/O matrices."""
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
    """Read the FFN geometry used when canonicalizing FFN matrices."""
    hidden_size = getattr(model.config, "hidden_size", None)
    intermediate_size = getattr(model.config, "intermediate_size", None)
    if hidden_size is None or intermediate_size is None:
        raise ValueError(
            "Model config must define hidden_size and intermediate_size to build the FFN tensor."
        )
    return intermediate_size, hidden_size


def tensor_to_matrix(tensor, matrix_shape, reshape_kind):
    """Invert the tensorization step so the adapter can be applied as a matrix."""
    if reshape_kind == "direct":
        return tensor.contiguous().reshape(matrix_shape)
    if reshape_kind == "transpose":
        d_out, d_in = matrix_shape
        return tensor.contiguous().reshape(d_in, d_out).t().contiguous()
    raise ValueError(f"Unsupported reshape kind: {reshape_kind}")


def matrix_to_attention_tensor(weight, family_name, num_heads):
    """
    Convert one attention weight matrix into a 3D tensor.

    For Q/K/V, the output dimension is split into heads.
    For O, the input dimension is split into heads, so a transpose is needed.
    """
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
    """
    Align FFN matrices to a shared orientation before stacking them into one tensor.

    Different FFN stages can appear as [m, n] or [n, m]. This helper normalizes all
    stages to the same canonical orientation so that one shared FFN tensor can be built.
    """
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
    """
    Build the fixed basis matrices used for ASTRA-Mode / ASTRA-Hybrid tuning.

    These basis matrices are frozen. Only the combination coefficients are trained.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    basis = torch.randn(num_bases, rank, rank, generator=generator, dtype=torch.float32)
    return (basis / math.sqrt(max(rank, 1))).contiguous()


def build_transform_seed(base_seed, family_name, mode_index):
    """Derive a deterministic but family-specific seed for one mode transform."""
    seed_payload = f"{family_name}:{mode_index}:{int(base_seed)}".encode("utf-8")
    return zlib.crc32(seed_payload) & 0xFFFFFFFF


class MultiplicativeModeAdapter(nn.Module):
    """
    Learn one square transform for one Tucker-core mode.

    The transform is parameterized as:
        I + sum_i c_i * B_i
    where:
    - `B_i` are fixed basis matrices
    - `c_i` are the learned coefficients
    """

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
        """Return the current square transform matrix for this Tucker mode."""
        scaled_basis = self.coefficients.to(dtype).view(-1, 1, 1) * self.basis_matrices.to(dtype)
        return self.identity.to(dtype) + scaled_basis.sum(dim=0)


class TuckerFamilySharedState(nn.Module):
    """
    Shared trainable state for one family of Tucker-adapted layers.

    Every layer in the same family shares:
    - one decomposed base core from Tucker/HOOI
    - optional additive delta_core
    - optional multiplicative mode transforms

    What changes from layer to layer is only which contracted factor row is selected.
    """

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
        """Return the additive-updated core for additive/both tuning modes."""
        base_core = self.base_core.to(dtype)
        if self.has_additive:
            return base_core + self.delta_core.to(dtype)
        return base_core

    def get_base_core(self, dtype):
        """Return the frozen Tucker core in the requested dtype."""
        return self.base_core.to(dtype)

    def get_contracted_factor_row(self, mode_idx, row_idx, dtype):
        """Return the factor row that corresponds to one specific layer/stage index."""
        factor = getattr(self, f"contracted_factor_matrix_{mode_idx}")
        return factor.to(dtype)[int(row_idx)]

    def get_remaining_factors(self, dtype):
        """Return the non-contracted factor matrices used after layer selection."""
        factors = []
        for idx in range(self.num_remaining_modes):
            factor = getattr(self, f"remaining_mode_factor_{idx}")
            factors.append(factor.to(dtype))
        return factors

    def apply_mode_transforms(self, core_tensor, dtype):
        """
        Apply the learned multiplicative transforms to every Tucker-core mode.

        This is the mode-adaptation step: before choosing a concrete layer, the shared core
        is first transformed along every mode by learned square matrices.
        """
        if not self.has_multiplicative:
            return core_tensor

        transforms = [adapter.get_transform(dtype) for adapter in self.mode_transform_adapters]
        return tl.tenalg.multi_mode_dot(
            core_tensor,
            transforms,
            modes=list(range(self.num_contracted_modes + self.num_remaining_modes)),
        )


class TuckerAdaptedLinear(nn.Module):
    """
    Frozen linear layer reconstructed from:
    - a Tucker prior obtained from the pretrained model
    - an optional additive core update
    - an optional multiplicative mode transform

    The `alpha` in this script controls how far we move from the prior Tucker weight
    toward the tuned Tucker weight.
    """

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

        # Split the original dense weight into:
        # 1) a Tucker prior reconstructed from the pretrained decomposition
        # 2) a residual term not explained by that Tucker prior
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
        """
        Rebuild the concrete tensor for this layer from the shared family state.

        Steps:
        1. optionally apply multiplicative transforms to the shared core
        2. contract the shared tensor along layer/stage modes using this layer's indices
        3. expand the remaining free modes back with the remaining factor matrices
        """
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
        """
        Forward pass for one adapted linear layer.

        The important logic is:
        - additive mode: tune the Tucker core
        - multiplicative mode: keep the base core, but transform each mode
        - both: do both operations before rebuilding the weight matrix

        After reconstruction, only the Tucker part is interpolated with `alpha`;
        the residual part from the original dense layer is always kept.
        """
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

        # alpha=1 means fully trust the tuned Tucker branch.
        # alpha<1 means interpolate between the pretrained Tucker prior and the tuned Tucker weight.
        tuned_tucker_weight = prior_weight + self.alpha * (adapted_weight - prior_weight)
        effective_weight = self.residual_weight.to(current_dtype) + tuned_tucker_weight

        output = torch.matmul(x, effective_weight.t())
        if self.base_layer.bias is not None:
            output = output + self.base_layer.bias.to(current_dtype)
        return output


def matches_attention_family(name, module, family_name):
    """Check whether a module name belongs to one attention family."""
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
    """
    Build the Tucker/HOOI decomposition cache for one attention family.

    Unlike the HOSVD script, this variant stores the full Tucker decomposition:
    - `base_core`
    - contracted factor matrices
    - remaining factor matrices
    """
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
    """Build the Tucker/HOOI decomposition cache for the shared FFN tensor."""
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
    """
    Clone a cached Tucker decomposition into a runtime family description.

    The cached decomposition stays fixed on disk; the runtime object only adds:
    - tuning mode
    - multiplicative basis count
    - alpha scaling
    """
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


def build_decomposition_cache_file(cache_root, family_name, ranks):
    """Build the cache-file path for one family and one rank setting."""
    rank_part = "-".join(str(x) for x in ranks)
    return cache_root / f"{family_name}_r{rank_part}.pt"


def build_decomposition_cache_key(model_path, family_name, ranks):
    """Build the in-memory cache key used during a sweep run."""
    return (str(Path(model_path).resolve()), family_name, tuple(ranks))


def load_cached_decomposition_entry(cache_file, resolved_model_path, family_name, ranks):
    """Load and validate a Tucker/HOOI cache file if it already exists."""
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
    """
    Load or compute the Tucker/HOOI cache for one family.

    This function enables two levels of reuse:
    - on disk across different runs
    - in memory across experiments within the same sweep
    """
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


def patch_model_with_tucker(model, decomposition_results):
    """
    Replace the selected linear layers with PM-style Tucker wrappers.

    Each family is represented by one shared state module. Every layer in that family
    reads from the same shared state and only differs by its contracted indices.
    """
    family_name = decomposition_results["family_name"]
    shared_state_name = f"astra_core_shared_state_{family_name}"
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
    """Train the task head together with the PM Tucker parameters."""
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
        return
    if hasattr(model, "score"):
        for param in model.score.parameters():
            param.requires_grad = True
        return
    print("Warning: no classifier head named 'classifier' or 'score' was found.")
