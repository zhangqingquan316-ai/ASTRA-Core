"""Static configuration shared across the Tucker-LoRA PM workflow."""

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
