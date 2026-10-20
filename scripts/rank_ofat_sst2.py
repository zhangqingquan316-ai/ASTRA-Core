"""One-factor-at-a-time rank study for ASTRA-Tangent on SST-2.

Waits until no other `python -m astra_core` job is using the GPU, then:

1. Trains with FFN included and ffn_ranks fixed at (2, 4, 64, 64),
   varying one attention mode rank at a time.
2. Adds extra ranks on attention modes whose validation accuracy moves the most.
3. Locks the best measured attention rank tuple.
4. Repeats the same procedure for FFN ranks.
5. Writes rank_ofat_sst2/results.json and rank_ofat_sst2/分析报告.md.

Does not launch a run while another astra_core process or a GPU compute
process is present.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
import traceback
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT / ".venv" / "bin" / "python"
RUN_ROOT = PROJECT / "astra_core_runs" / "sst2"
STUDY = PROJECT / "rank_ofat_sst2"
LOG_DIR = STUDY / "logs"

ATTN_BASE = (4, 4, 16, 32)
FFN_BASE = (2, 4, 64, 64)
ATTN_NAMES = ("layer", "head", "head_dim", "hidden")
FFN_NAMES = ("ffn_stage", "ffn_layer", "ffn_m", "ffn_n")
ATTN_LIMITS = (24, 16, 64, 1024)
FFN_LIMITS = (2, 24, 4096, 1024)
ATTN_GRID = {
    0: (2, 4, 8, 16),
    1: (2, 4, 8, 16),
    2: (8, 16, 32, 48),
    3: (8, 16, 32, 64, 128),
}
FFN_GRID = {
    0: (1, 2),
    1: (2, 4, 8, 16),
    2: (32, 64, 128, 256),
    3: (32, 64, 128, 256),
}
ATTN_LABELS = {
    "layer": "层（layer，上限 24）",
    "head": "头（head，上限 16）",
    "head_dim": "头维度（head_dim，上限 64）",
    "hidden": "隐层（hidden，上限 1024）",
}
FFN_LABELS = {
    "ffn_stage": "FFN 阶段（stage，上限 2）",
    "ffn_layer": "FFN 层（layer，上限 24）",
    "ffn_m": "FFN 大维 m（canonical 4096）",
    "ffn_n": "FFN 小维 n（canonical 1024）",
}

# SST-2 validation has 872 labeled examples. Around 93% accuracy the
# binomial standard error is about 0.86 percentage points.
VAL_SIZE = 872
ACC_NOISE_PP = 0.86


def log(message):
    STUDY.mkdir(parents=True, exist_ok=True)
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with (STUDY / "orchestrator.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def read_cmd(pid):
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", errors="replace")


def other_astra_pids():
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        command = read_cmd(entry.name)
        if " -m astra_core" in command or command.strip().endswith(" astra_core"):
            found.append(int(entry.name))
    return found


def gpu_compute_pids():
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    pids = []
    for line in output.splitlines():
        text = line.strip()
        if text.isdigit():
            pids.append(int(text))
    return pids


def wait_until_gpu_is_free():
    announced = False
    while True:
        astra = other_astra_pids()
        gpu = gpu_compute_pids()
        if not astra and not gpu:
            if announced:
                log("已有训练已结束，GPU 空闲，开始秩扫描。")
            return
        if not announced:
            log(
                "等待当前训练结束，不启动新进程。"
                f" astra_core PID={astra or '-'}，GPU 进程 PID={gpu or '-'}。"
            )
            announced = True
        time.sleep(30)


def rank_tuple(values):
    return tuple(int(value) for value in values)


def run_name_for(phase, dim_name, rank, attn, ffn, kind):
    if kind == "baseline" and phase == "attn":
        return "ofat_attn_baseline_r" + "-".join(str(x) for x in attn)
    if kind == "composed":
        joined = "-".join(str(x) for x in (attn if phase == "attn" else ffn))
        return f"ofat_{phase}_composed_r{joined}"
    if phase == "attn":
        return f"ofat_attn_{dim_name}_r{rank}"
    attn_tag = "-".join(str(x) for x in attn)
    return f"ofat_ffn_attn{attn_tag}_{dim_name}_r{rank}"


def build_initial_queue(phase, attn_fixed):
    names = ATTN_NAMES if phase == "attn" else FFN_NAMES
    grid = ATTN_GRID if phase == "attn" else FFN_GRID
    base = ATTN_BASE if phase == "attn" else FFN_BASE
    queue = []
    seen = set()

    def add(ranks, dim_index, kind):
        attn = ranks if phase == "attn" else attn_fixed
        ffn = FFN_BASE if phase == "attn" else ranks
        key = (rank_tuple(attn), rank_tuple(ffn))
        if key in seen:
            return
        seen.add(key)
        dim_name = "baseline" if kind == "baseline" else names[dim_index]
        varied_rank = None if kind == "baseline" else ranks[dim_index]
        queue.append(
            {
                "phase": phase,
                "kind": kind,
                "dim_index": None if kind == "baseline" else dim_index,
                "dim_name": dim_name,
                "varied_rank": varied_rank,
                "attn_ranks": list(attn),
                "ffn_ranks": list(ffn),
                "run_name": run_name_for(phase, dim_name, varied_rank, attn, ffn, kind),
            }
        )

    add(base, None, "baseline")
    for dim_index, values in grid.items():
        for value in values:
            ranks = list(base)
            ranks[dim_index] = int(value)
            add(tuple(ranks), dim_index, "grid")
    return queue


def extra_ranks(tested, best, limit):
    tested = sorted({int(value) for value in tested})
    best = int(best)
    candidates = []
    if best in tested:
        index = tested.index(best)
        if index > 0:
            mid = (tested[index - 1] + best) // 2
            if tested[index - 1] < mid < best:
                candidates.append(mid)
        if index + 1 < len(tested):
            mid = (best + tested[index + 1]) // 2
            if best < mid < tested[index + 1]:
                candidates.append(mid)
        if index == 0:
            lower = max(1, best // 2)
            if lower < best:
                candidates.append(lower)
        if index == len(tested) - 1 and best < limit:
            step = max(2, best // 2)
            upper = min(int(limit), best + step)
            if upper > best:
                candidates.append(upper)
    unique = []
    for value in candidates:
        if value not in tested and value not in unique and 1 <= value <= limit:
            unique.append(value)
    return unique[:2]


def load_history_csv(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def record_from_disk(run_dir):
    summary_path = run_dir / "experiment_summary.json"
    config_path = run_dir / "experiment_config.json"
    if not summary_path.exists() or not config_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    families = config.get("target_families") or []
    if "ffn" not in families or config.get("tuning_mode") != "manifold":
        return None
    if int(config.get("seed", -1)) != 42 or float(config.get("num_train_epochs", 0)) != 3.0:
        return None
    eval_rows = load_history_csv(run_dir / "eval_history.csv")
    train_rows = load_history_csv(run_dir / "train_loss_history.csv")
    init_eval = summary.get("initial_eval_metrics") or {}
    final_metrics = summary.get("final_metrics") or {}
    init_acc = float(init_eval.get("eval_init_accuracy"))
    init_loss = float(init_eval.get("eval_init_loss"))
    best_acc = float(summary["best_metric"])
    final_loss = float(final_metrics["eval_loss"])
    train_first = float(train_rows[0]["loss"]) if train_rows else None
    train_last = float(train_rows[-1]["loss"]) if train_rows else None
    return {
        "run_name": config["experiment_name"],
        "attn_ranks": rank_tuple(config["attn_ranks"]),
        "ffn_ranks": rank_tuple(config["ffn_ranks"]),
        "best_accuracy": best_acc,
        "best_epoch": summary.get("best_epoch"),
        "init_accuracy": init_acc,
        "init_eval_loss": init_loss,
        "final_eval_loss": final_loss,
        "accuracy_gain": best_acc - init_acc,
        "eval_loss_drop": init_loss - final_loss,
        "train_loss_first": train_first,
        "train_loss_last": train_last,
        "train_loss_drop": None if train_first is None else train_first - train_last,
        "adapter_params": summary.get("parameter_stats", {}).get("adapter_params"),
        "horizontal_dof": summary.get("parameter_stats", {}).get("manifold_horizontal_dof"),
        "train_seconds": (summary.get("training_resources") or {}).get("train_and_epoch_eval_seconds"),
        "eval_history": [
            {
                "epoch": float(row["epoch"]),
                "eval_loss": float(row["eval_loss"]),
                "eval_accuracy": float(row["eval_accuracy"]),
            }
            for row in eval_rows
        ],
        "run_dir": str(run_dir),
    }


def load_completed():
    records = []
    if not RUN_ROOT.exists():
        return records
    for run_dir in sorted(RUN_ROOT.glob("ofat_*")):
        record = record_from_disk(run_dir)
        if record is not None:
            records.append(record)
    return records


def find_record(records, attn, ffn):
    attn = rank_tuple(attn)
    ffn = rank_tuple(ffn)
    matches = [record for record in records if record["attn_ranks"] == attn and record["ffn_ranks"] == ffn]
    return matches[-1] if matches else None


def slice_records(records, phase, dim_index, attn_fixed):
    names = ATTN_NAMES if phase == "attn" else FFN_NAMES
    base = ATTN_BASE if phase == "attn" else FFN_BASE
    points = []
    for record in records:
        if phase == "attn":
            if record["ffn_ranks"] != FFN_BASE:
                continue
            ranks = record["attn_ranks"]
        else:
            if record["attn_ranks"] != rank_tuple(attn_fixed):
                continue
            ranks = record["ffn_ranks"]
        if any(ranks[index] != base[index] for index in range(4) if index != dim_index):
            continue
        points.append(record)
    points.sort(key=lambda record: (record["attn_ranks"] if phase == "attn" else record["ffn_ranks"])[dim_index])
    return points, names[dim_index]


def sensitive_dimensions(records, phase, attn_fixed):
    scored = []
    for dim_index in range(4):
        points, name = slice_records(records, phase, dim_index, attn_fixed)
        if len(points) < 2:
            continue
        accuracies = [point["best_accuracy"] for point in points]
        span = max(accuracies) - min(accuracies)
        scored.append((span, dim_index, name, points))
    if not scored:
        return []
    scored.sort(reverse=True, key=lambda item: item[0])
    largest = scored[0][0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    chosen = []
    for span, dim_index, name, points in scored:
        if span >= 0.008 or (span >= 0.005 and span >= 1.5 * second and dim_index == scored[0][1]):
            chosen.append((dim_index, name, points, span))
    if largest < 0.005:
        return []
    return chosen


def best_rank_on_slice(points, phase, dim_index):
    def sort_key(record):
        ranks = record["attn_ranks"] if phase == "attn" else record["ffn_ranks"]
        return (
            record["best_accuracy"],
            -record["final_eval_loss"],
            -(record["adapter_params"] or 0),
            -ranks[dim_index],
        )

    winner = max(points, key=sort_key)
    ranks = winner["attn_ranks"] if phase == "attn" else winner["ffn_ranks"]
    return ranks[dim_index], winner


def compose_best(records, phase, attn_fixed):
    base = list(ATTN_BASE if phase == "attn" else FFN_BASE)
    for dim_index in range(4):
        points, _ = slice_records(records, phase, dim_index, attn_fixed)
        if not points:
            continue
        best_value, _ = best_rank_on_slice(points, phase, dim_index)
        base[dim_index] = int(best_value)
    return tuple(base)


def select_best_record(records, phase, attn_fixed):
    if phase == "attn":
        pool = [record for record in records if record["ffn_ranks"] == FFN_BASE]
    else:
        pool = [record for record in records if record["attn_ranks"] == rank_tuple(attn_fixed)]
    if not pool:
        return None
    return max(
        pool,
        key=lambda record: (
            record["best_accuracy"],
            -record["final_eval_loss"],
            -(record["adapter_params"] or 0),
        ),
    )


def launch(job):
    command = [
        str(PYTHON),
        "-m",
        "astra_core",
        "--model-path",
        "./local_models/roberta-large",
        "--dataset-path",
        "./local_datasets/glue_sst2",
        "--glue-task",
        "sst2",
        "--target-families",
        "q",
        "k",
        "v",
        "o",
        "ffn",
        "--attn-ranks",
        *[str(value) for value in job["attn_ranks"]],
        "--ffn-ranks",
        *[str(value) for value in job["ffn_ranks"]],
        "--tuning-mode",
        "manifold",
        "--manifold-policy",
        "adaptive",
        "--learning-rate",
        "1e-3",
        "--num-train-epochs",
        "3",
        "--seed",
        "42",
        "--run-name",
        job["run_name"],
    ]
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{job['run_name']}.log"
    log(
        "开始 "
        f"{job['run_name']} attn={tuple(job['attn_ranks'])} ffn={tuple(job['ffn_ranks'])}"
    )
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
            check=False,
        )
    record = record_from_disk(RUN_ROOT / job["run_name"])
    if completed.returncode != 0 or record is None:
        log(f"失败 {job['run_name']} exit={completed.returncode}，日志 {log_path}")
        return None
    log(
        f"完成 {job['run_name']} acc={record['best_accuracy']:.4f} "
        f"eval_loss={record['final_eval_loss']:.4f} "
        f"adapter_params={record['adapter_params']}"
    )
    return record


def ensure_job(records, job):
    existing = find_record(records, job["attn_ranks"], job["ffn_ranks"])
    if existing is not None:
        log(f"跳过已完成 {existing['run_name']} attn={existing['attn_ranks']} ffn={existing['ffn_ranks']}")
        return records
    wait_until_gpu_is_free()
    record = launch(job)
    if record is None:
        log(f"重试 {job['run_name']}")
        wait_until_gpu_is_free()
        record = launch(job)
    if record is not None:
        records.append(record)
        write_state(records)
    return records


def write_state(records):
    STUDY.mkdir(parents=True, exist_ok=True)
    payload = {
        "attn_base": list(ATTN_BASE),
        "ffn_base": list(FFN_BASE),
        "records": [
            {**record, "attn_ranks": list(record["attn_ranks"]), "ffn_ranks": list(record["ffn_ranks"])}
            for record in records
        ],
    }
    (STUDY / "results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def pct(value):
    return f"{value * 100:.2f}%"


def pp(delta):
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta * 100:.2f} 个百分点"


def trend_text(points, phase, dim_index):
    ordered = sorted(
        points,
        key=lambda record: (record["attn_ranks"] if phase == "attn" else record["ffn_ranks"])[dim_index],
    )
    accuracies = [record["best_accuracy"] for record in ordered]
    span = max(accuracies) - min(accuracies)
    if span < 0.003:
        return "准确率几乎不随该维秩变化"
    best_index = max(range(len(ordered)), key=lambda index: accuracies[index])
    if best_index == 0:
        return "秩越小准确率越高，增大秩没有带来收益"
    if best_index == len(ordered) - 1:
        return "准确率随秩增大而提高，扫描范围内尚未看到饱和"
    best_rank = (ordered[best_index]["attn_ranks"] if phase == "attn" else ordered[best_index]["ffn_ranks"])[dim_index]
    return f"准确率在秩 {best_rank} 达到峰值，继续增大后回落"


def markdown_table(points, phase, dim_index, reference):
    lines = [
        "| 秩 | 验证准确率 | 相对基线 | 准确率提升 | 验证 loss | loss 下降 | 训练 loss 下降 | 适配器参数 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered = sorted(
        points,
        key=lambda record: (record["attn_ranks"] if phase == "attn" else record["ffn_ranks"])[dim_index],
    )
    for record in ordered:
        ranks = record["attn_ranks"] if phase == "attn" else record["ffn_ranks"]
        train_drop = "—" if record["train_loss_drop"] is None else f"{record['train_loss_drop']:.4f}"
        lines.append(
            "| {rank} | {acc} | {delta} | {gain} | {loss:.4f} | {ldrop:.4f} | {tdrop} | {params} |".format(
                rank=ranks[dim_index],
                acc=pct(record["best_accuracy"]),
                delta=pp(record["best_accuracy"] - reference["best_accuracy"]),
                gain=pp(record["accuracy_gain"]),
                loss=record["final_eval_loss"],
                ldrop=record["eval_loss_drop"],
                tdrop=train_drop,
                params=f"{record['adapter_params']:,}",
            )
        )
    return "\n".join(lines)


def describe_phase(records, phase, attn_fixed, reference):
    labels = ATTN_LABELS if phase == "attn" else FFN_LABELS
    limits = ATTN_LIMITS if phase == "attn" else FFN_LIMITS
    sections = []
    spans = []
    for dim_index in range(4):
        points, name = slice_records(records, phase, dim_index, attn_fixed)
        if len(points) < 2:
            continue
        accuracies = [point["best_accuracy"] for point in points]
        span = max(accuracies) - min(accuracies)
        spans.append((name, span, points, dim_index))
        best_value, best_point = best_rank_on_slice(points, phase, dim_index)
        loss_values = [point["final_eval_loss"] for point in points]
        best_loss_point = min(points, key=lambda record: record["final_eval_loss"])
        loss_rank = (best_loss_point["attn_ranks"] if phase == "attn" else best_loss_point["ffn_ranks"])[dim_index]
        same = best_value == loss_rank
        sections.append(
            "\n".join(
                [
                    f"### {labels[name]}",
                    "",
                    trend_text(points, phase, dim_index) + f"，极差 {pp(span)}。",
                    f"该维准确率最高的秩是 **{best_value}**（{pct(best_point['best_accuracy'])}，验证 loss {best_point['final_eval_loss']:.4f}）。",
                    (
                        "最低验证 loss 出现在同一秩。"
                        if same
                        else f"最低验证 loss 出现在秩 {loss_rank}（loss {min(loss_values):.4f}），和最高准确率不是同一个点。"
                    ),
                    f"该维秩上限是 {limits[dim_index]}。秩等于上限时对应因子是方阵，流形更新没有水平方向，只剩核心容量变化。",
                    "",
                    markdown_table(points, phase, dim_index, reference),
                ]
            )
        )
    spans.sort(key=lambda item: item[1], reverse=True)
    return sections, spans


def write_report(records, attn_fixed, ffn_winner):
    attn_reference = find_record(records, ATTN_BASE, FFN_BASE)
    if attn_reference is None:
        raise RuntimeError("缺少注意力基线结果，无法写报告。")
    attn_sections, attn_spans = describe_phase(records, "attn", ATTN_BASE, attn_reference)
    ffn_reference = find_record(records, attn_fixed, FFN_BASE)
    ffn_sections, ffn_spans = ([], [])
    if ffn_reference is not None:
        ffn_sections, ffn_spans = describe_phase(records, "ffn", attn_fixed, ffn_reference)

    attn_best = select_best_record(records, "attn", ATTN_BASE)
    composed_attn = compose_best(records, "attn", ATTN_BASE)
    composed_attn_record = find_record(records, composed_attn, FFN_BASE)

    def span_sentence(spans, labels):
        if not spans:
            return "没有足够的单维曲线。"
        parts = [f"{labels[name]} {pp(span)}" for name, span, _, _ in spans]
        return "各维验证准确率极差：" + "，".join(parts) + "。"

    lines = [
        "# SST-2 上 Tucker 秩的单维扫描",
        "",
        "## 实验设置",
        "",
        "模型是本地 `roberta-large`，任务是 GLUE SST-2，训练 3 个 epoch，学习率 1e-3，"
        "seed 42，batch size 32，`tuning-mode=manifold`，`manifold-policy=adaptive`。"
        "注意力族 q/k/v/o 和 FFN 都参加适配。注意力张量形状是 `(24, 16, 64, 1024)`，"
        "对应秩 `[layer, head, head_dim, hidden]`。FFN 张量形状是 `(2, 24, 4096, 1024)`，"
        "对应秩 `[stage, layer, m, n]`。",
        "",
        f"第一阶段把 FFN 秩固定为 `{FFN_BASE}`，注意力基线是代码默认值 `{ATTN_BASE}`。"
        "每次只改一个注意力维度。准确率变化达到约 0.8 个百分点，或明显大于其他维度时，"
        "在峰值两侧补点。然后把四个维度各自的最优秩拼成一组，若这组还没训练过就补跑一次。",
        "",
        f"第二阶段把注意力秩锁定为实测最优的 `{tuple(attn_fixed)}`，"
        f"再对 FFN 秩做同样的单维扫描，基线是 `{FFN_BASE}`。",
        "",
        f"验证集有 {VAL_SIZE} 条。准确率在 93% 附近时，二项分布标准误差大约是 "
        f"{ACC_NOISE_PP:.2f} 个百分点。小于这个幅度的差别，单次 seed 不能当成稳定优势。"
        "分类头从随机初始化开始，所以各配置的初始准确率都在 50% 附近，"
        "“准确率提升”指相对训练前验证准确率的增幅。",
        "",
        "## 注意力秩",
        "",
        span_sentence(attn_spans, ATTN_LABELS),
        "",
        f"基线 `{ATTN_BASE}` 的最佳验证准确率是 {pct(attn_reference['best_accuracy'])}，"
        f"最佳模型验证 loss {attn_reference['final_eval_loss']:.4f}，"
        f"相对训练前 loss 下降 {attn_reference['eval_loss_drop']:.4f}，"
        f"准确率提升 {pp(attn_reference['accuracy_gain'])}。",
        "",
        *attn_sections,
        "",
        "### 注意力配比",
        "",
        f"四个维度分开看时，拼出来的秩是 `{composed_attn}`。"
        + (
            f"这组实际跑到 {pct(composed_attn_record['best_accuracy'])}，验证 loss {composed_attn_record['final_eval_loss']:.4f}。"
            if composed_attn_record is not None
            else "这组没有成功跑完。"
        ),
        f"全部注意力实验里实测最高的是 `{attn_best['attn_ranks']}`，"
        f"准确率 {pct(attn_best['best_accuracy'])}，验证 loss {attn_best['final_eval_loss']:.4f}，"
        f"适配器参数 {attn_best['adapter_params']:,}。"
        "第二阶段锁定的是这组实测最优，而不是未训练的组合。",
        "",
        "## FFN 秩",
        "",
    ]
    if not ffn_sections:
        lines.append("FFN 扫描没有形成完整曲线。")
    else:
        lines.extend(
            [
                span_sentence(ffn_spans, FFN_LABELS),
                "",
                f"FFN 基线 `{FFN_BASE}` 在锁定注意力 `{tuple(attn_fixed)}` 下的准确率是 "
                f"{pct(ffn_reference['best_accuracy'])}，验证 loss {ffn_reference['final_eval_loss']:.4f}。",
                "",
                *ffn_sections,
                "",
                "### FFN 配比",
                "",
            ]
        )
        if ffn_winner is not None:
            lines.append(
                f"实测最优 FFN 秩是 `{ffn_winner['ffn_ranks']}`，注意力秩 `{ffn_winner['attn_ranks']}`，"
                f"准确率 {pct(ffn_winner['best_accuracy'])}，验证 loss {ffn_winner['final_eval_loss']:.4f}，"
                f"相对训练前准确率提升 {pp(ffn_winner['accuracy_gain'])}，"
                f"验证 loss 下降 {ffn_winner['eval_loss_drop']:.4f}，"
                f"适配器参数 {ffn_winner['adapter_params']:,}。"
            )
    lines.extend(["", "## 总结", ""])
    if attn_spans:
        top_name, top_span, _, _ = attn_spans[0]
        lines.append(
            f"注意力侧对验证准确率影响最大的是{ATTN_LABELS[top_name]}，极差 {pp(top_span)}。"
        )
    if ffn_spans:
        top_name, top_span, _, _ = ffn_spans[0]
        lines.append(
            f"FFN 侧对验证准确率影响最大的是{FFN_LABELS[top_name]}，极差 {pp(top_span)}。"
        )
    if attn_best is not None and ffn_winner is not None:
        lines.append(
            f"综合两次扫描，准确率最高的配置是注意力 `{ffn_winner['attn_ranks']}`、"
            f"FFN `{ffn_winner['ffn_ranks']}`，验证准确率 {pct(ffn_winner['best_accuracy'])}。"
            "单维扫描看不到维度之间的交互；拼出来的组合如果和某条单维曲线上的点不同，报告里采用的是实际训练过的结果。"
        )
    lines.append(
        "loss 下降和准确率提升大体同向，但个别秩上最低 loss 和最高准确率并不重合。"
        "这种时候以验证准确率为准，因为 SST-2 的主指标是准确率。"
    )
    lines.append("")
    text = "\n".join(lines) + "\n"
    (STUDY / "分析报告.md").write_text(text, encoding="utf-8")
    log(f"已写入报告 {STUDY / '分析报告.md'}")


def run_phase(records, phase, attn_fixed):
    names = ATTN_NAMES if phase == "attn" else FFN_NAMES
    limits = ATTN_LIMITS if phase == "attn" else FFN_LIMITS
    for job in build_initial_queue(phase, attn_fixed):
        records = ensure_job(records, job)

    for dim_index, name, points, span in sensitive_dimensions(records, phase, attn_fixed):
        tested = [
            (point["attn_ranks"] if phase == "attn" else point["ffn_ranks"])[dim_index]
            for point in points
        ]
        best_value, _ = best_rank_on_slice(points, phase, dim_index)
        extras = extra_ranks(tested, best_value, limits[dim_index])
        log(f"{phase} 维度 {name} 准确率极差 {span:.4f}，补点 {extras or '无'}。")
        base = list(ATTN_BASE if phase == "attn" else FFN_BASE)
        for value in extras:
            ranks = list(base)
            # After refinement the best may already have moved off the original baseline
            # on this dimension only. Other dimensions stay at the phase baseline.
            ranks[dim_index] = int(value)
            attn = ranks if phase == "attn" else attn_fixed
            ffn = FFN_BASE if phase == "attn" else ranks
            job = {
                "phase": phase,
                "kind": "refine",
                "dim_index": dim_index,
                "dim_name": names[dim_index],
                "varied_rank": int(value),
                "attn_ranks": list(attn),
                "ffn_ranks": list(ffn),
                "run_name": run_name_for(phase, names[dim_index], value, attn, ffn, "refine"),
            }
            records = ensure_job(records, job)

    composed = compose_best(records, phase, attn_fixed)
    attn = composed if phase == "attn" else attn_fixed
    ffn = FFN_BASE if phase == "attn" else composed
    job = {
        "phase": phase,
        "kind": "composed",
        "dim_index": None,
        "dim_name": "composed",
        "varied_rank": None,
        "attn_ranks": list(attn),
        "ffn_ranks": list(ffn),
        "run_name": run_name_for(phase, "composed", None, attn, ffn, "composed"),
    }
    records = ensure_job(records, job)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-plan", action="store_true", help="Print the initial queues and exit.")
    args = parser.parse_args()
    if args.print_plan:
        attn_queue = build_initial_queue("attn", ATTN_BASE)
        ffn_queue = build_initial_queue("ffn", ATTN_BASE)
        print(f"attn jobs {len(attn_queue)}")
        for job in attn_queue:
            print(job["run_name"], job["attn_ranks"], job["ffn_ranks"])
        print(f"ffn jobs {len(ffn_queue)}")
        for job in ffn_queue:
            print(job["run_name"], job["attn_ranks"], job["ffn_ranks"])
        return

    STUDY.mkdir(parents=True, exist_ok=True)
    log("秩扫描编排启动。当前训练结束前不会占用 GPU。")
    wait_until_gpu_is_free()
    records = load_completed()
    attn_fixed = ATTN_BASE
    ffn_winner = None
    try:
        log(f"已有完成结果 {len(records)} 组。")
        records = run_phase(records, "attn", ATTN_BASE)
        attn_best = select_best_record(records, "attn", ATTN_BASE)
        if attn_best is None:
            raise RuntimeError("注意力阶段没有成功结果。")
        attn_fixed = attn_best["attn_ranks"]
        log(f"锁定注意力秩 {attn_fixed}，准确率 {attn_best['best_accuracy']:.4f}。")
        (STUDY / "attn_choice.json").write_text(
            json.dumps(
                {
                    "attn_ranks": list(attn_fixed),
                    "best_accuracy": attn_best["best_accuracy"],
                    "run_name": attn_best["run_name"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        records = run_phase(records, "ffn", attn_fixed)
        ffn_winner = select_best_record(records, "ffn", attn_fixed)
        write_state(records)
        write_report(records, attn_fixed, ffn_winner)
        log("全部秩扫描结束。")
    except Exception:
        log(traceback.format_exc())
        if records:
            try:
                write_state(records)
                write_report(records, attn_fixed, ffn_winner)
            except Exception:
                log(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
