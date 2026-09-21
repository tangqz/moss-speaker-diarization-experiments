"""Evaluate the frozen dev-selected checkpoint on the untouched test split."""
import json
import html
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    class SummaryWriter:  # type: ignore[no-redef]
        """No-op fallback; prediction and metric JSON remain authoritative."""
        def __init__(self, *args, **kwargs):
            pass

        def add_scalar(self, *args, **kwargs):
            pass

        def add_text(self, *args, **kwargs):
            pass

        def close(self):
            pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "evaluation"))
from common import BASE, OLD_EVAL, read, write, sha, now

VLLM_PY = "/work/qt28/moss/envs/vllm-moss-20260914/bin/python"
METRIC_PY = "/work/qt28/moss/dkucc/evaluation/metrics-env/bin/python"


def manifest(path):
    return {
        "path": str(path),
        "files": {
            p.name: sha(p)
            for p in sorted(path.iterdir())
            if p.suffix in (".safetensors", ".py") or p.name == "config.json"
        },
    }


def progress(writer, run, stage, completed, started_at):
    stage_codes = {"starting": 0, "base": 1, "sft": 2, "scoring": 3, "complete": 4}
    writer.add_scalar("test/records_completed", completed, completed)
    writer.add_scalar("test/progress_percent", completed / 112 * 100, completed)
    writer.add_scalar("test/stage", stage_codes[stage], completed)
    writer.add_scalar("test/elapsed_minutes", (time.monotonic() - started_at) / 60, completed)
    if os.environ.get("MOSS_REUSE_BASE"):
        writer.add_scalar("test/reused_base_records", 56 if completed >= 56 else 0, completed)
        writer.add_scalar("test/sft_records_completed", max(0, completed - 56), completed)
        writer.add_scalar("test/sft_progress_percent", max(0, completed - 56) / 56 * 100, completed)
    writer.flush()
    write(run / "test_progress.json", {
        "stage": stage,
        "completed_predictions": completed,
        "expected_predictions": 112,
        "progress_percent": completed / 112 * 100,
        "elapsed_minutes": (time.monotonic() - started_at) / 60,
        "utc": now(),
    })


def publish_text_logs(writer, label, paths, offsets, text_steps):
    for rank, path in enumerate(paths):
        size = path.stat().st_size if path.exists() else 0
        offset = offsets.get(path, 0)
        if size <= offset:
            continue
        with path.open("rb") as stream:
            raw = stream.read(size)
        offsets[path] = size
        text_steps[path] = text_steps.get(path, 0) + 1
        content = raw.decode("utf-8", errors="replace").replace("\r", "\n")
        writer.add_text(
            f"test/logs/{label}/rank_{rank}",
            "<pre>" + html.escape(content) + "</pre>",
            text_steps[path],
        )
    writer.flush()


def controller_log(writer, step, message):
    writer.add_text("test/logs/controller", message, step)
    writer.flush()


def generate(run, folder, checkpoint, label, writer, completed_before, started_at):
    processes, handles = [], []
    log_paths, offsets, text_steps = [], {}, {}
    env = dict(os.environ)
    last_completed = -1
    last_publish = 0.0
    for key in ["NPROC_PER_NODE", "RANK", "LOCAL_RANK", "WORLD_SIZE", "PYTORCH_ALLOC_CONF"]:
        env.pop(key, None)
    try:
        for rank in range(4):
            log_path = run / "logs" / f"test-{label}-rank-{rank}.log"
            offsets[log_path] = log_path.stat().st_size if log_path.exists() else 0
            log_paths.append(log_path)
            log = log_path.open("a")
            handles.append(log)
            processes.append(
                subprocess.Popen(
                    [VLLM_PY, str(HERE / "evaluation/vllm_generate.py"),
                     "--run", str(folder), "--checkpoint", str(checkpoint),
                     "--rank", str(rank), "--model-label", label],
                    start_new_session=True, stdout=log, stderr=subprocess.STDOUT, env=env,
                )
            )
        deadline = time.monotonic() + 6 * 3600
        while True:
            codes = [p.poll() for p in processes]
            completed = sum(
                read(p).get("status") == "ok"
                for p in (folder / "predictions" / label).rglob("*.json")
            )
            if completed != last_completed or time.monotonic() - last_publish >= 15:
                progress(writer, run, label, completed_before + completed, started_at)
                last_completed = completed
                publish_text_logs(writer, label, log_paths, offsets, text_steps)
                last_publish = time.monotonic()
            if not all(code in (None, 0) for code in codes):
                raise RuntimeError(f"{label} vLLM generation failed: {codes}")
            if all(code == 0 for code in codes):
                break
            if time.monotonic() > deadline:
                raise TimeoutError(f"{label} vLLM generation exceeded six hours")
            time.sleep(2)
    finally:
        for p in processes:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p in processes:
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
        for handle in handles:
            handle.close()
        publish_text_logs(writer, label, log_paths, offsets, text_steps)
    rows = [read(folder / "predictions" / label / f"{item['key']}.json")
            for item in read(folder / "inputs.json")]
    assert len(rows) == 56
    assert all(row["status"] == "ok" and not row["engineering_smoke"] for row in rows)
    progress(writer, run, label, completed_before + 56, started_at)


def main():
    run = Path(os.environ["MOSS_RUN"])
    outcome = read(run / "outcome.json")
    selection = read(run / "selection.json")
    selected_step = int(outcome["selected_step"])
    assert selection["decision"]["selected_step"] == selected_step
    assert not outcome["test_evaluated"]

    checkpoint = run / f"vllm_exports/checkpoint-{selected_step}"
    folder = run / f"evaluations/test-{selected_step}"
    (run / "logs").mkdir(parents=True, exist_ok=True)
    assert checkpoint.is_dir()
    write(run / "status.json", {
        "stage": "test_running",
        "utc": now(),
        "selected_step": selected_step,
        "test_path": str(folder),
    })
    event_dir = "test_reuse_events" if os.environ.get("MOSS_REUSE_BASE") else "test_events"
    writer = SummaryWriter(log_dir=str(run / event_dir), flush_secs=5)
    started_at = time.monotonic()
    progress(writer, run, "starting", 0, started_at)
    controller_log(writer, 0, f"正式 Test 启动；Dev 冻结选点为 Step {selected_step}。")

    old_inputs = read(OLD_EVAL / "inputs.json")
    items = [row for row in old_inputs if row["split"] == "test"]
    counts = {}
    for row in items:
        counts[row["dataset"]] = counts.get(row["dataset"], 0) + 1
    assert len(items) == 56 and counts == {"alimeeting": 20, "aishell4": 20, "ami": 16}
    assert sorted({row["rank"] for row in items}) == [0, 1, 2, 3]
    references = read(OLD_EVAL / "references.json")
    write(folder / "inputs.json", items)
    write(folder / "references.json", {row["key"]: references[row["key"]] for row in items})
    write(folder / "protocol.json", {
        "scope": "untouched_test_after_dev_selection",
        "selected_step": selected_step,
        "selection_frozen_before_test": True,
        "datasets": counts,
        "models": {"base": str(BASE), "sft": str(checkpoint)},
        "generation": "vLLM 0.23.1rc1.dev949+g68b4a1d58 BF16 greedy, one engine per A40, complete meetings",
        "decoding": {"temperature": 0, "top_p": 1, "top_k": -1,
                     "repetition_penalty": 1, "presence_penalty": 0,
                     "frequency_penalty": 0, "max_context": 131072,
                     "max_new_tokens_cap": 65536},
        "metrics": "CER, cpCER, DeltaCP, DER collar 0 and 0.25, RTF",
        "test_not_used_for_selection": True,
        "utc": now(),
    })

    timings = {}
    base_manifest = read(run / "evaluations/dev-0/model_manifest.json")
    write(folder / "model_manifest.json", base_manifest)
    start = time.perf_counter()
    if os.environ.get("MOSS_REUSE_BASE"):
        source = Path(os.environ["MOSS_REUSE_BASE"])
        assert read(source / "test_complete.json")["complete"]
        assert read(source / "inputs.json") == items
        assert read(source / "references.json") == read(folder / "references.json")
        assert read(source / "protocol.json")["decoding"] == read(folder / "protocol.json")["decoding"]
        assert read(source / "model_manifests.json")["base"] == base_manifest
        assert manifest(BASE) == base_manifest
        expected_version = "0.23.1rc1.dev949+g68b4a1d58"
        hashes = {}
        for item in items:
            rel = Path("predictions/base") / (item["key"] + ".json")
            record = read(source / rel)
            assert record["status"] == "ok" and not record["engineering_smoke"]
            assert record["backend_version"] == expected_version
            assert record["model_manifest"] == base_manifest
            for key, value in item.items():
                assert record[key] == value, (item["key"], key)
            dest = folder / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / rel, dest)
            hashes[str(rel)] = sha(dest)
            assert hashes[str(rel)] == sha(source / rel)
        receipt = dict(source=str(source), reused_records=56, prediction_sha256=hashes,
                       backend_version=expected_version, utc=now())
        write(folder / "base_reuse.json", receipt)
        protocol = read(folder / "protocol.json")
        protocol.update(base_reused_from=str(source), base_reused_records=56,
                        base_runtime_is_historical=True)
        write(folder / "protocol.json", protocol)
        timings["base_wall_seconds_including_startup"] = 0.0
        timings["historical_base_wall_seconds_including_startup"] = read(source / "timing.json")["base_wall_seconds_including_startup"]
        progress(writer, run, "base", 56, started_at)
        controller_log(writer, 1, "Base 已校验并复用上一轮56条预测；总进度50%表示Base结果就绪。本轮只运行SFT的56场会议。")
        writer.add_text("test/base_reuse", "<pre>" + html.escape(json.dumps(receipt, indent=2)) + "</pre>", 0)
        historical_logs = [source.parent.parent / "logs" / f"test-base-rank-{i}.log" for i in range(4)]
        publish_text_logs(writer, "base_reused_historical", historical_logs, {}, {})
    else:
        generate(run, folder, BASE, "base", writer, 0, started_at)
        timings["base_wall_seconds_including_startup"] = time.perf_counter() - start
        controller_log(writer, 1, "Base 的 56 条 Test 推理全部完成。")

    sft_manifest = read(run / f"evaluations/dev-{selected_step}/model_manifest.json")
    write(folder / "model_manifest.json", sft_manifest)
    start = time.perf_counter()
    generate(run, folder, checkpoint, "sft", writer, 56, started_at)
    timings["sft_wall_seconds_including_startup"] = time.perf_counter() - start
    controller_log(writer, 2, f"Step {selected_step} SFT 的 56 条 Test 推理全部完成。")
    write(folder / "timing.json", timings)
    write(folder / "model_manifests.json", {"base": base_manifest, "sft": sft_manifest})

    progress(writer, run, "scoring", 112, started_at)
    score_log = (run / "logs/final-test-score.log").open("a")
    try:
        subprocess.run(
            [METRIC_PY, str(HERE / "evaluation/selection.py"), "--run", str(folder)],
            stdout=score_log, stderr=subprocess.STDOUT, check=True,
        )
    finally:
        score_log.close()
    score_text = (run / "logs/final-test-score.log").read_text(encoding="utf-8", errors="replace")
    writer.add_text("test/logs/scoring", "<pre>" + html.escape(score_text.replace("\r", "\n")) + "</pre>", 1)
    controller_log(writer, 3, "CER、cpCER、DeltaCP、DER 和 RTF 汇总评分完成。")
    summary = read(folder / "metrics_summary.json")
    assert summary["complete"] and summary["scored_predictions"] == 112
    complete = {
        "complete": True,
        "selected_step": selected_step,
        "predictions": 112,
        "metrics_sha256": sha(folder / "metrics_summary.json"),
        "utc": now(),
    }
    write(folder / "test_complete.json", complete)
    outcome.update(test_evaluated=True, test_path=str(folder),
                   test_metrics_sha256=complete["metrics_sha256"], utc=now())
    write(run / "outcome.json", outcome)
    write(run / "status.json", {"stage": "completed_with_test", "utc": now(),
                                 "selected_step": selected_step, "test_predictions": 112})
    progress(writer, run, "complete", 112, started_at)
    writer.close()
    print(json.dumps(complete, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
