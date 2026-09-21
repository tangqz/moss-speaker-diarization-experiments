import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[2]


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def link(path, label):
    return f"[{label}](<{path.as_posix()}>)"


def segments_text(segments):
    return "\n".join(
        f"[{s['start']:.2f}–{s['end']:.2f}] {s['speaker']}: {s['text']}"
        for s in segments
    ) or "此时间范围没有参考/已解析语音段。"


audit = read(HERE / "overlap_audit.json")
expanded = []
rows = [
    "# 复读起点附近：MS-Swift 案例与历史旁证",
    "",
    "MS-Swift 三份输出优先展示，随后附早期 20 份输出供参考。完全相同的副本已剔除；同一会议/模型的不同历史输出仍保留，不能作为独立样本。",
    "",
    "格式为 [开始时间][说话人]文本[结束时间]。前一句结束晚于后一句开始可以表示重叠语音，不能直接判定为负时长。各模型的 S01/S02 可以置换，不能直接按编号判断归属错误。",
    "",
    "稳定周期尾段不一定是最早错误位置；文字循环的时间锚点通常是整个生成语句的开头，不是第一重复字的强制对齐时间。",
]
ordered = sorted(enumerate(audit["cases"], 1),
                 key=lambda pair: ("/ms-swift-" not in pair[1]["model_path"], pair[0]))
for index, row in ordered:
    source = Path(row["source"])
    pred = read(source)
    reference = read(Path(row["reference_source"]))[row["key"]]
    anchor, onset = row["anchor_seconds"], row["loop"]["char_offset"]
    selected_refs = [s for s in reference["segments"]
                     if s["end"] >= anchor-15 and s["start"] <= anchor+25]
    predroot = next(p for p in source.parents if p.name == "predictions")
    basepath = predroot/"base"/(row["key"]+".json")
    if not basepath.exists():
        basepath = WORK/"dkucc/artifacts/eval-62994-63363/raw/results/eval-62994-63363/predictions/base"/(row["key"]+".json")
    base = read(basepath) if basepath.exists() else None
    base_context = [s for s in base["segments"]
                    if s["end"] >= anchor-5 and s["start"] <= anchor+20] if base else []
    details = dict(
        original_case_index=index, key=row["key"], source=str(source),
        source_sha256=row["raw_sha256"], reference_source=row["reference_source"],
        model_path=row["model_path"], generated_anchor_seconds=anchor,
        generated_before_stable_loop=pred["raw_text"][max(0,onset-1200):onset],
        generated_stable_loop_start=pred["raw_text"][onset:onset+180],
        reference_context=selected_refs, base_source=str(basepath) if base else None,
        base_context=base_context, base_ended_eos=base.get("ended_eos") if base else None,
        base_truncated=base.get("truncated") if base else None,
        base_decoding=base.get("decoding") if base else None,
        sft_decoding=pred.get("decoding"),
        equal_initial_prompt_hash=bool(base and base.get("prompt_ids_sha256") and
                                     base.get("prompt_ids_sha256")==pred.get("prompt_ids_sha256")),
    )
    unit = row["loop"].get("repeated_unit","")
    if unit and not any(c.isdigit() for c in unit) and row["loop"]["kind"]=="exact_periodic_tail":
        def longest(raw):
            return max((len(m[0])//len(unit) for m in re.finditer("(?:"+re.escape(unit)+")+",raw)), default=0)
        details["repeat_unit"] = unit
        details["sft_max_consecutive_unit_count"] = longest(pred["raw_text"])
        details["base_max_consecutive_unit_count"] = longest(base["raw_text"]) if base else None
    expanded.append(details)
    rows += [
        "", f"## 原始案例 {index:02d}：{row['key']}", "",
        f"模型：{row['model_path']}。生成锚点：{anchor:.2f} 秒。",
        "", f"证据：{link(source,'SFT 原始结果')}；{link(Path(row['reference_source']),'参考标注')}。",
        "", "**SFT 稳定循环前 1,200 字符与循环开头**", "", "~~~text",
        details["generated_before_stable_loop"]+"\n<<< 稳定循环尾段起点 >>>\n"+details["generated_stable_loop_start"],
        "~~~", "", "**参考：锚点前 15 秒至后 25 秒相交的全部语句**", "", "~~~text",
        segments_text(selected_refs), "~~~", "",
    ]
    if base:
        rows += [
            f"**Base 对照**：{link(basepath,'原始结果')}。整场 truncated={base.get('truncated')}，ended_eos={base.get('ended_eos')}。",
            "", "~~~text", segments_text(base_context), "~~~", "",
        ]
        if "repeat_unit" in details:
            rows += [f"全篇最长连续单元计数：单元「{unit}」；Base {details['base_max_consecutive_unit_count']}，SFT {details['sft_max_consecutive_unit_count']}。文字单元计数不等同模型 token 计数。", ""]
    else:
        rows += ["这份证据包未定位到对应 Base 文件。", ""]

assert len(expanded)==23
(HERE/"onset_comparisons.json").write_text(json.dumps(expanded,ensure_ascii=False,indent=2),encoding="utf-8")
(HERE/"逐例转录对照.md").write_text("\n".join(rows),encoding="utf-8")
print(json.dumps(dict(cases=len(expanded),base_comparisons=sum(r["base_source"] is not None for r in expanded),
                     verified_ms_swift_prompt_hash_pairs=sum(r["equal_initial_prompt_hash"] for r in expanded))))
