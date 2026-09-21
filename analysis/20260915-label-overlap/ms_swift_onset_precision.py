"""Separate exact generated token indices from approximate audio-time anchors."""
import json
from pathlib import Path
import re

from tokenizers import Tokenizer
from analyze_failures import timeline, interval_stats

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def exact_token_boundary(tokenizer, ids, raw, char_position):
    # Use original generation IDs: re-encoding text can choose different BPE IDs.
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(ids[:mid])) < char_position:
            lo = mid + 1
        else:
            hi = mid
    for index in range(max(0, lo-4), min(len(ids), lo+5)):
        if tokenizer.decode(ids[:index]) == raw[:char_position]:
            return index
    raise ValueError("No verified token boundary at the requested character")


tok = Tokenizer.from_file(str(HERE/"tokenizer.json"))
inventory = read(HERE/"ms_swift_inventory_fresh.json")
audit = read(HERE/"overlap_audit.json")
cases = [r for r in audit["cases"] if "/ms-swift-" in r["model_path"]]
failures = [r for r in inventory["records"] if r["truncated"]]
assert {r["raw_sha256"] for r in cases} == {r["raw_sha256"] for r in failures}
result = []
for r in cases:
    pred = read(Path(r["source"]))
    ref = read(Path(r["reference_source"]))[r["key"]]
    raw, ids = pred["raw_text"], pred["generated_ids"]
    assert tok.decode(ids) == raw
    row = dict(key=r["key"], model_path=r["model_path"], source=r["source"],
               reference_source=r["reference_source"], raw_sha256=r["raw_sha256"],
               original_generated_ids_verified=True)
    if r["key"].endswith("M8009"):
        index = pred["diagnostics"]["longest_identical_run"]["start"]
        assert ids[index] == 47815 and all(i == 47815 for i in ids[index:])
        assert len(tok.decode(ids[:index])) == r["loop"]["char_offset"]
        target = next(s for s in ref["segments"] if s["start"] == 983.66 and s["end"] == 994.21)
        slices = timeline(ref)
        row.update(
            first_loop_token_index_zero_based=index,
            first_loop_token_number_one_based=index+1,
            first_loop_token_id=ids[index], first_loop_token_text=tok.decode([ids[index]]),
            reference_candidate=target,
            candidate_utterance_overlap=interval_stats(slices,target["start"],target["end"]),
            candidate_overlap_intervals=[
                [max(a,target["start"]),min(b,target["end"])]
                for a,b,n in slices if n>=2 and max(a,target["start"])<min(b,target["end"])
            ],
            candidate_nonoverlap_intervals=[
                [max(a,target["start"]),min(b,target["end"])]
                for a,b,n in slices if n<2 and max(a,target["start"])<min(b,target["end"])
            ],
            generated_utterance_start=983.61,
            first_loop_token_audio_time=None,
            onset_overlap_class="unresolved",
            reason="The first 外 follows 就买那假一赔十的 within a 10.55s reference utterance; there is no word-level label time. Its utterance contains both overlapped and non-overlapped speech.",
        )
    else:
        match = re.search(r"(?P<tag>\[\d+\.\d+\])(?P=tag){8,}", raw)
        assert match
        char_position = match.start()+1
        index = exact_token_boundary(tok,ids,raw,char_position)
        stamps = list(re.finditer(r"\[(\d+\.\d+)\]",raw[:match.start()]))
        prior_start = float(stamps[-1][1])
        repeated_stamp = float(match["tag"][1:-1])
        lo,hi = sorted([prior_start,repeated_stamp])
        refs = [s for s in ref["segments"] if s["end"]>lo and s["start"]<=hi]
        slices = timeline(ref)
        counts = [n for a,b,n in slices if a<hi and b>lo]
        points = {}
        for point in [prior_start,repeated_stamp]:
            speakers = sorted({s["speaker"] for s in ref["segments"]
                               if s["start"]<=point<s["end"]})
            points[str(point)] = speakers
        assert min(counts)>=2 and all(len(v)>=2 for v in points.values())
        row.update(
            first_loop_token_index_zero_based=index,
            first_loop_token_number_one_based=index+1,
            first_loop_token_id=ids[index], first_loop_token_text=tok.decode([ids[index]]),
            first_repeated_timestamp_tag=match["tag"],
            token_definition="First numeric token in first repeated tag; shared delimiter ][ is not by itself the structural error.",
            preceding_new_utterance_start=prior_start,
            time_anchor_interval=[lo,hi],
            reference_speakers_at_endpoints=points,
            minimum_reference_speakers_across_anchor_interval=min(counts),
            maximum_reference_speakers_across_anchor_interval=max(counts),
            reference_utterances=refs,
            onset_overlap_class="overlap_under_generated_timestamp_alignment",
            first_loop_token_audio_time=None,
            limitation="The token's exact place in generated_ids is known; the numeric timestamps are semantic audio anchors, not acoustic forced alignment.",
        )
    result.append(row)
summary = dict(
    scope="MS-Swift trained-model full-generation loops ending at the output cap; excludes Base, finite local repetitions and engineering smoke tests",
    inventory_utc=inventory["utc"],
    completed_prediction_records_scanned=len(inventory["records"]),
    completed_prediction_identities=len({(r["model_path"],r["key"],r["raw_sha256"]) for r in inventory["records"]}),
    failures=3,
    onset_in_overlap_under_timestamp_alignment=2,
    onset_outside_overlap=0,
    unresolved=1,
    overlap_fraction_among_locatable=1.0,
    locatable_coverage=2/3,
    all_case_overlap_fraction=None,
    possible_all_case_fraction_if_timestamp_alignment_accepted=[2/3,1.0],
    bound_is_not_confidence_interval=True,
    no_forced_word_alignment_performed=True,
)
(HERE/"ms_swift_onset_precision.json").write_text(
    json.dumps(dict(summary=summary,cases=result),ensure_ascii=False,indent=2),encoding="utf-8")
lines = [
    "# MS-Swift 循环起点与 overlap：精确口径复核",
    "",
    f"服务器扫描时间：{inventory['utc']}。核查 {len(inventory['records'])} 份已完成的正式长度 SFT 预测，确认 3 个持续循环至截断的案例；不包括 Base、能结束的局部短重复和工程 smoke test。",
    "",
    "**按生成时间戳定位，能够定位的 2 个案例均处于 overlap（2/2=100%），定位覆盖率为 2/3=66.7%。另 1 个案例缺少第一个循环字的声学时间，不能把它算成非 overlap。全部 3 个案例的真实起点占比尚不能精确给出。**",
    "",
    "| 模型/会议 | 第一个循环 token 在原始输出中的编号（从 1 起） | 可用的音频时间证据 | 判断 |",
    "|---|---:|---|---|",
]
for r in result:
    if r["onset_overlap_class"]=="unresolved":
        evidence="整句 983.66–994.21 秒；首个“外”在“假一赔十的”之后，无字级时间"
        verdict="待定"
    else:
        lo,hi=r["time_anchor_interval"]
        evidence=f"{lo:.2f}–{hi:.2f} 秒；参考有 {r['minimum_reference_speakers_across_anchor_interval']}–{r['maximum_reference_speakers_across_anchor_interval']} 人"
        verdict="时间戳定位为 overlap"
    step=r["model_path"].rsplit("-",1)[-1]
    lines.append(f"| Step {step} / {r['key'].rsplit('/',1)[-1]} | {r['first_loop_token_number_one_based']} | {evidence} | {verdict} |")
lines += [
    "",
    "文字循环的第一个 token 为“外”（ID 47815）。时间戳循环由多个模型 token 构成，表中索引指第一个重复时间戳的首个数字 token；前面的共享分隔符 ][ 本身仍可用于合法说话人格式。所有索引均依据保存的 generated_ids，并验证其整体解码与 raw_text 完全一致，没有把重新分词的位置误当作实际生成位置。",
    "",
    "M8009 的 983.61 秒是整个未结束生成语句的起始时间；第一个“外”在该句已输出数十字之后，因此不能使用 983.61 秒处的说话人数判断它。原始 far 与 near TextGrid 对应标注均只有 983.66–994.21 秒的整句区间，句内同时存在 overlap 和非 overlap。句内重叠时长比例也不能当作该 token 落入 overlap 的概率。",
    "",
    "M8008 从新语句起点 1458.70 到循环时间戳 1458.88 秒，对应参考说话人数从 3 人变为 2 人，始终 overlap。M8001 从新语句起点 1321.81 到循环时间戳 1322.01 秒，始终为 3 人 overlap。这对选取哪一个相邻时间锚点较稳健，但生成时间戳本身仍不等于词级声学对齐。",
    "",
    "若接受以上时间戳定位口径，未知案例的两种归类使全体比例位于 66.7%–100%；这是未知样本造成的分类范围，不是统计置信区间，也不是已测得的最终比例。",
    "",
    "最严格的“错误 token 的真实声学位置”可能没有一一对应：错误字符本身未必来自某个真实语音字。补充强制对齐或人工听音时，应定义为参考转录中对应续写/分歧边界的位置，并同时标注时间不确定性。",
    "",
    "证据：[机器可读结果](ms_swift_onset_precision.json)；[服务器最新清单](ms_swift_inventory_fresh.json)；[逐例转录对照](逐例转录对照.md)。",
]
(HERE/"MS-Swift循环起点精确复核.md").write_text("\n".join(lines),encoding="utf-8")
print(json.dumps(dict(summary=summary,cases=[{k:v for k,v in r.items() if k in
    ["key","first_loop_token_number_one_based","first_loop_token_id","first_loop_token_text","time_anchor_interval","onset_overlap_class","reference_speakers_at_endpoints"]} for r in result]),ensure_ascii=False,indent=2))
