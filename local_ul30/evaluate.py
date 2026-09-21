#!/usr/bin/env python3
"""Fixed-injection/free-generation result utilities.

Heavy model execution is deliberately kept in explicit GPU commands.  The
portable `scan` command diagnoses raw IDs without filtering parser failures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import periodic_runs


def scan_jsonl(source: Path, output: Path) -> dict:
    records = []
    with source.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_status = row.get("status")
            if row_status == "not_applicable":
                records.append({"line": line_no, "status": "not_applicable",
                                "source_key": row.get("source_key"), "case": row.get("case"),
                                "reason": row.get("not_applicable_reason")})
                continue
            if row_status == "error":
                records.append({"line": line_no, "status": "execution_failure",
                                "source_key": row.get("source_key"), "case": row.get("case"),
                                "reason": row.get("error")})
                continue
            ids = row.get("raw_ids", row.get("generated_ids"))
            if not isinstance(ids, list):
                records.append({"line": line_no, "status": "execution_failure",
                                "reason": "raw token IDs missing"})
                continue
            runs = list(periodic_runs(ids))
            finish = row.get("finish_reason")
            raw_text = row.get("raw_text", "")
            scope = row.get("evaluation_scope", "unknown")
            local_censored = finish == "length" and scope == "local_injected_256"
            records.append({"line": line_no, "source_key": row.get("source_key"),
                            "case": row.get("case"), "token_count": len(ids),
                            "evaluation_scope": scope, "hit_cap": finish == "length",
                            "horizon_censored": local_censored,
                            "empty_output": len(ids) == 0 or not raw_text.strip(),
                            "periodic_runs": runs, "long_loop": bool(runs),
                            "parser_ok": row.get("parser_ok")})
    execution_failures = sum(x.get("status") == "execution_failure" for x in records)
    not_applicable = sum(x.get("status") == "not_applicable" for x in records)
    summary = {"schema_version": "short-recovery-diagnostics-v1", "input_rows": len(records),
               "status": "complete_with_errors" if execution_failures else "complete",
               "execution_failures": execution_failures, "not_applicable": not_applicable,
               "evaluated_rows": len(records) - execution_failures - not_applicable,
               "hard_failures": sum(bool((x.get("hit_cap") and not x.get("horizon_censored"))
                                          or x.get("empty_output") or x.get("long_loop")
                                          or x.get("parser_ok") is False
                                          or x.get("status") == "execution_failure")
                                    for x in records),
               "records": records,
               "limitation": "Early-EOS completion and structural/time progress require reference-aware evaluation."}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan")
    scan.add_argument("--predictions", required=True, type=Path)
    scan.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "scan":
        print(json.dumps(scan_jsonl(args.predictions, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
