"""Compare action horizons only on complete, hash-matched diagnostic pairs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import median
from zipfile import ZipFile

from run_overnight_native import HORIZONS, SCENES


def summarize(rows: list[dict], manifest: dict) -> dict:
    buckets = defaultdict(dict)
    failures = Counter()
    for row in rows:
        if row["phase"] != "direct_horizon_diagnostics":
            continue
        if not row["completed"]:
            failures[row["reason"]] += 1
            continue
        key = (row["scene"], row["seed"], row["init"])
        if row["horizon"] in buckets[key]:
            raise ValueError(f"duplicate horizon in {key}")
        buckets[key][row["horizon"]] = row

    paired = {}
    incomplete = []
    hash_mismatches = []
    for key, horizons in sorted(buckets.items()):
        if set(horizons) != set(HORIZONS):
            incomplete.append({"key": key, "observed_horizons": sorted(horizons)})
            continue
        hashes = {r["init_state_sha256"] for r in horizons.values()}
        if len(hashes) != 1 or None in hashes or "" in hashes:
            hash_mismatches.append(key)
            continue
        paired[key] = horizons

    def compare(cases, horizon):
        candidates = [entry[horizon] for entry in cases]
        baseline = [entry[50] for entry in cases]
        n = len(candidates)
        successes = sum(r["success"] is True for r in candidates)
        wins = sum(r["success"] is True and b["success"] is False for r, b in zip(candidates, baseline, strict=True))
        losses = sum(r["success"] is False and b["success"] is True for r, b in zip(candidates, baseline, strict=True))
        both = [r["steps"] - b["steps"] for r, b in zip(candidates, baseline, strict=True) if r["success"] and b["success"]]
        return {
            "horizon": horizon, "n": n, "successes": successes,
            "success_pct": round(100 * successes / n, 2) if n else None,
            "delta_vs_50_pp": round(100 * (wins - losses) / n, 2) if n else None,
            "wins_vs_50": wins, "losses_vs_50": losses,
            "median_steps_all": median(r["steps"] for r in candidates) if n else None,
            "both_succeeded": len(both), "median_step_delta_both_succeeded": median(both) if both else None,
            "terminal_reasons": dict(Counter(r["reason"] for r in candidates)),
        }

    return {
        "run": manifest["run"], "snapshot_utc": manifest["created_at_utc"], "status": manifest["status"],
        "indexed_rows": manifest["indexed_rows"], "baseline_horizon": 50,
        "paired_count": len(paired), "incomplete_pairs": incomplete, "hash_mismatches": hash_mismatches,
        "infrastructure_errors": dict(failures),
        "overall": [compare(list(paired.values()), h) for h in HORIZONS],
        "by_scene": {str(scene): [compare([v for k, v in paired.items() if k[0] == scene], h) for h in HORIZONS] for scene in SCENES},
    }


def markdown(report: dict) -> str:
    def table(rows):
        lines = [
            "| 动作块 | 成功 / 配对集 | 成功率 | 对比 50 步 | 多成功 / 少成功 | 都成功时的步数差中位数 |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            success = f"{r['success_pct']:.1f}%" if r["success_pct"] is not None else "—"
            delta = f"{r['delta_vs_50_pp']:+.1f} 个百分点" if r["delta_vs_50_pp"] is not None else "—"
            steps = r["median_step_delta_both_succeeded"]
            step_text = f"{steps:+g} 步（{r['both_succeeded']} 对）" if steps is not None else "—"
            lines.append(f"| {r['horizon']} | {r['successes']} / {r['n']} | {success} | {delta} | {r['wins_vs_50']} / {r['losses_vs_50']} | {step_text} |")
        return lines

    lines = [
        "# TableClear V1 动作块诊断", "",
        f"作业 `{report['run']}`。证据快照时间为 `{report['snapshot_utc']}`。状态为 `{report['status']}`。", "",
        f"按场景、seed 和官方初始状态索引配对，并核对四档的初始状态哈希。纳入 {report['paired_count']} 组。"
        f"排除 {len(report['incomplete_pairs'])} 组未齐全和 {len(report['hash_mismatches'])} 组哈希不匹配的条件。"
        f"执行故障记录共 {sum(report['infrastructure_errors'].values())} 条。", "",
        "基线是模型默认的 50 步动作块。步数差只在两档都成功的配对上计算，负数表示新档更省步。", "",
        *table(report["overall"]), "",
    ]
    for scene, rows in report["by_scene"].items():
        lines.extend([f"## `libero-10-{int(scene):02d}`", "", *table(rows), ""])
    lines.extend(["这是预留诊断初始状态上的直接执行对照。高层视觉闭环的效果需要正式方法对照。", ""])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    with ZipFile(args.archive) as archive:
        manifest = json.loads(archive.read("archive-manifest.json"))
        rows = [json.loads(line) for line in archive.read("index/results.jsonl").splitlines()]
    report = summarize(rows, manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "动作块配对结果.md").write_text(markdown(report), encoding="utf-8")
    (args.output_dir / "horizon-metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "paired_count", "overall")}, ensure_ascii=False))
