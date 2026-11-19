"""Combine two independently audited development batches without replaying them."""
import argparse
from itertools import combinations
from pathlib import Path

from dense_online_audit import ARMS, NORMAL, RUNNER_SHA, checked_path, read, require, sha, stamp, write


FIRST_SHA = "503da932fdc3792e3ae111e10cd5917e082ff054fc5cbeba68664906178c6fd4"
EXTRA_RUNNER_SHA = "68de384ab9dceaaa4e9a8f2d0f17e60fedc64848954cd9b26b3bfee18bc2a9f2"
AUDITORS = ("e8d2ead95b3c9cada15cebc034c87c12fbab06a5623ad3e10a93f8425a9d8dd3",
            "2b6b859167d9072fe49d56495c568856268d373933de7e2324200e40c2f5cc15")
LAYOUTS = tuple(range(931300, 931308))


def batch(root, audit_path, digest, tag, seeds, runner, auditor):
    root = root.resolve()
    require(sha(audit_path) == digest, f"Changed {tag} audit")
    audit = read(audit_path)
    require(audit["schema"] == "v2-dense-online-audit/1" and audit["audit_sha256"] == auditor
            and audit["runner_sha256"] == runner and audit["independent_pilot_release"] is False
            and audit["all_four_unique_initials_verified"] is True and [c["card"] for c in audit["cards"]] == [2, 3],
            f"Wrong or incomplete {tag} audit")
    records, fixtures = {}, {}
    for c in audit["cards"]:
        card = c["card"]
        index = read(checked_path(root, f"dense-online-{card}.json", c["manifest_sha256"]))
        plan = read(checked_path(root, f"dense-online-plan-{card}.json", c["plan_sha256"]))
        assigned = [s for s in seeds if s % 2 == card - 2]
        require(index["card"] == plan["card"] == card and index["complete"] is c["complete"]
                and index["plan_sha256"] == c["plan_sha256"] and plan["gate_evidence"] == audit["gate_evidence"]
                and [row["layout_seed"] for row in plan["cases"]] == assigned
                and [row["layout_seed"] for row in index["prefixes"]] == assigned
                and runner in plan["code"].values(), "Changed card assignment or provenance")
        expected = [(r["layout_seed"], arm) for r in plan["cases"] for arm in r["order"]]
        declared = {(r["layout_seed"], r["arm"]): r for r in index["episodes"]}
        require(list(declared) == expected[:len(declared)] and len(declared) == len(index["episodes"])
                and (len(declared) == 6 if c["complete"] else audit["adaptive_safety_stop"] is not None),
                "Card rollout order, completion or safety-stop provenance differs")
        for f in index["prefixes"]:
            seed = f["layout_seed"]
            require(f["fixture"] == f"fixtures/{seed}/fixture.json", "Unexpected fixture path")
            path = checked_path(root, f["fixture"], f["fixture_sha256"])
            meta = read(path)
            prefix = read(checked_path(path.parent, "prefix.json", meta["prefix_sha256"]))
            require(meta == {k: v for k, v in f.items() if k not in {"fixture", "fixture_sha256"}}
                    and stamp(prefix["initial"]) == stamp(f["initial"]) and stamp(prefix["end"]) == stamp(f["fork"])
                    and f["qualified"] is audit["prefix_eligible"][str(seed)], "Changed joint prefix evidence")
            fixtures[seed] = {"layout_seed": seed, "batch": tag, "eligible": f["qualified"], "reason": f["reason"],
                              "initial": f["initial"], "fork": f["fork"], "prefix_sha256": f["prefix_sha256"]}
        for key in expected:
            r, = [r for r in audit["records"] if (r["layout_seed"], r["arm"]) == key]
            row = declared.get(key)
            if row is None:
                require(r["status"] == "unrecorded_after_explicit_stop", "Unjustified missing aggregate record")
            else:
                require(r["status"] == row["status"] and r.get("result") == row["result"]
                        and r.get("result_sha256") == row["result_sha256"]
                        and r.get("official_success") is row.get("official_success")
                        and r.get("qualified_success") is row.get("qualified_success"), "Manifest and audit differ")
            require(r.get("prefix_sha256", fixtures[key[0]]["prefix_sha256"]) == fixtures[key[0]]["prefix_sha256"],
                    "Audited arm prefix changed")
            records[key] = {**r, "batch": tag}
    require(len(records) == len(audit["records"]) == 12, "Expected exactly twelve prespecified rows")
    return fixtures, records, {"batch": tag, "audit_sha256": digest, "runner_sha256": runner, "cards": audit["cards"],
                                "gate_evidence": audit["gate_evidence"], "adaptive_safety_stop": audit["adaptive_safety_stop"]}


def merge(inputs):
    fixtures = {k: v for f, _, _ in inputs for k, v in f.items()}
    records = {k: v for _, r, _ in inputs for k, v in r.items()}
    require(set(fixtures) == set(LAYOUTS) and set(records) == {(seed, arm) for seed in LAYOUTS for arm in ARMS}
            and all(len({f["initial"][field] for f in fixtures.values()}) == 8 for field in ("raw_sha256", "state_sha256"))
            and inputs[0][2]["gate_evidence"] == inputs[1][2]["gate_evidence"], "Changed shared gates or duplicate assignment")
    by_arm = {}
    for arm in ARMS:
        group = [records[seed, arm] for seed in LAYOUTS]
        by_arm[arm] = {"assigned": len(group), "valid_native_terminals": sum(r["valid_native_terminal"] for r in group),
                       "prefix_skips": sum(r["status"] == "prefix_unavailable" for r in group),
                       "interrupted": sum(r["interrupted"] for r in group),
                       "observed_complete_incomplete": sum(r["status"] == "observed_complete_incomplete" for r in group),
                       "official_successes": sum(r["official_success"] is True for r in group),
                       "qualified_successes": sum(r["qualified_success"] is True for r in group),
                       **{k: sum(r.get(k, 0) for r in group) for k in
                          ("repeats", "regrasp_steps", "incorrect_predicate_commits", "still_held_commits")}}
    pairs = []
    for a, b in combinations(ARMS, 2):
        values = [(s, records[s, a], records[s, b]) for s in LAYOUTS]
        admitted = [(s, x, y) for s, x, y in values
                    if all(r["status"] in NORMAL | {"observed_complete_incomplete"} for r in (x, y))]
        rows = [{"layout_seed": s, "result": "tie" if x["qualified_success"] == y["qualified_success"]
                 else a if x["qualified_success"] else b, "repeat_delta_right_minus_left": y["repeats"] - x["repeats"]}
                for s, x, y in admitted]
        pairs.append({"left": a, "right": b, "n_paired": len(rows), "wins_left": sum(r["result"] == a for r in rows),
                      "wins_right": sum(r["result"] == b for r in rows), "ties": sum(r["result"] == "tie" for r in rows),
                      "unscored_layouts": [s for s in LAYOUTS if s not in {r["layout_seed"] for r in rows}], "cases": rows})
    return {"schema": "v2-dense-online-combined/1", "scope": "seen_development_only", "audit_sha256": sha(Path(__file__)),
            "sources": [v for _, _, v in inputs], "layouts_checked": list(LAYOUTS), "planned_records": 24,
            "eligible_layouts": [seed for seed in LAYOUTS if fixtures[seed]["eligible"]], "all_eight_initials_unique": True,
            "fixtures": [fixtures[s] for s in LAYOUTS], "records": [records[s, a] for s in LAYOUTS for a in ARMS],
            "by_arm": by_arm, "paired_success": pairs, "independent_pilot_release": False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("first", "extra", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--extra-audit-sha", required=True)
    p.add_argument("--first-audit", type=Path)
    p.add_argument("--extra-audit", type=Path)
    a = p.parse_args()
    require(not a.out.exists() and a.extra.resolve() != a.first.resolve(), "Distinct runs and a fresh output are required")
    write(a.out, merge([batch(a.first, a.first_audit or a.first / "audit.json", FIRST_SHA, "first", LAYOUTS[:4], RUNNER_SHA, AUDITORS[0]),
                        batch(a.extra, a.extra_audit or a.extra / "audit.json", a.extra_audit_sha.lower(),
                              "extra", LAYOUTS[4:], EXTRA_RUNNER_SHA, AUDITORS[1])]))


if __name__ == "__main__":
    main()
