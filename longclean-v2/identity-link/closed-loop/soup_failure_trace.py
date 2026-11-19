"""Export evaluator-only grasp/loss evidence from three existing audited rollouts."""
import argparse
import json
import math
from pathlib import Path

import dense_online_diagnostic as base


DIAGNOSTIC_SHA = "693e5f937f66a8fa342560df02dc77d823767e5ad01eb9786380e5aee82a23ff"
HELPER_SHA = "5dcb9446bd4aec14712572c46f47a459f6806d94086b069ff2803a46d204dff4"
CASES = ((931301, "adaptive1", 270), (931303, "actual10", 260), (931303, "adaptive1", 260))


def subtract(a, b):
    return [x - y for x, y in zip(a, b, strict=True)]


def summarize(arm, reference):
    timeline, spans, current = [], [], None
    soup = base.SOUP
    physical = arm["physical"]
    before = reference
    for row in arm["native"]:
        step = row["step"]
        if step < reference:
            continue
        audit, held = row["audit"], physical[step]["held_by_target"][soup]
        pos, basket, eef = audit["positions"][soup], audit["positions"]["basket_1"], audit["site_xyz"]
        offset = subtract(eef, pos)
        timeline.append({"step": step, "source_start_step": row["start_step"], "held_soup": held,
            "goals": physical[step]["goals"], "soup_body_xyz_m": pos, "basket_body_xyz_m": basket, "eef_site_xyz_m": eef,
            "soup_minus_basket_m": subtract(pos, basket), "eef_minus_soup_m": offset,
            "eef_to_soup_body_distance_m": math.sqrt(sum(v * v for v in offset)),
            "gripper_command_last": row["actions"][-1][-1], "gripper_command_min": min(a[-1] for a in row["actions"]),
            "gripper_command_max": max(a[-1] for a in row["actions"]), "state_sha256": row["state_sha256"]})
        if held and current is None:
            current = {"first_held_sample": step, "previous_unheld_sample": None if step == reference else before,
                       "left_censored_at_reference": step == reference}
        if current is not None:
            if held:
                current["last_held_sample"] = step
            else:
                current["first_unheld_sample"] = step
                spans.append(current)
                current = None
        before = step
    if current is not None:
        spans.append({**current, "first_unheld_sample": None})
    at = {r["step"]: r for r in timeline}
    controls = arm["controls"]
    for span in spans:
        last, first = span["last_held_sample"], span["first_unheld_sample"]
        span["last_held_geometry"] = at[last]
        span["first_unheld_geometry"] = None if first is None else at[first]
        span["commands_in_loss_bracket"] = None if first is None else [
            {"step": step, "gripper_command": controls[step - 1][-1]} for step in range(last + 1, first + 1)]
        span["first_open_command_after_first_held_sample"] = next((step for step in range(span["first_held_sample"] + 1, len(controls) + 1)
                                                                 if controls[step - 1][-1] < -.5), None)
    closest = min(timeline, key=lambda r: r["eef_to_soup_body_distance_m"])
    result = arm["result"]
    return {"layout_seed": result["layout_seed"], "arm": result["arm"], "reference_step": reference,
            "sampled_soup_grasp_spans": spans, "closest_sampled_eef_to_soup_body": closest,
            "max_sampled_soup_body_z_after_reference_m": max(r["soup_body_xyz_m"][2] for r in timeline),
            "max_sampled_soup_body_z_while_held_m": max((r["soup_body_xyz_m"][2] for r in timeline if r["held_soup"]), default=None),
            "soup_official_transitions": [t for r in arm["native"] for t in r["transitions"] if t["object"] == soup],
            "timeline": timeline, "gripper_commands": [{"step": step, "command": controls[step - 1][-1]}
                for step in range(reference + 1, len(controls) + 1)],
            "terminal_goals": result["terminal_native_audit"]["goals"], "terminal_held": result["terminal_held_by_target"],
            "inputs": {k: arm["entry"][k] for k in ("result", "result_sha256", "actions_sha256", "grasp_audit_sha256")}}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "audit", "diagnostic", "out"):
        cli.add_argument("--" + name, type=Path, required=True)
    args = cli.parse_args()
    base.require(not args.out.exists() and base.sha(args.audit) == base.AUDIT_SHA
                 and base.sha(args.diagnostic) == DIAGNOSTIC_SHA and base.sha(Path(base.__file__)) == HELPER_SHA,
                 "Changed sealed input/helper or existing output")
    audit = base.read(args.audit)
    base.require(audit["schema"] == "v2-dense-online-audit/1" and audit["runner_sha256"] == base.RUNNER_SHA
                 and len(audit["records"]) == 12 and audit["valid_native_terminal_records"] == 12
                 and audit["adaptive_safety_stop"] is None, "Incomplete source experiment")
    sources, records = {}, []
    for layout, arm, reference in CASES:
        entry, = [r for r in audit["records"] if r["layout_seed"] == layout and r["arm"] == arm]
        loaded = base.load_arm(args.run.resolve(), entry)
        sources[layout, arm] = loaded
        record = summarize(loaded, reference)
        records.append(record)
        print("SOUP_FAILURE_TRACE", layout, arm, record["sampled_soup_grasp_spans"], flush=True)
    left, right = sources[931303, "actual10"], sources[931303, "adaptive1"]
    common = sorted(set(left["endpoint"]) & set(right["endpoint"]))
    same = left["controls"] == right["controls"] and all(
        left["endpoint"][s][k] == right["endpoint"][s][k] for s in common for k in ("state_sha256", "observation_sha256"))
    base.require(same, "The presumed duplicate 931303 trajectory differs")
    output = {"schema": "v2-soup-failure-trace/1", "source_sha256": base.sha(Path(__file__)),
        "audit_sha256": base.AUDIT_SHA, "diagnostic_sha256": DIAGNOSTIC_SHA, "cases": records,
        "931303_shared_controls_and_common_state_observation_anchors": {"verified": same, "anchor_steps": common},
        "scope": "Read-only evaluator geometry and sampled grasp evidence. No new policy calls or simulation.",
        "limits": ["Grip loss lies between the last held and first unheld sample; this is not the exact release step.",
            "Negative gripper commands do not measure finger opening; measured gap is absent from these raw native rows.",
            "Body-center offsets do not establish basket collision or rim support; contact evidence requires separate exact-action replay."]}
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
