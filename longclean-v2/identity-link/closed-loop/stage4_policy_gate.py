"""Keep a failed smoke from opening the preassigned pilot case pool."""
import argparse
from pathlib import Path

from stage4_fixture import GPU, read, require, sha, write
from common import CHUNK, SAUCE, SOUP


FROZEN = "428f6c06819003a2b31ec9e4563e0d89684a8eda7d286a5a7a919f9b13968f24"
AUDIT = "774a8a771af191b3fe9576551313895920d076630beaa7a30ccaf20082ba4303"
PARITY = "dcd5495de6d2eeef4d7259cf711983abbde30d02ce8cd9924637b6984e7fba6b"
NATIVE = "84ec009b2a304a705dfee5004bb64ddb42fde79864adea981be042518f506b9d"
PINNED_CASES = {2: (931000, 46, True), 3: (931005, 47, False)}


def check_smoke(smoke):
    require(sha(smoke / "frozen.json") == FROZEN, "Smoke freeze changed")
    sealed = read(smoke / "frozen.json")
    require({r["card"]: (r["layout_seed"], r["policy_seed"], r["moved"]) for r in sealed["cases"]}
            == PINNED_CASES, "Unexpected diagnostic layouts or seeds")
    here = Path(__file__).resolve().parent
    for name, digest in sealed["code"].items():
        require(sha(here / name) == digest, f"Frozen code changed: {name}")
    require(sha(here / "audit4_native.py") == NATIVE, "Unexpected pilot audit revision")
    require(sha(smoke / "audit4-native.json") == AUDIT and sha(smoke / "legacy-parity.json") == PARITY,
            "Original smoke evidence changed")
    audit, parity = read(smoke / "audit4-native.json"), read(smoke / "legacy-parity.json")
    require(audit["freeze_sha256"] == FROZEN == parity["smoke_freeze_sha256"] and parity["gate"]
            and audit["gate"]["smoke_release_pilot"] and audit["gate"]["zero_wrong_candidate_observations_on_both_arms"]
            and not any(r["arms"]["candidate"]["soup_writes"] for r in audit["cases"]),
            "Original smoke parity/safety is missing")


def score(output):
    reports, valid, wrong, wrong_shadow = [], 0, 0, 0
    path = output / "gate.json"
    require(not path.exists(), "Refusing to overwrite policy smoke gate")
    source = sha(Path(__file__).with_name("progress_policy.py"))
    for card in GPU:
        name = f"progress-policy-{card}.json"
        r = read(output / name)
        require(r["freeze_sha256"] == FROZEN and r["source_sha256"] == source and r["card"] == card
                and (r["layout_seed"], r["policy_seed"], r["moved"]) == PINNED_CASES[card],
                "Policy diagnostic case or code does not match the sealed inputs")
        evidence = []
        for v in r["valid_writes"]:
            event, before, after = v["event"], v["boundary_before"], v["boundary_after"]
            require(event["object"] == SOUP and event["event"] == "in_basket"
                    and event["writer"] == "stage3d_plane" and event["applied"] is True
                    and event["confirmed_in_audit"] is True and event["held_soup"] is False
                    and after["official_success"] is False and SOUP in before["order"] and SOUP not in after["order"]
                    and after["target"] == SAUCE and before["step"] + CHUNK == after["step"]
                    and before["step"] < event["step"] <= after["step"],
                    "Report did not prove a correct nonterminal write at the next decision")
            evidence.append({"write_step": event["step"], "decision_step": after["step"], "next_target": after["target"]})
        require(r["valid_write"] is bool(evidence) and type(r["wrong_writes"]) is int
                and type(r["wrong_candidate_observations"]) is int
                and min(r["wrong_writes"], r["wrong_candidate_observations"]) >= 0, "Invalid diagnostic counts")
        valid += len(evidence)
        wrong += r["wrong_writes"]
        wrong_shadow += r["wrong_candidate_observations"]
        reports.append({"card": card, "layout_seed": r["layout_seed"], "report_sha256": sha(output / name),
                        "valid_writes": evidence, "wrong_writes": r["wrong_writes"],
                        "wrong_candidate_observations": r["wrong_candidate_observations"]})
    passed = bool(valid and wrong == wrong_shadow == 0)
    write(path, {"schema": "v2-stage4-policy-smoke-gate/1", "smoke_freeze_sha256": FROZEN,
                 "diagnostic_source_sha256": source, "native_audit_sha256": NATIVE,
                 "gate_source_sha256": sha(Path(__file__)), "reports": reports,
                 "valid_nonterminal_writes": valid, "wrong_writes": wrong, "wrong_candidate_observations": wrong_shadow,
                 "release_pilot": passed})
    print(f"POLICY_GATE valid={valid} wrong_writes={wrong} wrong_observations={wrong_shadow} release={passed}", flush=True)
    return 0 if passed else 3


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("phase", choices=("preflight", "score"))
    cli.add_argument("--smoke", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    check_smoke(args.smoke)
    if args.phase == "preflight":
        require(not args.output.exists() and (Path(__file__).with_name("progress_policy.py")).is_file(),
                "Expected an unstarted policy diagnostic and its implementation")
        return 0
    return score(args.output)


if __name__ == "__main__":
    raise SystemExit(main())
