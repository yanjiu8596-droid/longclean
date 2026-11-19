"""Reconstruct the source basket site's rigid transform from the audited replay."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

SUMMARY_SHA = "0e464752ed75440c02fe6c9c267a239d22519b2183d95a173a1e940181142646"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    assert not a.out.exists()
    assert hashlib.sha256(a.summary.read_bytes()).hexdigest() == SUMMARY_SHA
    source = json.loads(a.summary.read_text())
    region = source["common_fork"]["region"]
    original = source["source_contact_450"]
    assert region["site_body_id"] == region["basket_body_id"]
    old_rotation = np.asarray(region["basket_body_rotation"])
    local_position = old_rotation.T @ (np.asarray(region["position_m"]) - region["basket_body_position_m"])
    local_rotation = old_rotation.T @ np.asarray(region["rotation"])
    basket = original["body_poses"]["basket_1"]
    rotation = np.asarray(basket["rotation_matrix"])
    site_position = np.asarray(basket["position_m"]) + rotation @ local_position
    site_rotation = rotation @ local_rotation
    size = np.asarray(region["size"])
    extent = np.abs(site_rotation @ size)
    lower, upper = site_position - extent, site_position + extent
    lower[2] -= .01
    objects = {}
    for obj, expected in original["goals"].items():
        center = np.asarray(original["body_poses"][obj]["position_m"])
        inside = bool(np.all(center > lower) and np.all(center < upper))
        assert inside == expected
        objects[obj] = {"center_m": center.tolist(), "lower_margin_m": (center - lower).tolist(),
                        "upper_margin_m": (upper - center).tolist(), "reconstructed_inside": inside,
                        "official_goal": expected, "failed_axes": [axis for i, axis in enumerate("xyz")
                            if not lower[i] < center[i] < upper[i]]}
    result = {"schema": "v2-source-region-check/1", "summary_sha256": SUMMARY_SHA,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "source_contacts_sha256": source["source_contacts_sha256"],
              "step": original["step"], "state_sha256": original["state_sha256"],
              "site_position_m": site_position.tolist(), "site_rotation": site_rotation.tolist(),
              "local_site_position_m": local_position.tolist(), "local_site_rotation": local_rotation.tolist(),
              "lower_m": lower.tolist(), "upper_m": upper.tolist(), "objects": objects,
              "method": "Rigid site attached to the same basket body; reconstruct from audited step-400 site and step-450 basket poses, then use the installed strict axis-aligned in_box formula.",
              "limit": "Matrix reconstruction has floating-point rounding; all three resulting Boolean predicates match original native values. No original data, simulator state or predicate was modified."}
    with a.out.open("x", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print("SOURCE_REGION_CHECK", objects["alphabet_soup_1"], flush=True)


if __name__ == "__main__":
    main()
