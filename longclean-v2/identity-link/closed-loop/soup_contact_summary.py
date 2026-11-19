"""Compact, evaluator-only contact evidence and labeled replay montages."""
import argparse
import hashlib
import json
from pathlib import Path


AUDIT_SHA = "503da932fdc3792e3ae111e10cd5917e082ff054fc5cbeba68664906178c6fd4"
REPLAY_SHA = "13dbd5a239f96be7d2fffaaedb9eec72ec331395819c2439c82a0b9501d656b5"
REPORTS = {931301: "72f8e63eeca8774b836778723558aad7c2c04e93520e1420fd0265e1fbe46d4c",
           931303: "4f4e70c0c9060acb862706cbfd64f9d2d9fe0d4b813ddc5b013b8be784892ea0"}
SELECTED = {931301: (350, 400, 410, 415, 420, 430, 450), 931303: (300, 350, 400, 450)}
SOUP = "alphabet_soup_1"
THRESHOLD_N = .01


def require(ok, reason):
    if not ok:
        raise RuntimeError(reason)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def checked(root, name, digest):
    path = (root / name).resolve()
    require(not Path(name).is_absolute() and path.is_relative_to(root.resolve()) and path.is_file()
            and sha(path) == digest, f"Changed or unsafe evidence: {name}")
    return path


def compact_contacts(records, layout):
    geometries, samples, object_steps = {}, [], {}
    for row in records:
        step = row["step"]
        if not 300 <= step <= 450:
            continue
        retained, small = [], []
        for c in row["soup_contacts"]:
            key = str(c["other_geom_id"])
            description = {k: c[k] for k in ("other_object", "other_geom_name", "other_body_name")}
            require(key not in geometries or geometries[key] == description, "Contact geometry identity changed")
            geometries[key] = description
            force = c["normal_force_N"]
            if force <= THRESHOLD_N:
                small.append(force)
                continue
            item = {"geom": c["other_geom_id"], "normal_N": round(force, 6), "distance_m": round(c["distance_m"], 7)}
            if layout == 931303 or step >= 400:
                item.update(xyz_m=[round(x, 6) for x in c["position_m"]],
                            normal_axis=[round(x, 6) for x in c["contact_frame"][0]],
                            soup_is_geom1=c["other_geom_id"] == c["geom2"])
            retained.append(item)
            object_steps.setdefault(c["other_object"], set()).add(step)
        samples.append({"step": step, "held_soup": row["held_soup"], "official_soup_goal": row["goals"][SOUP],
            "grip_gap_m": round(row["grip_gap_m"], 6),
            "soup_xyz_m": [round(x, 6) for x in row["body_poses"][SOUP]["position_m"]],
            "force_contacts": retained, "small_or_nonpositive_contact_count": len(small),
            "small_or_nonpositive_normal_N_sum": round(sum(small), 7),
            "small_or_nonpositive_normal_N_max": None if not small else round(max(small), 7)})
    require([r["step"] for r in samples] == list(range(300, 451, 5)), "Missing original five-step contact samples")
    held = [r["step"] for r in samples if r["held_soup"]]
    return {"layout_seed": layout, "geometry_dictionary": geometries, "samples": samples,
        "sample_steps_with_force_by_object": {obj: sorted(steps) for obj, steps in object_steps.items()},
        "sample_steps_with_soup_held": held,
        "first_sampled_unheld_after_first_held": None if not held else next(
            (r["step"] for r in samples if r["step"] > held[0] and not r["held_soup"]), None),
        "official_soup_goal_true_samples": [r["step"] for r in samples if r["official_soup_goal"]]}


def montage(directory, report, summary, path):
    from PIL import Image, ImageDraw
    require(not path.exists(), "Existing montage")
    selected = SELECTED[report["layout_seed"]]
    canvas = Image.new("RGB", (768, ((len(selected) + 1) // 2) * 216), "white")
    draw = ImageDraw.Draw(canvas)
    snapshots = {r["step"]: r for r in report["snapshots"]}
    samples = {r["step"]: r for r in summary["samples"]}
    sources = []
    for index, step in enumerate(selected):
        image_row, sample = snapshots[step], samples[step]
        source = checked(directory, image_row["png"], image_row["png_sha256"])
        with Image.open(source) as image:
            require(image.size == (512, 256), "Unexpected dual-camera keyframe size")
            tile = image.convert("RGB").resize((384, 192), Image.Resampling.LANCZOS)
        x, y = (index % 2) * 384, (index // 2) * 216
        draw.text((x + 5, y + 4), f"step {step}  held={int(sample['held_soup'])}  gap={sample['grip_gap_m']:.3f}m", fill="black")
        canvas.paste(tile, (x, y + 24))
        sources.append({"step": step, "png": image_row["png"], "sha256": image_row["png_sha256"]})
    with path.open("xb") as stream:
        canvas.save(stream, format="JPEG", quality=87, optimize=True)
    return {"file": path.name, "sha256": sha(path), "size_pixels": list(canvas.size), "source_pngs": sources}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "source-run", "audit", "out"):
        cli.add_argument("--" + name, required=True, type=Path)
    args = cli.parse_args()
    root, source, out = args.run.resolve(), args.source_run.resolve(), args.out.resolve()
    require(not out.exists() and sha(args.audit) == AUDIT_SHA, "Existing output or changed original audit")
    audited = read(args.audit)
    results = []
    for layout, digest in REPORTS.items():
        marker = read(root / f"soup-execution-{layout}.json")
        require(marker["complete"] is True and marker["report_sha256"] == digest, "Incomplete or changed replay marker")
        report_path = checked(root, marker["report"], digest)
        directory, report = report_path.parent, read(report_path)
        require(report["schema"] == "v2-soup-execution-replay/1" and report["source_sha256"] == REPLAY_SHA
                and report["source_audit_sha256"] == AUDIT_SHA and report["layout_seed"] == layout, "Unexpected replay provenance")
        entry, = [r for r in audited["records"] if r["layout_seed"] == layout and r["arm"] == "adaptive1"]
        require(report["source_result"] == entry["result"] and report["source_result_sha256"] == entry["result_sha256"], "Wrong source trajectory")
        original_path = checked(source, entry["result"], entry["result_sha256"])
        original = read(original_path)
        native = rows(checked(directory, "actions.jsonl", report["actions_sha256"]))
        grasp = rows(checked(directory, "grasp-audit.jsonl", report["grasp_audit_sha256"]))
        contact = rows(checked(directory, "contacts.jsonl", report["contacts_sha256"]))
        require(native == [r for r in rows(checked(original_path.parent, "actions.jsonl", original["actions_sha256"])) if r["step"] <= 450]
                and grasp == [r for r in rows(checked(original_path.parent, "grasp-audit.jsonl", original["grasp_audit_sha256"])) if r["step"] <= 450],
                "Replay actions or grasp audit differ from the sealed original")
        for n, g, c in zip([report["initial"], *native], grasp, contact, strict=True):
            require(n["step"] == g["step"] == c["step"] and n["state_sha256"] == g["state_sha256"] == c["state_sha256"] == c["state_sha256_after_read"]
                    and c["goals"] == g["goals"] and c["held_soup"] == g["held_by_target"][SOUP], "Contact sidecar state or grasp changed")
            if n["step"]:
                require(c["goals"] == n["audit"]["goals"] and all(p["position_m"] == n["audit"]["positions"][obj]
                        for obj, p in c["body_poses"].items()), "Contact body pose differs from native audit")
        native_at = {r["step"]: r for r in native}
        for snap in report["snapshots"]:
            require(all(snap[k] == native_at[snap["step"]][k] for k in ("state_sha256", "observation_sha256")), "Keyframe anchor changed")
            checked(directory, snap["npz"], snap["npz_sha256"])
            checked(directory, snap["png"], snap["png_sha256"])
        summary = compact_contacts(contact, layout)
        summary.update(report_sha256=digest, contacts_sha256=report["contacts_sha256"], verified_source_rows=len(native),
                       verified_contact_rows=len(contact), raw_contacts=f"{layout}/contacts.jsonl")
        summary["montage"] = montage(directory, report, summary, out.parent / f"soup-contacts-{layout}.jpg")
        results.append(summary)
    output = {"schema": "v2-soup-contact-summary/1", "source_sha256": sha(Path(__file__)), "source_audit_sha256": AUDIT_SHA,
        "replay_source_sha256": REPLAY_SHA, "listing_threshold_normal_force_N": THRESHOLD_N, "cases": results,
        "interpretation": "Only contacts with normal force >0.01N are listed individually; lower forces remain counted and in the pinned raw sidecar. This threshold is for display, not a support classifier. Basket height-band hints are not used to label rim or floor. Raw contact frame axes and geom ordering are retained; no vertical support inference is made.",
        "scope": "Read-only exact-action replay evidence. No new intervention, success claim, or controller input."}
    encoded = json.dumps(output, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n"
    with out.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print("SOUP_CONTACT_SUMMARY", len(encoded.encode("utf-8")), "bytes", flush=True)


if __name__ == "__main__":
    main()
