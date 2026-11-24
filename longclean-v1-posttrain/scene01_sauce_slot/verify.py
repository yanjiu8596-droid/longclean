"""Independently score the two archived branches and the matched native reference."""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import io
import json
import math
from pathlib import Path
import sys
import tarfile
import tempfile

import numpy as np


OBJECTS = ("alphabet_soup_1", "tomato_sauce_1")
CALIBRATION = {
    "entry75.py":"a2c965b67f3cf7d5968ad561d1ede59221a356e52582b33eef05a5792dee8e2b",
    "geometry.py":"53ee5c55cce36b9432b992c8acea0477e7c6d4eeb411aa14f48289f51060cb09",
    "hand.json":"bc4dc22b566df561c87fd5b8d5d78a306c7910db34746675d74d0cf88b0e34dc",
    "low_place.py":"d5ac9a4331963cf8987703d01304f5263d700ba389d19e999e0eee3f5f2c0616",
    "same_frame.py":"46903d6ff47d2f5c5d6babe2963e8fac7b6f55a5c2a2cc6fbae95410806a1e72",
}


def data(tar, path):
    return json.load(tar.extractfile(path))


def rows(tar, path):
    return [json.loads(line) for line in tar.extractfile(path)]


def checked(path):
    expected, name = path.with_name(path.name + ".sha256").read_text().split()
    assert Path(name).name == path.name
    assert sha256(path.read_bytes()).hexdigest() == expected
    return expected


@contextmanager
def calibrated(pair_path, project):
    checked(pair_path)
    project = project.resolve()
    with tarfile.open(pair_path) as tar, tempfile.TemporaryDirectory() as tmp:
        for name, expected in CALIBRATION.items():
            content = tar.extractfile("calibration/" + name).read()
            assert sha256(content).hexdigest() == expected
            (Path(tmp)/name).write_bytes(content)
        additions=[tmp,str(project/"scripts"),str(project/"src")]
        sys.path[:0]=additions
        try:
            yield tar
        finally:
            del sys.path[:len(additions)]


def reproduce_vision(pair_path, project, destination):
    with calibrated(pair_path,project) as t:
        from slot import Placement
        archived_code = t.extractfile("slot.py").read().replace(b"\r\n", b"\n")
        local_code = Path(__file__).with_name("slot.py").read_bytes().replace(b"\r\n", b"\n")
        assert archived_code == local_code
        p=data(t,"protocol.json")
        frames=[]
        for j in p["view_steps"]:
            with np.load(io.BytesIO(t.extractfile(f"A/view-{j}.npz").read()),allow_pickle=False) as f:
                frames.append(dict(f))
        decision=Placement(frames,p["geometry"]).decision
    destination.write_text(json.dumps(decision,ensure_ascii=False,indent=2)+"\n",encoding="utf8")


def spans(steps):
    runs = []
    for j in steps:
        if runs and runs[-1][1] + 1 == j:
            runs[-1][1] = j
        else:
            runs.append([j, j])
    return runs


def arm_scores(states, fork):
    soup, sauce = OBJECTS
    part = states[fork:]
    def steps(test):
        return [s["step"] for s in part if test(s)]
    def free(s, name):
        return not any(c.startswith("gripper0_") for c in s[name]["contacts"])
    def soup_hit(s):
        return any(c.startswith(soup + "_") for c in s[sauce]["contacts"])
    def wall(s):
        return any(c.startswith("basket_1_g") and c != "basket_1_g1" for c in s[sauce]["contacts"])
    def both(s):
        return s["dual"] and all(free(s, name) for name in OBJECTS)
    return {"official_dual_intervals": spans(steps(lambda s: s["dual"])),
            "dual_released_intervals": spans(steps(both)), "last20_dual_released": sum(both(s) for s in states[-20:]),
            "soup_goal_intervals": spans(steps(lambda s: s[soup]["in"])),
            "sauce_goal_intervals": spans(steps(lambda s: s[sauce]["in"])),
            "sauce_on_floor": spans(steps(lambda s: "basket_1_g1" in s[sauce]["contacts"])),
            "sauce_wall_and_soup_contact": spans(steps(lambda s: soup_hit(s) and wall(s))),
            "first_released_state": next((s["step"] for s in part if free(s, sauce)), None)}


def robust_targets(views, params):
    """Offline planning sensitivity. Both future centers have to fit every observed rim."""
    step = params["grid_m"]
    xy = np.median([f["rim"][:2] for f in views], axis=0)
    axis = np.arange(-100, 101) * step
    candidates = xy + np.stack(np.meshgrid(axis, axis), axis=-1).reshape(-1,2)
    for v in views:
        delta = candidates - np.array(v["rim"][:2])
        inside = np.abs(delta @ np.array(v["basis"])).max(axis=1)
        candidates = candidates[(inside <= params["inner_half_width_m"] - params["payload_radius_m"] - params["margin_m"])
                                & (np.abs(delta[:,1]) <= params["centerline_y_half_width_m"])]
    max_dist = 0.
    for pos in candidates:
        max_dist = max(max_dist, float(np.linalg.norm(candidates - pos, axis=1).max()))
    need = params["occupant_radius_m"] + params["payload_radius_m"] + params["margin_m"]
    return {"shared_candidate_centers":len(candidates), "maximum_separation_mm":max_dist*1000,
            "required_separation_mm":need*1000, "distinct_pair_available":max_dist >= need}


def verify(pair_path, old_path, vision):
    pair_sha, old_sha = checked(pair_path), checked(old_path)
    with tarfile.open(pair_path) as t, tarfile.open(old_path) as old:
        p = data(t, "protocol.json")
        assert p["reference_sha256"] == old_sha
        for line in t.extractfile("frozen.sha256").read().decode().splitlines():
            expected, path = line.split()
            assert sha256(t.extractfile(path).read()).hexdigest() == expected
        fork, end = p["fork"], p["end"]
        prefix = "runs/" + p["case"] + "-retry/integrated/"
        ep = data(old, prefix + "episode.json")
        assert ep["steps"] == p["formal_budget"] < fork and not ep["success"]
        chunks = rows(old, prefix + "actions.jsonl") + rows(old,"diagnostic/extension.jsonl")
        trace = np.array([a for c in chunks for a in c["actions"][:c["executed"]]], dtype="<f4")
        original = rows(old,"diagnostic/state.jsonl")
        parts = {r["step"]:r for r in rows(old,"diagnostic/placement-contacts.jsonl")}
        assert trace[fork-1,-1]>0 and trace[fork,-1]<0
        selection = data(t,"B/selection.json")
        assert data(t,"A/fork.json") == data(t,"B/fork.json")
        arms, actions, stats, device = {}, {}, {}, {}
        for arm in ("A","B"):
            path = arm + "/"
            assert t.extractfile(arm + ".exit").read().strip() == b"0"
            st = rows(t,path+"states.jsonl")
            assert [s["step"] for s in st] == list(range(end+1))
            log = rows(t,path+"actions.jsonl")
            assert [a["start"] for a in log] == list(range(fork,end))
            arr = np.array([r["action"] for r in log], dtype="<f4")
            finished = data(t,path+"finished.json")
            assert (finished["fork"],finished["end"],finished["source_sha256"]) == (fork,end,old_sha)
            assert sha256(arr.tobytes()).hexdigest() == finished["suffix_float32_sha256"]
            forked = data(t,path+"fork.json")
            assert st[fork]["sim_state_sha256"] == forked["physics"]
            for j, key in enumerate(p["view_steps"]):
                with np.load(io.BytesIO(t.extractfile(path+f"view-{key}.npz").read()),allow_pickle=False) as f:
                    assert int(f["step"]) == key
                    assert {k:sha256(f[k].tobytes()).hexdigest() for k in f.files} == forked["frames"][j]
            limit = end if arm == "A" else fork
            for sample in st[:limit+1]:
                j=sample["step"]
                assert sample["dual"] == all(sample[n]["in"] for n in OBJECTS)
                for n in OBJECTS:
                    assert sample[n]["xyz"] == original[j][n]["xyz"] and sample[n]["in"] == original[j][n]["in"]
                    if j in parts:
                        assert sample[n]["contacts"] == parts[j][n]
            assert st[0]["sim_state_sha256"] == ep["sim_state_sha256"]
            for c in chunks:
                at=c["start_step"]+c["executed"]
                if at<=limit:
                    assert st[at]["sim_state_sha256"] == c["sim_state_sha256"]
            arms[arm],actions[arm],stats[arm] = st,arr,arm_scores(st,fork)
            checks=finished["reference_observations"]
            device[arm]={"backend":finished["backend"],"visible_devices":finished["device"],"original_image_checkpoints":len(checks),
                         "original_image_both_exact":sum(x["camera1"] and x["camera2"] for x in checks.values())}
        assert arms["A"][:fork+1] == arms["B"][:fork+1] and np.array_equal(actions["A"],trace[fork:end])
        changed=int(np.any(actions["A"]!=actions["B"],axis=1).sum())
        if not selection["admitted"]:
            assert changed == 0 and arms["A"] == arms["B"]
        assert device["A"]["backend"]==device["B"]["backend"]
        # Development-only geometric cross-check with native scoring labels, never fed to B.
        center = np.asarray(original[fork]["basket"][:2])
        soup = np.asarray(original[fork][OBJECTS[0]]["xyz"][:2])
        offset = float(np.linalg.norm(soup-center))
        g = p["geometry"]
        bounds = []
        for half in (g["inner_half_width_m"], .073):
            upper = offset + math.sqrt(2) * (half-g["payload_radius_m"]-g["margin_m"])
            need = g["occupant_radius_m"] + g["payload_radius_m"] + g["margin_m"]
            bounds.append({"aperture_half_width_mm":half*1000,"upper_bound_can_separation_mm":upper*1000,
                           "required_mm":need*1000,"shortfall_mm":(need-upper)*1000})
        local = json.loads(vision.read_bytes())
        assert local["admitted"] == selection["admitted"] and local["reason"] == selection["reason"]
        assert len(local["views"])==len(selection["views"])==len(p["view_steps"])
        for a,b in zip(local["views"],selection["views"]):
            assert a["step"]==b["step"] and ("soup" in a)==("soup" in b)
            if "rim" in a:
                np.testing.assert_allclose(a["rim"],b["rim"],atol=1e-8,rtol=0)
            if a.get("soup") is not None:
                np.testing.assert_allclose(a["soup"]["center"],b["soup"]["center"],atol=1e-6,rtol=0)
        if "grid" in selection:
            for group in ("centerline","full_aperture"):
                assert local["grid"][group]["feasible"] == selection["grid"][group]["feasible"]
                assert local["grid"][group]["grid_candidates_inside_walls"] == selection["grid"][group]["grid_candidates_inside_walls"]
        rims=[v for v in selection["views"] if "rim" in v]
        return {"case":p["case"],"reference_sha256":old_sha,"evidence_sha256":pair_sha,"formal_success":ep["success"],
                "fork":fork,"end":end,"source_initial_sha256":ep["init_state_sha256"],"fork_state":data(t,"A/fork.json")["physics"],
                "sensor_frames_verified":p["view_steps"],"backend":device,"action_differences":changed,
                "identical_physics_frames":sum(a["sim_state_sha256"]==b["sim_state_sha256"] for a,b in zip(arms["A"],arms["B"])),
                "selection":selection,"arms":stats,"same_level_geometry_audit":{"native_soup_to_basket_center_mm":1000*offset,
                "sensitivity_bounds":bounds,"joint_centers_across_raw_rim_estimates":robust_targets(rims,g) if rims else None}}


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for argument in ("pair","reference","vision","output"):
        parser.add_argument(argument,type=Path)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    reproduce_vision(args.pair,args.project,args.vision)
    result=verify(args.pair,args.reference,args.vision)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(result["selection"]["reason"],result["action_differences"],result["arms"],result["same_level_geometry_audit"])
