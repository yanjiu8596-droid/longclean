"""Check whether two floor positions could be reserved before releasing the first can."""
import argparse
import io
import json
from pathlib import Path
import tarfile

import numpy as np
from verify import calibrated, checked, data, rows


def select_pair(rims, params, hand):
    center = np.median([r[0][:2] for r in rims],axis=0)
    axis = np.arange(-100,101) * params["grid_m"]
    pts = center + np.stack(np.meshgrid(axis,axis),axis=-1).reshape(-1,2)
    wall = np.full(len(pts),np.inf)
    band = np.ones(len(pts),dtype=bool)
    for rim,basis in rims:
        delta = pts-rim[:2]
        wall = np.minimum(wall, params["inner_half_width_m"]-params["payload_radius_m"]-abs(delta@basis).max(1))
        band &= abs(delta[:,1]) <= params["centerline_y_half_width_m"]
    pts,wall = pts[band & (wall>=0)],wall[band & (wall>=0)]
    need = params["payload_radius_m"]+params["occupant_radius_m"]
    best=None
    count=0
    for i,u in enumerate(pts[:-1]):
        separation=np.linalg.norm(pts[i+1:]-u,axis=1)
        margins=np.minimum(np.minimum(wall[i],wall[i+1:]),separation-need)
        count+=int((margins>=params["margin_m"]).sum())
        j=int(np.argmax(margins))
        if best is None or margins[j]>best[0]:
            best=(float(margins[j]),i,i+1+j,float(separation[j]))
    result={"candidate_centers":len(pts),"feasible_unordered_pairs":count}
    if best is not None:
        clearance,i,j,dist=best
        a,b=sorted((i,j),key=lambda k:np.linalg.norm(hand-pts[k]))
        result.update(best_clearance_mm=clearance*1000,first_soup_xy=pts[a].tolist(),
                      reserved_sauce_xy=pts[b].tolist(),separation_mm=dist*1000,
                      worst_wall_clearance_mm=1000*float(min(wall[a],wall[b])))
    return result


def run(pair,earlier,project):
    expected="30bd792b285094d0b2c554b3eb1a76715cbc61802057a407ee364a86f667507a"
    assert checked(earlier)==expected
    with calibrated(pair,project) as tested, tarfile.open(earlier) as prior:
        from low_place import clear_depth,clouds,fit_rim
        plan=data(tested,"protocol.json")
        prefix="runs/"+plan["case"]+"-retry/integrated/"
        trace=rows(prior,prefix+"state.jsonl")
        contacts=rows(prior,prefix+"contacts.jsonl")
        fits=[]
        details=[]
        previous=None
        for j in (395,396,397):
            with np.load(io.BytesIO(prior.extractfile(prefix+f"carry-sensors/{j}.npz").read()),allow_pickle=False) as f:
                views=clouds(dict(f))
                fit=fit_rim(views,np.asarray(trace[395]["site_position"]),previous)
            if fit is None:
                details.append({"step":j,"visible":False})
                continue
            _,rim,basis,edges=fit
            details.append({"step":j,"visible":True,"rim":rim.tolist(),"basis":basis.tolist(),"edges":edges.tolist()})
            fits.append((rim,basis))
            previous=rim
        result={"case":plan["case"],"earlier_sha256":expected,"view_steps":[395,396,397],
                "decision_state":397,"rim_fits":details,"threshold_mm":1000*plan["geometry"]["margin_m"],
                "native_grasp":{str(j):trace[j]["alphabet_soup_1"]["grasp"] for j in (395,396,397)},
                "physical_gripper_contact":{str(j):contacts[j]["objects"]["alphabet_soup_1"]["contact"] for j in (395,396,397)}}
        if len(fits)==3:
            result["joint"]=select_pair(fits,plan["geometry"],np.asarray(trace[397]["site_position"][:2]))
            if result["joint"]["feasible_unordered_pairs"]:
                # The existing straight-approach gate is a separate check from the floor packing.
                arm={k:trace[397][k] for k in ("site_position","site_rotation","gripper_qpos")}
                gap=np.abs(arm["gripper_qpos"])
                offset=(np.asarray(arm["site_rotation"])@np.array([(gap[1]-gap[0])/2,0.,0.]))[:2]
                result["straight_approach_depth_from_rim_m"]={
                    k:clear_depth(views,arm,rim,basis,np.asarray(result["joint"][k])-offset)
                    for k in ("first_soup_xy","reserved_sauce_xy")}
        return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    for arg in ("pair","earlier","output"):
        parser.add_argument(arg,type=Path)
    parser.add_argument("--project",type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    result=run(args.pair,args.earlier,args.project)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(result.get("joint"),result.get("straight_approach_depth_from_rim_m"))
