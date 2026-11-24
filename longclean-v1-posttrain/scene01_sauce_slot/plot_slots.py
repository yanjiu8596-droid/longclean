"""Plot the observed candidate centers and the two-can planning calculation."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
import numpy as np


def panel(ax, views):
    origin = np.median([v["rim"][:2] for v in views], axis=0)
    corners = np.array([[-1.,-1.],[1.,-1.],[1.,1.],[-1.,1.]]) * .063
    for i, v in enumerate(views):
        outline = corners @ np.asarray(v["basis"]).T + v["rim"][:2] - origin
        ax.add_patch(Polygon(1000*outline, fill=False, color="#687783", alpha=.35 if i+1<len(views) else .9))
    ax.set(xlim=(-96,96),ylim=(-96,96),aspect="equal",xlabel="World X offset (mm)",ylabel="World Y offset (mm)")
    ax.grid(alpha=.14)
    return origin


def draw(audit, earlier, destination):
    fig, axes = plt.subplots(1,2, figsize=(9.8,5.1), constrained_layout=True)
    a,b = axes
    origin = panel(a, earlier["rim_fits"])
    joint = earlier.get("joint",{})
    for key,color,label in (("first_soup_xy","#19667a","Soup"),("reserved_sauce_xy","#c46330","Sauce")):
        if key in joint:
            xy = 1000*(np.asarray(joint[key])-origin)
            a.add_patch(Circle(xy,33,color=color,alpha=.22))
            a.plot(*xy,"o",color=color,markersize=4)
            a.annotate(label,xy,xytext=(0,7),textcoords="offset points",ha="center",fontsize=10)
    a.set_title("State 397 · before the first placement")
    if "best_clearance_mm" in joint:
        a.set_xlabel(f"Best joint footprint clearance: {joint['best_clearance_mm']:.1f} mm\nWorld X offset (mm)")
    selection=audit["selection"]
    rims=[v for v in selection["views"] if "rim" in v]
    origin=panel(b,rims)
    rim=np.asarray(selection["rim"][:2])
    basis=np.asarray(selection["basis"])
    axis=np.arange(-126,127)*.001
    xy=np.stack(np.meshgrid(axis,axis),axis=-1).reshape(-1,2)
    mask=(.063-np.abs(xy@basis).max(1)-.033>=.002)&(abs(xy[:,1])<=.007)
    p=1000*(xy[mask]+rim-origin)
    b.scatter(p[:,0],p[:,1],s=3,color="#c46330",alpha=.48,label="Wall-clear sauce centers")
    soup=1000*(np.asarray(selection["occupant"][:2])-origin)
    b.add_patch(Circle(soup,33,color="#19667a",alpha=.22))
    b.add_patch(Circle(soup,68,fill=False,color="#19667a",linestyle="--",linewidth=1.4,label="Required center separation"))
    b.plot(*soup,"o",color="#19667a",markersize=4)
    b.set_title("State 569 · soup already in the basket")
    feasible=selection["grid"]["centerline"]["feasible"]
    b.set_xlabel(f"Feasible sauce centers with soup retained: {feasible}\nWorld X offset (mm)")
    b.legend(loc="lower left",fontsize=7,framealpha=.94)
    fig.suptitle("Scene 01 · same-level placement geometry",fontsize=14)
    fig.savefig(destination,dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    for arg in ("audit","earlier","output"):
        parser.add_argument(arg,type=Path)
    args=parser.parse_args()
    draw(json.loads(args.audit.read_bytes()),json.loads(args.earlier.read_bytes()),args.output)
