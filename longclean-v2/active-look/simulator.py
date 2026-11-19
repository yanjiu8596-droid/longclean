"""Extend the frozen simulator with a read-only wrist instance mask for scoring."""
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import simulator as frozen
from common import CAMERAS, SOUP, array_hash


class ScoredSimulator(frozen.Simulator):
    def capture(self, **metrics):
        result = super().capture(**metrics)
        if self.frame not in range(260, 296, 5):
            return result
        inner = self.env.env
        sim = inner.sim
        camera = CAMERAS["camera2"].removesuffix("_image")
        raw = self.obs[CAMERAS["camera2"]]
        h, w = raw.shape[:2]
        # Register to the captured RGB, then apply the observer's documented flip.
        rendered = sim.render(camera_name=camera, width=w, height=h)
        transforms = [(sy, sx) for sy in (1, -1) for sx in (1, -1)
                      if np.array_equal(raw, rendered[::sy, ::sx])]
        if len(transforms) != 1:
            raise RuntimeError(f"Wrist RGB does not uniquely register to mask render at {self.frame}")
        sy, sx = transforms[0]
        seg = sim.render(camera_name=camera, width=w, height=h, segmentation=True)[::sy, ::sx]
        body = inner.obj_body_id[SOUP]
        ids = []
        for g, owner in enumerate(sim.model.geom_bodyid):
            while owner not in (0, body):
                owner = sim.model.body_parentid[owner]
            if owner == body:
                ids.append(g)
        assert ids and seg.shape == (h, w, 2)
        mask = (seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[..., 1], ids)
        np.save(self.output / "score-mask.npy", mask[::-1, ::-1])
        assert array_hash(self.env.get_sim_state()) == result["state_sha256"]
        result["mask_pixels"] = int(mask.sum())
        return result


if __name__ == "__main__":
    frozen.Simulator = ScoredSimulator
    frozen.main()
