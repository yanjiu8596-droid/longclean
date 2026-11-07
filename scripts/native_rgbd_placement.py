"""Verify a visually identified package using camera depth after measured release."""

from __future__ import annotations

from native_visual import OBJECT_APPEARANCE


RGBD_STRATEGIES = ("rgbd_placement", "rgbd_hybrid")


def world_points(depth, intrinsic, extrinsic):
    import numpy as np

    rows, cols = np.indices(depth.shape)
    # Replay uses LeRobot's double flip, while the camera matrix expects a top-left image.
    x = (depth.shape[1] - 1 - cols - intrinsic[0, 2]) * depth / intrinsic[0, 0]
    y = (rows - intrinsic[1, 2]) * depth / intrinsic[1, 1]
    rays = np.stack((x, y, depth), axis=-1)
    return rays @ extrinsic[:3, :3].T + extrinsic[:3, 3]


def cloud(frame, camera):
    data = frame.geometry
    if "clouds" not in data:
        data["clouds"] = [None] * 2
    if data["clouds"][camera] is None:
        data["clouds"][camera] = world_points(
            data["depths"][camera], data["intrinsics"][camera], data["extrinsics"][camera],
        )
    return data["clouds"][camera]


def held_crops(before):
    import numpy as np
    from PIL import Image

    held = []
    center = before.geometry["eef"]
    for camera, source in enumerate(before.images):
        offset = cloud(before, camera) - center
        mask = (np.linalg.norm(offset[..., :2], axis=-1) < 0.065) & (offset[..., 2] > -0.12) & (offset[..., 2] < 0.015)
        rows, cols = np.nonzero(mask)
        if len(rows) < 12:
            continue
        pixels = np.asarray(source)
        highlight = np.where(mask[..., None], pixels, pixels // 5).astype(np.uint8)
        crop = Image.fromarray(highlight).crop((
            max(int(cols.min()) - 8, 0), max(int(rows.min()) - 8, 0),
            min(int(cols.max()) + 9, source.width), min(int(rows.max()) + 9, source.height),
        ))
        crop.thumbnail((384, 384))
        if max(crop.size) < 192:
            scale = 192 / max(crop.size)
            crop = crop.resize(tuple(max(1, round(side * scale)) for side in crop.size), Image.Resampling.BICUBIC)
        held.append((camera, crop))
    return held


def held_identity(planner, before):
    from PIL import Image, ImageDraw

    verifier = getattr(planner, "held_logo", None)
    local = verifier.inspect(before) if verifier is not None else None
    if local is not None and local["label"] is not None:
        return {"raw": "wrist logo matched at grasp depth", "label": local["label"], "parsed": True, "local_identity": local}, 0
    held = [crop for _, crop in held_crops(before)]
    if not held:
        return {"raw": "no held-object pixels", "label": None, "parsed": True}, 0
    examples = Image.new("RGB", (512, 148), "white")
    draw = ImageDraw.Draw(examples)
    labels = list(OBJECT_APPEARANCE)
    for index, label in enumerate(labels):
        examples.paste(planner.references[label].resize((128, 128)), (128 * index, 20))
        draw.text((128 * index + 4, 3), str(index + 1), fill="black")
    prompt = (
        "The first image shows four numbered reference packages. The following images highlight the 3D area "
        "immediately below the fingers before they open. Identify the object actually held by the fingers. "
        "Ignore darkened background and any package already resting farther below the fingers. "
        "1 is the blue and yellow cylindrical soup can. 2 is the red and green cylindrical tomato sauce can. "
        "3 is the flat blue and white cream cheese box. 4 is the thin orange-red butter carton. "
        "Answer 5 for an empty gripper, another item, or an item whose identity cannot be seen. Reply with one digit."
    )
    raw = planner.answer([examples, *held], prompt)
    token = raw.strip(". \n\t")
    identity = {str(index + 1): label for index, label in enumerate(labels)}
    return {
        "raw": raw, "label": identity.get(token), "parsed": token in (*identity, "5"),
        **({"local_identity": local} if local is not None else {}),
    }, 1


def color_mask(image, label):
    import numpy as np

    red, green, blue = np.moveaxis(np.asarray(image, dtype=np.int16), -1, 0)
    blue_ink = (blue > red + 18) & (blue > green + 6) & (blue > 52)
    if label == "cream cheese box":
        return blue_ink
    if label == "alphabet soup can":
        yellow_ink = (red > 70) & (green > 58) & (green > blue + 35) & (red > blue + 44)
        return blue_ink | yellow_ink
    if label == "tomato sauce":
        return (green > red + 10) & (green > blue + 10)
    return (red > green + 22) & (red > blue + 34)


def basket_pixels(frame, camera, release_xy):
    import numpy as np

    x, y, z = np.moveaxis(cloud(frame, camera), -1, 0)
    # Interior bounds measured from the original, upright basket in development RGB-D frames.
    return (
        (x > -0.055) & (x < 0.065) & (y > 0.195) & (y < 0.335) & (z > 0.468) & (z < 0.57)
        & (np.hypot(x - release_xy[0], y - release_xy[1]) < 0.075)
    )


def geometry_evidence(before, after, label):
    import numpy as np

    center = before.geometry["eef"]
    if not (-0.07 < center[0] < 0.085 and 0.18 < center[1] < 0.35 and 0.53 < center[2] < 0.80):
        return None
    change = before.geometry["depths"][0] - after.geometry["depths"][0]
    front = basket_pixels(after, 0, center[:2])
    foreground = int(np.count_nonzero(front & (change > 0.009)))
    colors = []
    for frame in (before, after):
        colors.append([
            int(np.count_nonzero(basket_pixels(frame, camera, center[:2]) & color_mask(image, label)))
            for camera, image in enumerate(frame.images)
        ])
    return {"fixed_foreground_pixels": foreground, "target_color_pixels": colors}


def has_deposit(evidence, *, upright):
    before, after = evidence["target_color_pixels"]
    return upright and evidence["fixed_foreground_pixels"] >= 45 or any(
        new - old >= 45 for old, new in zip(before, after, strict=True)
    )


def wall_still_visible(reference, current):
    import numpy as np

    wall = np.s_[156:193, :52]
    difference = current.geometry["depths"][0][wall] - reference.geometry["depths"][0][wall]
    # A nearer foreground can occlude the wall, but visible depths must still match.
    return float((np.abs(difference) <= 0.010).mean()) >= 0.35 and float((difference > 0.015).mean()) < 0.35
