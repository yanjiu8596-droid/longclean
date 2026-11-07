"""Review visible placement events with the same observation stream as native OSC."""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter

from native_visual import OBJECT_APPEARANCE, STATUSES
from native_rgbd_placement import (
    RGBD_RELEASE, RGBD_STRATEGIES, entering_basket, geometry_evidence, has_deposit, held_identity, wall_still_visible,
)
from tableclear.models import VerdictStatus


TEMPORAL_STRATEGIES = ("temporal_identity", "temporal_change")


@dataclass
class Frame:
    step: int
    images: list
    hashes: list[str]
    gripper_width: float
    geometry: dict | None = None


def release_steps(commands: list[float]) -> list[int]:
    """Find an open command sustained after a sustained close command."""
    closed = False
    high = low = 0
    releases = []
    for step, command in enumerate(commands, 1):
        high = high + 1 if command >= 0.5 else 0
        low = low + 1 if command <= -0.5 else 0
        if high >= 3:
            closed = True
        if closed and low >= 3:
            releases.append(step - 2)
            closed = False
    return releases


def basket_difference(a: Frame, b: Frame) -> float:
    import numpy as np

    def wall(frame):
        view = np.asarray(frame.images[0], dtype=np.int16)
        return view[156:193, :52]

    return float(np.abs(wall(a) - wall(b)).mean())


def lining_fraction(frame: Frame) -> float:
    import numpy as np

    wrist = np.asarray(frame.images[1], dtype=np.int16)[28:218, 25:231]
    light = wrist.min(axis=2) > 140
    neutral = np.ptp(wrist, axis=2) < 30
    return float((light & neutral).mean())


class TemporalVision:
    def __init__(self):
        self.frames: list[Frame] = []
        self.commands: list[float] = []
        self.static = {}
        self.events = {}
        self.identities = {}
        self.held_identities = {}
        self.current_holds = {}
        self.grasp_votes = {}

    def observe(self, step: int, images: list, commands: list[float], hashes: list[str], gripper_qpos, geometry=None) -> None:
        if step != len(self.commands) + len(commands) or self.frames and step <= self.frames[-1].step:
            raise ValueError("temporal observations must have contiguous native actions")
        self.commands.extend(commands)
        left, right = gripper_qpos
        self.frames.append(Frame(step, images, hashes, abs(float(left) - float(right)), geometry))

    def open_after(self, before, after, *, rgbd=False):
        full = self.frames[0].gripper_width
        if rgbd:
            increase = after.gripper_width - before.gripper_width
            return increase >= 0.018 or increase >= 0.010 and after.gripper_width >= 0.64 * full
        return after.gripper_width >= 0.8 * full

    def release_windows(self, *, rgbd=False):
        for released in release_steps(self.commands):
            before = next((frame for frame in reversed(self.frames) if frame.step < released), None)
            if before is None:
                continue
            after = next((
                frame for frame in self.frames
                if released + 3 <= frame.step <= released + 26 and self.open_after(before, frame, rgbd=rgbd)
            ), None)
            if after is not None:
                yield released, before, after

    def static_at(self, planner, frame, target_id, label):
        key = frame.step, label
        if key in self.static:
            return self.static[key], 0
        answer = planner.inspect_consensus(frame.images, target_id, label)
        result = {**answer, "frame": frame.step, "image_sha256": frame.hashes}
        self.static[key] = result
        return result, answer["inferences"]

    def event_at(self, planner, label, released, before, after):
        from PIL import Image

        key = released, label
        if key in self.events:
            return self.events[key], 0
        crop = after.images[0].crop((0, 80, 96, 202)).resize((336, 448), Image.Resampling.BICUBIC)
        views = [planner.references[label], before.images[1], after.images[1], crop]
        prompt = (
            f"Image 1 shows the reference target, the {OBJECT_APPEARANCE[label]}. Image 2 is from the wrist "
            "just before a gripper opening. Images 3 and 4 show the scene after opening, from the wrist "
            "and from the side of the basket. Decide where that exact object is AFTER opening. "
            "Answer 1 only if the target is visibly resting inside the white-lined basket, "
            "2 if the same target is visibly outside, 3 if still between the gripper fingers, "
            "or 4 if the target cannot be identified. A different object does not count. Reply with one digit."
        )
        raw, margin = planner.generate(views, prompt, score_digits=True)
        token = raw.strip(". \n\t")
        status = STATUSES.get(token, VerdictStatus.UNKNOWN)
        if status == VerdictStatus.IN_BASKET and margin < 2.0:
            status = VerdictStatus.UNKNOWN
        result = {
            "release_step": released, "before": before.step, "after": after.step, "raw": raw,
            "status": status.value, "parsed": token in STATUSES, "basket_margin": margin,
            "image_sha256": [before.hashes, after.hashes],
            "measured_gripper_width": [before.gripper_width, after.gripper_width],
        }
        self.events[key] = result
        return result, 1

    def confirmation_valid(self, frame_step: int, *, rgbd=False) -> bool:
        confirmed = next(frame for frame in self.frames if frame.step == frame_step)
        current = self.frames[-1]
        if rgbd:
            return wall_still_visible(self.frames[0], current) and wall_still_visible(confirmed, current)
        return basket_difference(self.frames[0], current) < 60 and basket_difference(confirmed, current) < 25

    def grasp_view(self, released, before):
        prior_release = max((step for step in release_steps(self.commands) if step < released), default=0)
        full_width = self.frames[0].gripper_width
        return next((
            frame for frame in reversed(self.frames)
            if prior_release < frame.step <= before.step
            and 0.018 < frame.gripper_width < 0.94 * full_width
            and frame.geometry["eef"][1] < 0.17 and frame.geometry["eef"][2] > 0.51
        ), before)

    def vote_grasp(self, planner, released, before):
        previous = max((opened.step for step, _, opened in self.release_windows(rgbd=True) if step < released), default=0)
        # Brief open commands can leave the measured grip unchanged. Do not cross a measured release.
        oldest = max(previous, before.step - 55)
        full_width = self.frames[0].gripper_width
        views = []
        for frame in reversed(self.frames):
            if not oldest < frame.step <= before.step or views and views[-1].step - frame.step < 5:
                continue
            if (0.018 < frame.gripper_width < 0.94 * full_width
                    and abs(frame.gripper_width - before.gripper_width) < 0.007
                    and self.commands[frame.step - 1] >= 0.5
                    and frame.geometry["eef"][1] < 0.195 and frame.geometry["eef"][2] > 0.50):
                views.append(frame)
            if len(views) == 3:
                break
        votes, calls = [], 0
        for frame in views:
            if frame.step not in self.grasp_votes:
                self.grasp_votes[frame.step], count = held_identity(planner, frame)
                calls += count
            votes.append({**self.grasp_votes[frame.step], "frame": frame.step, "image_sha256": frame.hashes})
        labels = Counter(row["label"] for row in votes if row["parsed"] and row["label"] is not None)
        label = next((label for label, count in labels.items() if count >= 2), None)
        return {"label": label, "parsed": all(row["parsed"] for row in votes), "votes": votes}, calls

    def pending_placement(self, planner, label, current, invalidated_through):
        # Look only at the ongoing opening run, never a frame after the decision.
        count = next((i for i, value in enumerate(reversed(self.commands)) if value > -0.5), len(self.commands))
        released = current.step - count + 1
        if not 1 <= count <= 12 or released <= max(3, invalidated_through):
            return None, 0
        if not all(value >= 0.5 for value in self.commands[released - 4:released - 1]):
            return None, 0
        before = next((frame for frame in reversed(self.frames) if frame.step < released), None)
        if before is None or self.open_after(before, current, rgbd=True) or basket_difference(self.frames[0], current) >= 22:
            return None, 0
        evidence = entering_basket(before, current, label)
        if evidence is None or not evidence["supported"]:
            return None, 0
        identity, calls = self.vote_grasp(planner, released, before)
        return {"frame": current.step, "release_step": released, "before": before.step, "identity": identity,
                "in_region": identity["parsed"] and identity["label"] == label, "geometry": evidence}, calls

    def current_hold(self, planner):
        current = self.frames[-1]
        if current.step in self.current_holds:
            return self.current_holds[current.step], 0
        last_open = max((after.step for _, _, after in self.release_windows(rgbd=True)), default=0)
        after_open = self.commands[last_open:current.step]
        if (
            len(after_open) < 3 or not 0.014 < current.gripper_width < 0.94 * self.frames[0].gripper_width
            or not any(all(value >= 0.5 for value in after_open[i:i + 3]) for i in range(len(after_open) - 2))
        ):
            self.current_holds[current.step] = None
            return None, 0
        view = next((
            frame for frame in reversed(self.frames)
            if frame.step > last_open and abs(frame.gripper_width - current.gripper_width) < 0.007
            and frame.geometry["eef"][1] < 0.17 and frame.geometry["eef"][2] > 0.51
        ), current)
        # Opening commands alone do not establish release while the measured gap is unchanged.
        identity, calls = held_identity(planner, view)
        result = {**identity, "frame": view.step, "image_sha256": view.hashes, "current_gripper_width": current.gripper_width}
        self.current_holds[current.step] = result
        return result, calls

    def depth_event(self, planner, target_id, label, released, before, after, current, use_static, *, multi_view=False):
        observations = []
        eligible = [
            frame for frame in self.frames
            if after.step <= frame.step <= released + 36 and self.open_after(before, frame, rgbd=True)
        ]
        confirmed = None
        calls = 0
        stable_frames = []
        for frame in eligible:
            evidence = geometry_evidence(before, frame, label)
            if evidence is None:
                break
            stable = wall_still_visible(frame, current)
            upright = basket_difference(self.frames[0], frame) < 22 and basket_difference(before, frame) < 22
            observations.append({
                "frame": frame.step, "basket_stable": stable, "basket_upright": upright,
                "image_sha256": frame.hashes, **evidence,
            })
            if stable and upright:
                stable_frames.append(frame)
            if stable and has_deposit(evidence, upright=upright):
                confirmed = frame.step
                break
        identity = self.held_identities.get(released)
        if (confirmed is not None or use_static and len(stable_frames) >= 2) and identity is None:
            if multi_view:
                identity, count = self.vote_grasp(planner, released, before)
            else:
                view = self.grasp_view(released, before)
                identity, count = held_identity(planner, view)
                if identity["label"] is None and view.step != before.step:
                    calls += count
                    view = before
                    identity, count = held_identity(planner, view)
                identity = {**identity, "frame": view.step, "image_sha256": view.hashes}
            calls += count
            self.held_identities[released] = identity
        if identity is None or identity["label"] != label or not identity["parsed"]:
            confirmed = None
        corroboration = []
        if confirmed is None and use_static and identity is not None and identity["label"] == label:
            for frame in stable_frames:
                verdict, count = self.static_at(planner, frame, target_id, label)
                calls += count
                corroboration.append(verdict)
                if sum(item["status"] == "in_basket" and item["parsed"] for item in corroboration) >= 2:
                    confirmed = frame.step
                    break
        return {
            "release_step": released, "before": before.step, "after": after.step, "confirmed": confirmed,
            "status": "in_basket" if confirmed is not None else "unknown",
            "parsed": identity is None or identity["parsed"], "depth_evidence": observations,
            "held_identity": identity, "corroboration": corroboration,
        }, calls

    def identify_at(self, planner, released, before, after, strategy):
        import numpy as np
        from PIL import Image, ImageDraw

        use_change = strategy == "temporal_change" and basket_difference(before, after) < 12
        key = released, use_change
        if key in self.identities:
            return self.identities[key], 0
        labels = list(OBJECT_APPEARANCE)
        examples = Image.new("RGB", (512, 148), "white")
        draw = ImageDraw.Draw(examples)
        for index, label in enumerate(labels):
            examples.paste(planner.references[label].resize((128, 128)), (index * 128, 20))
            draw.text((index * 128 + 4, 3), str(index + 1), fill="black")
        # The fixed external camera covers the gripper and basket in this crop.
        before_crop, after_crop = (
            frame.images[0].crop((0, 48, 100, 201)).resize((300, 459), Image.Resampling.BICUBIC)
            for frame in (before, after)
        )
        prompt = (
            "Image 1 shows four numbered reference packages. Images 2 and 3 show the same basket before "
            "and after the robot opened its fingers. Image 4 is the wrist camera just before opening. "
            "Which exact package is being released here? Match the label colors and shape of the item "
            "that moved from the gripper. Ignore packages that were already lying still in the basket. "
            "1 is the blue/yellow cylindrical soup can, 2 is the red/green cylindrical tomato sauce can, "
            "3 is the blue/white rectangular cream cheese box, 4 is the orange-red butter carton. "
            "Answer 5 if the released object is a different item, is hidden, or cannot be identified. "
            "Reply with one digit."
        )
        views = [examples, before_crop, after_crop, before.images[1]]
        if use_change:
            old, new = (
                np.asarray(frame.images[0], dtype=np.int16)[103:160, :88]
                for frame in (before, after)
            )
            changed = np.abs(new - old).max(axis=2) > 26
            highlighted = np.where(changed[..., None], new, new * 0.30).astype(np.uint8)
            closeup = Image.fromarray(highlighted).resize((352, 228), Image.Resampling.BICUBIC)
            views = [examples, closeup, before.images[1]]
            prompt = (
                "Image 1 shows four numbered reference packages. Image 2 highlights pixels that changed "
                "near the basket when the robot opened its fingers. In that image, ignore darkened regions "
                "where objects were already present. Image 3 shows the wrist view before opening. "
                "Which package is newly released? 1 blue/yellow cylindrical soup can, "
                "2 red/green cylindrical tomato sauce can, 3 blue/white rectangular cream cheese box, "
                "4 orange-red butter carton. Answer 5 if no exact package can be identified. Reply with one digit."
            )
        raw = planner.answer(views, prompt)
        token = raw.strip(". \n\t")
        match = {str(index + 1): label for index, label in enumerate(labels)}
        result = {"raw": raw, "label": match.get(token), "parsed": token in (*match, "5"), "highlighted_change": use_change}
        self.identities[key] = result
        return result, 1

    def inspect(self, planner, target_id: str, label: str, strategy: str, *, invalidated_through: int = -1) -> dict:
        current = self.frames[-1]
        current_vote, calls = self.static_at(planner, current, target_id, label)
        status = current_vote["status"]
        initial_delta = basket_difference(self.frames[0], current)
        rgbd = strategy in RGBD_STRATEGIES
        release_aware = strategy == RGBD_RELEASE
        basket_valid = wall_still_visible(self.frames[0], current) if rgbd else initial_delta < 60
        evidence = []
        accepted = None
        reason = "current_nonplacement"
        if basket_valid:
            for released, before, after in reversed(list(self.release_windows(rgbd=rgbd))):
                if after.step <= invalidated_through:
                    break
                if rgbd:
                    depth, count = self.depth_event(
                        planner, target_id, label, released, before, after, current,
                        use_static=strategy in ("rgbd_hybrid", RGBD_RELEASE) and status == "in_basket",
                        multi_view=release_aware,
                    )
                    calls += count
                    evidence.append(depth)
                    if depth["confirmed"] is not None:
                        accepted = depth["confirmed"]
                        break
                    if strategy == "rgbd_placement" or depth["held_identity"] is not None:
                        continue
                visible_lining = lining_fraction(after)
                if visible_lining < 0.03:
                    evidence.append({
                        "release_step": released, "before": before.step, "after": after.step,
                        "status": "unknown", "parsed": True, "raw": "basket_lining_not_visible",
                        "wrist_lining_fraction": visible_lining, "image_sha256": [before.hashes, after.hashes],
                    })
                    continue
                event, count = self.event_at(planner, label, released, before, after)
                calls += count
                delta = basket_difference(after, current)
                evidence.append({**event, "wall_difference_to_current": delta, "wrist_lining_fraction": visible_lining})
                if event["status"] == "in_basket" and delta < 25:
                    if release_aware:
                        identity, count = self.vote_grasp(planner, released, before)
                    else:
                        identity, count = self.identify_at(planner, released, before, after, strategy)
                    calls += count
                    evidence[-1]["identity"] = identity
                    if identity["parsed"] and identity["label"] == label:
                        accepted = after.step
                        break
            if accepted is not None:
                status, reason = "in_basket", "visible_after_release"
        if status == "in_basket" and accepted is None:
            status, reason = "unknown", "basket_moved" if not basket_valid else "placement_unconfirmed"
        holding = None
        if rgbd and status == "unknown" and accepted is None:
            holding, count = self.current_hold(planner)
            calls += count
            if holding is not None and holding["parsed"] and holding["label"] == label:
                status, reason = "grasped", "target_still_held"
        pending = None
        if release_aware and basket_valid and accepted is None and current_vote["status"] == "in_basket":
            pending, count = self.pending_placement(planner, label, current, invalidated_through)
            calls += count
            if pending is not None and pending["in_region"]:
                status, reason = "grasped", "inside_basket_release_unconfirmed"
        return {
            "target_id": target_id, "target": label, "status": status, "inferences": calls,
            "parsed": current_vote["parsed"] and all(
                item["parsed"] and item.get("identity", {}).get("parsed", True) for item in evidence
            ) and (holding is None or holding["parsed"]) and (pending is None or pending["identity"]["parsed"]),
            "raw": reason, "current": current_vote, "history": evidence,
            "current_hold": holding,
            "initial_wall_difference": initial_delta, "confirmed_frame": accepted,
            **({"in_basket_region": True if accepted is not None or pending and pending["in_region"] else
                False if status in ("on_table", "grasped") else None,
                "release_confirmed": accepted is not None, "pending_placement": pending} if release_aware else {}),
        }
