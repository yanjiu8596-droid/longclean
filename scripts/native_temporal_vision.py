"""Review visible placement events with the same observation stream as native OSC."""

from __future__ import annotations

from dataclasses import dataclass

from native_visual import OBJECT_APPEARANCE, STATUSES
from tableclear.models import VerdictStatus


TEMPORAL_STRATEGIES = ("temporal_identity", "temporal_change")


@dataclass
class Frame:
    step: int
    images: list
    hashes: list[str]
    gripper_width: float


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

    def observe(self, step: int, images: list, commands: list[float], hashes: list[str], gripper_qpos) -> None:
        if step != len(self.commands) + len(commands) or self.frames and step <= self.frames[-1].step:
            raise ValueError("temporal observations must have contiguous native actions")
        self.commands.extend(commands)
        left, right = gripper_qpos
        self.frames.append(Frame(step, images, hashes, abs(float(left) - float(right))))

    def release_windows(self):
        open_width = 0.8 * self.frames[0].gripper_width
        for released in release_steps(self.commands):
            before = next((frame for frame in reversed(self.frames) if frame.step < released), None)
            after = next((
                frame for frame in self.frames
                if released + 3 <= frame.step <= released + 26 and frame.gripper_width >= open_width
            ), None)
            if before is not None and after is not None:
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

    def confirmation_valid(self, frame_step: int) -> bool:
        confirmed = next(frame for frame in self.frames if frame.step == frame_step)
        current = self.frames[-1]
        return basket_difference(self.frames[0], current) < 60 and basket_difference(confirmed, current) < 25

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
        evidence = []
        accepted = None
        reason = "current_nonplacement"
        if initial_delta < 60:
            for released, before, after in reversed(list(self.release_windows())):
                if after.step <= invalidated_through:
                    break
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
                    identity, count = self.identify_at(planner, released, before, after, strategy)
                    calls += count
                    evidence[-1]["identity"] = identity
                    if identity["parsed"] and identity["label"] == label:
                        accepted = after.step
                        break
            if accepted is not None:
                status, reason = "in_basket", "visible_after_release"
        if status == "in_basket" and accepted is None:
            status, reason = "unknown", "basket_moved" if initial_delta >= 60 else "placement_unconfirmed"
        return {
            "target_id": target_id, "target": label, "status": status, "inferences": calls,
            "parsed": current_vote["parsed"] and all(
                item["parsed"] and item.get("identity", {}).get("parsed", True) for item in evidence
            ),
            "raw": reason, "current": current_vote, "history": evidence,
            "initial_wall_difference": initial_delta, "confirmed_frame": accepted,
        }
