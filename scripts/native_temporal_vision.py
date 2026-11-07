"""Review visible placement events with the same observation stream as native OSC."""

from __future__ import annotations

from dataclasses import dataclass

from native_visual import OBJECT_APPEARANCE, STATUSES
from tableclear.models import VerdictStatus


TEMPORAL_STRATEGIES = ("temporal_release", "temporal_hybrid")


@dataclass
class Frame:
    step: int
    images: list
    hashes: list[str]


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


class TemporalVision:
    def __init__(self):
        self.frames: list[Frame] = []
        self.commands: list[float] = []
        self.static = {}
        self.events = {}

    def observe(self, step: int, images: list, commands: list[float], hashes: list[str]) -> None:
        if step != len(self.commands) + len(commands) or self.frames and step <= self.frames[-1].step:
            raise ValueError("temporal observations must have contiguous native actions")
        self.commands.extend(commands)
        self.frames.append(Frame(step, images, hashes))

    def release_windows(self):
        for released in release_steps(self.commands):
            before = next((frame for frame in reversed(self.frames) if frame.step < released), None)
            after = next((frame for frame in self.frames if frame.step >= released + 3), None)
            if before is not None and after is not None and after.step - released <= 16:
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
        }
        self.events[key] = result
        return result, 1

    def inspect(self, planner, target_id: str, label: str, strategy: str) -> dict:
        current = self.frames[-1]
        current_vote, calls = self.static_at(planner, current, target_id, label)
        status = current_vote["status"]
        initial_delta = basket_difference(self.frames[0], current)
        evidence = []
        accepted = None
        reason = "current_nonplacement"
        if initial_delta < 60:
            for released, before, after in reversed(list(self.release_windows())):
                event, count = self.event_at(planner, label, released, before, after)
                calls += count
                delta = basket_difference(after, current)
                evidence.append({**event, "wall_difference_to_current": delta})
                if event["status"] == "in_basket" and delta < 25:
                    accepted = after.step
                    break
            if accepted is not None:
                status, reason = "in_basket", "visible_after_release"
            elif status == "in_basket" and strategy == "temporal_hybrid" and len(self.frames) > 1:
                previous = self.frames[-2]
                prior_vote, count = self.static_at(planner, previous, target_id, label)
                calls += count
                evidence.append(prior_vote)
                if prior_vote["status"] == "in_basket" and basket_difference(previous, current) < 25:
                    accepted = current.step
                    reason = "consistent_recent_views"
        if status == "in_basket" and accepted is None:
            status, reason = "unknown", "basket_moved" if initial_delta >= 60 else "placement_unconfirmed"
        return {
            "target_id": target_id, "target": label, "status": status, "inferences": calls,
            "parsed": current_vote["parsed"] and all(item["parsed"] for item in evidence),
            "raw": reason, "current": current_vote, "history": evidence,
            "initial_wall_difference": initial_delta, "confirmed_frame": accepted,
        }
