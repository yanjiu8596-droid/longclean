"""Verify a known wrist-camera label by local features and grasp depth."""

from __future__ import annotations

from hashlib import sha256

from native_rgbd_placement import cloud


CREAM_LABEL = "cream cheese box"
LOGO_BOUNDS = (96, 130, 161, 213)
WRIST_BOUNDS = (54, 100, 203, 228)


def load_logo(directory, templates):
    from PIL import Image

    images = []
    for row in templates:
        name = row["file"]
        path = directory / name
        if path.name != name or sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"logo template failed verification: {path}")
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    return PlanarLogo(images)


class PlanarLogo:
    def __init__(self, images):
        import cv2

        self.cv = cv2
        self.sift = cv2.SIFT_create(contrastThreshold=0.012, edgeThreshold=15)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.references = [self.features(image) for image in images]
        if len(self.references) < 2 or any(descriptors is None for _, descriptors in self.references):
            raise ValueError("at least two usable logo reference images are required")

    def features(self, image):
        import numpy as np

        gray = self.cv.cvtColor(np.asarray(image.convert("RGB")), self.cv.COLOR_RGB2GRAY)
        enlarged = self.cv.resize(gray, None, fx=3, fy=3, interpolation=self.cv.INTER_CUBIC)
        return self.sift.detectAndCompute(enlarged, None)

    def inspect(self, frame):
        import numpy as np

        x0, y0, _, _ = WRIST_BOUNDS
        keypoints, descriptors = self.features(frame.images[1].crop(WRIST_BOUNDS))
        depth = cloud(frame, 1)
        records = []
        for reference, vectors in self.references:
            pairs = [] if descriptors is None or len(descriptors) < 2 else self.matcher.knnMatch(vectors, descriptors, k=2)
            good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.78 * pair[1].distance]
            inliers = support = 0
            if len(good) >= 6:
                left = np.float32([reference[pair.queryIdx].pt for pair in good])
                right = np.float32([keypoints[pair.trainIdx].pt for pair in good])
                _, accepted = self.cv.findHomography(left, right, self.cv.RANSAC, 4.0)
                if accepted is not None:
                    points = np.rint(right[accepted.ravel() > 0] / 3 + [x0, y0]).astype(int)
                    points = np.unique(points, axis=0)
                    inliers = len(points)
                    pixels = depth[points[:, 1], points[:, 0]] - frame.geometry["eef"]
                    attached = (np.linalg.norm(pixels[:, :2], axis=1) < 0.055) & (pixels[:, 2] > -0.12) & (pixels[:, 2] < 0.015)
                    valid = points[attached]
                    if len(valid) and np.ptp(valid, axis=0).min() >= 5:
                        support = len(valid)
            records.append({"matches": len(good), "inliers": inliers, "grasp_depth_inliers": support})
        agreeing = sum(
            row["grasp_depth_inliers"] >= 6 and row["grasp_depth_inliers"] >= 0.7 * row["inliers"]
            for row in records
        )
        return {"label": CREAM_LABEL if agreeing >= 2 else None, "agreeing_references": agreeing, "evidence": records}
