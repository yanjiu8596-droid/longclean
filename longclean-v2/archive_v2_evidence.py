"""Save and verify the LongClean V2 lab evidence without staging model weights."""

import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile


MANIFEST = "MANIFEST.json"
TEXT = {".bddl", ".csv", ".exit", ".json", ".jsonl", ".lock", ".log", ".md", ".ps1", ".py", ".sh", ".sha256", ".tsv"}
REPLAY_DIRS = {
    "release-diagnosis-20261118",
    "study4",
    *(f"identity-link/closed-loop/stage4-{name}" for name in (
        "basket-stall-probe", "basket-retreat-pair", "basket-retreat-holdout",
        "dense-online", "dense-online-extra", "near-table-grasp", "sauce-controls",
        "sauce-holdout-replay", "soup-execution", "soup-placement-feasibility-v2",
    )),
}
SKIP_DIRS = {"__pycache__", "cache", "models", "weights", "checkpoints", "node_modules"}


def select(root):
    for directory, names, files in os.walk(root):
        folder = Path(directory)
        names[:] = sorted(name for name in names if not name.startswith(".") and name not in SKIP_DIRS
                          and not (folder / name).is_symlink())
        parents = {parent.as_posix() for parent in folder.relative_to(root).parents}
        parents.add(folder.relative_to(root).as_posix())
        replay = not REPLAY_DIRS.isdisjoint(parents)
        for name in sorted(files):
            path = folder / name
            if name.startswith(".") or path.is_symlink() or not path.is_file():
                continue
            suffix = path.suffix.lower()
            source_bundle = name.endswith(".tar.gz") and ("source" in name or "stage3d" in name)
            image = suffix in {".jpg", ".jpeg", ".png"} and "snapshots" not in path.parts
            if suffix in TEXT or source_bundle or image or (replay and suffix in {".npz", ".npy"}):
                yield path


class DigestReader:
    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()

    def read(self, size):
        data = self.source.read(size)
        self.digest.update(data)
        return data


def pack(root, paths, archive):
    partial = archive.with_suffix(archive.suffix + ".partial")
    if archive.exists() or partial.exists():
        raise FileExistsError(archive)
    entries = []
    with tarfile.open(partial, "x:gz", compresslevel=1) as bundle:
        for path in paths:
            info = bundle.gettarinfo(path, path.relative_to(root).as_posix())
            with path.open("rb") as source:
                before = os.fstat(source.fileno())
                reader = DigestReader(source)
                bundle.addfile(info, reader)
                after = os.fstat(source.fileno())
            if before.st_size != info.size or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError(f"File changed during archival: {path}")
            entries.append({"path": info.name, "bytes": info.size, "sha256": reader.digest.hexdigest()})
        manifest = (json.dumps({"files": entries}, ensure_ascii=False, indent=2) + "\n").encode()
        info = tarfile.TarInfo(MANIFEST)
        info.size = len(manifest)
        bundle.addfile(info, io.BytesIO(manifest))
    partial.replace(archive)


def verify(archive):
    entries = []
    manifest = None
    with tarfile.open(archive, "r|gz") as bundle:
        for info in bundle:
            if not info.isfile():
                raise ValueError(f"Non-file member: {info.name}")
            with bundle.extractfile(info) as source:
                if info.name == MANIFEST:
                    if manifest is not None:
                        raise ValueError("Duplicate manifest")
                    manifest = json.load(source)["files"]
                    continue
                reader = DigestReader(source)
                while reader.read(1024 * 1024):
                    pass
                entries.append({"path": info.name, "bytes": info.size, "sha256": reader.digest.hexdigest()})
    if manifest != entries or len({entry["path"] for entry in entries}) != len(entries):
        raise ValueError("Mismatch between members and manifest")
    print(f"VERIFIED {len(entries)} files, {sum(entry['bytes'] for entry in entries)} bytes")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_subparsers(dest="command", required=True)
    mode.add_parser("inventory").add_argument("root", type=Path)
    gather = mode.add_parser("pack")
    gather.add_argument("root", type=Path)
    gather.add_argument("archive", type=Path)
    mode.add_parser("verify").add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.command == "verify":
        verify(args.archive)
        return
    root = args.root.resolve(strict=True)
    paths = list(select(root))
    if not paths:
        raise ValueError(f"No evidence found under {root}")
    if args.command == "pack":
        pack(root, paths, args.archive)
        return
    size = Counter()
    for path in paths:
        size[path.suffix.lower()] += path.stat().st_size
    print(json.dumps({"files": len(paths), "bytes": sum(size.values()), "by_extension": size}, indent=2))


if __name__ == "__main__":
    main()
