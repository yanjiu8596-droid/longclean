from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SceneConfig:
    id: str
    target_count: int
    source_task: str
    objects: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("scene id must not be empty")
        if self.target_count != len(self.objects):
            raise ValueError(
                f"{self.id}: target_count={self.target_count}, "
                f"but {len(self.objects)} object labels were provided"
            )
        if self.target_count < 1:
            raise ValueError(f"{self.id}: target_count must be positive")


@dataclass(frozen=True, slots=True)
class MethodConfig:
    id: str
    max_actions: int = 10
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("method id must not be empty")
        if self.max_actions < 1:
            raise ValueError(f"{self.id}: max_actions must be positive")
        if self.max_attempts < 1:
            raise ValueError(f"{self.id}: max_attempts must be positive")


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    project_name: str
    scenes: tuple[SceneConfig, ...]
    methods: tuple[MethodConfig, ...]
    seeds: tuple[int, ...]
    episodes_per_seed: int

    def __post_init__(self) -> None:
        if not self.scenes:
            raise ValueError("at least one scene is required")
        if not self.methods:
            raise ValueError("at least one method is required")
        if not self.seeds:
            raise ValueError("at least one seed is required")
        if any(seed < 0 for seed in self.seeds):
            raise ValueError("seeds must be non-negative")
        if self.episodes_per_seed < 1:
            raise ValueError("episodes_per_seed must be positive")
        _ensure_unique("scene", [scene.id for scene in self.scenes])
        _ensure_unique("method", [method.id for method in self.methods])
        _ensure_unique("seed", [str(seed) for seed in self.seeds])

    @property
    def episode_count(self) -> int:
        return (
            len(self.scenes)
            * len(self.methods)
            * len(self.seeds)
            * self.episodes_per_seed
        )


@dataclass(frozen=True, slots=True)
class EpisodeJob:
    run_id: str
    scene_id: str
    method_id: str
    seed: int
    episode: int
    init_state_index: int


def _ensure_unique(label: str, values: list[str]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label} id in configuration")


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    config_path = Path(path)
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    project = raw.get("project", {})
    evaluation = raw.get("evaluation", {})

    scenes = tuple(
        SceneConfig(
            id=item["id"],
            target_count=int(item["target_count"]),
            source_task=item["source_task"],
            objects=tuple(item["objects"]),
        )
        for item in raw.get("scenes", [])
    )
    methods = tuple(
        MethodConfig(
            id=item["id"],
            max_actions=int(item.get("max_actions", 10)),
            max_attempts=int(item.get("max_attempts", 3)),
        )
        for item in raw.get("methods", [])
    )

    return EvaluationConfig(
        project_name=project.get("name", "tableclear"),
        scenes=scenes,
        methods=methods,
        seeds=tuple(int(seed) for seed in evaluation.get("seeds", [])),
        episodes_per_seed=int(evaluation.get("episodes_per_seed", 0)),
    )


def expand_matrix(config: EvaluationConfig) -> tuple[EpisodeJob, ...]:
    jobs: list[EpisodeJob] = []
    for scene in config.scenes:
        for method in config.methods:
            for seed in config.seeds:
                for episode in range(config.episodes_per_seed):
                    init_state_index = seed * config.episodes_per_seed + episode
                    run_id = (
                        f"{scene.id}__{method.id}__s{seed:02d}__e{episode:02d}"
                    )
                    jobs.append(
                        EpisodeJob(
                            run_id=run_id,
                            scene_id=scene.id,
                            method_id=method.id,
                            seed=seed,
                            episode=episode,
                            init_state_index=init_state_index,
                        )
                    )
    return tuple(jobs)

