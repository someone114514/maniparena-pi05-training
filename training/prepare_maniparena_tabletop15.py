#!/usr/bin/env python3
"""Convert ManipArena tabletop tasks into a single LeRobot dataset for pi0.5.

The source dataset is organized as one LeRobot dataset per task. This script
creates a single mixed tabletop dataset with the 14D EE state/action interface
used by the ManipArena submission server.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import av
import numpy as np
import pandas as pd

TABLETOP_GROUPS = ("execution_reasoning", "semantic_reasoning")
CAMERA_KEYS = (
    "observation.images.faceImg",
    "observation.images.leftImg",
    "observation.images.rightImg",
)
STATE_NAMES = (
    "left_x",
    "left_y",
    "left_z",
    "left_roll",
    "left_pitch",
    "left_yaw",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_roll",
    "right_pitch",
    "right_yaw",
    "right_gripper",
)


def _load_lerobot_dataset_class():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise SystemExit(
            "Cannot import LeRobotDataset. Activate the conda env created by "
            "training/setup_pi05_env.sh first."
        ) from exc
    return LeRobotDataset


def _read_task_text(task_dir: Path) -> str:
    path = task_dir / "meta" / "tasks.jsonl"
    if not path.exists():
        return task_dir.name
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                return str(item.get("task", task_dir.name))
    return task_dir.name


def _iter_task_dirs(source: Path) -> Iterable[Path]:
    real_root = source / "real"
    for group in TABLETOP_GROUPS:
        group_dir = real_root / group
        if not group_dir.exists():
            raise FileNotFoundError(f"Missing ManipArena group directory: {group_dir}")
        for task_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            if (task_dir / "data").exists():
                yield task_dir


def _iter_episode_files(task_dir: Path) -> Iterable[Path]:
    data_root = task_dir / "data"
    for chunk_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and p.name.startswith("chunk-")):
        yield from sorted(chunk_dir.glob("episode_*.parquet"))


def _video_path(task_dir: Path, parquet_path: Path, camera_key: str) -> Path:
    chunk = parquet_path.parent.name
    episode = parquet_path.stem
    path = task_dir / "videos" / chunk / camera_key / f"{episode}.mp4"
    if path.exists():
        return path
    short = camera_key.rsplit(".", 1)[-1]
    short_path = task_dir / "videos" / chunk / short / f"{episode}.mp4"
    if short_path.exists():
        return short_path
    raise FileNotFoundError(f"Missing video for {camera_key}: {path}")


def _read_video_rgb(path: Path) -> list[np.ndarray]:
    container = av.open(str(path))
    try:
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    finally:
        container.close()


def _make_features(image_shape: tuple[int, int, int]) -> dict:
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (14,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (14,),
            "names": STATE_NAMES,
        },
        **{
            key: {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            }
            for key in CAMERA_KEYS
        },
    }




def _add_frame(dataset, frame: dict, task_text: str) -> None:
    try:
        dataset.add_frame(frame, task=task_text)
    except TypeError:
        dataset.add_frame(frame)


def _save_episode(dataset, task_text: str) -> None:
    try:
        dataset.save_episode(task=task_text)
    except TypeError:
        dataset.save_episode()


def _create_dataset(repo_id: str, root: Path, fps: int, features: dict):
    LeRobotDataset = _load_lerobot_dataset_class()
    kwargs = {
        "repo_id": repo_id,
        "root": root,
        "fps": fps,
        "features": features,
        "robot_type": "maniparena-bimanual-ee",
        "use_videos": True,
        "image_writer_threads": 8,
        "image_writer_processes": 0,
    }
    try:
        return LeRobotDataset.create(**kwargs)
    except TypeError:
        kwargs.pop("image_writer_processes", None)
        return LeRobotDataset.create(**kwargs)


def convert(args: argparse.Namespace) -> None:
    source = Path(args.source).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)

    task_dirs = list(_iter_task_dirs(source))
    if len(task_dirs) != 15:
        raise RuntimeError(f"Expected 15 tabletop tasks, found {len(task_dirs)} under {source / 'real'}")

    first_episode = next(_iter_episode_files(task_dirs[0]))
    first_frame = _read_video_rgb(_video_path(task_dirs[0], first_episode, CAMERA_KEYS[0]))[0]
    dataset = _create_dataset(
        repo_id=args.repo_id,
        root=root,
        fps=args.fps,
        features=_make_features(tuple(first_frame.shape)),
    )

    total_episodes = 0
    total_frames = 0
    for task_dir in task_dirs:
        task_text = _read_task_text(task_dir)
        episode_files = list(_iter_episode_files(task_dir))
        print(f"[TASK] {task_dir.parent.name}/{task_dir.name}: {len(episode_files)} episodes, task={task_text!r}")
        for parquet_path in episode_files:
            df = pd.read_parquet(parquet_path)
            if "observation.state" not in df.columns or "action" not in df.columns:
                raise KeyError(f"{parquet_path} missing observation.state/action")

            videos = {key: _read_video_rgb(_video_path(task_dir, parquet_path, key)) for key in CAMERA_KEYS}
            n = min(len(df), *(len(frames) for frames in videos.values()))
            if n == 0:
                raise RuntimeError(f"No aligned frames for {parquet_path}")

            for idx in range(n):
                state = np.asarray(df["observation.state"].iloc[idx], dtype=np.float32).reshape(-1)
                action = np.asarray(df["action"].iloc[idx], dtype=np.float32).reshape(-1)
                if state.shape[0] < 14 or action.shape[0] < 14:
                    raise ValueError(f"{parquet_path} has state/action dims {state.shape}/{action.shape}")
                frame = {
                    "observation.state": state[:14],
                    "action": action[:14],
                    **{key: videos[key][idx] for key in CAMERA_KEYS},
                }
                _add_frame(dataset, frame, task_text)

            _save_episode(dataset, task_text)
            total_episodes += 1
            total_frames += n
            print(f"  [EP] {parquet_path.parent.name}/{parquet_path.name}: {n} frames")

    if hasattr(dataset, "finalize"):
        dataset.finalize()
    elif hasattr(dataset, "consolidate"):
        dataset.consolidate()
    print(f"[DONE] repo_id={args.repo_id} root={root} episodes={total_episodes} frames={total_frames}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="ManipArena dataset root from HF download.")
    parser.add_argument("--repo-id", default="maniparena/tabletop15_pi05")
    parser.add_argument("--root", default="data/lerobot/maniparena_tabletop15_pi05")
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    convert(parse_args())
