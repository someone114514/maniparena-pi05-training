"""ManipArena pi0.5 / LeRobot serving adapter.

Start the server with:
    python serve.py --checkpoint outputs/pi05_tabletop15/checkpoints/last/pretrained_model --port 8000
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict

import numpy as np

from maniparena.policy import ModelPolicy

logger = logging.getLogger(__name__)

CAMERA_MAP = {
    "camera_front": "observation.images.faceImg",
    "camera_left": "observation.images.leftImg",
    "camera_right": "observation.images.rightImg",
}


def _decode_image(value: Any) -> np.ndarray:
    import cv2

    if isinstance(value, np.ndarray):
        return value.astype(np.uint8, copy=False)
    raw = base64.b64decode(value) if isinstance(value, str) else bytes(value)
    bgr = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("cv2.imdecode failed")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _as_7d(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 7:
        raise ValueError(f"Expected at least 7D arm state, got {arr.size}D")
    return arr[:7]


def _to_numpy(value: Any) -> np.ndarray:
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().float().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(value, dtype=np.float32)


def _extract_actions(value: Any) -> Any:
    if isinstance(value, dict):
        for key in ("action", "actions", "pred_action", "pred_actions"):
            if key in value:
                return value[key]
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _import_pi05_policy():
    try:
        from lerobot.policies.pi05 import PI05Policy
        return PI05Policy
    except ImportError:
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        return PI05Policy


def _load_preprocessors(policy: Any, checkpoint_path: str, device: str):
    try:
        from lerobot.policies.factory import make_pre_post_processors
    except ImportError:
        logger.warning("LeRobot make_pre_post_processors not found; using raw policy inputs.")
        return (lambda frame: frame), (lambda out: out)

    try:
        return make_pre_post_processors(
            policy.config,
            checkpoint_path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
    except TypeError:
        return make_pre_post_processors(policy.config, checkpoint_path)


def _load_pi05_policy(checkpoint_path: str, device: str) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for the LeRobot pi0.5 adapter.") from exc

    PI05Policy = _import_pi05_policy()
    torch_device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
    policy = PI05Policy.from_pretrained(checkpoint_path).to(torch_device).eval()
    preprocess, postprocess = _load_preprocessors(policy, checkpoint_path, str(torch_device))
    logger.info("Loaded LeRobot pi0.5 policy from %s on %s", checkpoint_path, torch_device)
    return {"policy": policy, "preprocess": preprocess, "postprocess": postprocess, "device": torch_device}


def _normalize_actions(actions: Any, action_horizon: int, previous: np.ndarray | None) -> np.ndarray:
    actions = _extract_actions(actions)
    arr = _to_numpy(actions).astype(np.float32, copy=False)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] < 14:
        raise ValueError(f"Expected action shape (T, >=14), got {arr.shape}")
    arr = arr[:, :14]

    if arr.shape[0] < action_horizon:
        pad = np.repeat(arr[-1:], action_horizon - arr.shape[0], axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    elif arr.shape[0] > action_horizon:
        arr = arr[:action_horizon]

    if previous is not None:
        seed = previous.astype(np.float32, copy=False)
        clipped = arr.copy()
        limits = np.array([0.035, 0.035, 0.035, 0.20, 0.20, 0.20, 1.0] * 2, dtype=np.float32)
        last = seed
        for idx in range(clipped.shape[0]):
            delta = np.clip(clipped[idx] - last, -limits, limits)
            clipped[idx] = last + delta
            last = clipped[idx]
        arr = clipped

    arr[:, 6] = np.clip(arr[:, 6], 0.0, 1.0)
    arr[:, 13] = np.clip(arr[:, 13], 0.0, 1.0)
    return arr


class MyPolicy(ModelPolicy):
    def load_model(self, checkpoint_path: str, device: str) -> Any:
        backend = os.getenv("MANIPARENA_PI05_POLICY", "lerobot").lower()
        if backend != "lerobot":
            raise ValueError(f"Unsupported MANIPARENA_PI05_POLICY={backend!r}")
        return _load_pi05_policy(checkpoint_path, device)

    def convert_input(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        if self.control_mode != "end_pose":
            raise ValueError("pi0.5 adapter expects --control-mode end_pose")

        state_dict = obs.get("state", {})
        left = _as_7d(state_dict.get("follow1_pos", np.zeros(7, dtype=np.float32)))
        right = _as_7d(state_dict.get("follow2_pos", np.zeros(7, dtype=np.float32)))
        state = np.concatenate([left, right]).astype(np.float32)
        self._last_state = state

        instruction = obs.get("instruction", "") or ""
        frame: Dict[str, Any] = {
            "observation.state": state,
            "task": instruction,
            "prompt": instruction,
        }
        views = obs.get("views", {})
        for client_key, lerobot_key in CAMERA_MAP.items():
            raw = views.get(client_key)
            frame[lerobot_key] = _decode_image(raw) if raw is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        return frame

    def run_inference(self, model_input: Dict[str, Any]) -> Any:
        import torch

        policy = self.model["policy"]
        batch = self.model["preprocess"](model_input)
        with torch.inference_mode():
            if hasattr(policy, "predict_action_chunk"):
                output = policy.predict_action_chunk(batch)
            else:
                output = policy.select_action(batch)
        try:
            return self.model["postprocess"](output)
        except Exception:
            logger.debug("Postprocess failed; returning raw policy output", exc_info=True)
            return output

    def convert_output(self, model_output: Any) -> Dict[str, Any]:
        actions = _normalize_actions(
            model_output,
            action_horizon=self.action_horizon,
            previous=getattr(self, "_last_state", None),
        )
        return {
            "follow1_pos": actions[:, :7].tolist(),
            "follow2_pos": actions[:, 7:14].tolist(),
        }
