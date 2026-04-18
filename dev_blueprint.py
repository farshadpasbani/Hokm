"""
Developer console API: training, evaluation, model listing, play configuration.
"""

from __future__ import annotations

import json
import numbers
import os
import threading
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, render_template, request

from dev_eval import run_evaluation
from train_backend import TrainBackend

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEV_CACHE = os.path.join(_PROJECT_DIR, "dev_cache")
os.makedirs(DEV_CACHE, exist_ok=True)
METRICS_JSON = os.path.join(DEV_CACHE, "last_training_metrics.json")
MODELS_DIR = os.path.join(_PROJECT_DIR, "models")
EVAL_JSON = os.path.join(DEV_CACHE, "last_eval.json")
PLAY_CONFIG_JSON = os.path.join(DEV_CACHE, "play_checkpoints.json")

training_lock = threading.Lock()
training_thread: Optional[threading.Thread] = None
stop_event: Optional[threading.Event] = None

training_state: Dict[str, Any] = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "current_game": 0,
    "total_games": 0,
    "message": "",
    "error": None,
    "metrics": None,
    "logs": [],
}

MAX_LOG_LINES = 200
# Persist metrics JSON to disk every N completed games (in-memory metrics update every game).
METRICS_DISK_EVERY = 10


def _json_safe(obj: Any) -> Any:
    """Convert numpy/pandas/torch scalars and nested structures to JSON-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    # NumPy integer/floating scalars are numbers.Number but not built-in int/float; handle
    # before importing numpy so serialization still works if numpy import fails in a worker.
    if isinstance(obj, numbers.Number):
        if isinstance(obj, numbers.Integral):
            return int(obj)
        if isinstance(obj, numbers.Real):
            return float(obj)
        if isinstance(obj, numbers.Complex):
            c = complex(obj)
            return {"real": c.real, "imag": c.imag}
        return float(obj)
    if isinstance(obj, str):
        return obj
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return _json_safe(obj.tolist())
    except ImportError:
        pass
    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            return _json_safe(obj.to_dict(orient="list"))
        if isinstance(obj, pd.Series):
            return _json_safe(obj.tolist())
    except ImportError:
        pass
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            if obj.ndim == 0:
                return _json_safe(obj.detach().cpu().item())
            return _json_safe(obj.detach().cpu().tolist())
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            return _json_safe(obj.item())
        except Exception:
            pass
    return obj


def _log(line: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {line}"
    training_state["logs"].append(entry)
    training_state["logs"] = training_state["logs"][-MAX_LOG_LINES:]


def _save_metrics_file(metrics: Dict[str, Any]) -> None:
    payload = {
        "saved_at": datetime.now().isoformat(),
        "metrics": _json_safe(metrics),
    }
    with open(METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _load_play_config() -> Dict[str, Any]:
    if not os.path.isfile(PLAY_CONFIG_JSON):
        return {"ai_policy_paths": [None, None, None]}
    with open(PLAY_CONFIG_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_play_config(cfg: Dict[str, Any]) -> None:
    with open(PLAY_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def create_dev_blueprint() -> Blueprint:
    bp = Blueprint("dev", __name__, url_prefix="/dev")

    @bp.route("/console")
    def dev_console_page():
        return render_template("dev_console.html")

    @bp.route("/api/status", methods=["GET"])
    def api_status():
        with training_lock:
            st = dict(training_state)
        return jsonify(
            _json_safe(
                {"ok": True, "training": st, "play_config": _load_play_config()}
            )
        )

    @bp.route("/api/training/start", methods=["POST"])
    def training_start():
        global training_thread, stop_event
        data = request.get_json(silent=True) or {}
        num_games = int(data.get("num_games", 100))
        save_interval = int(data.get("save_interval", 50))
        if num_games < 1 or num_games > 100000:
            return jsonify({"ok": False, "error": "num_games must be 1–100000"}), 400
        if save_interval < 1:
            return jsonify({"ok": False, "error": "save_interval must be >= 1"}), 400

        with training_lock:
            if training_state["running"]:
                return jsonify({"ok": False, "error": "Training already running"}), 409
            training_state["running"] = True
            training_state["error"] = None
            training_state["started_at"] = datetime.now().isoformat()
            training_state["finished_at"] = None
            training_state["current_game"] = 0
            training_state["total_games"] = num_games
            training_state["message"] = "Starting…"
            training_state["logs"] = []
            training_state["metrics"] = None
        stop_event = threading.Event()

        def on_progress(done: int, metrics: Dict[str, Any]) -> None:
            with training_lock:
                training_state["current_game"] = done
                training_state["metrics"] = metrics
                training_state["message"] = f"Completed game {done}/{num_games}"
            if done % METRICS_DISK_EVERY == 0 or done == num_games:
                _save_metrics_file(metrics)
            if done % 100 == 0 or done == num_games:
                _log(f"Game {done}/{num_games} — metrics snapshot saved")

        def log_train(msg: str) -> None:
            with training_lock:
                _log(msg)

        def run() -> None:
            global training_thread, stop_event
            try:
                _log(f"Initializing TrainBackend({num_games} games, save every {save_interval})")
                backend = TrainBackend(
                    num_games=num_games, model_save_interval=save_interval
                )
                backend.train(
                    stop_event=stop_event,
                    on_progress=on_progress,
                    log_fn=log_train,
                )
                with training_lock:
                    training_state["message"] = "Training finished"
                    training_state["finished_at"] = datetime.now().isoformat()
                _log("TrainBackend.train() returned normally")
            except Exception as e:
                with training_lock:
                    training_state["error"] = str(e)
                    training_state["message"] = "Training failed"
                    training_state["finished_at"] = datetime.now().isoformat()
                _log(f"ERROR: {e}\n{traceback.format_exc()}")
            finally:
                with training_lock:
                    training_state["running"] = False
                training_thread = None

        training_thread = threading.Thread(target=run, daemon=True)
        training_thread.start()
        return jsonify({"ok": True, "message": "Training started"})

    @bp.route("/api/training/stop", methods=["POST"])
    def training_stop():
        global stop_event
        with training_lock:
            if not training_state["running"]:
                return jsonify({"ok": False, "error": "Not running"}), 400
            if stop_event:
                stop_event.set()
            training_state["message"] = "Stop requested…"
        return jsonify({"ok": True, "message": "Stop signal sent"})

    @bp.route("/api/metrics/latest", methods=["GET"])
    def metrics_latest():
        if os.path.isfile(METRICS_JSON):
            with open(METRICS_JSON, "r", encoding="utf-8") as f:
                return jsonify({"ok": True, "data": json.load(f)})
        return jsonify({"ok": True, "data": None})

    @bp.route("/api/models", methods=["GET"])
    def list_models():
        root = MODELS_DIR
        os.makedirs(root, exist_ok=True)
        files = []
        for name in sorted(os.listdir(root)):
            if not name.endswith(".pth"):
                continue
            path = os.path.join(root, name)
            try:
                st = os.stat(path)
                files.append(
                    {
                        "name": name,
                        "path": path,
                        "size_bytes": st.st_size,
                        "mtime": st.st_mtime,
                    }
                )
            except OSError:
                continue
        return jsonify({"ok": True, "models": files})

    @bp.route("/api/evaluate", methods=["POST"])
    def evaluate():
        data = request.get_json(silent=True) or {}
        num_games = int(data.get("num_games", 20))
        paths = data.get("checkpoint_paths")
        if paths is None:
            paths = [None, None, None, None]
        if not isinstance(paths, list) or len(paths) != 4:
            return jsonify(
                {"ok": False, "error": "checkpoint_paths must be a list of 4 strings or nulls"}
            ), 400
        norm: List[Optional[str]] = []
        for p in paths:
            if p is None or p == "":
                norm.append(None)
            else:
                ap = os.path.abspath(p)
                if not os.path.isfile(ap):
                    return jsonify({"ok": False, "error": f"Missing file: {ap}"}), 400
                norm.append(ap)
        try:
            # Deterministic greedy evaluation: ε = η = 0, learning disabled.
            # Seed is accepted from the request body if provided, otherwise
            # a fixed default is used so repeated clicks give comparable runs.
            seed_val = data.get("seed", 42)
            try:
                seed_val = int(seed_val) if seed_val is not None else None
            except (TypeError, ValueError):
                seed_val = 42
            result = run_evaluation(
                num_games, norm, epsilon=0.0, eta=0.0, seed=seed_val
            )
            result["evaluated_at"] = datetime.now().isoformat()
            with open(EVAL_JSON, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            return jsonify({"ok": True, "result": result})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500

    @bp.route("/api/eval/latest", methods=["GET"])
    def eval_latest():
        if os.path.isfile(EVAL_JSON):
            with open(EVAL_JSON, "r", encoding="utf-8") as f:
                return jsonify({"ok": True, "data": json.load(f)})
        return jsonify({"ok": True, "data": None})

    @bp.route("/api/play/config", methods=["GET", "POST"])
    def play_config():
        if request.method == "GET":
            return jsonify({"ok": True, "config": _load_play_config()})
        data = request.get_json(silent=True) or {}
        paths = data.get("ai_policy_paths")
        if paths is None:
            return jsonify({"ok": False, "error": "ai_policy_paths required"}), 400
        if not isinstance(paths, list) or len(paths) != 3:
            return jsonify(
                {"ok": False, "error": "ai_policy_paths must be a list of 3 paths (North, East, West)"}
            ), 400
        cleaned = []
        for p in paths:
            if p is None or p == "":
                cleaned.append(None)
            else:
                ap = os.path.abspath(p)
                if not os.path.isfile(ap):
                    return jsonify({"ok": False, "error": f"Missing file: {ap}"}), 400
                cleaned.append(ap)
        _save_play_config({"ai_policy_paths": cleaned})
        return jsonify({"ok": True, "config": _load_play_config()})

    return bp
