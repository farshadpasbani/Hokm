"""
Developer console API: training, evaluation (seat & baseline), model listing,
play configuration, and housekeeping.

All endpoints live under `/dev/api/...`. The UI in `templates/dev_console.html`
is the reference client, but everything is plain JSON so `curl` / scripts work
too. See each route's docstring for the accepted body shape.
"""

from __future__ import annotations

import json
import numbers
import os
import threading
import traceback
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, render_template, request

from config import DEFAULT_CONFIG, HokmConfig
from dev_eval import run_evaluation
from evaluate import OPPONENTS, run_matchup
from train_backend import TrainBackend

_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEV_CACHE = os.path.join(_PROJECT_DIR, "dev_cache")
os.makedirs(DEV_CACHE, exist_ok=True)
METRICS_JSON = os.path.join(DEV_CACHE, "last_training_metrics.json")
MODELS_DIR = os.path.join(_PROJECT_DIR, "models")
EVAL_JSON = os.path.join(DEV_CACHE, "last_eval.json")
BASELINE_EVAL_JSON = os.path.join(DEV_CACHE, "last_baseline_eval.json")
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
    "config": None,  # snapshot of HokmConfig used for the last start
}

MAX_LOG_LINES = 200
# Persist metrics JSON to disk every N completed games (in-memory metrics update every game).
METRICS_DISK_EVERY = 10


# =============================================================================
# JSON helpers
# =============================================================================

def _json_safe(obj: Any) -> Any:
    """Convert numpy/pandas/torch scalars and nested structures to JSON-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
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


# =============================================================================
# State helpers
# =============================================================================

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


# =============================================================================
# Path safety
# =============================================================================

def _is_under_models_dir(path: str) -> bool:
    """
    Defense-in-depth: any user-supplied model path must resolve to a file
    inside <project>/models/ before we touch it (list / delete / load).
    """
    try:
        ap = os.path.realpath(os.path.abspath(path))
        root = os.path.realpath(MODELS_DIR)
        return ap.startswith(root + os.sep) or ap == root
    except Exception:
        return False


# =============================================================================
# Blueprint
# =============================================================================

def create_dev_blueprint() -> Blueprint:
    bp = Blueprint("dev", __name__, url_prefix="/dev")

    # ---------- pages & read-only status ----------

    @bp.route("/console")
    def dev_console_page():
        return render_template("dev_console.html")

    @bp.route("/api/status", methods=["GET"])
    def api_status():
        with training_lock:
            st = dict(training_state)
        return jsonify(
            _json_safe(
                {
                    "ok": True,
                    "training": st,
                    "play_config": _load_play_config(),
                }
            )
        )

    @bp.route("/api/config/defaults", methods=["GET"])
    def config_defaults():
        """Return the full default HokmConfig as nested JSON for the UI form."""
        return jsonify({"ok": True, "config": asdict(DEFAULT_CONFIG)})

    # ---------- training ----------

    @bp.route("/api/training/start", methods=["POST"])
    def training_start():
        """
        Start a training run.

        Body:
          {
            "num_games": int,
            "save_interval": int,
            "config": {                    # optional; any subset of HokmConfig
               "seed": int | null,
               "nfsp": { "reward_mode": "outcome" | "heuristic" | "mixed",
                         "learn_every": int, "epsilon_start": float,
                         "shaping_weight": float, "win_bonus": float, ... },
               "opponents": { "self_play": 0.7, "random": 0.1, "heuristic": 0.2,
                              "frozen_pool": 0.0, "frozen_pool_dir": null,
                              "trainable_seats": [0, 2] }
            }
          }
        """
        global training_thread, stop_event
        data = request.get_json(silent=True) or {}

        try:
            num_games = int(data.get("num_games", 100))
            save_interval = int(data.get("save_interval", 50))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "num_games / save_interval must be integers"}), 400
        if num_games < 1 or num_games > 100000:
            return jsonify({"ok": False, "error": "num_games must be 1–100000"}), 400
        if save_interval < 1:
            return jsonify({"ok": False, "error": "save_interval must be >= 1"}), 400

        # Build a HokmConfig from the (optional) partial override.
        cfg_override = data.get("config") or {}
        try:
            # Start from defaults serialized, then deep-merge user fields.
            base = asdict(DEFAULT_CONFIG)
            _deep_merge(base, cfg_override)
            cfg = HokmConfig.from_dict(base)
            # Caller-supplied num_games/save_interval win (they're the simple inputs).
            cfg.num_games = num_games
            cfg.model_save_interval = save_interval
        except Exception as e:
            return jsonify({"ok": False, "error": f"Invalid config: {e}"}), 400

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
            training_state["config"] = asdict(cfg)
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
                _log(
                    f"Initializing TrainBackend({num_games} games, save every {save_interval}, "
                    f"reward={cfg.nfsp.reward_mode}, seed={cfg.seed})"
                )
                backend = TrainBackend(
                    num_games=num_games,
                    model_save_interval=save_interval,
                    config=cfg,
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
        return jsonify({"ok": True, "message": "Training started", "config": _json_safe(asdict(cfg))})

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

    # ---------- models ----------

    @bp.route("/api/models", methods=["GET"])
    def list_models():
        root = MODELS_DIR
        os.makedirs(root, exist_ok=True)
        files = []
        for name in os.listdir(root):
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
        # Sort newest first for nicer default UI.
        files.sort(key=lambda d: d["mtime"], reverse=True)
        return jsonify({"ok": True, "models": files})

    @bp.route("/api/models/delete", methods=["POST"])
    def delete_model():
        """
        Delete a .pth checkpoint. Body: {"path": "<absolute path under models/>"}.

        Path traversal is blocked: the resolved real path must live under MODELS_DIR.
        """
        data = request.get_json(silent=True) or {}
        path = data.get("path") or ""
        if not path or not _is_under_models_dir(path):
            return jsonify({"ok": False, "error": "Path must be inside the models/ directory"}), 400
        ap = os.path.abspath(path)
        if not os.path.isfile(ap):
            return jsonify({"ok": False, "error": f"Not a file: {ap}"}), 400
        if not ap.endswith(".pth"):
            return jsonify({"ok": False, "error": "Refusing to delete non-.pth file"}), 400
        try:
            os.remove(ap)
        except OSError as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        return jsonify({"ok": True, "deleted": ap})

    # ---------- evaluation ----------

    @bp.route("/api/evaluate", methods=["POST"])
    def evaluate():
        """
        Four-seat deterministic evaluation. Body:
          { "num_games": int, "checkpoint_paths": [p0, p1, p2, p3], "seed": int | null }
        ε = η = 0, learning disabled — pure greedy Q play.
        """
        data = request.get_json(silent=True) or {}
        try:
            num_games = int(data.get("num_games", 20))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "num_games must be an integer"}), 400
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

    @bp.route("/api/evaluate/baselines", methods=["POST"])
    def evaluate_baselines():
        """
        Evaluate a trained checkpoint (seated on Team 1) against one or more
        baseline families (seated on Team 2). This is the web equivalent of:

            python evaluate.py --checkpoint <path> --opponent <o> --games N --seed S

        Body:
          {
            "checkpoint": "<abs path under models/>",
            "opponents":  ["random", "heuristic", "self", "untrained"] | ["all"],
            "games": int, "seed": int
          }

        Response includes win-rate, Wilson 95% CI, tie rate, and mean trick
        differential per matchup.
        """
        data = request.get_json(silent=True) or {}
        checkpoint = data.get("checkpoint") or ""
        if not checkpoint or not os.path.isfile(os.path.abspath(checkpoint)):
            return jsonify({"ok": False, "error": f"Checkpoint not found: {checkpoint}"}), 400
        if not _is_under_models_dir(checkpoint):
            return jsonify({"ok": False, "error": "Checkpoint must live under models/"}), 400

        try:
            games = int(data.get("games", 500))
            seed = int(data.get("seed", 42))
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "games / seed must be integers"}), 400
        if games < 1 or games > 20000:
            return jsonify({"ok": False, "error": "games must be 1–20000"}), 400

        req = data.get("opponents") or ["all"]
        if not isinstance(req, list):
            return jsonify({"ok": False, "error": "opponents must be a list"}), 400
        if "all" in req:
            target_opps = list(OPPONENTS)
        else:
            target_opps = [o for o in req if o in OPPONENTS]
            if not target_opps:
                return jsonify({"ok": False, "error": f"No valid opponents in {req!r}"}), 400

        try:
            results = []
            for i, opp in enumerate(target_opps):
                r = run_matchup(
                    os.path.abspath(checkpoint), opp, games, seed=seed + i
                )
                results.append(
                    {
                        "opponent": r.opponent,
                        "games": r.games,
                        "wins": r.wins,
                        "losses": r.losses,
                        "ties": r.ties,
                        "win_rate": r.win_rate,
                        "ci95_low": r.ci95_low,
                        "ci95_high": r.ci95_high,
                        "mean_trick_diff": r.mean_trick_diff,
                        "seed": r.seed,
                    }
                )
            payload = {
                "checkpoint": os.path.abspath(checkpoint),
                "games_per_matchup": games,
                "base_seed": seed,
                "evaluated_at": datetime.now().isoformat(),
                "results": results,
            }
            with open(BASELINE_EVAL_JSON, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            return jsonify({"ok": True, "result": payload})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e), "trace": traceback.format_exc()}), 500

    @bp.route("/api/eval/baselines/latest", methods=["GET"])
    def baseline_eval_latest():
        if os.path.isfile(BASELINE_EVAL_JSON):
            with open(BASELINE_EVAL_JSON, "r", encoding="utf-8") as f:
                return jsonify({"ok": True, "data": json.load(f)})
        return jsonify({"ok": True, "data": None})

    # ---------- play configuration (human-vs-AI web app) ----------

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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> None:
    """In-place deep-merge of `override` into `base` (dict of dicts)."""
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
