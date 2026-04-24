from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from flask import Flask, render_template, request, jsonify

from game_constants import Card, STATE_DIM, ACTION_DIM, ranks, suits
from dev_blueprint import create_dev_blueprint
from hokm import Hokm
from enhanced_player import EnhancedPlayer

app = Flask(__name__)
app.register_blueprint(create_dev_blueprint())

game = None
human_player = None

_DEV_PLAY_PATH = os.path.join("dev_cache", "play_checkpoints.json")


def _load_saved_ai_checkpoints() -> Optional[List[Optional[str]]]:
    if not os.path.isfile(_DEV_PLAY_PATH):
        return None
    try:
        with open(_DEV_PLAY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        paths = data.get("ai_policy_paths")
        if isinstance(paths, list) and len(paths) == 3:
            return [p or None for p in paths]
    except (json.JSONDecodeError, OSError):
        pass
    return None


def create_game(
    human_player_name: str,
    ai_policy_paths: Optional[List[Optional[str]]] = None,
) -> Hokm:
    """
    Build a 4-player Hokm game for human play. AI seats run purely greedy:
    ε = η = 0 (no training-time exploration) and learning disabled (no weight
    updates at inference). This matches the evaluation-time protocol and
    prevents the AI from making random-looking moves against a human.
    """
    labels = ["AI North", "AI East", "AI West"]
    ai_players = []
    for i, label in enumerate(labels):
        p = EnhancedPlayer(label, STATE_DIM, ACTION_DIM, epsilon=0.0, eta=0.0)
        p.learning_enabled = False
        if ai_policy_paths and i < len(ai_policy_paths):
            path = ai_policy_paths[i]
            if path and os.path.isfile(path):
                p.load_policy_state(path)
        ai_players.append(p)
    human = EnhancedPlayer(
        human_player_name or "You", STATE_DIM, ACTION_DIM, is_human=True
    )
    human.learning_enabled = False
    return Hokm([human] + ai_players)


def scores_for_json(g: Hokm) -> dict:
    return {"Team 1": g.scores[1], "Team 2": g.scores[2]}


def seats_for_json(g: Hokm) -> dict:
    """Clockwise from South: South=human (0), West=3, North=2, East=1."""
    p = g.players
    return {
        "south": p[0].name,
        "east": p[1].name,
        "north": p[2].name,
        "west": p[3].name,
    }


def game_over(g: Hokm) -> bool:
    return g.scores[1] >= 7 or g.scores[2] >= 7


def all_hands_empty(g: Hokm) -> bool:
    return all(len(p.hand) == 0 for p in g.players)


def sort_human_hand(human: EnhancedPlayer) -> None:
    """Sort human's hand by suit then rank (standard display order)."""
    if not human.hand:
        return
    human.hand.sort(key=lambda c: (suits.index(c.suit), ranks.index(c.rank)))


def run_ai_turns(g: Hokm, human: EnhancedPlayer) -> List[Dict[str, Any]]:
    """
    Advance the game with AI plays until it is the human's turn or the trick resolves
    to a human lead, or the hand/game ends.
    Returns events including:
      {"type": "play", "player": str, "card": str} for each AI card played
      {"type": "trick", "winner": str, "trick": [...]} when a trick completes
    """
    events = []
    safety = 0
    while safety < 200:
        safety += 1
        if game_over(g) or all_hands_empty(g):
            return events

        if len(g.current_trick) == 4:
            winner, snapshot = g.resolve_trick_if_complete()
            if winner is not None:
                events.append(
                    {
                        "type": "trick",
                        "winner": winner.name,
                        "trick": snapshot,
                    }
                )
            continue

        nxt = g.get_next_to_play()
        if nxt == human:
            return events

        card, _ = nxt.play_card(g.lead_suit)
        err = g.apply_play(nxt, card)
        if err:
            raise RuntimeError(f"AI play failed ({nxt.name}): {err}")
        events.append(
            {"type": "play", "player": nxt.name, "card": str(card)}
        )

    raise RuntimeError("run_ai_turns exceeded safety limit — possible game loop bug")


def build_payload(
    g: Hokm,
    human: EnhancedPlayer,
    *,
    status: str,
    last_events: Optional[List[Dict[str, Any]]] = None,
    trick_before_ai: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    sort_human_hand(human)
    trick_display = [
        {"player": p.name, "card": str(c)} for p, c in g.current_trick
    ]
    payload = {
        "status": status,
        "scores": scores_for_json(g),
        "seats": seats_for_json(g),
        "human_player_hand": [str(c) for c in human.hand],
        "current_trick": trick_display,
        "trump_suit": g.trump_suit,
        "hakem": g.hakem.name if g.hakem else None,
        "next_player": g.get_next_to_play().name,
        "your_turn": g.get_next_to_play() == human,
        "game_over": game_over(g) or all_hands_empty(g),
        "last_events": last_events or [],
    }
    if trick_before_ai is not None:
        payload["trick_before_ai"] = trick_before_ai
    if payload["game_over"]:
        t1, t2 = g.scores[1], g.scores[2]
        if t1 > t2:
            payload["result"] = "Team 1 wins"
        elif t2 > t1:
            payload["result"] = "Team 2 wins"
        else:
            payload["result"] = "Draw"
    return payload


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_game", methods=["POST"])
def start_game():
    global game, human_player
    data = request.get_json(silent=True) or {}
    human_name = data.get("player_name", "You")
    ai_paths = data.get("ai_policy_paths")
    if ai_paths is None:
        ai_paths = _load_saved_ai_checkpoints()
    elif isinstance(ai_paths, list) and len(ai_paths) == 3:
        ai_paths = [p or None for p in ai_paths]
    else:
        ai_paths = None
    game = create_game(human_name, ai_policy_paths=ai_paths)
    human_player = game.players[0]

    game.start_game()
    if game.hakem == human_player:
        sort_human_hand(human_player)
        return jsonify(
            {
                "status": "success",
                "hakem": game.hakem.name,
                "hakem_cards": [
                    {"rank": c.rank, "suit": c.suit} for c in human_player.hand
                ],
                "scores": scores_for_json(game),
                "seats": seats_for_json(game),
                "message": "Choose trump from one of your dealt suits.",
            }
        )

    game.choose_trump_suit()
    events = run_ai_turns(game, human_player)
    return jsonify(
        {
            "status": "success",
            "message": f"{game.hakem.name} is Hakem; trump is {game.trump_suit}.",
            **build_payload(
                game,
                human_player,
                status="Playing",
                last_events=events,
                trick_before_ai=[],
            ),
        }
    )


@app.route("/set_trump_suit", methods=["POST"])
def set_trump_suit():
    global game, human_player
    if not game:
        return jsonify({"status": "error", "message": "Game not started"})

    data = request.get_json(silent=True) or {}
    trump_suit = data.get("trump_suit")
    if not trump_suit:
        return jsonify({"status": "error", "message": "No trump suit provided"})

    try:
        game.set_trump_suit(trump_suit)
        events = run_ai_turns(game, human_player)
        return jsonify(
            {
                "status": "success",
                "message": f"Trump suit set to {trump_suit}",
                **build_payload(
                    game,
                    human_player,
                    status="Playing",
                    last_events=events,
                    trick_before_ai=[],
                ),
            }
        )
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/play_card", methods=["POST"])
def play_card():
    global game, human_player
    if not game:
        return jsonify({"status": "error", "message": "Game not started"})

    data = request.get_json(silent=True) or {}
    card_str = data.get("card")
    if not card_str:
        return jsonify({"status": "error", "message": "No card provided"})

    try:
        card = Card.from_string(card_str)
        if game.get_next_to_play() != human_player:
            return jsonify({"status": "error", "message": "Not your turn"})
        err = game.apply_play(human_player, card)
        if err:
            return jsonify({"status": "error", "message": err})

        trick_before_ai = [
            {"player": p.name, "card": str(c)} for p, c in game.current_trick
        ]
        events = run_ai_turns(game, human_player)
        status = (
            "Hand complete"
            if all_hands_empty(game)
            else ("Game over" if game_over(game) else "Playing")
        )
        body = build_payload(
            game,
            human_player,
            status=status,
            last_events=events,
            trick_before_ai=trick_before_ai,
        )
        body["status"] = "success"
        if game_over(game) or all_hands_empty(game):
            game = None
            human_player = None
        return jsonify(body)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/game_state", methods=["GET"])
def get_game_state():
    global game, human_player
    if not game:
        return jsonify({"status": "error", "message": "Game not started"})
    events = run_ai_turns(game, human_player)
    body = build_payload(
        game,
        human_player,
        status="Playing",
        last_events=events,
        trick_before_ai=[],
    )
    body["status"] = "success"
    return jsonify(body)


if __name__ == "__main__":
    app.run(debug=True)
