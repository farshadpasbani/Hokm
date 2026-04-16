from __future__ import annotations

from typing import Any, Dict, List, Optional

from flask import Flask, render_template, request, jsonify

from game_constants import Card, STATE_DIM, ACTION_DIM
from hokm import Hokm
from enhanced_player import EnhancedPlayer

app = Flask(__name__)

game = None
human_player = None


def create_game(human_player_name: str) -> Hokm:
    ai_players = [
        EnhancedPlayer("AI North", STATE_DIM, ACTION_DIM),
        EnhancedPlayer("AI East", STATE_DIM, ACTION_DIM),
        EnhancedPlayer("AI West", STATE_DIM, ACTION_DIM),
    ]
    human = EnhancedPlayer(
        human_player_name or "You", STATE_DIM, ACTION_DIM, is_human=True
    )
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


def run_ai_turns(g: Hokm, human: EnhancedPlayer) -> List[Dict[str, Any]]:
    """
    Advance the game with AI plays until it is the human's turn or the trick resolves
    to a human lead, or the hand/game ends.
    Returns a list of events: {"type": "trick", "winner": str, "trick": [...]}.
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

    raise RuntimeError("run_ai_turns exceeded safety limit — possible game loop bug")


def build_payload(
    g: Hokm,
    human: EnhancedPlayer,
    *,
    status: str,
    last_events: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
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
    game = create_game(human_name)
    human_player = game.players[0]

    game.start_game()
    if game.hakem == human_player:
        return jsonify(
            {
                "status": "success",
                "hakem": game.hakem.name,
                "hakem_cards": [
                    {"rank": c.rank, "suit": c.suit} for c in game.hakem_cards
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

        events = run_ai_turns(game, human_player)
        status = (
            "Hand complete"
            if all_hands_empty(game)
            else ("Game over" if game_over(game) else "Playing")
        )
        body = build_payload(game, human_player, status=status, last_events=events)
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
    )
    body["status"] = "success"
    return jsonify(body)


if __name__ == "__main__":
    app.run(debug=True)
