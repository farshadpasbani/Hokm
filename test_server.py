from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Simple in-memory game state
game = None


@app.route("/api/new-game", methods=["POST"])
def new_game():
    global game
    data = request.json
    chat_id = data.get("chatId")

    if not chat_id:
        return jsonify({"error": "chatId is required"}), 400

    # Create a simple mock game
    game = {
        "chatId": chat_id,
        "humanHand": [
            {"suit": "Hearts", "rank": "A"},
            {"suit": "Hearts", "rank": "K"},
            {"suit": "Diamonds", "rank": "Q"},
            {"suit": "Clubs", "rank": "J"},
        ],
        "currentTrick": [],
        "trumpSuit": "Hearts",
        "team1Score": 0,
        "team2Score": 0,
        "currentPlayer": "Player 1",
        "gameStatus": "playing",
        "message": "Game started",
    }

    return jsonify({"message": "Game created successfully", "status": "success"})


@app.route("/api/game-state", methods=["GET"])
def get_game_state():
    chat_id = request.args.get("chatId")

    if not chat_id:
        return jsonify({"error": "chatId is required"}), 400

    if not game:
        return jsonify({"error": "No active game"}), 404

    if str(chat_id) != str(game["chatId"]):
        return jsonify({"error": "Game not found for this chat ID"}), 404

    return jsonify(game)


@app.route("/api/play-card", methods=["POST"])
def play_card():
    if not game:
        return jsonify({"error": "No active game"}), 404

    data = request.json
    chat_id = data.get("chatId")
    card_str = data.get("card")

    if not chat_id or not card_str:
        return jsonify({"error": "chatId and card are required"}), 400

    # Simulate playing a card
    return jsonify({"message": f"Card {card_str} played successfully"})


@app.route("/api/select-trump", methods=["POST"])
def select_trump():
    global game
    if not game:
        return jsonify({"error": "No active game"}), 404

    data = request.json
    chat_id = data.get("chatId")
    suit = data.get("suit")

    if not chat_id or not suit:
        return jsonify({"error": "chatId and suit are required"}), 400

    # Update the trump suit
    game["trumpSuit"] = suit

    return jsonify({"message": f"Trump suit set to {suit}"})


@app.route("/api/end-game", methods=["POST"])
def end_game():
    global game
    if not game:
        return jsonify({"error": "No active game"}), 404

    data = request.json
    chat_id = data.get("chatId")

    if not chat_id:
        return jsonify({"error": "chatId is required"}), 400

    # End the game
    game = None

    return jsonify({"message": "Game ended successfully"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
