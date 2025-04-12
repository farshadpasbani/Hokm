from flask import Flask, render_template, request, jsonify
from hokm_game import Hokm, DQNPlayer, Card
import torch
import json

app = Flask(__name__)

# Initialize game state
game = None
human_player = None
ai_players = []


def create_game(human_player_name="Player 1"):
    global game, human_player, ai_players

    # Create players
    human_player = DQNPlayer(human_player_name, 52, 13)
    ai_players = [
        DQNPlayer("AI Player 1", 52, 13),
        DQNPlayer("AI Player 2", 52, 13),
        DQNPlayer("AI Player 3", 52, 13),
    ]

    # Load trained models for AI players
    for i, player in enumerate(ai_players, 1):
        try:
            player.policy_net.load_state_dict(torch.load(f"Player {i}_policy_net.pth"))
            player.policy_net.eval()
        except:
            print(f"Could not load model for AI Player {i}")

    # Create game with all players
    all_players = [human_player] + ai_players
    game = Hokm(all_players)
    game.deal_cards()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_game", methods=["POST"])
def start_game():
    data = request.json
    human_player_name = data.get("player_name", "Player 1")
    create_game(human_player_name)
    return jsonify(
        {
            "status": "success",
            "hand": [str(card) for card in human_player.hand],
            "trump_suit": game.trump_suit,
        }
    )


@app.route("/play_card", methods=["POST"])
def play_card():
    data = request.json
    card_index = data.get("card_index")

    if not game or not human_player.hand:
        return jsonify({"error": "Game not started or no cards in hand"})

    try:
        # Human player plays card
        card = human_player.hand[card_index]
        human_player.hand.pop(card_index)

        # Store the current trick
        game.current_trick = [(human_player, card)]

        # AI players play their cards
        ai_moves = []
        for ai_player in ai_players:
            ai_card, _ = ai_player.play_card(game.trump_suit)
            game.current_trick.append((ai_player, ai_card))
            ai_moves.append({"player": ai_player.name, "card": str(ai_card)})

        # Determine trick winner
        winner = game.determine_trick_winner(game.current_trick, game.trump_suit)

        # Update game state
        game.tricks_won[winner] += 1
        game.last_trick_winner = winner

        # Check if game is over
        team1_tricks = sum(game.tricks_won[player] for player in game.team1)
        team2_tricks = sum(game.tricks_won[player] for player in game.team2)
        game_over = team1_tricks >= 7 or team2_tricks >= 7

        # If game is not over and all cards are played, start a new round
        if not game_over and len(human_player.hand) == 0:
            # Reset all players' hands
            for player in [human_player] + ai_players:
                player.hand = []

            # Rotate hakem and deal new cards
            game.rotate_hakem()
            game.deal_cards()

            # Clear current trick
            game.current_trick = []

            # Update human player's hand
            human_hand = [str(card) for card in human_player.hand]
        else:
            human_hand = [str(card) for card in human_player.hand]

        return jsonify(
            {
                "status": "success",
                "human_card": str(card),
                "ai_moves": ai_moves,
                "winner": winner.name,
                "remaining_cards": human_hand,
                "game_over": game_over,
                "winning_team": (
                    "Team 1"
                    if team1_tricks >= 7
                    else "Team 2" if team2_tricks >= 7 else None
                ),
                "team1_score": team1_tricks,
                "team2_score": team2_tricks,
                "new_round": len(human_player.hand)
                == 13,  # Indicates if a new round has started
                "current_trick": [
                    {"player": p.name, "card": str(c)} for p, c in game.current_trick
                ],
            }
        )
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/game_state", methods=["GET"])
def get_game_state():
    if not game:
        return jsonify({"error": "Game not started"})

    team1_tricks = sum(game.tricks_won[player] for player in game.team1)
    team2_tricks = sum(game.tricks_won[player] for player in game.team2)

    # Get current trick information
    current_trick = []
    if hasattr(game, "current_trick") and game.current_trick:
        for player, card in game.current_trick:
            current_trick.append({"player": player.name, "card": str(card)})

    return jsonify(
        {
            "trump_suit": game.trump_suit,
            "human_hand": [str(card) for card in human_player.hand],
            "tricks_won": {
                player.name: game.tricks_won[player]
                for player in [human_player] + ai_players
            },
            "current_hakem": game.current_hakem.name,
            "team1_score": team1_tricks,
            "team2_score": team2_tricks,
            "game_over": team1_tricks >= 7 or team2_tricks >= 7,
            "last_winner": (
                game.last_trick_winner.name if game.last_trick_winner else None
            ),
            "current_trick": current_trick,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
