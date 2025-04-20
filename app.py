from flask import Flask, render_template, request, jsonify
from game_constants import Card
from hokm import Hokm, Deck
from enhanced_player import EnhancedPlayer
import torch
import threading
import time

app = Flask(__name__)

# Global game state
game = None
human_player = None
ai_players = None
training_thread = None
training_active = False


def create_game(human_player_name):
    global game, human_player, ai_players

    # Create enhanced AI players
    state_dim = 52 + 52 + (4 * 52) + 2 + 4  # hand + played + trick + scores + trump
    action_dim = 52  # Maximum possible actions

    ai_players = [
        EnhancedPlayer("AI Player 1", state_dim, action_dim),
        EnhancedPlayer("AI Player 2", state_dim, action_dim),
        EnhancedPlayer("AI Player 3", state_dim, action_dim),
    ]

    # Create human player
    human_player = EnhancedPlayer(human_player_name, state_dim, action_dim)

    # Create game with all players
    game = Hokm([human_player] + ai_players)

    return game


def train_ai_players():
    global training_active
    training_active = True

    while training_active:
        # Create a new game with only AI players
        state_dim = 52 + 52 + (4 * 52) + 2 + 4
        action_dim = 52

        training_players = [
            EnhancedPlayer("Training AI 1", state_dim, action_dim),
            EnhancedPlayer("Training AI 2", state_dim, action_dim),
            EnhancedPlayer("Training AI 3", state_dim, action_dim),
            EnhancedPlayer("Training AI 4", state_dim, action_dim),
        ]

        training_game = Hokm(training_players)
        training_game.play_game()

        # Save models periodically
        if training_game.game_count % 100 == 0:
            for i, player in enumerate(training_players):
                torch.save(
                    player.policy_net.state_dict(), f"training_ai_{i+1}_policy_net.pth"
                )

        time.sleep(0.1)  # Prevent CPU overload


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/start_game", methods=["POST"])
def start_game():
    global game
    if not game:
        data = request.get_json()
        human_player_name = data.get("player_name", "Player 1")
        game = create_game(human_player_name)

    # Start the game and get Hakem's cards
    hakem_cards = game.start_game()

    return jsonify(
        {
            "status": "success",
            "hakem": game.hakem.name,
            "hakem_cards": [
                {"rank": card.rank, "suit": card.suit} for card in hakem_cards
            ],
        }
    )


@app.route("/set_trump_suit", methods=["POST"])
def set_trump_suit():
    global game
    if not game:
        return jsonify({"status": "error", "message": "Game not started"})

    data = request.get_json()
    trump_suit = data.get("trump_suit")

    if not trump_suit:
        return jsonify({"status": "error", "message": "No trump suit provided"})

    try:
        game.set_trump_suit(trump_suit)
        return jsonify(
            {
                "status": "success",
                "message": f"Trump suit set to {trump_suit}",
                "current_player": game.players[1].name,  # Player after Hakem starts
                "human_player_hand": [str(card) for card in game.players[0].hand],
                "scores": game.scores,
                "trump_suit": trump_suit,
            }
        )
    except ValueError as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/play_card", methods=["POST"])
def play_card():
    global game
    if not game:
        return jsonify({"status": "error", "message": "Game not started"})

    data = request.get_json()
    card_str = data.get("card")

    if not card_str:
        return jsonify({"status": "error", "message": "No card provided"})

    try:
        # Parse the card string
        card = Card.from_string(card_str)

        # Get the human player
        human_player = game.players[0]

        # Validate game state
        if not human_player.hand:
            return jsonify({"status": "error", "message": "No cards in hand"})

        # Check if it's the human player's turn
        if len(game.current_trick) > 0:
            return jsonify({"status": "error", "message": "Not your turn"})

        # Check if the card is in the player's hand
        if card not in human_player.hand:
            return jsonify({"status": "error", "message": "Card not in hand"})

        # Check if the card follows suit
        if game.lead_suit and card.suit != game.lead_suit:
            # Check if player has any cards of the lead suit
            has_lead_suit = any(c.suit == game.lead_suit for c in human_player.hand)
            if has_lead_suit:
                return jsonify({"status": "error", "message": "Must follow lead suit"})

        # Play the card
        game.current_trick = [(human_player, card)]
        human_player.hand.remove(card)

        # Set the lead suit for this trick
        game.lead_suit = card.suit

        # AI players play their cards with delay
        current_player_index = 1  # Start with the first AI player
        while len(game.current_trick) < 4:
            ai_player = game.players[current_player_index]
            try:
                ai_card, _ = ai_player.play_card(game.lead_suit)
                if ai_card not in ai_player.hand:
                    raise ValueError(
                        f"AI player {ai_player.name} attempted to play card not in hand"
                    )
                game.current_trick.append((ai_player, ai_card))
            except Exception as e:
                print(f"Error with AI player {ai_player.name}: {str(e)}")
                return jsonify(
                    {"status": "error", "message": f"AI player error: {str(e)}"}
                )
            current_player_index = (current_player_index + 1) % 4

        # Determine the winner of the trick
        winner = game.determine_trick_winner()

        # Update scores
        team = 1 if winner in [game.players[0], game.players[2]] else 2
        game.scores[team] += 1

        # Clear the current trick and lead suit for the next round
        game.current_trick = []
        game.lead_suit = None

        # Check if game is over
        game_over = any(score >= 7 for score in game.scores.values())

        # Get the current game state
        game_state = {
            "status": "success",
            "message": f"{winner.name} won the trick!"
            + (" Game Over!" if game_over else ""),
            "current_player": winner.name,  # Winner leads the next trick
            "human_player_hand": [str(card) for card in human_player.hand],
            "current_trick": [],  # Clear for next trick
            "scores": {"Team 1": game.scores[1], "Team 2": game.scores[2]},
            "trump_suit": game.trump_suit,
            "winner": winner.name,
            "game_over": game_over,
        }

        # If game is over, reset the game
        if game_over:
            game = None

        return jsonify(game_state)

    except Exception as e:
        print(f"Error playing card: {str(e)}")
        return jsonify({"status": "error", "message": f"Error playing card: {str(e)}"})


@app.route("/game_state", methods=["GET"])
def get_game_state():
    global game
    if not game:
        return jsonify({"status": "error", "message": "Game not started"})

    try:
        game_state = {
            "current_player": game.players[0].name,  # Human player
            "human_player_hand": [str(card) for card in game.players[0].hand],
            "current_trick": [(p.name, str(c)) for p, c in game.current_trick],
            "scores": game.scores,
            "trump_suit": game.trump_suit,
            "status": "Your turn to play",
            "game_over": False,
        }
        return jsonify(game_state)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
