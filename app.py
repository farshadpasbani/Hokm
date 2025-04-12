from flask import Flask, render_template, request, jsonify
from game_constants import Card
from hokm_game import Hokm, Deck
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

    # Deal initial cards
    game.deal_cards()


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
    data = request.json
    human_player_name = data.get("player_name", "Player 1")
    create_game(human_player_name)

    # Start training thread if not already running
    global training_thread
    if not training_thread or not training_thread.is_alive():
        training_thread = threading.Thread(target=train_ai_players)
        training_thread.daemon = True
        training_thread.start()

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

        # Store the current trick and set the lead suit
        game.current_trick = [(human_player, card)]
        lead_suit = card.suit

        # AI players play their cards
        ai_moves = []
        for ai_player in ai_players:
            # Update AI player's state
            ai_player.current_trick = game.current_trick
            ai_player.trump_suit = game.trump_suit

            ai_card, _ = ai_player.play_card(lead_suit)
            game.current_trick.append((ai_player, ai_card))
            ai_moves.append({"player": ai_player.name, "card": str(ai_card)})

        # Determine trick winner
        winner = game.determine_trick_winner(game.current_trick, lead_suit)

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
                player.played_cards_memory.clear()

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
                "new_round": len(human_player.hand) == 13,
                "current_trick": [
                    {"player": p.name, "card": str(c)} for p, c in game.current_trick
                ],
                "difficulty_level": game.difficulty_level,
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
            "difficulty_level": game.difficulty_level,
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
