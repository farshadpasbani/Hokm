import os
import logging
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
)
import torch
from hokm import Hokm
from enhanced_player import EnhancedPlayer
from game_constants import suits, ranks, Card

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "https://hokm-mini-app.vercel.app"}})

# Load Telegram bot token from environment variable
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    logger.error("BOT_TOKEN not set in environment variables")
    raise ValueError("BOT_TOKEN is required")


class HokmBot:
    def __init__(self):
        self.games = {}  # Store active games by chat_id
        self.model_path = os.getenv("MODEL_PATH", "models/Player_{}_game_1000.pth")
        self.app = app  # Store Flask app for routing
        self.telegram_app = Application.builder().token(BOT_TOKEN).build()

        # Register routes
        self.app.add_url_rule(
            "/api/game-state", view_func=self.get_game_state, methods=["GET"]
        )
        self.app.add_url_rule(
            "/api/select-trump", view_func=self.select_trump, methods=["POST"]
        )
        self.app.add_url_rule(
            "/api/play-card", view_func=self.play_card, methods=["POST"]
        )
        self.app.add_url_rule(
            "/api/new-game", view_func=self.api_new_game, methods=["POST"]
        )
        self.app.add_url_rule(
            "/api/end-game", view_func=self.api_end_game, methods=["POST"]
        )

        # Register Telegram handlers
        self.telegram_app.add_handler(CommandHandler("start", self.start))
        self.telegram_app.add_handler(CommandHandler("newgame", self.new_game))
        self.telegram_app.add_handler(CommandHandler("endgame", self.end_game))
        self.telegram_app.add_handler(CallbackQueryHandler(self.button))
        self.telegram_app.add_error_handler(self.error_handler)

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command with inline button to launch web app."""
        web_app_url = "https://hokm-mini-app.vercel.app"
        keyboard = [
            [InlineKeyboardButton("Play Hokm", web_app=WebAppInfo(url=web_app_url))]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            "Welcome to the Hokm Bot! 🎮\n"
            "Click below to play Hokm or use /newgame ai to start a text-based game against AI opponents.\n"
            "You'll play as Player 1, with AI as Players 2, 3, and 4.",
            reply_markup=reply_markup,
        )
        logger.info(f"Start command received for chat_id: {update.effective_chat.id}")

    async def new_game(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /newgame ai command."""
        chat_id = update.message.chat_id
        if chat_id in self.games:
            await update.message.reply_text(
                "A game is already in progress! Use /endgame to stop it."
            )
            return

        # Initialize players
        human_player = EnhancedPlayer("Player 1", is_human=True)
        ai_players = [EnhancedPlayer(f"Player {i+2}") for i in range(3)]
        players = [human_player] + ai_players

        # Load AI models
        for player in ai_players:
            model_path = self.model_path.format(player.name.split()[-1])
            try:
                if os.path.exists(model_path):
                    player.model.load_state_dict(
                        torch.load(model_path, map_location=torch.device("cpu"))
                    )
                    player.target_model.load_state_dict(
                        torch.load(model_path, map_location=torch.device("cpu"))
                    )
                    logger.info(f"Loaded model for {player.name} from {model_path}")
                else:
                    logger.warning(f"Model for {player.name} not found at {model_path}")
            except Exception as e:
                logger.error(f"Error loading model for {player.name}: {e}")
                await update.message.reply_text(
                    f"Failed to load model for {player.name}. Starting with untrained AI."
                )

        # Initialize game
        game = Hokm(players)
        self.games[chat_id] = {
            "game": game,
            "human_player": human_player,
            "current_state": "init",
        }

        # Start game and deal hakem cards
        try:
            hakem_cards = game.start_game()
            if game.hakem == human_player:
                self.games[chat_id]["current_state"] = "choose_trump"
                keyboard = [
                    [InlineKeyboardButton(suit, callback_data=f"trump_{suit}")]
                    for suit in suits
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text(
                    f"You are the Hakem! Your cards: {[str(c) for c in hakem_cards]}\n"
                    "Choose the trump suit:",
                    reply_markup=reply_markup,
                )
            else:
                game.choose_trump_suit()
                self.games[chat_id]["current_state"] = "playing"
                await self.start_round(chat_id, update, context)
        except Exception as e:
            logger.error(f"Error starting game for chat_id {chat_id}: {e}")
            await update.message.reply_text("Failed to start game.")

    async def button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button presses (e.g., trump suit selection)."""
        query = update.callback_query
        chat_id = query.message.chat_id
        data = query.data

        if chat_id not in self.games:
            await query.message.reply_text(
                "No active game. Start one with /newgame ai."
            )
            return

        game_data = self.games[chat_id]
        game = game_data["game"]

        if data.startswith("trump_") and game_data["current_state"] == "choose_trump":
            trump_suit = data.split("_")[1]
            if trump_suit not in suits:
                await query.message.reply_text("Invalid trump suit selected.")
                return
            game.set_trump_suit(trump_suit)
            game_data["current_state"] = "playing"
            await query.message.reply_text(f"Trump suit set to {trump_suit}.")
            await self.start_round(chat_id, update, context)
            await query.answer()

        elif data.startswith("card_") and game_data["current_state"] == "playing":
            card_str = data.split("_", 1)[1]
            rank, suit = card_str.rsplit(" of ", 1)
            if rank not in ranks or suit not in suits:
                await query.message.reply_text("Invalid card selected.")
                return
            card = Card(suit, rank)
            human_player = game_data["human_player"]
            if card not in human_player.hand:
                await query.message.reply_text(
                    f"You don't have {card_str} in your hand."
                )
                return
            try:
                human_player.play_card(game.lead_suit, selected_card=card)
                await query.message.reply_text(f"You played {card_str}.")
                await self.continue_round(chat_id, update, context)
            except Exception as e:
                await query.message.reply_text(f"Error playing card: {e}")
            await query.answer()

    async def start_round(
        self, chat_id: int, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Start a new round and handle AI/human turns."""
        if chat_id not in self.games:
            await update.message.reply_text(
                "No active game. Start one with /newgame ai."
            )
            return

        game_data = self.games[chat_id]
        game = game_data["game"]
        human_player = game_data["human_player"]

        try:
            winner = game.play_round()
            game_data["game"] = game

            team1_score = game.scores[1]
            team2_score = game.scores[2]
            trick_winner = winner.name
            human_hand = [str(c) for c in human_player.hand]
            trick_cards = ", ".join(
                [f"{p.name}: {str(c)}" for p, c in game.current_trick]
            )
            message = (
                f"Trick won by {trick_winner}\n"
                f"Trick cards: {trick_cards}\n"
                f"Team 1 Score: {team1_score}\n"
                f"Team 2 Score: {team2_score}\n"
                f"Your hand: {human_hand}"
            )

            if team1_score >= 7 or team2_score >= 7:
                game_winner = "Team 1" if team1_score >= 7 else "Team 2"
                game_data["current_state"] = "ended"
                await update.message.reply_text(
                    f"Game ended! Winner: {game_winner}\n{message}"
                )
                game.save_game_log(save_full_log=True)
                del self.games[chat_id]
                return

            if not any(len(player.hand) > 0 for player in game.players):
                game_data["current_state"] = "ended"
                await update.message.reply_text(
                    f"Game ended with no clear winner (all cards played).\n{message}"
                )
                game.save_game_log(save_full_log=True)
                del self.games[chat_id]
                return

            next_player = game.players[(game.players.index(winner) + 1) % 4]
            if next_player == human_player:
                keyboard = [
                    [InlineKeyboardButton(str(c), callback_data=f"card_{str(c)}")]
                    for c in human_player.hand
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text(
                    f"Your turn to play a card:\n{message}", reply_markup=reply_markup
                )
            else:
                await update.message.reply_text(message)
                await self.continue_round(chat_id, update, context)

        except Exception as e:
            logger.error(f"Error in start_round for chat_id {chat_id}: {e}")
            await update.message.reply_text(f"Error in round: {e}")
            game.log_game_state(f"Round error: {str(e)}", player_hands=True)
            game.save_game_log(save_full_log=True)
            del self.games[chat_id]

    async def continue_round(
        self, chat_id: int, update: Update, context: ContextTypes.DEFAULT_TYPE
    ):
        """Continue the current round for AI players or prompt human."""
        if chat_id not in self.games:
            await update.message.reply_text(
                "No active game. Start one with /newgame ai."
            )
            return

        game_data = self.games[chat_id]
        game = game_data["game"]
        human_player = game_data["human_player"]

        current_player_index = (
            (game.players.index(game.current_trick[-1][0]) + 1) % 4
            if game.current_trick
            else game.players.index(game.hakem)
        )
        current_player = game.players[current_player_index]

        if current_player == human_player and game_data["current_state"] == "playing":
            keyboard = [
                [InlineKeyboardButton(str(c), callback_data=f"card_{str(c)}")]
                for c in human_player.hand
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            trick_cards = ", ".join(
                [f"{p.name}: {str(c)}" for p, c in game.current_trick]
            )
            await update.message.reply_text(
                f"Your turn to play a card.\nCurrent trick: {trick_cards}\nYour hand: {[str(c) for c in human_player.hand]}",
                reply_markup=reply_markup,
            )
        else:
            try:
                card, _ = current_player.play_card(game.lead_suit)
                game.current_trick.append((current_player, card))
                current_player.hand.remove(card)
                trick_cards = ", ".join(
                    [f"{p.name}: {str(c)}" for p, c in game.current_trick]
                )
                await update.message.reply_text(
                    f"{current_player.name} played {str(card)}.\nCurrent trick: {trick_cards}"
                )

                if len(game.current_trick) == 4:
                    await self.start_round(chat_id, update, context)
                else:
                    await self.continue_round(chat_id, update, context)
            except Exception as e:
                logger.error(f"Error in continue_round for chat_id {chat_id}: {e}")
                await update.message.reply_text(f"Error in AI play: {e}")
                game.log_game_state(f"AI play error: {str(e)}", player_hands=True)
                game.save_game_log(save_full_log=True)
                del self.games[chat_id]

    async def end_game(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /endgame command."""
        chat_id = update.message.chat_id
        if chat_id in self.games:
            game = self.games[chat_id]["game"]
            game.log_game_state(
                "Game ended by user", player_hands=True, game_winner="None"
            )
            game.save_game_log(save_full_log=True)
            del self.games[chat_id]
            await update.message.reply_text(
                "Game ended. Start a new one with /newgame ai."
            )
        else:
            await update.message.reply_text("No active game to end.")

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors."""
        logger.error(f"Update {update} caused error {context.error}")
        if update and update.message:
            await update.message.reply_text(
                "An error occurred. Please try again or use /endgame."
            )

    def get_game_state(self):
        """Fetch the current game state for the web app."""
        chat_id = request.args.get("chatId")
        if not chat_id or chat_id not in self.games:
            return jsonify({"error": "No active game"}), 404
        try:
            chat_id = int(chat_id)
            game_data = self.games[chat_id]
            game = game_data["game"]
            human_player = game_data["human_player"]
            current_player = (
                game.players[
                    (game.players.index(game.current_trick[-1][0]) + 1) % 4
                ].name
                if game.current_trick
                else game.hakem.name
            )
            response = {
                "humanHand": [
                    {"suit": c.suit, "rank": c.rank} for c in human_player.hand
                ],
                "currentTrick": [
                    {"player": p.name, "card": {"suit": c.suit, "rank": c.rank}}
                    for p, c in game.current_trick
                ],
                "trumpSuit": game.trump_suit,
                "team1Score": game.scores[1],
                "team2Score": game.scores[2],
                "currentPlayer": current_player,
                "gameStatus": game_data["current_state"],
                "message": (
                    "Game in progress"
                    if game_data["current_state"] != "ended"
                    else "Game ended"
                ),
            }
            if (
                game_data["current_state"] == "choose_trump"
                and game.hakem == human_player
            ):
                response["hakemCards"] = [
                    {"suit": c.suit, "rank": c.rank} for c in game.hakem_cards
                ]
            return jsonify(response)
        except Exception as e:
            logger.error(f"Error in get_game_state for chat_id {chat_id}: {e}")
            return jsonify({"error": str(e)}), 500

    def select_trump(self):
        """Handle trump suit selection from the web app."""
        data = request.json
        chat_id = data.get("chatId")
        suit = data.get("suit")
        if not chat_id or chat_id not in self.games or not suit:
            return jsonify({"error": "Invalid request"}), 400
        try:
            chat_id = int(chat_id)
            game_data = self.games[chat_id]
            if game_data["current_state"] != "choose_trump":
                return jsonify({"error": "Not in trump selection phase"}), 400
            game = game_data["game"]
            if suit not in suits:
                return jsonify({"error": "Invalid suit"}), 400
            game.set_trump_suit(suit)
            game_data["current_state"] = "playing"
            return jsonify({"message": "Trump suit set"})
        except Exception as e:
            logger.error(f"Error in select_trump for chat_id {chat_id}: {e}")
            return jsonify({"error": str(e)}), 500

    def play_card(self):
        """Handle card play from the web app."""
        data = request.json
        chat_id = data.get("chatId")
        card_str = data.get("card")
        if not chat_id or chat_id not in self.games or not card_str:
            return jsonify({"error": "Invalid request"}), 400
        try:
            chat_id = int(chat_id)
            game_data = self.games[chat_id]
            game = game_data["game"]
            human_player = game_data["human_player"]
            if game_data["current_state"] != "playing":
                return jsonify({"error": "Not in playing phase"}), 400
            rank, suit = card_str.rsplit(" of ", 1)
            if rank not in ranks or suit not in suits:
                return jsonify({"error": "Invalid card"}), 400
            card = Card(suit, rank)
            if card not in human_player.hand:
                return jsonify({"error": "Card not in hand"}), 400
            human_player.play_card(game.lead_suit, selected_card=card)
            return jsonify({"message": "Card played"})
        except Exception as e:
            logger.error(f"Error in play_card for chat_id {chat_id}: {e}")
            return jsonify({"error": str(e)}), 500

    def api_new_game(self):
        """Start a new game via API."""
        data = request.json
        chat_id = data.get("chatId")
        if not chat_id:
            return jsonify({"error": "chatId is required"}), 400
        try:
            chat_id = int(chat_id)
            if chat_id in self.games:
                return jsonify({"error": "A game is already in progress"}), 400

            human_player = EnhancedPlayer("Player 1", is_human=True)
            ai_players = [EnhancedPlayer(f"Player {i+2}") for i in range(3)]
            players = [human_player] + ai_players

            for player in ai_players:
                model_path = self.model_path.format(player.name.split()[-1])
                try:
                    if os.path.exists(model_path):
                        player.model.load_state_dict(
                            torch.load(model_path, map_location=torch.device("cpu"))
                        )
                        player.target_model.load_state_dict(
                            torch.load(model_path, map_location=torch.device("cpu"))
                        )
                        logger.info(f"Loaded model for {player.name} from {model_path}")
                    else:
                        logger.warning(
                            f"Model for {player.name} not found at {model_path}"
                        )
                except Exception as e:
                    logger.error(f"Error loading model for {player.name}: {e}")

            game = Hokm(players)
            self.games[chat_id] = {
                "game": game,
                "human_player": human_player,
                "current_state": "init",
            }

            hakem_cards = game.start_game()
            if game.hakem == human_player:
                self.games[chat_id]["current_state"] = "choose_trump"
                return jsonify(
                    {
                        "message": "Game started, choose trump suit",
                        "hakemCards": [
                            {"suit": c.suit, "rank": c.rank} for c in hakem_cards
                        ],
                    }
                )
            else:
                game.choose_trump_suit()
                self.games[chat_id]["current_state"] = "playing"
                return jsonify(
                    {"message": "Game started, trump suit set automatically"}
                )
        except Exception as e:
            logger.error(f"Error in api_new_game for chat_id {chat_id}: {e}")
            return jsonify({"error": str(e)}), 500

    def api_end_game(self):
        """End the current game via API."""
        data = request.json
        chat_id = data.get("chatId")
        if not chat_id:
            return jsonify({"error": "chatId is required"}), 400
        try:
            chat_id = int(chat_id)
            if chat_id in self.games:
                game = self.games[chat_id]["game"]
                game.log_game_state(
                    "Game ended by user via API", player_hands=True, game_winner="None"
                )
                game.save_game_log(save_full_log=True)
                del self.games[chat_id]
                return jsonify({"message": "Game ended"})
            else:
                return jsonify({"error": "No active game"}), 404
        except Exception as e:
            logger.error(f"Error in api_end_game for chat_id {chat_id}: {e}")
            return jsonify({"error": str(e)}), 500

    def run(self):
        """Run Flask and Telegram bot."""
        flask_thread = threading.Thread(target=self.run_flask)
        flask_thread.daemon = True
        flask_thread.start()
        logger.info("Flask thread started")

        bot_thread = threading.Thread(target=self.run_bot)
        bot_thread.daemon = True
        bot_thread.start()
        logger.info("Telegram bot thread started")

        flask_thread.join()
        bot_thread.join()

    def run_flask(self):
        """Run Flask server."""
        port = int(os.getenv("PORT", 5000))
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
        logger.info(f"Flask server started on port {port}")

    def run_bot(self):
        """Run Telegram bot polling."""
        logger.info("Starting Telegram bot polling")
        self.telegram_app.run_polling(allowed_updates=Update.ALL_TYPES)


def main():
    """Initialize and run the bot."""
    bot = HokmBot()
    bot.run()


if __name__ == "__main__":
    main()
