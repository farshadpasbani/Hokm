import os
import logging
import asyncio
import signal
from flask import Flask, request, jsonify
from flask_cors import CORS
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes,
    MessageHandler,
)
import torch
from hokm import Hokm
from enhanced_player import EnhancedPlayer
from game_constants import suits, ranks, Card
from dotenv import load_dotenv
from multiprocessing import Process
from fastapi import FastAPI, Request

# Configure logging with detailed format
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

# Load environment variables
logger.info("Loading environment variables")
load_dotenv()

# Check environment variables
BOT_TOKEN = os.getenv("BOT_TOKEN")
PORT = os.getenv("PORT", "5001")
MODEL_PATH = os.getenv("MODEL_PATH", "models/player_{}_model.pth")
ENV = os.getenv(
    "ENV", "development"
)  # 'development' for local, 'production' for Render

if not BOT_TOKEN:
    logger.error("BOT_TOKEN not set in environment variables")
    raise ValueError("BOT_TOKEN is required")


def create_flask_app():
    """Create and configure Flask app."""
    app = Flask(__name__)
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": [
                    "http://localhost:3000",  # Local frontend
                    "https://hokm-mini-app.vercel.app",  # Deployed frontend
                ],
                "methods": ["GET", "POST", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
            }
        },
    )
    return app


def run_flask(port, use_gunicorn=False):
    """Run Flask server with error handling."""
    try:
        app = create_flask_app()
        if use_gunicorn and ENV == "production":
            logger.info("Starting Flask server with gunicorn")
            try:
                from gunicorn.app.base import BaseApplication

                class FlaskApplication(BaseApplication):
                    def __init__(self, app, options=None):
                        self.options = options or {}
                        self.application = app
                        super().__init__()

                    def load_config(self):
                        for key, value in self.options.items():
                            self.cfg.set(key.lower(), value)

                    def load(self):
                        return self.application

                options = {
                    "bind": f"0.0.0.0:{port}",
                    "workers": 2,
                    "timeout": 60,
                }
                FlaskApplication(app, options).run()
            except ImportError:
                logger.error("gunicorn not installed, falling back to Flask dev server")
                app.run(host="0.0.0.0", port=int(port), debug=False, use_reloader=False)
        else:
            logger.info(f"Starting Flask dev server on port {port}")
            app.run(host="0.0.0.0", port=int(port), debug=True, use_reloader=False)
    except Exception as e:
        logger.error(f"Error starting Flask server: {e}", exc_info=True)
        raise


async def run_telegram_bot(telegram_app):
    """Run Telegram bot with proper shutdown handling."""
    try:
        logger.info("Starting Telegram bot")
        await telegram_app.initialize()
        await telegram_app.start()
        logger.info("Starting polling")
        await telegram_app.updater.start_polling(allowed_updates=Update.ALL_TYPES)
        # Keep polling running until stopped
        while telegram_app.updater.running:
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Error in Telegram bot: {e}", exc_info=True)
        raise
    finally:
        logger.info("Stopping Telegram bot")
        try:
            if telegram_app.updater.running:
                logger.info("Stopping updater polling")
                await telegram_app.updater.stop()
            logger.info("Stopping application")
            await telegram_app.stop()
            logger.info("Shutting down application")
            await telegram_app.shutdown()
            logger.info("Telegram bot fully stopped")
        except Exception as e:
            logger.error(f"Error during Telegram bot shutdown: {e}", exc_info=True)


class HokmBot:
    def __init__(self):
        logger.info("Initializing HokmBot")
        self.games = {}
        self.model_path = MODEL_PATH
        self.telegram_app = Application.builder().token(BOT_TOKEN).build()
        self.app = create_flask_app()
        self.register_routes()
        self.register_handlers()

    def register_routes(self):
        """Register Flask routes with error handling."""
        try:
            self.app.add_url_rule(
                "/api/new-game", view_func=self.api_new_game, methods=["POST"]
            )
            self.app.add_url_rule(
                "/api/end-game", view_func=self.api_end_game, methods=["POST"]
            )
            self.app.add_url_rule(
                "/api/game-state", view_func=self.get_game_state, methods=["GET"]
            )
            self.app.add_url_rule(
                "/api/select-trump", view_func=self.select_trump, methods=["POST"]
            )
            self.app.add_url_rule(
                "/api/play-card", view_func=self.play_card, methods=["POST"]
            )
            logger.info("API routes registered successfully")
        except Exception as e:
            logger.error(f"Error registering routes: {e}", exc_info=True)
            raise

    def register_handlers(self):
        """Register Telegram handlers with error handling."""
        try:
            self.telegram_app.add_handler(CommandHandler("start", self.start))
            self.telegram_app.add_handler(CommandHandler("newgame", self.new_game))
            self.telegram_app.add_handler(CommandHandler("endgame", self.end_game))
            self.telegram_app.add_handler(CallbackQueryHandler(self.button))
            self.telegram_app.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message)
            )
            self.telegram_app.add_error_handler(self.error_handler)
            logger.info("Telegram handlers registered successfully")
        except Exception as e:
            logger.error(f"Error registering handlers: {e}", exc_info=True)
            raise

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command with inline button to launch web app."""
        web_app_url = (
            "https://hokm-mini-app.vercel.app"
            if ENV == "production"
            else "http://localhost:3000"
        )
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
                    logger.warning(f"Model for {player.name} not found at {model_path}")
            except Exception as e:
                logger.error(f"Error loading model for {player.name}: {e}")
                await update.message.reply_text(
                    f"Failed to load model for {player.name}. Starting with untrained AI."
                )

        game = Hokm(players)
        self.games[chat_id] = {
            "game": game,
            "human_player": human_player,
            "current_state": "init",
        }

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
                    f"You are the Hakem! Your cards: {[str(c) for c in hakem_cards]}\nChoose the trump suit:",
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

    async def handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle non-command messages."""
        await update.message.reply_text(
            "Please use /start to begin or /help for a list of commands."
        )

    async def error_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle errors."""
        logger.error(f"Update {update} caused error {context.error}")
        if update and update.message:
            await update.message.reply_text(
                "An error occurred. Please try again or use /endgame."
            )

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
                    self.continue_round(chat_id, update, context)
            except Exception as e:
                logger.error(f"Error in continue_round for chat_id {chat_id}: {e}")
                await update.message.reply_text(f"Error in AI play: {e}")
                game.log_game_state(f"AI play error: {str(e)}", player_hands=True)
                game.save_game_log(save_full_log=True)
                del self.games[chat_id]

    def api_new_game(self):
        """Start a new game via API."""
        try:
            logger.info("Received new game request")
            data = request.json
            logger.info(f"Request data: {data}")
            if not data:
                logger.error("No JSON data in request")
                return jsonify({"error": "No JSON data provided"}), 400
            chat_id = data.get("chatId")
            if not chat_id:
                logger.error("No chatId provided in request")
                return jsonify({"error": "chatId is required"}), 400
            try:
                chat_id = int(chat_id)
            except ValueError:
                logger.error(f"Invalid chatId format: {chat_id}")
                return jsonify({"error": "chatId must be a number"}), 400
            logger.info(f"Starting new game for chat_id: {chat_id}")
            if chat_id in self.games:
                logger.warning(f"Game already in progress for chat_id: {chat_id}")
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
                    return (
                        jsonify(
                            {
                                "error": f"Error loading model for {player.name}: {str(e)}"
                            }
                        ),
                        500,
                    )
            game = Hokm(players)
            self.games[chat_id] = {
                "game": game,
                "human_player": human_player,
                "current_state": "init",
            }
            hakem_cards = game.start_game()
            if game.hakem == human_player:
                self.games[chat_id]["current_state"] = "choose_trump"
                logger.info("Human player is hakem, waiting for trump selection")
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
                logger.info("AI is hakem, setting trump suit automatically")
                return jsonify(
                    {"message": "Game started, trump suit set automatically"}
                )
        except Exception as e:
            logger.error(f"Unexpected error in api_new_game: {e}", exc_info=True)
            return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

    def api_end_game(self):
        """End the current game via API."""
        try:
            logger.info("Received end game request")
            data = request.json
            logger.info(f"Request data: {data}")
            chat_id = data.get("chatId")
            if not chat_id:
                logger.error("No chatId provided in request")
                return jsonify({"error": "chatId is required"}), 400
            try:
                chat_id = int(chat_id)
            except ValueError:
                logger.error(f"Invalid chatId format: {chat_id}")
                return jsonify({"error": "chatId must be a number"}), 400
            if chat_id in self.games:
                game = self.games[chat_id]["game"]
                game.log_game_state(
                    "Game ended by user via API", player_hands=True, game_winner="None"
                )
                game.save_game_log(save_full_log=True)
                del self.games[chat_id]
                logger.info(f"Game ended for chat_id: {chat_id}")
                return jsonify({"message": "Game ended"})
            else:
                logger.warning(f"No active game for chat_id: {chat_id}")
                return jsonify({"error": "No active game"}), 404
        except Exception as e:
            logger.error(f"Error in api_end_game: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    def get_game_state(self):
        """Fetch the current game state for the web app."""
        try:
            logger.info("Received game state request")
            chat_id = request.args.get("chatId")
            logger.info(f"Request for chat_id: {chat_id}")
            if not chat_id:
                logger.error("No chatId provided in request")
                return jsonify({"error": "chatId is required"}), 400
            try:
                chat_id = int(chat_id)
            except ValueError:
                logger.error(f"Invalid chatId format: {chat_id}")
                return jsonify({"error": "chatId must be a number"}), 400
            if chat_id not in self.games:
                logger.warning(f"No active game for chat_id: {chat_id}")
                return jsonify({"error": "No active game"}), 404
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
            logger.info("Game state retrieved successfully")
            return jsonify(response)
        except Exception as e:
            logger.error(f"Error in get_game_state: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    def select_trump(self):
        """Handle trump suit selection from the web app."""
        try:
            logger.info("Received trump selection request")
            data = request.json
            logger.info(f"Request data: {data}")
            chat_id = data.get("chatId")
            suit = data.get("suit")
            if not chat_id or not suit:
                logger.error("Missing required parameters")
                return jsonify({"error": "chatId and suit are required"}), 400
            try:
                chat_id = int(chat_id)
            except ValueError:
                logger.error(f"Invalid chatId format: {chat_id}")
                return jsonify({"error": "chatId must be a number"}), 400
            if chat_id not in self.games:
                logger.warning(f"No active game for chat_id: {chat_id}")
                return jsonify({"error": "No active game"}), 404
            game_data = self.games[chat_id]
            if game_data["current_state"] != "choose_trump":
                logger.error("Not in trump selection phase")
                return jsonify({"error": "Not in trump selection phase"}), 400
            game = game_data["game"]
            if suit not in suits:
                logger.error(f"Invalid suit: {suit}")
                return jsonify({"error": "Invalid suit"}), 400
            game.set_trump_suit(suit)
            game_data["current_state"] = "playing"
            logger.info(f"Trump suit set to {suit}")
            return jsonify({"message": "Trump suit set"})
        except Exception as e:
            logger.error(f"Error in select_trump: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    def play_card(self):
        """Handle card play from the web app."""
        try:
            logger.info("Received play card request")
            data = request.json
            logger.info(f"Request data: {data}")
            chat_id = data.get("chatId")
            card_str = data.get("card")
            if not chat_id or not card_str:
                logger.error("Missing required parameters")
                return jsonify({"error": "chatId and card are required"}), 400
            try:
                chat_id = int(chat_id)
            except ValueError:
                logger.error(f"Invalid chatId format: {chat_id}")
                return jsonify({"error": "chatId must be a number"}), 400
            if chat_id not in self.games:
                logger.warning(f"No active game for chat_id: {chat_id}")
                return jsonify({"error": "No active game"}), 404
            game_data = self.games[chat_id]
            game = game_data["game"]
            human_player = game_data["human_player"]
            if game_data["current_state"] != "playing":
                logger.error("Not in playing phase")
                return jsonify({"error": "Not in playing phase"}), 400
            rank, suit = card_str.rsplit(" of ", 1)
            if rank not in ranks or suit not in suits:
                logger.error(f"Invalid card: {card_str}")
                return jsonify({"error": "Invalid card"}), 400
            card = Card(suit, rank)
            if card not in human_player.hand:
                logger.error(f"Card not in hand: {card_str}")
                return jsonify({"error": "Card not in hand"}), 400
            human_player.play_card(game.lead_suit, selected_card=card)
            logger.info(f"Card played: {card_str}")
            return jsonify({"message": "Card played"})
        except Exception as e:
            logger.error(f"Error in play_card: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    def run(self):
        """Run Flask and Telegram bot with proper error handling."""

        def handle_shutdown(signum, frame):
            logger.info("Received shutdown signal, stopping application")
            if flask_process.is_alive():
                flask_process.terminate()
                flask_process.join()
                logger.info("Flask process terminated")
            if self.telegram_app.updater.running:
                asyncio.run(self.telegram_app.updater.stop())
                asyncio.run(self.telegram_app.stop())
                asyncio.run(self.telegram_app.shutdown())
                logger.info("Telegram bot stopped")
            raise SystemExit("Shutdown complete")

        try:
            flask_process = Process(target=run_flask, args=(PORT, ENV == "production"))
            flask_process.start()
            logger.info("Flask process started")

            # Register signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, handle_shutdown)
            signal.signal(signal.SIGTERM, handle_shutdown)

            # Configure for webhook mode if WEBHOOK_URL is set in environment
            # This helps avoid conflicts with other bot instances
            webhook_url = os.getenv("WEBHOOK_URL")

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                if webhook_url:
                    logger.info(f"Using webhook mode with URL: {webhook_url}")
                    # Use webhook instead of polling to avoid conflicts
                    webhook_app = FastAPI()

                    async def setup():
                        await self.telegram_app.initialize()
                        await self.telegram_app.start()
                        # Set webhook
                        await self.telegram_app.bot.set_webhook(url=webhook_url)
                        logger.info("Webhook set successfully")

                    @webhook_app.post(f"/{BOT_TOKEN}")
                    async def telegram_webhook(request: Request):
                        req_body = await request.json()
                        await self.telegram_app.update_queue.put(
                            Update.de_json(data=req_body, bot=self.telegram_app.bot)
                        )
                        return {"ok": True}

                    @webhook_app.on_event("startup")
                    async def on_startup():
                        await setup()

                    @webhook_app.on_event("shutdown")
                    async def on_shutdown():
                        await self.telegram_app.bot.delete_webhook()
                        await self.telegram_app.stop()
                        await self.telegram_app.shutdown()

                    # Run webhook with uvicorn
                    import uvicorn

                    uvicorn.run(webhook_app, host="0.0.0.0", port=8443)
                else:
                    # Use regular polling mode
                    logger.info("Using polling mode")
                    loop.run_until_complete(run_telegram_bot(self.telegram_app))
            except KeyboardInterrupt:
                logger.info("Received keyboard interrupt, shutting down...")
            finally:
                loop.close()
                if flask_process.is_alive():
                    flask_process.terminate()
                    flask_process.join()
                    logger.info("Flask process terminated")
        except Exception as e:
            logger.error(f"Error in main run method: {e}", exc_info=True)
            if flask_process.is_alive():
                flask_process.terminate()
                flask_process.join()
            raise


def main():
    """Initialize and run the bot with error handling."""
    try:
        logger.info("Starting HokmBot application")
        bot = HokmBot()
        bot.run()
    except Exception as e:
        logger.error(f"Fatal error in main: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
