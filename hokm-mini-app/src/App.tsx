import React, { useEffect, useState, useCallback } from 'react';
import { init, isTMA, viewport } from '@telegram-apps/sdk';
import { motion } from 'framer-motion';
import axios from 'axios';
// @ts-ignore - react-playing-cards doesn't have type definitions
import { PlayingCard } from 'react-playing-cards';
// @ts-ignore - react-toastify types are included in the package
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import './App.css';

// Extend existing Telegram WebApp type instead of redeclaring
declare module '@telegram-apps/sdk' {
  interface TelegramWebApp {
    initDataUnsafe: {
      user?: {
        id: number;
      };
    };
    ready: () => void;
  }
}

interface Card {
  suit: string;
  rank: string;
}

interface GameState {
  humanHand: Card[];
  currentTrick: { player: string; card: Card }[];
  trumpSuit: string | null;
  team1Score: number;
  team2Score: number;
  currentPlayer: string;
  gameStatus: 'init' | 'choose_trump' | 'playing' | 'ended';
  message: string;
  hakemCards?: Card[];
}

const App: React.FC = () => {
  const [gameState, setGameState] = useState<GameState>({
    humanHand: [],
    currentTrick: [],
    trumpSuit: null,
    team1Score: 0,
    team2Score: 0,
    currentPlayer: '',
    gameStatus: 'init',
    message: 'Welcome to Hokm! Start a new game.',
  });
  const [isLoading, setIsLoading] = useState(false);

  const fetchGameState = useCallback(async () => {
    try {
      const chatId = window.Telegram.WebApp.initDataUnsafe.user?.id;
      if (!chatId) {
        toast.error('User ID not available');
        setGameState((prev) => ({ ...prev, message: 'User ID not available' }));
        return;
      }
      const response = await axios.get('https://hokm-backend.onrender.com/api/game-state', {
        params: { chatId },
      });
      setGameState(response.data);
    } catch (error) {
      console.error('Error fetching game state:', error);
      toast.error('Error fetching game state');
      setGameState((prev) => ({ ...prev, message: 'Error fetching game state' }));
    }
  }, []);

  useEffect(() => {
    // Initialize Telegram WebApp
    async function initWebApp() {
      try {
        if (await isTMA()) {
          init();
          window.Telegram.WebApp.ready();
          if (viewport.mount.isAvailable()) {
            await viewport.mount();
            viewport.expand();
          }
          if (viewport.requestFullscreen.isAvailable()) {
            await viewport.requestFullscreen();
          }
        } else {
          console.warn('Not running in Telegram environment');
          toast.warn('Please open in Telegram for full functionality');
        }
      } catch (error) {
        console.error('Error initializing WebApp:', error);
        toast.error('Error initializing app');
      }
    }
    initWebApp();
  }, []);

  // Poll game state during AI turns
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (gameState.gameStatus === 'playing' && gameState.currentPlayer !== 'Player 1') {
      interval = setInterval(fetchGameState, 3000); // Poll every 3 seconds for AI moves
    }
    return () => clearInterval(interval);
  }, [gameState.gameStatus, gameState.currentPlayer, fetchGameState]);

  const startNewGame = async () => {
    setIsLoading(true);
    try {
      const chatId = window.Telegram.WebApp.initDataUnsafe.user?.id;
      if (!chatId) {
        toast.error('User ID not available');
        setGameState((prev) => ({ ...prev, message: 'User ID not available' }));
        return;
      }
      const response = await axios.post('https://hokm-backend.onrender.com/api/new-game', { chatId });
      setGameState({ ...gameState, ...response.data, gameStatus: response.data.hakemCards ? 'choose_trump' : 'playing' });
      toast.success('New game started!');
    } catch (error) {
      console.error('Error starting new game:', error);
      toast.error('Error starting new game');
      setGameState((prev) => ({ ...prev, message: 'Error starting new game' }));
    } finally {
      setIsLoading(false);
    }
  };

  const endGame = async () => {
    setIsLoading(true);
    try {
      const chatId = window.Telegram.WebApp.initDataUnsafe.user?.id;
      if (!chatId) {
        toast.error('User ID not available');
        setGameState((prev) => ({ ...prev, message: 'User ID not available' }));
        return;
      }
      await axios.post('https://hokm-backend.onrender.com/api/end-game', { chatId });
      setGameState({
        humanHand: [],
        currentTrick: [],
        trumpSuit: null,
        team1Score: 0,
        team2Score: 0,
        currentPlayer: '',
        gameStatus: 'init',
        message: 'Game ended. Start a new game.',
      });
      toast.success('Game ended');
    } catch (error) {
      console.error('Error ending game:', error);
      toast.error('Error ending game');
      setGameState((prev) => ({ ...prev, message: 'Error ending game' }));
    } finally {
      setIsLoading(false);
    }
  };

  const handleTrumpSelection = async (suit: string) => {
    setIsLoading(true);
    try {
      const chatId = window.Telegram.WebApp.initDataUnsafe.user?.id;
      if (!chatId) {
        toast.error('User ID not available');
        setGameState((prev) => ({ ...prev, message: 'User ID not available' }));
        return;
      }
      await axios.post('https://hokm-backend.onrender.com/api/select-trump', { chatId, suit });
      await fetchGameState();
      toast.success(`Trump suit set to ${suit}`);
    } catch (error) {
      console.error('Error selecting trump:', error);
      toast.error('Error selecting trump suit');
      setGameState((prev) => ({ ...prev, message: 'Error selecting trump suit' }));
    } finally {
      setIsLoading(false);
    }
  };

  const handlePlayCard = useCallback(
    async (card: Card) => {
      setIsLoading(true);
      try {
        const chatId = window.Telegram.WebApp.initDataUnsafe.user?.id;
        if (!chatId) {
          toast.error('User ID not available');
          setGameState((prev) => ({ ...prev, message: 'User ID not available' }));
          return;
        }
        await axios.post('https://hokm-backend.onrender.com/api/play-card', {
          chatId,
          card: `${card.rank} of ${card.suit}`,
        });
        await fetchGameState();
        toast.success(`Played ${card.rank} of ${card.suit}`);
      } catch (error) {
        console.error('Error playing card:', error);
        toast.error('Error playing card');
        setGameState((prev) => ({ ...prev, message: 'Error playing card' }));
      } finally {
        setIsLoading(false);
      }
    },
    [fetchGameState]
  );

  return (
    <div className="app">
      <h1>Hokm Card Game</h1>
      <div className="game-controls">
        <button
          onClick={startNewGame}
          disabled={isLoading || gameState.gameStatus !== 'init'}
          aria-label="Start a new game"
        >
          {isLoading ? 'Starting...' : 'New Game'}
        </button>
        <button
          onClick={endGame}
          disabled={isLoading || gameState.gameStatus === 'init'}
          aria-label="End current game"
        >
          {isLoading ? 'Ending...' : 'End Game'}
        </button>
      </div>
      <div className="game-info">
        <p>Trump Suit: {gameState.trumpSuit || 'Not set'}</p>
        <p>Scores: Team 1: {gameState.team1Score} | Team 2: {gameState.team2Score}</p>
        <p>Current Player: {gameState.currentPlayer}</p>
        <p>{gameState.message}</p>
      </div>

      {gameState.gameStatus === 'choose_trump' && (
        <div className="trump-selection">
          <h2>Choose Trump Suit</h2>
          {['Hearts', 'Diamonds', 'Clubs', 'Spades'].map((suit) => (
            <button
              key={suit}
              onClick={() => handleTrumpSelection(suit)}
              disabled={isLoading}
              aria-label={`Select ${suit} as trump suit`}
            >
              {suit}
            </button>
          ))}
          {gameState.hakemCards && (
            <div className="hakem-cards">
              <h3>Your Hakem Cards</h3>
              <div className="card-container">
                {gameState.hakemCards.map((card, index) => (
                  <motion.div
                    key={index}
                    initial={{ y: 50, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: index * 0.1 }}
                  >
                    <PlayingCard
                      suit={card.suit.toLowerCase()}
                      rank={card.rank}
                      width={100}
                    />
                  </motion.div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      <div className="trick">
        <h2>Current Trick</h2>
        <div className="card-container">
          {gameState.currentTrick.map(({ player, card }, index) => (
            <motion.div
              key={index}
              className="trick-card"
              initial={{ x: 50, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: index * 0.1 }}
            >
              <p>{player}</p>
              <PlayingCard
                suit={card.suit.toLowerCase()}
                rank={card.rank}
                width={100}
              />
            </motion.div>
          ))}
        </div>
      </div>

      <div className="hand">
        <h2>Your Hand</h2>
        <div className="card-container">
          {gameState.humanHand.map((card, index) => (
            <motion.div
              key={index}
              initial={{ y: 50, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: index * 0.1 }}
            >
              <PlayingCard
                suit={card.suit.toLowerCase()}
                rank={card.rank}
                width={100}
                onClick={() => handlePlayCard(card)}
                disabled={gameState.currentPlayer !== 'Player 1' || gameState.gameStatus !== 'playing' || isLoading}
                className={gameState.currentPlayer === 'Player 1' && gameState.gameStatus === 'playing' ? 'playable' : ''}
                aria-label={`Play ${card.rank} of ${card.suit}`}
              />
            </motion.div>
          ))}
        </div>
      </div>
      <ToastContainer position="top-center" autoClose={3000} />
    </div>
  );
};

export default App;