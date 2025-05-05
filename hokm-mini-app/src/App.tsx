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
import { WebAppProvider, useWebApp } from '@vkruglikov/react-telegram-web-app';

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

// Get API URL from environment or fallback to localhost
const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

interface Card {
  suit: string;
  rank: string;
}

interface TrickCard {
  player: string;
  card: Card;
}

interface GameState {
  humanHand: Card[];
  currentTrick: TrickCard[];
  trumpSuit: string | null;
  team1Score: number;
  team2Score: number;
  currentPlayer: string;
  gameStatus: 'init' | 'choose_trump' | 'playing' | 'ended';
  message: string;
  hakemCards?: Card[];
}

const App: React.FC = () => {
  const webApp = useWebApp();
  const [gameState, setGameState] = useState<GameState | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Get user information from Telegram
  const userId = webApp?.initDataUnsafe?.user?.id || '12345'; // Fallback for testing
  const chatId = userId;

  const fetchGameState = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_URL}/api/game-state`, {
        params: { chatId },
      });
      setGameState(response.data);
    } catch (err) {
      console.error('Error fetching game state:', err);
      setError('Failed to load game state. Please try again.');
    } finally {
      setIsLoading(false);
    }
  }, [chatId]);

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
    if (gameState?.gameStatus === 'playing' && gameState.currentPlayer !== 'Player 1') {
      interval = setInterval(fetchGameState, 3000); // Poll every 3 seconds for AI moves
    }
    return () => clearInterval(interval);
  }, [gameState?.gameStatus, gameState?.currentPlayer, fetchGameState]);

  useEffect(() => {
    if (chatId) {
      fetchGameState();
      // Poll for game state updates every 2 seconds
      const interval = setInterval(fetchGameState, 2000);
      return () => clearInterval(interval);
    }
  }, [chatId]);

  const startNewGame = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.post(`${API_URL}/api/new-game`, {
        chatId,
      });
      console.log('New game started:', response.data);
      fetchGameState();
    } catch (err) {
      console.error('Error starting new game:', err);
      setError('Failed to start a new game. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const endGame = async () => {
    setIsLoading(true);
    setError(null);
    try {
      await axios.post(`${API_URL}/api/end-game`, { chatId });
      setGameState(null);
    } catch (err) {
      console.error('Error ending game:', err);
      setError('Failed to end the game. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleTrumpSelection = async (suit: string) => {
    setIsLoading(true);
    setError(null);
    try {
      await axios.post(`${API_URL}/api/select-trump`, { chatId, suit });
      fetchGameState();
      toast.success(`Trump suit set to ${suit}`);
    } catch (err) {
      console.error('Error selecting trump:', err);
      setError('Failed to select trump suit. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handlePlayCard = useCallback(
    async (card: Card) => {
      setIsLoading(true);
      setError(null);
      try {
        await axios.post(`${API_URL}/api/play-card`, {
          chatId,
          card: `${card.rank} of ${card.suit}`,
        });
        fetchGameState();
        toast.success(`Played ${card.rank} of ${card.suit}`);
      } catch (err) {
        console.error('Error playing card:', err);
        setError('Failed to play card. Please try again.');
      } finally {
        setIsLoading(false);
      }
    },
    [fetchGameState]
  );

  // Function to determine card colors
  const getCardColor = (suit: string) => {
    return suit === 'Hearts' || suit === 'Diamonds' ? 'red' : 'black';
  };

  // Function to render a card
  const renderCard = (card: Card, index: number, onClick?: () => void) => {
    const color = getCardColor(card.suit);
    const suitSymbol = {
      'Hearts': '♥',
      'Diamonds': '♦',
      'Clubs': '♣',
      'Spades': '♠',
    }[card.suit] || card.suit;

    return (
      <div 
        key={`${card.suit}-${card.rank}-${index}`} 
        className={`card ${color} ${onClick ? 'clickable' : ''}`} 
        onClick={onClick}
      >
        <div className="card-top-left">
          <div className="card-rank">{card.rank}</div>
          <div className="card-suit">{suitSymbol}</div>
        </div>
        <div className="card-center">
          <div className="card-big-suit">{suitSymbol}</div>
        </div>
        <div className="card-bottom-right">
          <div className="card-rank">{card.rank}</div>
          <div className="card-suit">{suitSymbol}</div>
        </div>
      </div>
    );
  };

  // Render trump selection UI
  const renderTrumpSelection = () => {
    if (!gameState || !gameState.hakemCards) return null;
    
    return (
      <div className="trump-selection">
        <h3>You are the Hakem! Choose a trump suit:</h3>
        <div className="hakem-cards">
          {gameState.hakemCards.map((card, index) => renderCard(card, index))}
        </div>
        <div className="trump-options">
          {['Hearts', 'Diamonds', 'Clubs', 'Spades'].map(suit => (
            <button 
              key={suit} 
              className={`trump-button ${getCardColor(suit)}`}
              onClick={() => handleTrumpSelection(suit)}
              disabled={isLoading}
            >
              {suit} {
                {
                  'Hearts': '♥',
                  'Diamonds': '♦',
                  'Clubs': '♣',
                  'Spades': '♠',
                }[suit]
              }
            </button>
          ))}
        </div>
      </div>
    );
  };

  // Render current trick
  const renderCurrentTrick = () => {
    if (!gameState || !gameState.currentTrick || gameState.currentTrick.length === 0) {
      return <div className="trick-area"><p>No cards played yet</p></div>;
    }

    return (
      <div className="trick-area">
        <h3>Current Trick</h3>
        <div className="trick-cards">
          {gameState.currentTrick.map((trickCard, index) => (
            <div key={index} className="trick-card-container">
              {renderCard(trickCard.card, index)}
              <div className="player-name">{trickCard.player}</div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Main render
  if (isLoading && !gameState) {
    return <div className="loading">Loading game...</div>;
  }

  if (error) {
    return (
      <div className="error">
        <p>{error}</p>
        <button onClick={fetchGameState}>Retry</button>
      </div>
    );
  }

  if (!gameState) {
    return (
      <div className="no-game">
        <h2>Hokm Card Game</h2>
        <p>No active game. Start a new one!</p>
        <button onClick={startNewGame} disabled={isLoading}>
          {isLoading ? 'Starting...' : 'Start New Game'}
        </button>
      </div>
    );
  }

  return (
    <div className="game-container">
      <div className="game-header">
        <h2>Hokm Card Game</h2>
        <div className="game-info">
          <p>Trump Suit: {gameState.trumpSuit || 'Not set'} {
            gameState.trumpSuit ? {
              'Hearts': '♥',
              'Diamonds': '♦',
              'Clubs': '♣',
              'Spades': '♠',
            }[gameState.trumpSuit] : ''
          }</p>
          <p>Team 1 Score: {gameState.team1Score}</p>
          <p>Team 2 Score: {gameState.team2Score}</p>
          <p>Current Player: {gameState.currentPlayer}</p>
        </div>
        <button 
          className="end-game-button" 
          onClick={endGame}
          disabled={isLoading}
        >
          End Game
        </button>
      </div>

      {gameState.gameStatus === 'choose_trump' && renderTrumpSelection()}
      
      {gameState.gameStatus === 'playing' && (
        <>
          {renderCurrentTrick()}
          
          <div className="player-hand">
            <h3>Your Hand</h3>
            <div className="hand-cards">
              {gameState.humanHand.map((card, index) => 
                renderCard(card, index, () => handlePlayCard(card))
              )}
            </div>
          </div>
        </>
      )}

      {gameState.gameStatus === 'ended' && (
        <div className="game-ended">
          <h3>Game Ended</h3>
          <p>Final Score: Team 1: {gameState.team1Score} - Team 2: {gameState.team2Score}</p>
          <button onClick={startNewGame} disabled={isLoading}>
            {isLoading ? 'Starting...' : 'Start New Game'}
          </button>
        </div>
      )}

      {isLoading && <div className="loading-overlay">Processing...</div>}
    </div>
  );
};

// Wrap with WebAppProvider
const AppWrapper = () => (
  <WebAppProvider>
    <App />
  </WebAppProvider>
);

export default AppWrapper;