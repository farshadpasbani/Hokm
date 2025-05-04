import React, { useEffect, useState } from 'react';
import { init, isTMA, viewport } from '@telegram-apps/sdk';
import axios from 'axios';
import './App.css';

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

  useEffect(() => {
    // Initialize Telegram WebApp
    async function initWebApp() {
      if (await isTMA()) {
        init();
        window.Telegram.WebApp.ready(); // Notify Telegram the app is ready
        if (viewport.mount.isAvailable()) {
          await viewport.mount();
          viewport.expand(); // Maximize height
        }
        if (viewport.requestFullscreen.isAvailable()) {
          await viewport.requestFullscreen(); // Full-screen mode
        }
      } else {
        console.warn('Not running in Telegram environment');
      }
    }
    initWebApp();
  }, []);

  // Fetch game state from backend
  const fetchGameState = async () => {
    try {
      const chatId = window.Telegram.WebApp.initDataUnsafe.user?.id;
      if (!chatId) {
        console.error('No user ID available');
        return;
      }
      const response = await axios.get('/api/game-state', {
        params: { chatId },
      });
      setGameState(response.data);
    } catch (error) {
      console.error('Error fetching game state:', error);
    }
  };

  // Handle trump suit selection
  const handleTrumpSelection = async (suit: string) => {
    try {
      const chatId = window.Telegram.WebApp.initDataUnsafe.user?.id;
      if (!chatId) {
        console.error('No user ID available');
        return;
      }
      await axios.post('/api/select-trump', {
        chatId,
        suit,
      });
      fetchGameState();
    } catch (error) {
      console.error('Error selecting trump:', error);
    }
  };

  // Handle card play
  const handlePlayCard = async (card: Card) => {
    try {
      const chatId = window.Telegram.WebApp.initDataUnsafe.user?.id;
      if (!chatId) {
        console.error('No user ID available');
        return;
      }
      await axios.post('/api/play-card', {
        chatId,
        card: `${card.rank} of ${card.suit}`,
      });
      fetchGameState();
    } catch (error) {
      console.error('Error playing card:', error);
    }
  };

  return (
    <div className="app">
      <h1>Hokm Card Game</h1>
      <div className="game-info">
        <p>Trump Suit: {gameState.trumpSuit || 'Not set'}</p>
        <p>Scores: Team 1: {gameState.team1Score} | Team 2: {gameState.team2Score}</p>
        <p>Status: {gameState.message}</p>
      </div>

      {gameState.gameStatus === 'choose_trump' && (
        <div className="trump-selection">
          <h2>Choose Trump Suit</h2>
          {['Hearts', 'Diamonds', 'Clubs', 'Spades'].map((suit) => (
            <button key={suit} onClick={() => handleTrumpSelection(suit)}>
              {suit}
            </button>
          ))}
        </div>
      )}

      <div className="trick">
        <h2>Current Trick</h2>
        {gameState.currentTrick.map(({ player, card }, index) => (
          <div key={index}>
            {player}: {card.rank} of {card.suit}
          </div>
        ))}
      </div>

      <div className="hand">
        <h2>Your Hand</h2>
        {gameState.humanHand.map((card, index) => (
          <button
            key={index}
            onClick={() => handlePlayCard(card)}
            disabled={gameState.currentPlayer !== 'Player 1' || gameState.gameStatus !== 'playing'}
          >
            {card.rank} of {card.suit}
          </button>
        ))}
      </div>
    </div>
  );
};

export default App;