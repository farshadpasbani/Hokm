# Hokm Card Game with AI Players

A Python implementation of the Persian card game Hokm (حکم) with AI players using Deep Q-Learning (DQN).

## Game Description

Hokm is a trick-taking card game similar to Bridge, played with 4 players in 2 teams. The game features:
- Standard 52-card deck
- Trump suit selection by the Hakem
- Team-based gameplay
- First team to win 7 tricks wins the game

## Features

- AI players using Deep Q-Learning
- Strategic trump suit selection
- Team-based rewards system
- Game logging and statistics
- Customizable AI parameters

## Requirements

- Python 3.7+
- PyTorch
- pandas
- numpy

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hokm.git
cd hokm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic game setup and play:
```python
from hokm_game import Hokm, DQNPlayer

# Create players
players = [
    DQNPlayer("Player 1", 52, 13, epsilon=0.1),
    DQNPlayer("Player 2", 52, 13, epsilon=0.1),
    DQNPlayer("Player 3", 52, 13, epsilon=0.1),
    DQNPlayer("Player 4", 52, 13, epsilon=0.1)
]

# Create and play game
game = Hokm(players)
game.play_game()
```

## Project Structure

- `hokm_game.py`: Main game implementation
- `requirements.txt`: Project dependencies
- `README.md`: Project documentation
- `game_logs/`: Directory for game statistics (created automatically)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 