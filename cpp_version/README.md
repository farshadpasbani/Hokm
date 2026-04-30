# Hokm C++ Conversion

This directory contains the C++ conversion of the Hokm game, including the core game logic, the reinforcement learning agents (using LibTorch), the training loop, and a lightweight web server.

## Architecture

The C++ version is structured as follows:

- `include/`: Header files for the core game logic and RL agents.
  - `game_constants.h`: Constants and utility functions.
  - `card.h`, `deck.h`, `player.h`, `hokm.h`: Core game logic.
  - `enhanced_player.h`: RL agent implementation using LibTorch.
  - `httplib.h`: Lightweight header-only C++ web framework.
- `src/core/`: Implementation of the core game logic.
- `src/rl/`: Implementation of the RL agents (Neural Fictitious Self-Play).
- `src/train_hokm.cpp`: Training loop executable.
- `src/app.cpp`: Web server executable.
- `tests/`: Unit tests using Google Test (or standalone).

## Prerequisites

To build the full C++ project, you need:

1. **CMake** (version 3.14 or higher)
2. **C++17** compatible compiler (GCC, Clang, or MSVC)
3. **LibTorch** (PyTorch C++ API):
   - Download the pre-built LibTorch binaries from the [PyTorch website](https://pytorch.org/get-started/locally/).
   - Extract it to a directory (e.g., `/path/to/libtorch`).

## Building the Project

1. Create a build directory:
   ```bash
   mkdir build && cd build
   ```

2. Run CMake, pointing it to your LibTorch installation:
   ```bash
   cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
   ```

3. Build the project:
   ```bash
   make
   ```

This will generate several executables:
- `HokmTests`: Runs the unit tests.
- `TrainHokm`: Runs the RL training loop.
- `HokmApp`: Runs the web server.

## Running Tests (Without LibTorch)

If you don't have LibTorch installed yet and just want to test the core game logic, you can use the provided `Makefile` in the root of `cpp_version`:

```bash
make
./test_core
```

This will compile the core game logic (`Card`, `Deck`, `Player`, `Hokm`) and run a standalone test suite.

## Running the Web Server

Once built, you can start the web server:

```bash
./build/HokmApp
```

The server will listen on `http://localhost:8080`.

## Running the Training Loop

To train the RL agents:

```bash
./build/TrainHokm 10000
```

Where `10000` is the number of episodes to train.

## Documentation

The C++ code is heavily documented and mirrors the Python implementation:
- `Card` and `Deck` handle the physical cards.
- `Player` is an interface that can be implemented by humans or AI.
- `Hokm` manages the game state, trick resolution, and scoring.
- `EnhancedPlayer` uses two PyTorch Neural Networks (`QNetwork` and `AveragePolicyNetwork`) to implement NFSP (Neural Fictitious Self-Play) for the AI agents.
