# Hokm Card Game

Hokm is a Persian trick-taking card game. This project provides a Telegram Mini App version of the game with a Flask backend and React frontend.

## Docker Setup (Recommended)

The easiest way to run this application is using Docker:

### Prerequisites

1. [Docker](https://docs.docker.com/get-docker/)
   - Docker Desktop is recommended as it includes both Docker Engine and Docker Compose
   - Docker Desktop for Mac: https://www.docker.com/products/docker-desktop/
   - Docker Desktop for Windows: https://www.docker.com/products/docker-desktop/
   - On Linux, you may need to install Docker Compose separately
2. Telegram Bot token (from [@BotFather](https://t.me/botfather))
3. Ngrok authtoken (from [ngrok.com](https://ngrok.com/))

### Quick Start

1. Set up your environment variables:
   ```bash
   cp .env.example .env
   # Edit .env file with your actual tokens
   ```

2. Run the application:
   ```bash
   ./run_docker.sh
   ```

3. Follow the instructions displayed in the terminal to configure your Telegram bot's Mini App URL.

## Manual Setup

If you prefer to run the components manually:

### Backend Setup

1. Create a Python virtual environment:
   ```bash
   python -m venv hokm-venv
   source hokm-venv/bin/activate  # On Windows: hokm-venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file:
   ```
   BOT_TOKEN=your_telegram_bot_token
   PORT=5001
   ENV=development
   ```

4. Run the backend:
   ```bash
   python bot.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd hokm-mini-app
   ```

2. Install dependencies:
   ```bash
   npm install --legacy-peer-deps
   ```

3. Create a `.env` file:
   ```
   REACT_APP_API_URL=http://localhost:5001
   REACT_APP_TELEGRAM_BOT_USERNAME=your_bot_username
   ```

4. Run the frontend:
   ```bash
   npm start
   ```

### Exposing Your Local Server

To make your local development environment accessible to Telegram:

1. Install [ngrok](https://ngrok.com/)
2. Run ngrok to expose your frontend:
   ```bash
   ngrok http 3000
   ```
3. Use the https URL provided by ngrok as your Telegram Mini App URL in BotFather.

## Development

- Backend API is available at http://localhost:5001
- Frontend development server runs at http://localhost:3000

## Troubleshooting

If you encounter issues:

1. Check Docker logs:
   ```bash
   docker-compose logs
   ```

2. For port conflicts:
   ```bash
   # Check if ports are in use
   lsof -i :5001
   lsof -i :3000
   
   # Kill processes if needed
   kill -9 <PID>
   ```

3. For webhook errors, set `ENV=development` in your .env file to use polling mode. 