#!/bin/bash

# Exit script if any command fails
set -e

# Check if .env file exists, if not create from example
if [ ! -f ".env" ]; then
  echo "Creating .env file from .env.example..."
  cp .env.example .env
  echo "Please edit the .env file with your actual tokens and credentials."
  exit 1
fi

# Extract the Telegram bot token and username
BOT_TOKEN=$(grep BOT_TOKEN .env | cut -d '=' -f2)
TELEGRAM_BOT_USERNAME=$(grep TELEGRAM_BOT_USERNAME .env | cut -d '=' -f2)
NGROK_AUTHTOKEN=$(grep NGROK_AUTHTOKEN .env | cut -d '=' -f2)

# Check if any of the required values are missing or default
if [[ "$BOT_TOKEN" == "your_telegram_bot_token_here" || "$TELEGRAM_BOT_USERNAME" == "your_bot_username_here" || "$NGROK_AUTHTOKEN" == "your_ngrok_authtoken_here" ]]; then
  echo "ERROR: Please update the .env file with your actual tokens and credentials."
  exit 1
fi

# Check for docker compose
if command -v docker-compose &> /dev/null; then
  DOCKER_COMPOSE_CMD="docker-compose"
  echo "Using docker-compose command"
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
  DOCKER_COMPOSE_CMD="docker compose"
  echo "Using docker compose command"
else
  echo "ERROR: Neither docker-compose nor docker compose commands are available"
  echo "Please install Docker Desktop from https://www.docker.com/products/docker-desktop/"
  exit 1
fi

# Build and start the containers
echo "Building and starting containers..."
$DOCKER_COMPOSE_CMD up --build -d

# Wait for ngrok to start
echo "Waiting for ngrok to initialize (this may take a few seconds)..."
sleep 10

# Get the ngrok URL
NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"https://[^"]*' | sed 's/"public_url":"//' | head -1)

if [ -z "$NGROK_URL" ]; then
  echo "ERROR: Failed to get ngrok URL"
  echo "Try checking the ngrok logs with: $DOCKER_COMPOSE_CMD logs ngrok"
  exit 1
fi

echo ""
echo "==================================================================="
echo "Hokm game is now running in Docker containers!"
echo "==================================================================="
echo "Frontend: $NGROK_URL"
echo "Backend API: http://localhost:5001"
echo ""
echo "IMPORTANT INSTRUCTIONS FOR BOTFATHER:"
echo "1. Open Telegram and message @BotFather"
echo "2. Type /mybots and select your bot"
echo "3. Click 'Bot Settings' -> 'Menu Button' -> 'Configure Menu Button'"
echo "4. Select 'Add a menu button' -> 'Web App'"
echo "5. Set the 'Button Text' to 'Play Hokm'"
echo "6. Set the 'Web App URL' to: $NGROK_URL"
echo "7. Save your changes"
echo ""
echo "Your Mini App should now be accessible from your Telegram bot's menu button!"
echo "==================================================================="
echo ""
echo "To view logs: $DOCKER_COMPOSE_CMD logs -f"
echo "To stop: $DOCKER_COMPOSE_CMD down"
echo "" 