#!/bin/bash

# Exit script if any command fails
set -e

# Check if ngrok auth token is provided
if [ -z "$1" ]; then
  echo "Please provide your ngrok auth token as the first argument"
  echo "Usage: ./run_local.sh your_ngrok_auth_token"
  exit 1
fi

NGROK_AUTH_TOKEN=$1

# Configure ngrok with the auth token
echo "Configuring ngrok with your auth token..."
ngrok config add-authtoken $NGROK_AUTH_TOKEN

# Create or update the .env file
echo "Updating .env file..."
if [ -z "$2" ]; then
  # If Telegram bot token is not provided, ask for it
  read -p "Enter your Telegram bot token: " BOT_TOKEN
else
  BOT_TOKEN=$2
fi

# Make sure ngrok isn't already running
pkill -f "ngrok http" || true
rm -f ngrok.log || true

# Start background process for ngrok with improved output capturing
echo "Starting ngrok tunnel for Flask backend (port 5001)..."
ngrok http 5001 --log=stdout > ngrok.log 2>&1 &
NGROK_PID=$!

# Wait for ngrok to start and get the public URL
echo "Waiting for ngrok to initialize (this may take a few seconds)..."
sleep 5

# Try multiple methods to extract the URL
NGROK_URL=$(grep -o 'url=https://[^[:space:]]*' ngrok.log | sed 's/url=//' | head -1)

if [ -z "$NGROK_URL" ]; then
  NGROK_URL=$(grep -o 'https://[^[:space:]]*\.ngrok\.io' ngrok.log | head -1)
fi

if [ -z "$NGROK_URL" ]; then
  NGROK_URL=$(grep -o 'https://[^[:space:]]*\.ngrok\.[a-z]*' ngrok.log | head -1)
fi

if [ -z "$NGROK_URL" ]; then
  echo "Failed to get ngrok URL from log file. Trying API method..."
  # Try to get URL from ngrok API
  NGROK_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"https://[^"]*' | sed 's/"public_url":"//' | head -1)
fi

if [ -z "$NGROK_URL" ]; then
  echo "ERROR: Failed to get ngrok URL. Check ngrok.log for details:"
  cat ngrok.log
  kill $NGROK_PID
  exit 1
fi

echo "ngrok tunnel established at: $NGROK_URL"

# Configure Telegram webhook with ngrok URL
echo "Setting Telegram webhook URL to: $NGROK_URL"
WEBHOOK_RESPONSE=$(curl -s -F "url=$NGROK_URL" https://api.telegram.org/bot$BOT_TOKEN/setWebhook)
echo "Webhook response: $WEBHOOK_RESPONSE"

# Create .env file with webhook URL
cat > .env << EOL
BOT_TOKEN=$BOT_TOKEN
PORT=5001
WEBHOOK_URL=$NGROK_URL
EOL

# Create .env file for React frontend
echo "Creating .env file for React frontend..."
cat > hokm-mini-app/.env << EOL
REACT_APP_API_URL=$NGROK_URL
REACT_APP_TELEGRAM_BOT_USERNAME=$(curl -s https://api.telegram.org/bot$BOT_TOKEN/getMe | grep -o '"username":"[^"]*"' | cut -d'"' -f4)
EOL

# Start the Flask backend and Telegram bot
echo "Starting the Flask backend and Telegram bot..."
source hokm-venv/bin/activate
python bot.py &
BACKEND_PID=$!

# Start the frontend (React app)
echo "Starting the frontend (React app)..."
cd hokm-mini-app
npm install --legacy-peer-deps
npm start &
FRONTEND_PID=$!

# Setup trap to kill all processes on exit
trap "kill $NGROK_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null || true; exit" INT TERM EXIT

echo "Development environment is running!"
echo "- Backend API accessible at: $NGROK_URL"
echo "- Frontend running at: http://localhost:3000"
echo "- Telegram webhook configured to: $NGROK_URL"
echo ""
echo "Press Ctrl+C to stop all services."

# Keep script running
wait 