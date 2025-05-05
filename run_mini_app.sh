#!/bin/bash

# Exit script if any command fails
set -e

# Check if ngrok auth token is provided
if [ -z "$1" ]; then
  echo "Please provide your ngrok auth token as the first argument"
  echo "Usage: ./run_mini_app.sh your_ngrok_auth_token"
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

# Kill any existing ngrok processes
echo "Cleaning up any existing processes..."
pkill -f "ngrok" || true
pkill -f "python bot.py" || true
pkill -f "npm start" || true
sleep 2

# Check if ports are in use
echo "Checking if ports are available..."
PORT_5001_PID=$(lsof -ti:5001)
if [ ! -z "$PORT_5001_PID" ]; then
  echo "Port 5001 is in use by process $PORT_5001_PID. Killing it..."
  kill -9 $PORT_5001_PID || true
  sleep 1
fi

PORT_3000_PID=$(lsof -ti:3000)
if [ ! -z "$PORT_3000_PID" ]; then
  echo "Port 3000 is in use by process $PORT_3000_PID. Killing it..."
  kill -9 $PORT_3000_PID || true
  sleep 1
fi

# Start the Flask backend and Telegram bot
echo "Starting the Flask backend and Telegram bot..."
source hokm-venv/bin/activate
python bot.py &
BACKEND_PID=$!
sleep 2

# Start ngrok for the backend API
echo "Starting ngrok tunnel for Flask backend (port 5001)..."
ngrok http 5001 --log=stdout > ngrok_backend.log 2>&1 &
BACKEND_NGROK_PID=$!

# Wait for ngrok to start
echo "Waiting for backend ngrok to initialize..."
sleep 5

# Get the backend URL
BACKEND_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o '"public_url":"https://[^"]*' | sed 's/"public_url":"//' | head -1)
if [ -z "$BACKEND_URL" ]; then
  echo "ERROR: Failed to get backend ngrok URL"
  pkill -f "ngrok" || true
  kill $BACKEND_PID 2>/dev/null || true
  exit 1
fi

echo "Backend API URL: $BACKEND_URL"

# Configure Telegram webhook
echo "Setting Telegram webhook URL to: $BACKEND_URL"
WEBHOOK_RESPONSE=$(curl -s -F "url=$BACKEND_URL" https://api.telegram.org/bot$BOT_TOKEN/setWebhook)
echo "Webhook response: $WEBHOOK_RESPONSE"

# Create .env file for the backend
cat > .env << EOL
BOT_TOKEN=$BOT_TOKEN
PORT=5001
WEBHOOK_URL=$BACKEND_URL
EOL

# Create .env file for React frontend
echo "Creating .env file for React frontend..."
cat > hokm-mini-app/.env << EOL
REACT_APP_API_URL=$BACKEND_URL
REACT_APP_TELEGRAM_BOT_USERNAME=$(curl -s https://api.telegram.org/bot$BOT_TOKEN/getMe | grep -o '"username":"[^"]*"' | cut -d'"' -f4)
EOL

# Make sure React dependencies are properly installed
echo "Installing React dependencies..."
cd hokm-mini-app
npm install --legacy-peer-deps

# Start the frontend (React app)
echo "Starting the React app in the foreground to see any errors..."
npm start &
FRONTEND_PID=$!

# Check if React app starts (more robust approach)
echo "Checking if React app starts..."
MAX_WAIT=90  # Increased timeout
counter=0
# Poll for React startup for first 10 seconds
while [ $counter -lt 10 ]; do
  sleep 1
  counter=$((counter + 1))
  echo "Waiting for React initial startup... ($counter seconds)"
  
  # Check if the process is still running
  if ! ps -p $FRONTEND_PID > /dev/null; then
    echo "ERROR: React process died. Checking npm-debug.log for errors..."
    [ -f npm-debug.log ] && cat npm-debug.log
    echo "Failed to start React app. Killing all processes..."
    pkill -f "ngrok" || true
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
  fi
done

# Now wait for port to be listening
counter=0
while ! nc -z localhost 3000 && [ $counter -lt $MAX_WAIT ]; do
  sleep 1
  counter=$((counter + 1))
  if [ $((counter % 5)) -eq 0 ]; then
    echo "Still waiting for React app to start on port 3000... ($counter seconds)"
    
    # Check if the process is still running
    if ! ps -p $FRONTEND_PID > /dev/null; then
      echo "ERROR: React process died while waiting. Checking for error logs..."
      [ -f npm-debug.log ] && cat npm-debug.log
      echo "Failed to start React app. Killing all processes..."
      pkill -f "ngrok" || true
      kill $BACKEND_PID 2>/dev/null || true
      exit 1
    fi
  fi
done

if ! nc -z localhost 3000; then
  echo "ERROR: React app failed to start within $MAX_WAIT seconds"
  # Show any possible error logs
  echo "Checking for error logs..."
  [ -f npm-debug.log ] && cat npm-debug.log
  
  pkill -f "ngrok" || true
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
  exit 1
fi

echo "React app is running on port 3000!"
cd ..

# Start ngrok for the frontend
echo "Starting ngrok tunnel for React frontend (port 3000)..."
ngrok http 3000 --log=stdout > ngrok_frontend.log 2>&1 &
FRONTEND_NGROK_PID=$!

# Wait for frontend ngrok to start
echo "Waiting for frontend ngrok to initialize..."
sleep 5

# Get the frontend URL
FRONTEND_URL=$(curl -s http://localhost:4041/api/tunnels | grep -o '"public_url":"https://[^"]*' | sed 's/"public_url":"//' | head -1)
if [ -z "$FRONTEND_URL" ]; then
  echo "ERROR: Failed to get frontend ngrok URL"
  pkill -f "ngrok" || true
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
  exit 1
fi

echo "Frontend URL: $FRONTEND_URL"

# Print instructions for BotFather
echo ""
echo "==================================================================="
echo "IMPORTANT INSTRUCTIONS FOR BOTFATHER:"
echo "==================================================================="
echo "1. Open Telegram and message @BotFather"
echo "2. Type /mybots and select your bot"
echo "3. Click 'Bot Settings' -> 'Menu Button' -> 'Configure Menu Button'"
echo "4. Select 'Add a menu button' -> 'Web App'"
echo "5. Set the 'Button Text' to 'Play Hokm'"
echo "6. Set the 'Web App URL' to: $FRONTEND_URL"
echo "7. Save your changes"
echo ""
echo "Your Mini App should now be accessible from your Telegram bot's menu button!"
echo "==================================================================="
echo ""
echo "Press Ctrl+C to stop all services."

# Setup trap to kill all processes on exit
trap "pkill -f 'ngrok' || true; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true; exit" INT TERM EXIT

# Keep script running
wait 