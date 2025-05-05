#!/bin/bash

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

echo "Stopping Docker containers..."
$DOCKER_COMPOSE_CMD down

echo "Containers stopped successfully." 