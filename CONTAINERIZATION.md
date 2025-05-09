# Containerization Architecture

This document outlines the containerization methodology used in the Hokm project, explaining the Docker setup, services, and networking architecture.

## Overview

The project uses Docker and Docker Compose to containerize multiple services:
- Backend (Node.js)
- Frontend (React)
- Nginx (Reverse Proxy)
- Ngrok (Tunnel Service)

## Service Architecture

### 1. Backend Service
- **Base Image**: Node.js 18 Alpine
- **Port**: 5001
- **Environment Variables**:
  - `BOT_TOKEN`: Telegram bot token
  - `PORT`: Service port (5001)
  - `ENV`: Environment setting
- **Volumes**:
  - Application code mounted at `/app`
  - Models directory mounted at `/app/models`
- **Networking**: Connected to `hokm-network`

### 2. Frontend Service
- **Base Image**: Node.js 18 Alpine
- **Port**: 3000
- **Environment Variables**:
  - `REACT_APP_API_URL`: API endpoint configuration
  - `REACT_APP_TELEGRAM_BOT_USERNAME`: Telegram bot username
  - `DANGEROUSLY_DISABLE_HOST_CHECK`: Development host check bypass
  - `WDS_SOCKET_HOST` & `WDS_SOCKET_PORT`: WebSocket configuration
  - `HOST`: Binding address
- **Volumes**:
  - Application code mounted at `/app`
  - Node modules volume for dependency isolation
- **Dependencies**: Requires backend service
- **Networking**: Connected to `hokm-network`

### 3. Nginx Service
- **Base Image**: Nginx Alpine
- **Port**: 8080 (mapped to container port 80)
- **Volumes**:
  - Custom nginx configuration mounted at `/etc/nginx/conf.d/default.conf`
- **Dependencies**: Requires both frontend and backend services
- **Networking**: Connected to `hokm-network`

### 4. Ngrok Service
- **Base Image**: Official Ngrok image
- **Port**: 4040 (Ngrok web interface)
- **Environment Variables**:
  - `NGROK_AUTHTOKEN`: Authentication token for Ngrok
- **Command**: Configures HTTP tunnel to nginx service
- **Dependencies**: Requires nginx service
- **Networking**: Connected to `hokm-network`

## Backend-Frontend Connection Architecture

The backend and frontend services are connected through multiple layers:

1. **Direct Container Communication**:
   - Both services are connected to the same `hokm-network` bridge network
   - Services can communicate using their service names as hostnames
   - Frontend can reach backend at `http://backend:5001`

2. **Nginx Reverse Proxy**:
   - Acts as an intermediary between frontend and backend
   - Routes API requests from frontend to backend
   - Handles static file serving for the frontend
   - Provides a single entry point for external access

3. **Environment Configuration**:
   - Frontend is configured with `REACT_APP_API_URL=/api`
   - This URL is proxied through Nginx to the backend service
   - API requests from frontend are automatically routed to the backend

4. **Development Mode**:
   - Frontend runs on port 3000 with hot-reloading
   - Backend runs on port 5001
   - Nginx handles the routing between these services
   - WebSocket connections are properly configured for development

5. **Production Mode**:
   - All traffic is routed through Nginx
   - Frontend static files are served by Nginx
   - API requests are proxied to the backend
   - Single domain access for both services

## Network Architecture

The project uses a custom bridge network called `hokm-network` that enables:
- Inter-service communication
- Isolated network environment
- Controlled service discovery

## Volume Management

The project implements several volume mounts for:
- Development hot-reloading
- Persistent data storage
- Dependency isolation
- Configuration management

## Environment Configuration

Environment variables are managed through:
- Docker Compose environment section
- `.env` file for sensitive data
- Service-specific environment configurations

## Development Workflow

1. **Local Development**:
   - Services run in development mode with hot-reloading
   - Code changes are reflected immediately
   - Debugging ports are exposed

2. **Production Considerations**:
   - Services are configured for production use
   - Proper security measures are implemented
   - Performance optimizations are in place

## Security Considerations

1. **Network Security**:
   - Isolated network environment
   - Controlled service communication
   - Port exposure management

2. **Environment Variables**:
   - Sensitive data handled through environment variables
   - No hardcoded credentials
   - Secure configuration management

## Best Practices Implemented

1. **Container Optimization**:
   - Alpine-based images for smaller footprint
   - Multi-stage builds where applicable
   - Proper layer caching

2. **Service Management**:
   - Automatic restart policies
   - Dependency management
   - Health checks

3. **Development Experience**:
   - Hot-reloading support
   - Volume mounts for development
   - Debugging capabilities

## Usage

To start the containerized environment:

```bash
docker-compose up
```

To stop the environment:

```bash
docker-compose down
```

For development with rebuild:

```bash
docker-compose up --build
```

## Troubleshooting

Common issues and solutions:

1. **Port Conflicts**:
   - Check for existing services using required ports
   - Modify port mappings in docker-compose.yml if needed

2. **Volume Mount Issues**:
   - Ensure proper file permissions
   - Check volume mount paths

3. **Network Connectivity**:
   - Verify network creation
   - Check service dependencies
   - Ensure proper service discovery

## Future Improvements

Potential enhancements to consider:

1. **Container Optimization**:
   - Implement multi-stage builds
   - Optimize image sizes
   - Add health checks

2. **Security Enhancements**:
   - Implement secrets management
   - Add network policies
   - Enhance access controls

3. **Monitoring**:
   - Add logging solutions
   - Implement metrics collection
   - Set up monitoring dashboards 