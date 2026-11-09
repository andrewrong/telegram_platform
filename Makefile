# Telegram ASR Bot - Docker Management

.PHONY: help build up down logs shell clean prune

# Default target
help:
	@echo "Telegram ASR Bot - Docker Management"
	@echo ""
	@echo "Usage:"
	@echo "  make build    - Build Docker image"
	@echo "  make up       - Start bot with docker-compose"
	@echo "  make down     - Stop bot"
	@echo "  make restart  - Restart bot (build + down + up)"
	@echo "  make logs     - View logs"
	@echo "  make shell    - Open shell in container"
	@echo "  make clean    - Remove container and volumes"
	@echo "  make prune    - Clean up unused Docker resources"

# Build Docker image (default: latest)
build:
	@echo "Building Docker image..."
	docker build --target production -t telegram-asr-bot:latest .
	@echo "Build complete: telegram-asr-bot:latest"

# Build with version tag
build-version:
	@echo "Building Docker image version $(VERSION)..."
	docker build --target production -t telegram-asr-bot:$(VERSION) -t telegram-asr-bot:latest .
	@echo "Build complete: telegram-asr-bot:$(VERSION) and telegram-asr-bot:latest"

# Start services
up:
	@echo "Starting bot..."
	docker-compose up -d
	@echo "Bot started. Use 'make logs' to view logs."

# Stop services
down:
	@echo "Stopping bot..."
	docker-compose down

# Restart (build + stop + start)
restart: build down up

# View logs
logs:
	docker-compose logs -f telegram-bot

# Open shell in running container
shell:
	docker-compose exec telegram-bot /bin/sh

# Clean up containers and volumes
clean:
	@echo "Cleaning up..."
	docker-compose down -v
	docker rm -f telegram-asr-bot 2>/dev/null || true

# Prune unused Docker resources
prune:
	@echo "Pruning unused Docker resources..."
	docker system prune -f
	docker volume prune -f
