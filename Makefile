# Docker image settings
DOCKER_REPO_API := aorwall/moatless-api
VERSION := $(shell git describe --tags --always --dirty || echo latest)
DOCKER_TAG ?= latest
PLATFORMS := linux/amd64,linux/arm64

# Docker Compose settings
DOCKER_DIR := docker
API_DOCKERFILE := $(DOCKER_DIR)/Dockerfile.api
COMPOSE_FILE := docker-compose.yml

# Environment file
ENV_FILE ?= .env

.PHONY: help build-api build-multiarch run dev stop logs status restart-api

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ''
	@echo 'Environment:'
	@echo '  ENV_FILE=<file>    Use a specific environment file (default: .env)'

run: ## Start all services with docker-compose
	@if ! grep -q "^MOATLESS_DIR=" ${ENV_FILE} 2>/dev/null || [ -z "$$(grep "^MOATLESS_DIR=" ${ENV_FILE} 2>/dev/null | cut -d'=' -f2)" ]; then \
		echo "Error: MOATLESS_DIR is not set in ${ENV_FILE}"; \
		echo "Please add MOATLESS_DIR=/path/to/your/moatless/data to ${ENV_FILE}"; \
		echo "This directory will store Moatless configuration and trajectory data."; \
		exit 1; \
	fi; \
	$(eval DETECTED_ARCH := $(shell uname -m | sed 's/x86_64/amd64/')) \
	echo "Starting all services for $(DETECTED_ARCH) with environment: ${ENV_FILE}..."; \
	DOCKER_DEFAULT_PLATFORM=linux/$(DETECTED_ARCH) docker-compose --env-file ${ENV_FILE} -f $(COMPOSE_FILE) up -d

stop: ## Stop all services
	echo "Stopping services..."; \
	docker-compose --env-file ${ENV_FILE} -f $(COMPOSE_FILE) down

restart-api: ## Restart only the API service
	echo "Restarting API service with environment: ${ENV_FILE}..."; \
	docker-compose --env-file ${ENV_FILE} -f $(COMPOSE_FILE) restart api

logs: ## Show logs from all services
	docker-compose --env-file ${ENV_FILE} -f $(COMPOSE_FILE) logs -f

status: ## Show status of all services
	docker-compose --env-file ${ENV_FILE} -f $(COMPOSE_FILE) ps


build-api: ## Build API Docker image only
	$(eval TARGETPLATFORM := linux/$(shell uname -m | sed 's/x86_64/amd64/'))
	@echo "Building API Docker image with platform: $(TARGETPLATFORM) and tag: $(DOCKER_TAG)"
	docker build -f $(API_DOCKERFILE) \
		--platform $(TARGETPLATFORM) \
		--build-arg TARGETPLATFORM=$(TARGETPLATFORM) \
		-t $(DOCKER_REPO_API):$(DOCKER_TAG) \
		--no-cache .

build-multiarch: ## Build multi-architecture Docker images for both UI and API
	@echo "Building multi-architecture Docker images with tag: $(DOCKER_TAG)"
	docker buildx build --platform $(PLATFORMS) -f $(API_DOCKERFILE) \
		-t $(DOCKER_REPO_API):$(DOCKER_TAG) \
		--push .