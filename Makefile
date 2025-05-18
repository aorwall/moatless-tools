# Docker image settings
DOCKER_REPO_UI := aorwall/moatless-ui
DOCKER_REPO_API := aorwall/moatless-api
VERSION := $(shell git describe --tags --always --dirty || echo latest)
DOCKER_TAG ?= latest
PLATFORMS := linux/amd64,linux/arm64

# Docker Compose settings
DOCKER_DIR := docker
API_DIR := moatless-api
API_DOCKERFILE := $(DOCKER_DIR)/Dockerfile.api
UI_DOCKERFILE := $(DOCKER_DIR)/Dockerfile.ui
COMPOSE_FILE := docker-compose.yml

# Moatless tools path
MOATLESS_TOOLS_PATH := /Users/albert/repos/moatless/moatless-tools/moatless

# Environment file
ENV_FILE ?= .env
ENV_NAME ?= default

.PHONY: help build build-ui build-api build-multiarch push run run-local dev stop logs status tag-version release list-envs restart-api

help: ## Show this help message
	@echo 'Usage: make -f Makefile.docker [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ''
	@echo 'Environment:'
	@echo '  ENV_NAME=<name>    Use a specific environment (.env.<name> file)'
	@echo '  ENV_FILE=<path>    Use a specific environment file (default: .env)'

build-ui: ## Build UI Docker image only
	@echo "Building UI Docker image with tag: $(DOCKER_TAG)"
	docker build -f $(UI_DOCKERFILE) -t $(DOCKER_REPO_UI):$(DOCKER_TAG) .

build-api: ## Build API Docker image only
	@echo "Building API Docker image with tag: $(DOCKER_TAG)"
	mkdir -p temp_build
	cp -r $(MOATLESS_TOOLS_PATH) temp_build/moatless
	docker build -f $(API_DOCKERFILE) \
		--build-arg TARGETPLATFORM=linux/$(shell uname -m | sed 's/x86_64/amd64/' | sed 's/arm64/arm64/') \
		-t $(DOCKER_REPO_API):$(DOCKER_TAG) \
		--no-cache .
	rm -rf temp_build

build: build-ui build-api ## Build both UI and API Docker images
	@echo "Building both UI and API Docker images with tag: $(DOCKER_TAG) completed"

build-multiarch: ## Build multi-architecture Docker images for both UI and API
	@echo "Building multi-architecture Docker images with tag: $(DOCKER_TAG)"
	mkdir -p temp_build
	cp -r $(MOATLESS_TOOLS_PATH) temp_build/moatless
	docker buildx build --platform $(PLATFORMS) -f $(UI_DOCKERFILE) \
		-t $(DOCKER_REPO_UI):$(DOCKER_TAG) \
		--push .
	docker buildx build --platform $(PLATFORMS) -f $(API_DOCKERFILE) \
		-t $(DOCKER_REPO_API):$(DOCKER_TAG) \
		--push .
	rm -rf temp_build

push: ## Push both Docker images to Docker Hub
	@echo "Pushing both UI and API images to Docker Hub..."
	docker push $(DOCKER_REPO_UI):$(DOCKER_TAG)
	docker push $(DOCKER_REPO_API):$(DOCKER_TAG)

run: ## Start all services with docker-compose
	@if [ "$(ENV_NAME)" != "default" ]; then \
		ENV_FILE=.env.$(ENV_NAME); \
	fi; \
	echo "Starting all services (UI and API) with environment: $${ENV_FILE}..."; \
	docker-compose --env-file $${ENV_FILE} -f $(COMPOSE_FILE) up -d

dev: ## Start services in development mode
	@if [ "$(ENV_NAME)" != "default" ]; then \
		ENV_FILE=.env.$(ENV_NAME); \
	fi; \
	echo "Starting services in development mode with environment: $${ENV_FILE}..."; \
	docker-compose --env-file $${ENV_FILE} -f $(COMPOSE_FILE) up -d

stop: ## Stop all services
	@if [ "$(ENV_NAME)" != "default" ]; then \
		ENV_FILE=.env.$(ENV_NAME); \
	fi; \
	echo "Stopping services..."; \
	docker-compose --env-file $${ENV_FILE} -f $(COMPOSE_FILE) down

restart-api: ## Restart only the API service
	@if [ "$(ENV_NAME)" != "default" ]; then \
		ENV_FILE=.env.$(ENV_NAME); \
	fi; \
	echo "Restarting API service with environment: $${ENV_FILE}..."; \
	docker-compose --env-file $${ENV_FILE} -f $(COMPOSE_FILE) restart api

logs: ## Show logs from all services
	@if [ "$(ENV_NAME)" != "default" ]; then \
		ENV_FILE=.env.$(ENV_NAME); \
	fi; \
	docker-compose --env-file $${ENV_FILE} -f $(COMPOSE_FILE) logs -f

status: ## Show status of all services
	@if [ "$(ENV_NAME)" != "default" ]; then \
		ENV_FILE=.env.$(ENV_NAME); \
	fi; \
	docker-compose --env-file $${ENV_FILE} -f $(COMPOSE_FILE) ps

list-envs: ## List available environment files
	@echo "Available environment files:"
	@ls -1 .env* 2>/dev/null || echo "No environment files found"

tag-version: ## Tag Docker images with git version
	@echo "Tagging with version: $(VERSION)"
	docker tag $(DOCKER_REPO_UI):$(DOCKER_TAG) $(DOCKER_REPO_UI):$(VERSION)
	docker push $(DOCKER_REPO_UI):$(VERSION)
	docker tag $(DOCKER_REPO_API):$(DOCKER_TAG) $(DOCKER_REPO_API):$(VERSION)
	docker push $(DOCKER_REPO_API):$(VERSION)

release: build tag-version push ## Build, tag with version, and push to Docker Hub

# Default target
all: build 