# Docker image settings
DOCKER_REPO := aorwall/moatless-tools
VERSION := $(shell git describe --tags --always --dirty)
DOCKER_TAG ?= latest

# Docker Compose settings
COMPOSE_FILE := docker-compose.yml
WORKER_COUNT ?= 2

.PHONY: help build push run stop clean logs scale test

help: ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Targets:'
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Initial setup of directories and environment
	@echo "Setting up environment..."
	mkdir -p data/moatless data/repos data/index_stores
	test -f .env || cp .env.docker .env
	@echo "Setup complete. Edit .env file to configure your environment."

build: ## Build Docker images
	@echo "Building Docker images with tag: $(DOCKER_TAG)"
	docker build -t $(DOCKER_REPO):$(DOCKER_TAG) .

build-nc: ## Build Docker images without cache
	@echo "Building Docker images without cache, tag: $(DOCKER_TAG)"
	docker build --no-cache -t $(DOCKER_REPO):$(DOCKER_TAG) .

push: ## Push Docker images to Docker Hub
	@echo "Pushing images to Docker Hub..."
	docker push $(DOCKER_REPO):$(DOCKER_TAG)

tag-version: ## Tag Docker image with git version
	@echo "Tagging with version: $(VERSION)"
	docker tag $(DOCKER_REPO):$(DOCKER_TAG) $(DOCKER_REPO):$(VERSION)
	docker push $(DOCKER_REPO):$(VERSION)

run: ## Start all services
	@echo "Starting services..."
	WORKER_COUNT=$(WORKER_COUNT) docker-compose up -d

dev: ## Start services in development mode (with logs)
	@echo "Starting services in development mode..."
	WORKER_COUNT=$(WORKER_COUNT) docker-compose up

stop: ## Stop all services
	@echo "Stopping services..."
	docker-compose down

clean: stop ## Stop services and clean up
	@echo "Cleaning up..."
	docker-compose down -v
	rm -rf data/moatless/* data/repos/* data/index_stores/*

logs: ## Show logs from all services
	docker-compose logs -f

scale: ## Scale workers (usage: make scale WORKER_COUNT=N)
	@echo "Scaling workers to $(WORKER_COUNT)..."
	docker-compose up -d --scale worker=$(WORKER_COUNT)

status: ## Show status of all services
	docker-compose ps

test: ## Run tests in Docker environment
	@echo "Running tests..."
	docker-compose run --rm api pytest

release: build tag-version push ## Build, tag with version, and push to Docker Hub

# Default target
all: build 