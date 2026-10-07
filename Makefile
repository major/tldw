.PHONY: help install typecheck ty pyright test coverage check container-build container-up container-down

.DEFAULT_GOAL := check

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Install the project and dev dependencies
	uv sync --locked --all-extras --dev

ty: ## Type check with ty
	uv run ty check

pyright: ## Type check with pyright
	uv run pyright

typecheck: ty pyright ## Run all type checkers

test: ## Run tests with branch coverage
	uv run pytest

coverage: ## Run tests and write coverage.xml for Codecov
	uv run --locked pytest --cov-report=xml:coverage.xml

check: typecheck test ## Run all required checks

container-build: ## Build the container image with podman compose
	podman compose build

container-up: ## Start the container with podman compose
	podman compose up -d

container-down: ## Stop the container with podman compose
	podman compose down
