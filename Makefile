.PHONY: help install typecheck ty test check version container-build container-up container-down

.DEFAULT_GOAL := check

# Build identity stamped into the container image. The values are recomputed
# on every make invocation so a rebuild always picks up the current HEAD and
# wall-clock time. Falling back to "unknown" keeps the build working when the
# command runs outside a git checkout (for example inside a tarball build).
GIT_SHA = $(shell git rev-parse HEAD 2>/dev/null || echo unknown)
BUILD_TIME = $(shell date -u +%FT%TZ)

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

install: ## Install the project and dev dependencies
	uv sync --locked --all-extras --dev

ty: ## Type check with ty
	uv run ty check

typecheck: ty ## Run all type checkers

test: ## Run tests with branch coverage
	uv run pytest

check: typecheck test ## Run all required checks

version: ## Print the build identity that would be baked into the next image
	@echo "GIT_SHA=$(GIT_SHA)"
	@echo "BUILD_TIME=$(BUILD_TIME)"

container-build: ## Build the container image with podman compose
	podman compose build \
		--build-arg GIT_SHA=$(GIT_SHA) \
		--build-arg BUILD_TIME=$(BUILD_TIME)

container-up: ## Start the container with podman compose
	podman compose up -d

container-down: ## Stop the container with podman compose
	podman compose down
