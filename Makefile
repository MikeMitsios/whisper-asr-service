.DEFAULT_GOAL := help
.PHONY: help install install-gpu dev test test-all test-slow lint format eval viz chart docker docker-gpu docker-down clean

PYTHON ?= python
CONFIG ?= configs/cpu.yml

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package and dev/eval/viz extras (CPU torch)
	$(PYTHON) -m pip install -r requirements-cpu.txt
	$(PYTHON) -m pip install -e ".[dev,eval,viz]" --no-deps

install-gpu: ## Install with CUDA torch wheels
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e ".[dev,eval,viz]" --no-deps

dev: ## Run the API locally with hot reload (CPU config)
	CONFIG_PATH=$(CONFIG) $(PYTHON) -m uvicorn app.main:app --reload

test: ## Fast tests: offline, no weights, no server
	$(PYTHON) -m pytest

test-slow: ## Real-model tests (downloads whisper-tiny, CPU)
	$(PYTHON) -m pytest -m slow

test-all: ## Every test, including those needing Redis
	$(PYTHON) -m pytest -m ""

lint: ## Lint
	$(PYTHON) -m ruff check .

format: ## Auto-fix what ruff can
	$(PYTHON) -m ruff check . --fix

eval: ## Evaluate on Emilia-Dataset and append to evaluation_results.csv
	$(PYTHON) scripts/evaluation.py --config $(CONFIG) --num-samples 50

viz: ## Build the interactive Bokeh dashboard
	$(PYTHON) scripts/visualize.py

chart: ## Regenerate the static results SVG used by the README
	$(PYTHON) scripts/make_results_chart.py

docker: ## Build and start the CPU stack
	docker compose up -d --build

docker-gpu: ## Build and start the GPU stack
	docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build

docker-down: ## Stop the stack
	docker compose down

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .ruff_cache build dist src/*.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
