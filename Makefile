# NeuroChain: one command per result. Run `make help` for the list.
.DEFAULT_GOAL := help
.PHONY: help setup data data-status data-describe mearec u0 u1 u1-key u2 u2-key u0-key readme test test-fast lint format clean

help: ## Show this help
	@grep -E '^[a-zA-Z0-9_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Create the environment and install the git hooks
	uv sync
	uv run pre-commit install

data: ## Download and verify every dataset that can be scripted
	uv run neurochain-data fetch

data-status: ## Show which dataset files are on disk and verified
	uv run neurochain-data status

data-describe: ## Print what the Nason and NoMAD files contain (subject, channels, sessions)
	uv run python -m neurochain.data.nason
	uv run python -m neurochain.data.nomad

u0: ## U0: fetch Neuralink data, write both figures, the metrics and the README table
	uv run neurochain-data fetch neuralink --strict
	uv run python scripts/u0_figures.py
	uv run python scripts/update_readme.py
	uv run python scripts/u0_key_number.py

u0-key: ## U0: print the key numbers (add ARGS="--threshold-sigma 3.5" to change a parameter)
	uv run python scripts/u0_key_number.py $(ARGS)

u1: ## U1: electrode and front-end figures, metrics, netlists and the README table
	uv run python scripts/u1_figures.py --strict
	uv run python scripts/update_readme.py
	uv run python scripts/u1_key_number.py

u1-key: ## U1: print the key numbers (add ARGS="--bias-scale 0.5" to change a parameter)
	uv run python scripts/u1_key_number.py $(ARGS)

mearec: ## Generate the MEArec ground-truth recording (about a minute; needs a C compiler)
	uv run --group sim python scripts/make_mearec.py

u2: ## U2: ADC and feature figures, metrics and the README table (needs `make mearec` and Nason data)
	uv run python scripts/u2_figures.py --strict
	uv run python scripts/update_readme.py
	uv run python scripts/u2_key_number.py

u2-key: ## U2: print the key numbers (add ARGS="--bits 6" to change a setting)
	uv run python scripts/u2_key_number.py $(ARGS)

readme: ## Regenerate the README results tables from results/*.json
	uv run python scripts/update_readme.py

test: ## Run every test (real-data tests skip if the data is absent)
	uv run pytest --cov=neurochain --cov-report=term-missing:skip-covered

test-fast: ## Run only the tests that need no downloaded data
	uv run pytest -m "not data"

lint: ## Lint, check formatting, type-check, and check the README is in sync
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy
	uv run python scripts/update_readme.py --check

format: ## Auto-format and apply safe lint fixes
	uv run ruff format .
	uv run ruff check --fix .

clean: ## Remove caches (never touches data/)
	rm -rf results/cache .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov
