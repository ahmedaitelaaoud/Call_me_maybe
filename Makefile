PYTHON = uv run python
MODULE = -m src

.PHONY: install run debug clean lint lint-strict

install:
	@echo "Installing dependencies using uv..."
	uv sync

run:
	@echo "Running the main pipeline..."
	$(PYTHON) $(MODULE)

debug:
	@echo "Running in debug mode (pdb)..."
	$(PYTHON) -m pdb $(MODULE)

clean:
	@echo "Cleaning Python cache files..."
	rm -rf .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

lint:
	@echo "Running standard linting (flake8 & mypy)..."
	uv run flake8 src
	uv run mypy --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs src

lint-strict:
	@echo "Running strict linting..."
	uv run flake8 src
	uv run mypy --strict src
