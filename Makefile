NAME = src

install:
	@uv sync

run:
	@uv run python -m $(NAME)

lint:
	@flake8
	@mypy . --warn-return-any --warn-unused-ignores --ignores-missing-imports --disalow-untyped-defs

lint-strict:
	@mypy . --strict

clean:
	@rm -rf */__pycache__ */.mypy_cashe __paycache__

debug:
	@uv run python -m pdb -m $(NAME)

setup:
	@echo "Setting up goinfre environment..."

	@GOINFRE_BASE="/home/$$USER/goinfre"; \
	UV_CACHE="$$GOINFRE_BASE/uv"; \
	VENV_DIR="$$GOINFRE_BASE/.venv"; \
	CACHE_LINK="$$HOME/.cache/uv"; \
	PROJECT_VENV_LINK=".venv"; \
	\
	mkdir -p "$$UV_CACHE"; \
	mkdir -p "$$VENV_DIR"; \
	mkdir -p "$$HOME/.cache"; \
	\
	if [ -d "$$CACHE_LINK" ] && [ ! -L "$$CACHE_LINK" ]; then \
		echo "Removing existing uv cache directory..."; \
		rm -rf "$$CACHE_LINK"; \
	fi; \
	\
	if [ -L "$$CACHE_LINK" ]; then \
		echo "Removing existing uv cache symlink..."; \
		rm -f "$$CACHE_LINK"; \
	fi; \
	\
	ln -s "$$UV_CACHE" "$$CACHE_LINK"; \
	echo "Linked uv cache -> $$UV_CACHE"; \
	\
	if [ -L "$$PROJECT_VENV_LINK" ] || [ -d "$$PROJECT_VENV_LINK" ]; then \
		echo "Removing existing project .venv..."; \
		rm -rf "$$PROJECT_VENV_LINK"; \
	fi; \
	\
	ln -s "$$VENV_DIR" "$$PROJECT_VENV_LINK"; \
	echo "Linked project .venv -> $$VENV_DIR"; \
	\
	echo "Done! Now run:"; \
	echo "  make install"
