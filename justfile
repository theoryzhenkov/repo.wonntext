default:
    @just --list

# Project setup (run inside the flake devShell). `copier copy` has already wired direnv.
setup:
    ./setup.sh

# Install/sync dependencies into .venv
sync:
    uv sync

# Run the main module
run *args:
    uv run python -m wonntext {{args}}

# Train a masked-language WONN on a character corpus
train *args:
    uv run python -m wonntext.train {{args}}

# Run pytest
test *args:
    uv run pytest {{args}}

# Lint with ruff (pass --fix to auto-fix)
lint *args:
    uv run ruff check {{args}}

# Format with ruff
fmt:
    uv run ruff format

# Type-check with ty
typecheck:
    uv run ty check

# Lint + typecheck + test (CI-equivalent)
check: lint typecheck test
