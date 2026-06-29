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

# Compile the Typst preprint to PDF
paper-compile:
    nix run nixpkgs#typst -- compile paper/main.typ

# Live-preview the Typst preprint (open http://127.0.0.1:23626 in a browser)
paper-preview:
    nix run nixpkgs#tinymist -- preview paper/main.typ
