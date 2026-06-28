# Agent Instructions

## Project

`wonntext` adapts the Sudoku WONN experiment to a 1-D masked-language-model setting.
The four intentional changes are documented in `README.md`.

## Conventions

- Use `jj` for version control. See `jj log` for history.
- Use `just` for task running. See `just --list` for available commands.
- Secrets are managed with `sops` + `age`. Never commit `.env` or `.age-key`.
- Documentation follows the [SPECial](https://the-o-space.github.io/special/) standard.

## Python

- Package manager: `uv`. Never use `pip`, `poetry`, or ad-hoc venvs.
- Layout: `src/wonntext/`. Tests: `tests/`.
- Lint + format: `ruff` (`just lint`, `just fmt`).
- Type-check: `ty` (`just typecheck`).
- Test: `pytest` (`just test`).
- Run everything: `just check`.
- For one-off scripts, prefer `uv run --with <pkg> python ...` over installing into the project.
