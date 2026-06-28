#!/usr/bin/env bash
# Project setup: requires the flake devShell to be active (age, git, sops, jj, uv in PATH).
# Run from inside the devShell. `copier copy` has already created .env/.envrc and run direnv allow.
set -euo pipefail

# -- tool check --
missing=()
for tool in age-keygen git sops jj uv; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [ ${#missing[@]} -gt 0 ]; then
    echo "ERROR: missing tools: ${missing[*]}"
    echo "Enter the devShell with 'nix develop' (or step into the directory so direnv activates the flake)."
    exit 1
fi

# -- age key --
SOPS_UPDATED=0
if [ ! -f .age-key ]; then
    age-keygen -o .age-key 2>&1
    PUBLIC_KEY=$(age-keygen -y .age-key)
    sed -i "s|REPLACE_WITH_AGE_PUBLIC_KEY|$PUBLIC_KEY|" .sops.yaml
    SOPS_UPDATED=1
    echo "Generated .age-key and updated .sops.yaml"
else
    echo ".age-key already exists, skipping"
fi

# -- jj --
if [ ! -d .jj ]; then
    jj git init --colocate
    echo "Initialized colocated jj repository"
else
    echo "jj repository already exists, skipping"
fi

# -- uv --
uv sync
echo "Synced Python dependencies into .venv"

echo "Done."

if [ "$SOPS_UPDATED" -eq 1 ]; then
    echo
    echo "WARNING: .sops.yaml has been updated with your age public key."
    echo "         Commit this change: jj desc -m 'chore: initial project setup'"
fi
