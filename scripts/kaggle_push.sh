#!/usr/bin/env bash
# Pushes (and runs) a Kaggle kernel from kaggle/<name>/ under the account of the local Kaggle CLI.
# The metadata in the repo says "{user}" instead of the username, so no account name is committed;
# it is filled in a temporary copy. Usage: scripts/kaggle_push.sh kaggle/panda_mosaicos
set -euo pipefail
src="${1:?usage: scripts/kaggle_push.sh kaggle/<kernel-dir>}"
[ -f "$src/kernel-metadata.json" ] || { echo "no kernel-metadata.json in $src" >&2; exit 1; }
kaggle_cli() { if command -v kaggle >/dev/null; then kaggle "$@"; else uvx --with six kaggle "$@"; fi; }
user="${KAGGLE_USERNAME:-$(kaggle_cli config view | sed -n 's/^- username: //p')}"
[ -n "$user" ] || { echo "Kaggle username not found (set KAGGLE_USERNAME)" >&2; exit 1; }
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp -r "$src"/. "$tmp"/
sed -i "s/{user}/$user/g" "$tmp/kernel-metadata.json"
kaggle_cli kernels push -p "$tmp"
