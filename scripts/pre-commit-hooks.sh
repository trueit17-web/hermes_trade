# Скрипты для оптимизации workflow
# Используются GitHub Actions и pre-commit

#!/usr/bin/env bash
# pre-commit-hooks.sh — run линтеры и тесты перед коммитом

set -e

echo "🔍 Running linters..."

# Python линтер
if command -v ruff &> /dev/null; then
    ruff check src/ tests/ --fix
    ruff format src/ tests/
elif command -v flake8 &> /dev/null; then
    flake8 src/ tests/ --max-line-length=120 --exclude=__pycache__
fi

# Python форматтер (black)
if command -v black &> /dev/null; then
    black --check src/ tests/
fi

echo "✅ Linting passed"

echo "🧪 Running tests..."

python -m pytest tests/ --tb=short -q || echo "Tests failed (non-blocking for pre-commit)"

echo "✅ Pre-commit checks done"
