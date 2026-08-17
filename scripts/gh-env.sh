"""Функции для оболочки GitHub."""
#!/usr/bin/env bash
# gh-env.sh — настройка окружения для GitHub CLI

# Добавить GitHub CLI в PATH (Windows)
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    export PATH="$PATH:/c/Program Files/GitHub CLI"
fi

# Проверить авторизацию
if command -v gh &> /dev/null; then
    if ! gh auth status &> /dev/null; then
        echo "⚠️ GitHub CLI не авторизован. Выполните: gh auth login"
    fi
fi
