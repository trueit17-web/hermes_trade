"""Git credential helper — извлечение GitHub токена из git credentials."""
import os
import sys


def get_token() -> str:
    """Получить GitHub токен из git credential store."""
    credential_file = os.path.expanduser("~/.git-credentials")
    if not os.path.exists(credential_file):
        return ""

    with open(credential_file, "r") as f:
        for line in f:
            line = line.strip()
            if "github.com" in line and "x-oauth-basic" not in line:
                # Формат: https://user:token@github.com
                parts = line.split("@")[0].split("://")[1].split(":")
                if len(parts) == 2:
                    return parts[1]
    return ""


if __name__ == "__main__":
    token = get_token()
    if token:
        print(token)
    else:
        sys.exit(1)
