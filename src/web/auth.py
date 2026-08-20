"""Простая cookie-based авторизация для веб-панели (единственный админ-пользователь)."""
import hmac
import secrets
import time

from src.config import settings

SESSION_COOKIE_NAME = "session_token"
SESSION_TTL_SECONDS = 7 * 24 * 3600

# Токены сессий хранятся в памяти процесса — этого достаточно для одного
# инстанса бота; при перезапуске все сессии слетают, пользователю нужно
# войти заново.
_sessions: dict[str, float] = {}


def authenticate(username: str, password: str) -> bool:
    """Сверить логин/пароль с настройками (constant-time сравнение)."""
    valid_user = hmac.compare_digest((username or "").encode(), settings.web_admin_username.encode())
    valid_pass = hmac.compare_digest((password or "").encode(), settings.web_admin_password.encode())
    return valid_user and valid_pass


def create_session() -> str:
    """Создать новую сессию, вернуть токен для cookie."""
    token = secrets.token_urlsafe(32)
    _sessions[token] = time.time() + SESSION_TTL_SECONDS
    return token


def verify_session(token: str | None) -> bool:
    """Проверить, что токен существует и не истёк."""
    if not token:
        return False
    expiry = _sessions.get(token)
    if expiry is None:
        return False
    if expiry < time.time():
        _sessions.pop(token, None)
        return False
    return True


def revoke_session(token: str | None):
    """Удалить сессию (logout)."""
    if token:
        _sessions.pop(token, None)
