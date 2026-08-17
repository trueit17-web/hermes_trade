"""Утилиты: шифрование API ключей."""
import base64
import logging
import os

from cryptography.fernet import Fernet, InvalidToken

from src.config import settings

logger = logging.getLogger(__name__)

# Глобальный экземпляр шифровщика
_crypto: Optional[Fernet] = None


def get_crypto() -> Fernet:
    """Получить или создать Fernet шифровщик."""
    global _crypto
    if _crypto is not None:
        return _crypto

    key = settings.encryption_key
    if not key:
        # Генерация ключа для разработки (не для production!)
        key = base64.urlsafe_b64encode(os.urandom(32))
        logger.warning("ENCRYPTION_KEY не установлен, используем временный ключ")

    _crypto = Fernet(key)
    return _crypto


def encrypt_api_key(api_key: str) -> str:
    """Зашифровать API ключ."""
    return get_crypto().encrypt(api_key.encode()).decode()


def decrypt_api_key(encrypted: str) -> str:
    """Расшифровать API ключ."""
    try:
        return get_crypto().decrypt(encrypted.encode()).decode()
    except InvalidToken:
        logger.error("Не удалось расшифровать API ключ — возможно, изменён ключ шифрования")
        raise


def encrypt_secret(secret: str) -> str:
    """Зашифровать секрет."""
    return get_crypto().encrypt(secret.encode()).decode()


def decrypt_secret(encrypted: str) -> str:
    """Расшифровать секрет."""
    return decrypt_api_key(encrypted)


def generate_encryption_key() -> str:
    """Сгенерировать новый ключ шифрования (для первого запуска)."""
    return Fernet.generate_key().decode()
