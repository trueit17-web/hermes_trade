# CryptoBot Pro — заметки для Claude

## Продовое окружение

- Прод-сервер: `/opt/cryptobot` (root), за nginx, дашборд на
  `https://admininfou.pro` (логин tony/Qwerty1411).
- **venv на проде — Python 3.10.** Дев/тестовое окружение в песочнице Claude
  обычно Python 3.11+ — это НЕ то же самое. Прежде чем использовать любой
  синтаксис/stdlib-фичу новее 3.10 (например `datetime.UTC`, появившийся
  только в 3.11, `tomllib`, `except*`/ExceptionGroup, `typing.Self` и т.п.),
  проверьте, что она доступна в 3.10 — иначе код упадёт ImportError/
  SyntaxError на реальном сервере, хотя все тесты в песочнице пройдут.
  Реальный инцидент: `from datetime import UTC` в `src/utils/timeutils.py`
  (и ещё 6 файлах) ронял импорт `src.db.models` при любой попытке запустить
  alembic или сам бот на проде — молча работало только потому, что уже
  запущенный процесс не перечитывал код с диска.
- `pyproject.toml` → `[tool.ruff] target-version` должен соответствовать
  РЕАЛЬНОЙ версии на проде (сейчас `py310`), а не последней/удобной. Ruff с
  неверным target-version активно РЕКОМЕНДУЕТ (правило UP017 и другие UP*)
  писать код, несовместимый с продом — именно так возник инцидент выше.
  Если вдруг сервер обновят на более новый Python — обновите тут и
  перепроверьте: `python3 --version` внутри `/opt/cryptobot/venv`.
- Деплой — ручной: `git push` сам по себе ничего не обновляет на сервере.
  После каждого фикса пользователю нужно на сервере: `git pull` в
  `/opt/cryptobot`, затем рестарт процесса бота. Всегда напоминайте об этом.
- `trading_mode=real` + `use_exchange_sandbox=true` — бот НАМЕРЕННО торгует
  на демо-счету Bybit (`api-demo.bybit.com`), это осознанное решение
  пользователя, а не баг. Не поднимать тревогу из-за упоминаний
  `api-demo.bybit.com` в логах/ошибках баланса.

## Рабочий процесс при фиксах

1. Найти root cause в коде (не патчить симптом).
2. Написать/обновить тесты в `tests/test_all.py`.
3. `PYTHONPATH=. /tmp/hermes_venv/bin/pytest tests/test_all.py -q` — должно
   быть зелено, кроме 3 заведомо известных pre-existing падений из-за
   отсутствия реального `pandas-ta` в песочнице (стаб-пакет):
   `TestFeatureEngine::test_compute_macd`, `test_compute_rsi`,
   `test_extract_ml_features`.
4. `/tmp/hermes_venv/bin/ruff check src/ --output-format=github` — должно
   быть чисто. Ruff проверяется только для `src/`, не для `tests/`
   (в `tests/test_all.py` есть давние неисправленные I001/F401).
5. Коммит в `claude/repo-analysis-dev-plan-ll6lk2`, push, fast-forward-merge
   в `main`, push `main`.
6. Напомнить пользователю про `git pull` + рестарт на проде — без этого
   исправление не применится.

## `pandas-ta` в requirements.txt

`pandas-ta>=0.4.67b0` требует Python 3.12+ и не ставится ни в песочнице
(Python 3.11), ни, судя по всему, в чистом виде на проде (Python 3.10) —
там он, видимо, стоит из другого источника/версии. Не пытайтесь
переустанавливать `pandas-ta` на проде через `pip install -r
requirements.txt` "заодно" — если он уже работает, лишний раз его трогать
не нужно; ставьте только конкретный недостающий пакет (например
`pip install "alembic>=1.13.0"`), а не весь requirements.txt целиком.
