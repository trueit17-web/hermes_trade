# CryptoBot Pro — заметки для Claude

## Продовое окружение

- Прод-сервер: `/opt/cryptobot` (root), дашборд на `https://admininfou.pro`
  (логин tony/Qwerty1411).
- **Прод — Docker Compose** (`docker-compose.yml` в корне репозитория),
  НЕ голый процесс на хосте. Сервисы: `bot` (сам трейдер+веб-панель,
  контейнер `cryptobot-bot-1`), `db` (`postgres:16-alpine`,
  `cryptobot-db-1`), `redis` (`redis:7-alpine`, `cryptobot-redis-1`),
  `deploy-agent` (`cryptobot-deploy-agent`, см. ниже).
- **Версия Python в проде — та, что в `Dockerfile` (`FROM python:3.12-slim`)
  — это ЕДИНСТВЕННЫЙ источник истины.** На хосте сервера может существовать
  свой venv (например, `/opt/cryptobot/venv`) для ручных операций — это
  СОВЕРШЕННО ОТДЕЛЬНОЕ окружение, не то, что реально исполняет код бота.
  Реальный инцидент: приняли ошибки в этом хостовом venv (Python 3.10,
  устаревший/неполный набор пакетов) за характеристики прода и чуть не
  переписали код под несуществующее ограничение (заменяли `datetime.UTC`
  на `datetime.timezone.utc` по всему коду) — прежде чем менять код под
  "требования прода", проверяйте `Dockerfile`, а если сомневаетесь, что он
  соответствует реально запущенному образу — просите пользователя выполнить
  `docker exec cryptobot-bot-1 python3 --version`.
- **`pyproject.toml` → `[tool.ruff] target-version` должен соответствовать
  базовому образу в `Dockerfile`** (сейчас `py312`), не хостовому venv.
- **Деплой — через `deploy-agent`, НЕ через ручной `git pull` + рестарт на
  хосте.** В дашборде есть кнопка «Редеплой» — она бьёт в
  `cryptobot-deploy-agent` (`scripts/deploy_agent.py`), который сам
  выполняет:
  ```
  git pull origin main
  docker compose build bot
  docker compose up -d bot
  docker compose exec -T bot alembic upgrade head
  ```
  `docker-compose.yml` монтирует как volume только `./data` — папка `src/`
  копируется в образ ПРИ СБОРКЕ (`COPY src/ ./src/` в Dockerfile). Значит
  простой `git pull` на хосте + рестарт контейнера (без `docker compose
  build`) НЕ применит новый код вообще — контейнер продолжит работать на
  старом образе. Всегда напоминайте пользователю: либо нажать «Редеплой» в
  дашборде, либо руками выполнить все 4 команды выше по порядку (build
  строго до up, alembic строго после up — на новом образе).
- `trading_mode=real` + `use_exchange_sandbox=true` — бот НАМЕРЕННО торгует
  на демо-счету Bybit (`api-demo.bybit.com`), это осознанное решение
  пользователя, а не баг. Не поднимать тревогу из-за упоминаний
  `api-demo.bybit.com` в логах/ошибках баланса.

## Рабочий процесс при фиксах

1. Найти root cause в коде (не патчить симптом). Если гипотеза о причине
   опирается на характеристики окружения (версия Python, установленные
   пакеты и т.п.) — сверяйте их с `Dockerfile`/`docker-compose.yml`, а не с
   тем, что видно в произвольной shell-сессии на хосте сервера (см. выше).
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
6. Напомнить пользователю про редеплой — см. пункт про deploy-agent выше.
   Просто "git pull + рестарт" НЕДОСТАТОЧНО для этого проекта.

## `pandas-ta` в requirements.txt

`pandas-ta>=0.4.67b0` требует Python 3.12+ — в песочнице (Python 3.11)
поэтому не ставится, тесты используют самодельный стаб-пакет вместо него
(см. `TestFeatureEngine` failures выше). В прод-образе (Python 3.12, см.
`Dockerfile`) он ставится нормально при `pip install -r requirements.txt`
на этапе сборки — трогать его в проде отдельно не нужно.
