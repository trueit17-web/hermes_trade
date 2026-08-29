#!/usr/bin/env python3
"""
Деплой-агент CryptoBot Pro: маленький HTTP-сервис ВНЕ контейнера бота,
который по запросу с дашборда выполняет обновление — `git pull` +
`docker compose build` + `docker compose up -d` + `alembic upgrade head`.

Специально вынесен в отдельный контейнер (см. Dockerfile.deploy-agent,
docker-compose.yml, сервис "deploy-agent"), а не встроен в сам бот: только
у ЭТОГО минимального сервиса есть доступ к docker.sock хоста (нужен, чтобы
управлять соседними контейнерами). Сам бот с его большой веб-панелью
(парсинг Telegram-сообщений, LLM-фолбэки, десятки эндпоинтов) — гораздо
более широкая поверхность атаки; дать ЕЙ доступ к docker.sock означало бы,
что любая уязвимость в дашборде даёт root на всём сервере. Здесь же —
единственная привилегированная операция (POST /deploy), защищённая общим
секретом (DEPLOY_AGENT_TOKEN), и никакой сетевой поверхности снаружи
docker-compose сети (сервис не публикует порт наружу).

Использует только стандартную библиотеку Python — намеренно, чтобы не
тащить лишние зависимости в единственный сервис с доступом к docker.sock.

Переменные окружения (все, кроме DEPLOY_AGENT_TOKEN, необязательны):
    DEPLOY_AGENT_TOKEN    — общий секрет (ОБЯЗАТЕЛЕН, сервис откажется
                            стартовать без него)
    DEPLOY_AGENT_HOST     — интерфейс для listen (по умолчанию 0.0.0.0 —
                            это ок: сервис не публикует порт наружу в
                            docker-compose.yml, виден только внутри сети)
    DEPLOY_AGENT_PORT     — порт (по умолчанию 8091)
    DEPLOY_REPO_DIR       — путь к репозиторию (по умолчанию /opt/cryptobot;
                            ДОЛЖЕН совпадать с реальным путём на ХОСТЕ —
                            см. комментарий в docker-compose.yml)
    DEPLOY_COMPOSE_SERVICE — имя сервиса бота в docker-compose.yml (bot)
    DEPLOY_GIT_BRANCH     — ветка для git pull (main)
    DEPLOY_LOG_PATH       — куда писать лог деплоя
"""
from __future__ import annotations

import hmac
import json
import os
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HOST = os.environ.get("DEPLOY_AGENT_HOST", "0.0.0.0")
PORT = int(os.environ.get("DEPLOY_AGENT_PORT", "8091"))
TOKEN = os.environ.get("DEPLOY_AGENT_TOKEN", "")
REPO_DIR = os.environ.get("DEPLOY_REPO_DIR", "/opt/cryptobot")
COMPOSE_SERVICE = os.environ.get("DEPLOY_COMPOSE_SERVICE", "bot")
GIT_BRANCH = os.environ.get("DEPLOY_GIT_BRANCH", "main")
LOG_PATH = os.environ.get("DEPLOY_LOG_PATH", os.path.join(REPO_DIR, "data", "deploy_agent.log"))
LOG_TAIL_LINES = 200

_lock = threading.Lock()
_state: dict = {
    "running": False,
    "started_at": None,
    "finished_at": None,
    "exit_code": None,
}


def build_deploy_command(
    repo_dir: str = REPO_DIR,
    branch: str = GIT_BRANCH,
    service: str = COMPOSE_SERVICE,
) -> str:
    """
    Собрать shell-команду редеплоя. Отдельная функция (не инлайн в
    _run_deploy) — чтобы её можно было протестировать без реального
    запуска subprocess: параметры (пути/имя сервиса/ветка) берутся из
    переменных окружения, заданных оператором в docker-compose.yml, а не
    из HTTP-запроса — инъекция через сеть невозможна в принципе, здесь
    важно лишь не перепутать порядок шагов (миграции — ПОСЛЕ пересоздания
    контейнера, на новом образе, иначе применятся ещё по старому alembic
    env/файлам миграций).
    """
    return (
        f"cd {repo_dir} && "
        f"git pull origin {branch} && "
        f"docker compose build {service} && "
        f"docker compose up -d {service} && "
        f"docker compose exec -T {service} alembic upgrade head"
    )


def _run_deploy():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "a") as log_file:
        log_file.write(f"\n=== Деплой запущен {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        log_file.flush()
        proc = subprocess.run(
            ["bash", "-c", build_deploy_command()],
            cwd=REPO_DIR,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
        log_file.write(f"=== Деплой завершён, код выхода {proc.returncode} ===\n")
    with _lock:
        _state["running"] = False
        _state["finished_at"] = time.time()
        _state["exit_code"] = proc.returncode


def _tail_log(n: int = LOG_TAIL_LINES) -> list[str]:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        lines = f.readlines()
    return [line.rstrip("\n") for line in lines[-n:]]


class Handler(BaseHTTPRequestHandler):
    def _authorized(self) -> bool:
        if not TOKEN:
            return False
        auth_header = self.headers.get("Authorization", "")
        return hmac.compare_digest(auth_header, f"Bearer {TOKEN}")

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/deploy":
            self._send_json(404, {"error": "not found"})
            return
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        with _lock:
            if _state["running"]:
                self._send_json(409, {"error": "деплой уже выполняется"})
                return
            _state["running"] = True
            _state["started_at"] = time.time()
            _state["exit_code"] = None
        threading.Thread(target=_run_deploy, daemon=True).start()
        self._send_json(202, {"success": True, "message": "Деплой запущен"})

    def do_GET(self):
        if self.path != "/status":
            self._send_json(404, {"error": "not found"})
            return
        if not self._authorized():
            self._send_json(401, {"error": "unauthorized"})
            return
        with _lock:
            state = dict(_state)
        state["log_tail"] = _tail_log(50)
        self._send_json(200, state)

    def log_message(self, format, *args):  # noqa: A002 - сигнатура BaseHTTPRequestHandler
        pass  # тихо на stdout контейнера — свой лог деплоя пишем в _run_deploy


def main():
    if not TOKEN:
        raise SystemExit("DEPLOY_AGENT_TOKEN не задан — отказ запускаться без секрета")
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"deploy-agent слушает {HOST}:{PORT}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
