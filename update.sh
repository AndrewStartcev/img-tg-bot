#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "${APP_DIR}"
git pull --ff-only
"${APP_DIR}/.venv/bin/pip" install --upgrade -r requirements.txt
sudo systemctl restart img-tg-bot

echo "✅ Обновлено и перезапущено."
sudo systemctl status img-tg-bot --no-pager
