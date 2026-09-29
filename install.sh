#!/usr/bin/env bash
set -Eeuo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="img-tg-bot"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HF_CACHE_DEFAULT="/opt/img-tg-bot-hf-cache"

if [[ "${EUID}" -eq 0 ]]; then
  SUDO=""
  SERVICE_USER="${SUDO_USER:-root}"
else
  SUDO="sudo"
  SERVICE_USER="${USER}"
fi

echo "==> img-tg-bot installer"
echo "    app:  ${APP_DIR}"
echo "    user: ${SERVICE_USER}"

if [[ ! -f "${APP_DIR}/.env" ]]; then
  cp "${APP_DIR}/.env.example" "${APP_DIR}/.env"
  echo
  echo "Создан .env из примера."
  echo "Добавь TELEGRAM_BOT_TOKEN в:"
  echo "  ${APP_DIR}/.env"
  echo
  echo "После этого снова запусти:"
  echo "  sudo bash install.sh"
  exit 1
fi

if ! grep -Eq '^TELEGRAM_BOT_TOKEN=.+$' "${APP_DIR}/.env"; then
  echo "ERROR: TELEGRAM_BOT_TOKEN пустой в .env"
  exit 1
fi

echo "==> Проверяю NVIDIA GPU"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi не найден. Нужен образ Ubuntu + CUDA/NVIDIA driver."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "==> Ставлю системные пакеты"
${SUDO} apt-get update -y
${SUDO} env DEBIAN_FRONTEND=noninteractive apt-get install -y \
  git \
  python3 \
  python3-venv \
  python3-pip \
  ca-certificates \
  libgl1 \
  libglib2.0-0

echo "==> Создаю Python venv"
"${PYTHON_BIN}" -m venv "${APP_DIR}/.venv"
"${APP_DIR}/.venv/bin/python" -m pip install --upgrade pip wheel setuptools

echo "==> Ставлю PyTorch CUDA 12.8"
"${APP_DIR}/.venv/bin/pip" install --upgrade \
  torch torchvision \
  --index-url https://download.pytorch.org/whl/cu128

echo "==> Ставлю зависимости бота/Qwen"
"${APP_DIR}/.venv/bin/pip" install --upgrade -r "${APP_DIR}/requirements.txt"

HF_HOME_VALUE="$(grep -E '^HF_HOME=' "${APP_DIR}/.env" | tail -n1 | cut -d= -f2- || true)"
HF_HOME_VALUE="${HF_HOME_VALUE:-${HF_CACHE_DEFAULT}}"

${SUDO} mkdir -p "${HF_HOME_VALUE}"
${SUDO} chown -R "${SERVICE_USER}":"${SERVICE_USER}" "${HF_HOME_VALUE}" || true
mkdir -p "${APP_DIR}/temp"

SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

echo "==> Создаю systemd service: ${SERVICE_FILE}"
${SUDO} tee "${SERVICE_FILE}" >/dev/null <<EOF
[Unit]
Description=Qwen Image 2.1 Telegram Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/.env
Environment=PYTHONUNBUFFERED=1
Environment=TOKENIZERS_PARALLELISM=false
Environment=HF_HUB_DISABLE_TELEMETRY=1
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ExecStart=${APP_DIR}/.venv/bin/python ${APP_DIR}/bot.py
Restart=on-failure
RestartSec=5
TimeoutStopSec=30

[Install]
WantedBy=multi-user.target
EOF

${SUDO} systemctl daemon-reload
${SUDO} systemctl enable "${SERVICE_NAME}"
${SUDO} systemctl restart "${SERVICE_NAME}"

echo
echo "✅ Установка завершена."
echo
echo "Статус:"
echo "  systemctl status ${SERVICE_NAME} --no-pager"
echo
echo "Логи:"
echo "  journalctl -u ${SERVICE_NAME} -f"
echo
echo "Остановить:"
echo "  systemctl stop ${SERVICE_NAME}"
echo
echo "Перезапустить:"
echo "  systemctl restart ${SERVICE_NAME}"
