from __future__ import annotations

import html
import json
import logging
import os
import socket
import time
from io import BytesIO

import requests
import urllib3.util.connection as urllib3_connection
from PIL import Image, UnidentifiedImageError

from config import Settings, load_settings
from qwen_engine import QwenEngine


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("img-tg-bot")

# Force IPv4 for requests/urllib3: this VM has unrouted IPv6 for Telegram API.
urllib3_connection.allowed_gai_family = lambda: socket.AF_INET

settings: Settings = load_settings()
engine = QwenEngine(settings)


class TelegramAPI:
    def __init__(self, token: str) -> None:
        self.base = f"https://api.telegram.org/bot{token}"
        self.file_base = f"https://api.telegram.org/file/bot{token}"
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "img-tg-bot/1.0"})

    def call(
        self,
        method: str,
        *,
        data: dict | None = None,
        params: dict | None = None,
        files: dict | None = None,
        timeout: tuple[int, int] = (10, 60),
    ):
        url = f"{self.base}/{method}"
        response = self.session.post(
            url,
            data=data,
            params=params,
            files=files,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()

        if not payload.get("ok"):
            raise RuntimeError(
                f"Telegram {method}: {payload.get('error_code')} "
                f"{payload.get('description', 'unknown error')}"
            )
        return payload.get("result")

    def get_updates(self, offset: int | None) -> list[dict]:
        params = {
            "timeout": 25,
            "allowed_updates": json.dumps(["message"]),
        }
        if offset is not None:
            params["offset"] = offset

        # getUpdates лучше делать GET: это максимально близко к curl,
        # который стабильно работает на Intelion.
        response = self.session.get(
            f"{self.base}/getUpdates",
            params=params,
            timeout=(10, 40),
        )
        response.raise_for_status()
        payload = response.json()

        if not payload.get("ok"):
            raise RuntimeError(
                f"Telegram getUpdates: {payload.get('error_code')} "
                f"{payload.get('description', 'unknown error')}"
            )

        return payload.get("result", [])

    def get_me(self) -> dict:
        response = self.session.get(
            f"{self.base}/getMe",
            timeout=(10, 20),
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(payload.get("description", "getMe failed"))
        return payload["result"]

    def delete_webhook(self) -> None:
        response = self.session.get(
            f"{self.base}/deleteWebhook",
            params={"drop_pending_updates": "false"},
            timeout=(10, 20),
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            raise RuntimeError(payload.get("description", "deleteWebhook failed"))

    def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        parse_mode: str | None = None,
    ) -> dict:
        data = {"chat_id": str(chat_id), "text": text}
        if parse_mode:
            data["parse_mode"] = parse_mode
        return self.call("sendMessage", data=data)

    def edit_message(
        self,
        chat_id: int,
        message_id: int,
        text: str,
        *,
        parse_mode: str | None = None,
    ) -> None:
        data = {
            "chat_id": str(chat_id),
            "message_id": str(message_id),
            "text": text,
        }
        if parse_mode:
            data["parse_mode"] = parse_mode
        self.call("editMessageText", data=data)

    def delete_message(self, chat_id: int, message_id: int) -> None:
        try:
            self.call(
                "deleteMessage",
                data={
                    "chat_id": str(chat_id),
                    "message_id": str(message_id),
                },
                timeout=(10, 20),
            )
        except Exception:
            logger.debug("Could not delete status message", exc_info=True)

    def send_result(
        self,
        chat_id: int,
        image_bytes: bytes,
        seed: int,
    ) -> None:
        filename = f"qwen-{seed}.png"
        caption = f"✅ Готово · seed {seed}"

        if settings.send_as_document:
            self.call(
                "sendDocument",
                data={"chat_id": str(chat_id), "caption": caption},
                files={
                    "document": (
                        filename,
                        image_bytes,
                        "image/png",
                    )
                },
                timeout=(15, 180),
            )
        else:
            self.call(
                "sendPhoto",
                data={"chat_id": str(chat_id), "caption": caption},
                files={
                    "photo": (
                        filename,
                        image_bytes,
                        "image/png",
                    )
                },
                timeout=(15, 180),
            )

    def download_file(self, file_id: str) -> bytes:
        info = self.call(
            "getFile",
            data={"file_id": file_id},
            timeout=(10, 30),
        )
        path = info["file_path"]

        response = self.session.get(
            f"{self.file_base}/{path}",
            timeout=(10, 120),
        )
        response.raise_for_status()
        return response.content


tg = TelegramAPI(settings.telegram_bot_token)


def user_allowed(message: dict) -> bool:
    if not settings.allowed_user_ids:
        return True

    user = message.get("from") or {}
    user_id = user.get("id")
    return bool(user_id and int(user_id) in settings.allowed_user_ids)


def extract_prompt(raw: str | None) -> str | None:
    if raw is None:
        return None

    text = raw.strip()
    prefix = settings.access_prefix

    if not text.startswith(prefix):
        return None

    tail = text[len(prefix):]

    # "12345 ..." не считаем паролем "1234".
    if tail and not (tail[0].isspace() or tail[0] in ":,-"):
        return None

    prompt = tail.lstrip(" \t\r\n:,-")

    if len(prompt) > settings.max_prompt_length:
        prompt = prompt[: settings.max_prompt_length]

    return prompt


def status_error(chat_id: int, status_id: int, exc: Exception) -> None:
    error = html.escape(str(exc)[:700])
    try:
        tg.edit_message(
            chat_id,
            status_id,
            "❌ Ошибка. Посмотри журнал сервиса:\n"
            "<code>journalctl -u img-tg-bot -n 100 --no-pager</code>\n\n"
            f"<code>{type(exc).__name__}: {error}</code>",
            parse_mode="HTML",
        )
    except Exception:
        logger.exception("Could not report error to Telegram")


def run_generate(chat_id: int, prompt: str) -> None:
    if not prompt:
        tg.send_message(
            chat_id,
            f"Напиши после пароля, что создать.\n"
            f"Например: <code>{settings.access_prefix} "
            "фотореалистичный автомобиль ночью</code>",
            parse_mode="HTML",
        )
        return

    status = tg.send_message(chat_id, "⏳ Генерирую изображение…")
    status_id = status["message_id"]

    try:
        image, seed = engine.generate(prompt)
        image_bytes = engine.to_png_bytes(image)
        tg.send_result(chat_id, image_bytes, seed)
        tg.delete_message(chat_id, status_id)
        logger.info("Generation completed seed=%s", seed)
    except Exception as exc:
        logger.exception("Generation failed")
        status_error(chat_id, status_id, exc)


def image_from_message(message: dict) -> Image.Image:
    file_id: str | None = None

    photos = message.get("photo") or []
    if photos:
        file_id = photos[-1].get("file_id")

    if not file_id:
        document = message.get("document") or {}
        mime = document.get("mime_type", "")
        if mime.startswith("image/"):
            file_id = document.get("file_id")

    if not file_id:
        raise ValueError("В сообщении нет поддерживаемого изображения.")

    raw = tg.download_file(file_id)

    try:
        image = Image.open(BytesIO(raw))
        image.load()
        return image
    except UnidentifiedImageError as exc:
        raise ValueError("Telegram прислал неподдерживаемый файл изображения.") from exc


def run_edit(chat_id: int, message: dict, prompt: str) -> None:
    if not prompt:
        tg.send_message(
            chat_id,
            "Добавь к фотографии подпись, начинающуюся с пароля.\n"
            f"Например: <code>{settings.access_prefix} "
            "убери очки, остальное не меняй</code>",
            parse_mode="HTML",
        )
        return

    status = tg.send_message(chat_id, "⏳ Загружаю фото и редактирую…")
    status_id = status["message_id"]

    try:
        source = image_from_message(message)
        image, seed = engine.edit(prompt, source)
        image_bytes = engine.to_png_bytes(image)
        tg.send_result(chat_id, image_bytes, seed)
        tg.delete_message(chat_id, status_id)
        logger.info("Edit completed seed=%s", seed)
    except Exception as exc:
        logger.exception("Edit failed")
        status_error(chat_id, status_id, exc)


def handle_text(chat_id: int, prompt: str) -> None:
    command = prompt.strip().lower()

    if command in {"/help", "help"}:
        tg.send_message(
            chat_id,
            f"<b>Генерация</b>\n"
            f"<code>{settings.access_prefix} твой промпт</code>\n\n"
            f"<b>Редактирование</b>\n"
            "Отправь фотографию с подписью:\n"
            f"<code>{settings.access_prefix} "
            "убери очки, остальное не меняй</code>\n\n"
            f"<b>Команды</b>\n"
            f"<code>{settings.access_prefix} /status</code>\n"
            f"<code>{settings.access_prefix} /warmup</code>",
            parse_mode="HTML",
        )
        return

    if command == "/status":
        tg.send_message(
            chat_id,
            f"Модель: {'✅ загружена' if engine.loaded else '⏸ ещё не загружена'}\n"
            f"{engine.cuda_status()}",
        )
        return

    if command == "/warmup":
        status = tg.send_message(chat_id, "⏳ Загружаю модель в GPU…")
        status_id = status["message_id"]
        try:
            engine.warmup()
            tg.edit_message(chat_id, status_id, "✅ Модель загружена и готова.")
        except Exception as exc:
            logger.exception("Warmup failed")
            status_error(chat_id, status_id, exc)
        return

    run_generate(chat_id, prompt)


def handle_message(message: dict) -> None:
    if not user_allowed(message):
        return

    chat = message.get("chat") or {}
    chat_id = chat.get("id")
    if chat_id is None:
        return

    text = message.get("text")
    caption = message.get("caption")

    # /start специально не раскрывает пароль.
    if text == "/start":
        tg.send_message(
            int(chat_id),
            "Бот запущен. Для работы используй секретный префикс.",
        )
        return

    has_photo = bool(message.get("photo"))
    document = message.get("document") or {}
    has_image_document = bool(
        document
        and str(document.get("mime_type", "")).startswith("image/")
    )

    if has_photo or has_image_document:
        prompt = extract_prompt(caption)
        if prompt is None:
            if not settings.silent_auth_failure:
                tg.send_message(int(chat_id), "🔒 Неверный префикс.")
            return
        run_edit(int(chat_id), message, prompt)
        return

    if text is not None:
        prompt = extract_prompt(text)
        if prompt is None:
            if not settings.silent_auth_failure:
                tg.send_message(int(chat_id), "🔒 Неверный префикс.")
            return
        handle_text(int(chat_id), prompt)


def startup() -> None:
    # Не даём transient network error убить systemd-процесс.
    for attempt in range(1, 6):
        try:
            tg.delete_webhook()
            break
        except Exception as exc:
            logger.warning(
                "deleteWebhook attempt %s/5 failed: %s: %s",
                attempt,
                type(exc).__name__,
                exc,
            )
            time.sleep(min(attempt * 2, 10))

    me = tg.get_me()
    logger.info("Bot @%s started via requests transport", me.get("username"))

    if settings.model_load_on_start:
        try:
            engine.warmup()
            logger.info("Model preload complete")
        except Exception:
            logger.exception("Model preload failed")


def main() -> None:
    startup()

    offset: int | None = None
    backoff = 2

    while True:
        try:
            updates = tg.get_updates(offset)
            backoff = 2

            for update in updates:
                update_id = update.get("update_id")
                if update_id is not None:
                    offset = int(update_id) + 1

                message = update.get("message")
                if not message:
                    continue

                try:
                    handle_message(message)
                except Exception:
                    logger.exception("Unhandled message error")

        except KeyboardInterrupt:
            logger.info("Stopped by user")
            break
        except Exception as exc:
            logger.warning(
                "Telegram polling error: %s: %s; retry in %ss",
                type(exc).__name__,
                exc,
                backoff,
            )
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)


if __name__ == "__main__":
    main()
