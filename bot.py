from __future__ import annotations

import asyncio
import html
import logging
import os
from io import BytesIO

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, Message
from PIL import Image, UnidentifiedImageError

from config import Settings, load_settings
from qwen_engine import QwenEngine


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("img-tg-bot")

settings: Settings = load_settings()
engine = QwenEngine(settings)
dp = Dispatcher()

# Одна генерация/правка за раз. Для одной A10 это правильнее и стабильнее.
gpu_queue = asyncio.Lock()


def user_allowed(message: Message) -> bool:
    if not settings.allowed_user_ids:
        return True
    return bool(message.from_user and message.from_user.id in settings.allowed_user_ids)


def extract_prompt(raw: str | None) -> str | None:
    if raw is None:
        return None

    text = raw.strip()
    prefix = settings.access_prefix

    if not text.startswith(prefix):
        return None

    tail = text[len(prefix):]

    # Не считаем "12345 ..." корректным паролем "1234".
    if tail and not (tail[0].isspace() or tail[0] in ":,-"):
        return None

    prompt = tail.lstrip(" \t\r\n:,-")

    if len(prompt) > settings.max_prompt_length:
        prompt = prompt[: settings.max_prompt_length]

    return prompt


async def auth_prompt(message: Message, raw: str | None) -> str | None:
    if not user_allowed(message):
        return None

    prompt = extract_prompt(raw)
    if prompt is None:
        if not settings.silent_auth_failure:
            await message.answer("🔒 В начале сообщения нужен пароль.")
        return None

    return prompt


async def send_result(message: Message, image_bytes: bytes, seed: int) -> None:
    filename = f"qwen-{seed}.png"
    upload = BufferedInputFile(image_bytes, filename=filename)

    if settings.send_as_document:
        await message.answer_document(
            upload,
            caption=f"✅ Готово · seed {seed}",
        )
    else:
        await message.answer_photo(
            upload,
            caption=f"✅ Готово · seed {seed}",
        )


async def run_generate(message: Message, prompt: str) -> None:
    if not prompt:
        await message.answer(
            f"Напиши после пароля, что создать.\n"
            f"Например: <code>{settings.access_prefix} "
            f"фотореалистичный автомобиль ночью</code>",
            parse_mode="HTML",
        )
        return

    status = await message.answer("⏳ Генерирую изображение…")

    try:
        async with gpu_queue:
            image, seed = await asyncio.to_thread(engine.generate, prompt)
            image_bytes = await asyncio.to_thread(engine.to_png_bytes, image)

        await send_result(message, image_bytes, seed)
        await status.delete()
        logger.info(
            "generation completed user_id=%s seed=%s",
            message.from_user.id if message.from_user else None,
            seed,
        )

    except Exception as exc:
        logger.exception("Generation failed")
        await status.edit_text(
            "❌ Ошибка генерации. Посмотри журнал сервиса:\n"
            "<code>journalctl -u img-tg-bot -n 100 --no-pager</code>\n\n"
            f"<code>{type(exc).__name__}: {html.escape(str(exc)[:500])}</code>",
            parse_mode="HTML",
        )


async def download_input_image(message: Message) -> Image.Image:
    buffer = BytesIO()

    if message.photo:
        await message.bot.download(message.photo[-1], destination=buffer)
    elif (
        message.document
        and message.document.mime_type
        and message.document.mime_type.startswith("image/")
    ):
        await message.bot.download(message.document, destination=buffer)
    else:
        raise ValueError("В сообщении нет изображения.")

    buffer.seek(0)

    try:
        image = Image.open(buffer)
        image.load()
        return image
    except UnidentifiedImageError as exc:
        raise ValueError("Telegram прислал неподдерживаемый файл изображения.") from exc


async def run_edit(message: Message, prompt: str) -> None:
    if not prompt:
        await message.answer(
            f"Добавь к фотографии подпись, начинающуюся с пароля.\n"
            f"Например: <code>{settings.access_prefix} "
            f"убери очки, остальное не меняй</code>",
            parse_mode="HTML",
        )
        return

    status = await message.answer("⏳ Загружаю фото и редактирую…")

    try:
        source = await download_input_image(message)

        async with gpu_queue:
            image, seed = await asyncio.to_thread(engine.edit, prompt, source)
            image_bytes = await asyncio.to_thread(engine.to_png_bytes, image)

        await send_result(message, image_bytes, seed)
        await status.delete()
        logger.info(
            "edit completed user_id=%s seed=%s",
            message.from_user.id if message.from_user else None,
            seed,
        )

    except Exception as exc:
        logger.exception("Edit failed")
        await status.edit_text(
            "❌ Ошибка редактирования. Посмотри журнал сервиса:\n"
            "<code>journalctl -u img-tg-bot -n 100 --no-pager</code>\n\n"
            f"<code>{type(exc).__name__}: {html.escape(str(exc)[:500])}</code>",
            parse_mode="HTML",
        )


@dp.message(Command("start"))
async def start_handler(message: Message) -> None:
    # /start не запускает модель и не раскрывает пароль.
    if not user_allowed(message):
        return

    await message.answer(
        "Бот запущен.\n\n"
        "• Текст с правильным префиксом → генерация.\n"
        "• Фото + подпись с правильным префиксом → редактирование.\n"
        "• Команды также пишутся после префикса.",
    )


@dp.message(F.photo)
async def photo_handler(message: Message) -> None:
    prompt = await auth_prompt(message, message.caption)
    if prompt is None:
        return
    await run_edit(message, prompt)


@dp.message(F.document)
async def document_handler(message: Message) -> None:
    if not (
        message.document
        and message.document.mime_type
        and message.document.mime_type.startswith("image/")
    ):
        return

    prompt = await auth_prompt(message, message.caption)
    if prompt is None:
        return
    await run_edit(message, prompt)


@dp.message(F.text)
async def text_handler(message: Message) -> None:
    prompt = await auth_prompt(message, message.text)
    if prompt is None:
        return

    command = prompt.strip().lower()

    if command in {"/help", "help"}:
        await message.answer(
            f"<b>Генерация</b>\n"
            f"<code>{settings.access_prefix} твой промпт</code>\n\n"
            f"<b>Редактирование</b>\n"
            f"Отправь фотографию с подписью:\n"
            f"<code>{settings.access_prefix} убери очки, остальное не меняй</code>\n\n"
            f"<b>Команды</b>\n"
            f"<code>{settings.access_prefix} /status</code>\n"
            f"<code>{settings.access_prefix} /warmup</code>",
            parse_mode="HTML",
        )
        return

    if command == "/status":
        await message.answer(
            f"Модель: {'✅ загружена' if engine.loaded else '⏸ ещё не загружена'}\n"
            f"{engine.cuda_status()}"
        )
        return

    if command == "/warmup":
        status = await message.answer("⏳ Загружаю модель в GPU…")
        try:
            async with gpu_queue:
                await asyncio.to_thread(engine.warmup)
            await status.edit_text("✅ Модель загружена и готова.")
        except Exception as exc:
            logger.exception("Warmup failed")
            await status.edit_text(
                f"❌ Не удалось загрузить модель:\n"
                f"<code>{type(exc).__name__}: {html.escape(str(exc)[:500])}</code>",
                parse_mode="HTML",
            )
        return

    await run_generate(message, prompt)


async def preload_if_enabled() -> None:
    if not settings.model_load_on_start:
        return

    logger.info("MODEL_LOAD_ON_START=true, preloading model")
    try:
        async with gpu_queue:
            await asyncio.to_thread(engine.warmup)
        logger.info("Model preload complete")
    except Exception:
        logger.exception("Model preload failed")


async def main() -> None:
    bot = Bot(token=settings.telegram_bot_token)

    # Убираем старый webhook: используем long polling, открытые порты не нужны.
    await bot.delete_webhook(drop_pending_updates=False)

    if settings.model_load_on_start:
        asyncio.create_task(preload_if_enabled())

    me = await bot.get_me()
    logger.info("Bot @%s started", me.username)

    await dp.start_polling(
        bot,
        allowed_updates=dp.resolve_used_update_types(),
    )


if __name__ == "__main__":
    asyncio.run(main())
