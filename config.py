from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value and value.strip() else default


def _float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value and value.strip() else default


def _ids(name: str) -> frozenset[int]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return frozenset()
    return frozenset(int(item.strip()) for item in raw.split(",") if item.strip())


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    access_prefix: str
    allowed_user_ids: frozenset[int]

    model_id: str
    hf_token: str | None
    hf_home: Path

    generate_width: int
    generate_height: int
    edit_max_side: int
    inference_steps: int
    true_cfg_scale: float

    send_as_document: bool
    silent_auth_failure: bool
    max_prompt_length: int
    model_load_on_start: bool


def load_settings() -> Settings:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN не задан. Скопируй .env.example в .env и добавь токен."
        )

    prefix = os.getenv("ACCESS_PREFIX", "1234").strip()
    if not prefix:
        raise RuntimeError("ACCESS_PREFIX не может быть пустым.")

    hf_home = Path(
        os.getenv("HF_HOME", "/opt/img-tg-bot-hf-cache").strip()
        or "/opt/img-tg-bot-hf-cache"
    )

    return Settings(
        telegram_bot_token=token,
        access_prefix=prefix,
        allowed_user_ids=_ids("ALLOWED_USER_IDS"),
        model_id=os.getenv(
            "QWEN_MODEL_ID", "Qwen/Qwen-Image-2.1"
        ).strip(),
        hf_token=os.getenv("HF_TOKEN", "").strip() or None,
        hf_home=hf_home,
        generate_width=_int("GENERATE_WIDTH", 1024),
        generate_height=_int("GENERATE_HEIGHT", 1024),
        edit_max_side=_int("EDIT_MAX_SIDE", 1024),
        inference_steps=_int("INFERENCE_STEPS", 30),
        true_cfg_scale=_float("TRUE_CFG_SCALE", 1.0),
        send_as_document=_bool("SEND_AS_DOCUMENT", True),
        silent_auth_failure=_bool("SILENT_AUTH_FAILURE", True),
        max_prompt_length=_int("MAX_PROMPT_LENGTH", 4000),
        model_load_on_start=_bool("MODEL_LOAD_ON_START", False),
    )
