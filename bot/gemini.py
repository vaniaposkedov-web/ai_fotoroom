import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import time
import uuid
from typing import Optional

import aiohttp
import aiosqlite

logger = logging.getLogger(__name__)

# Определение базовой директории для сохранения истории
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Настройки сжатия и лимитов для работы с API
MAX_IMAGE_DIM = 1536
JPEG_QUALITY = 98

_POLZA_KEY_SLOT_LOCK = asyncio.Lock()
_POLZA_CACHE_LOCK = asyncio.Lock()
_POLZA_KEYS_CACHE: list[str] = []
_POLZA_KEYS_CACHE_TS = 0.0
_POLZA_ACTIVE_BY_KEY: dict[str, int] = {}


def get_polza_api_key() -> str:
    """Резервный API ключ Polza.ai из переменных окружения (основные ключи — в админке)."""
    return (os.getenv("POLZA_API_KEY") or "").strip()


def _is_enabled(flag_name: str, default: str = "1") -> bool:
    raw = str(os.getenv(flag_name, default)).strip().lower()
    return raw in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, str(default))).strip())
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return int(default)


def _env_csv(name: str, default: str = "") -> list[str]:
    raw = str(os.getenv(name, default) or "").strip()
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_polza_image_provider_preferences() -> dict | None:
    if not _is_enabled("POLZA_IMAGE_PROVIDER_ENABLED", "1"):
        return None
    provider: dict = {}
    only = _env_csv("POLZA_IMAGE_PROVIDER_ONLY")
    order = _env_csv("POLZA_IMAGE_PROVIDER_ORDER")
    # Раньше дефолт был google-ai-studio — из‑за этого 502 на gemini-3.1-flash-image-preview
    # (у preview единственный endpoint как раз google-ai-studio). Ignore только если явно задан в env.
    ignore = _env_csv("POLZA_IMAGE_PROVIDER_IGNORE", "")
    if only:
        provider["only"] = only
    if order:
        provider["order"] = order
    if ignore:
        provider["ignore"] = ignore
    provider["allow_fallbacks"] = _is_enabled("POLZA_IMAGE_PROVIDER_ALLOW_FALLBACKS", "1")
    return provider or None


def _mask_key_preview(key: str) -> str:
    if not key:
        return "n/a"
    if len(key) <= 8:
        return f"{key[:2]}***"
    return f"{key[:4]}...{key[-4:]}"


def _bot_db_path() -> str:
    raw = (os.getenv("DATABASE_URL") or "").strip()
    if raw.startswith("sqlite+aiosqlite:///"):
        path = raw.replace("sqlite+aiosqlite:///", "", 1)
        if not path.startswith("/"):
            path = "/" + path
        return path
    if raw.startswith("sqlite:///"):
        path = raw.replace("sqlite:///", "", 1)
        if not path.startswith("/"):
            path = "/" + path
        return path
    return os.getenv("BOT_DB_PATH", "/app/data/bot.db")


async def ensure_api_keys_provider_column(conn: aiosqlite.Connection) -> None:
    """
    Колонка api_keys.provider: ключи, добавленные до перехода на Polza.ai, помечаются 'zveno'
    и больше не используются. Новые ключи из админки сохраняются с provider='polza'.
    """
    async with conn.execute("PRAGMA table_info(api_keys)") as cur:
        cols = {str(r[1]) for r in await cur.fetchall()}
    if cols and "provider" not in cols:
        await conn.execute("ALTER TABLE api_keys ADD COLUMN provider TEXT NOT NULL DEFAULT 'zveno'")
        await conn.commit()


async def _load_active_polza_keys_from_db() -> list[str]:
    db_path = _bot_db_path()
    keys: list[str] = []
    try:
        async with aiosqlite.connect(db_path) as conn:
            await ensure_api_keys_provider_column(conn)
            # priority column optional in old schemas.
            try:
                async with conn.execute(
                    "SELECT token FROM api_keys WHERE is_active=1 AND provider='polza' "
                    "ORDER BY COALESCE(priority, 0) DESC, id ASC"
                ) as cur:
                    rows = await cur.fetchall()
            except Exception:
                async with conn.execute(
                    "SELECT token FROM api_keys WHERE is_active=1 AND provider='polza' ORDER BY id ASC"
                ) as cur:
                    rows = await cur.fetchall()
            for row in rows:
                token = (row[0] or "").strip()
                if token and token not in keys:
                    keys.append(token)
    except Exception as exc:
        logger.warning("[POLZA] failed to read api_keys from db %s: %s", db_path, exc)
    return keys


async def get_polza_api_keys_cached() -> list[str]:
    global _POLZA_KEYS_CACHE, _POLZA_KEYS_CACHE_TS
    cache_ttl = max(3.0, _env_float("POLZA_KEYS_CACHE_TTL_SECONDS", 15.0))
    now = time.time()
    async with _POLZA_CACHE_LOCK:
        if _POLZA_KEYS_CACHE and (now - _POLZA_KEYS_CACHE_TS) < cache_ttl:
            return list(_POLZA_KEYS_CACHE)
        keys = await _load_active_polza_keys_from_db()
        # Fallback to env key for bootstrap and recovery.
        env_key = get_polza_api_key()
        if env_key and env_key not in keys:
            keys.insert(0, env_key)
        _POLZA_KEYS_CACHE = keys
        _POLZA_KEYS_CACHE_TS = now
        return list(keys)


async def acquire_polza_key_slot(wait_seconds: float | None = None) -> str | None:
    keys = await get_polza_api_keys_cached()
    if not keys:
        return None
    per_key_limit = max(1, _env_int("POLZA_MAX_CONCURRENT_PER_KEY", 20))
    wait_for = max(0.5, wait_seconds if wait_seconds is not None else _env_float("POLZA_KEY_WAIT_SECONDS", 4.0))
    sleep_step = max(0.03, _env_float("POLZA_KEY_WAIT_POLL_SECONDS", 0.06))
    deadline = asyncio.get_running_loop().time() + wait_for
    while True:
        async with _POLZA_KEY_SLOT_LOCK:
            for key in keys:
                _POLZA_ACTIVE_BY_KEY.setdefault(key, 0)
            available = [k for k in keys if _POLZA_ACTIVE_BY_KEY.get(k, 0) < per_key_limit]
            if available:
                picked = min(available, key=lambda k: _POLZA_ACTIVE_BY_KEY.get(k, 0))
                _POLZA_ACTIVE_BY_KEY[picked] = _POLZA_ACTIVE_BY_KEY.get(picked, 0) + 1
                return picked
        if asyncio.get_running_loop().time() >= deadline:
            return None
        await asyncio.sleep(sleep_step)


async def release_polza_key_slot(key: str | None) -> None:
    if not key:
        return
    async with _POLZA_KEY_SLOT_LOCK:
        if key in _POLZA_ACTIVE_BY_KEY:
            _POLZA_ACTIVE_BY_KEY[key] = max(0, _POLZA_ACTIVE_BY_KEY[key] - 1)


def _provider_rub_from_polza_payload(usage: dict | None, full_response: dict | None) -> float | None:
    """Если Polza вернул стоимость в ответе — используем её как total_rub (точнее оценки по токенам)."""
    blobs: list[dict] = []
    if isinstance(usage, dict):
        blobs.append(usage)
    if isinstance(full_response, dict):
        blobs.append(full_response)
        nested = full_response.get("usage")
        if isinstance(nested, dict) and nested is not usage:
            blobs.append(nested)
    for b in blobs:
        for key in ("total_cost_rub", "cost_rub", "total_cost", "cost", "billed_rub", "polza_cost_rub"):
            val = b.get(key)
            if val is None:
                continue
            try:
                x = float(val)
                if x >= 0:
                    return x
            except (TypeError, ValueError):
                continue
    return None


async def record_polza_usage(
    *,
    api_key: str,
    model: str,
    usage: dict | None,
    latency_ms: int,
    status: str,
    error_text: str = "",
    full_response: dict | None = None,
    user_id: int | None = None,
) -> None:
    db_path = _bot_db_path()
    raw_prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
    completion_tokens = int((usage or {}).get("completion_tokens") or 0)
    prompt_tokens = raw_prompt_tokens
    total_tokens = int((usage or {}).get("total_tokens") or (prompt_tokens + completion_tokens))
    # Фиксированная ставка за 1 API-вызов (из env или 6.3 ₽ по умолчанию).
    flat_rate = _env_float("POLZA_FLAT_RATE_PER_CALL_RUB", 6.3)
    provider_rub = _provider_rub_from_polza_payload(usage, full_response)
    if provider_rub is not None:
        total_rub = float(provider_rub)
    else:
        total_rub = flat_rate
    input_rub = total_rub
    output_rub = 0.0
    key_preview = _mask_key_preview(api_key)
    try:
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS polza_usage_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    user_id INTEGER,
                    key_id INTEGER,
                    key_preview TEXT,
                    model TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ok',
                    prompt_tokens INTEGER NOT NULL DEFAULT 0,
                    completion_tokens INTEGER NOT NULL DEFAULT 0,
                    total_tokens INTEGER NOT NULL DEFAULT 0,
                    input_rub REAL NOT NULL DEFAULT 0,
                    output_rub REAL NOT NULL DEFAULT 0,
                    total_rub REAL NOT NULL DEFAULT 0,
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    error_text TEXT
                )
                """
            )
            try:
                async with conn.execute("PRAGMA table_info(polza_usage_log)") as cur:
                    cols = await cur.fetchall()
                col_names = {str(r[1]) for r in (cols or [])}
                if "user_id" not in col_names:
                    await conn.execute("ALTER TABLE polza_usage_log ADD COLUMN user_id INTEGER")
            except Exception:
                pass
            key_id = None
            try:
                async with conn.execute("SELECT id FROM api_keys WHERE token=? LIMIT 1", (api_key,)) as cur:
                    row = await cur.fetchone()
                    if row:
                        key_id = int(row[0])
            except Exception:
                key_id = None
            await conn.execute(
                """
                INSERT INTO polza_usage_log
                (user_id, key_id, key_preview, model, status, prompt_tokens, completion_tokens, total_tokens,
                 input_rub, output_rub, total_rub, latency_ms, error_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(user_id) if user_id is not None else None,
                    key_id,
                    key_preview,
                    str(model or ""),
                    str(status or "ok"),
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    float(input_rub),
                    float(output_rub),
                    float(total_rub),
                    int(latency_ms),
                    (error_text or "")[:500],
                ),
            )
            await conn.commit()
    except Exception as exc:
        logger.warning("[POLZA] failed to persist usage stats: %s", exc)


def _extract_chat_content_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("text", "output_text"):
                txt = str(part.get("text") or "").strip()
                if txt:
                    chunks.append(txt)
        return "\n".join(chunks).strip()
    return ""


async def enrich_prompt_with_polza(
    session: aiohttp.ClientSession,
    base_prompt: str,
    images_bytes: list[bytes],
    aspect_ratio: str,
    category: str,
) -> tuple[str, bool]:
    """Prompt-opt отключен: в image-запрос отправляем только готовый промпт + фото."""
    return base_prompt, False


def _resolve_polza_image_model(model_name: str | None) -> str:
    """Модель для генерации изображения в Polza (формат vendor/model)."""
    default_raw = (os.getenv("POLZA_IMAGE_MODEL") or "google/gemini-nano-banana-2.1").strip()
    mn = (model_name or "").strip()
    if not mn or "vertex" in mn.lower():
        return default_raw if "/" in default_raw else f"google/{default_raw.removeprefix('google/')}"
    if "/" in mn:
        return mn
    if mn.startswith("google/"):
        return mn
    return f"google/{mn}"


def _is_hard_generation_category(category: str | None) -> bool:
    cat = (category or "").strip().lower()
    return cat in {"own", "own_variant"}


def _pick_polza_image_model(model_name: str | None, generation_meta: dict | None, quality: str = "HD") -> str:
    # Явно заданная модель имеет приоритет.
    if (model_name or "").strip():
        return _resolve_polza_image_model(model_name)
    if str(quality).lower() == "4k":
        # Все 4K → nano banana pro (детальная модель). Нативный 4K (3712×4608)
        # обеспечивает image_size="4K" + чтение последнего кадра из ответа.
        pro = (os.getenv("POLZA_IMAGE_MODEL_PRO") or "google/gemini-nano-banana-2.1").strip()
        return _resolve_polza_image_model(pro)
    model = os.getenv("POLZA_IMAGE_MODEL_CHEAP") or "google/gemini-nano-banana-2.1"
    return _resolve_polza_image_model(model)


def prepare_image_for_upload(img_bytes: bytes, fallback_mime_type: str = "image/jpeg") -> tuple[bytes, str]:
    """
    Подготавливает изображение к загрузке: проверяет формат и сжимает до 1536px,
    если это необходимо для ускорения передачи данных.
    """
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(img_bytes))
        fmt = (image.format or "").upper()
        width, height = image.size

        if width > MAX_IMAGE_DIM or height > MAX_IMAGE_DIM:
            scale = min(MAX_IMAGE_DIM / width, MAX_IMAGE_DIM / height, 1.0)
            new_size = (int(width * scale), int(height * scale))
            image = image.convert("RGB").resize(new_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, "JPEG", quality=JPEG_QUALITY, optimize=True)
            return buffer.getvalue(), "image/jpeg"

        if fmt in {"JPEG", "JPG"}: return img_bytes, "image/jpeg"
        if fmt == "PNG": return img_bytes, "image/png"
        if fmt == "WEBP": return img_bytes, "image/webp"

        image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=JPEG_QUALITY, optimize=True)
        return buffer.getvalue(), "image/jpeg"
    except Exception as exc:
        logger.debug("[image] Ошибка подготовки изображения, используем оригинал: %s", exc)
        return img_bytes, fallback_mime_type


def compress_output_image(img_bytes: bytes) -> bytes:
    """Сжимает итоговое изображение перед отправкой в Telegram."""
    try:
        from PIL import Image
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        width, height = image.size
        if width <= MAX_IMAGE_DIM and height <= MAX_IMAGE_DIM:
            return img_bytes

        scale = min(MAX_IMAGE_DIM / width, MAX_IMAGE_DIM / height, 1.0)
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=JPEG_QUALITY, optimize=True)
        return buffer.getvalue()
    except Exception as exc:
        logger.debug("[image] Ошибка финального сжатия: %s", exc)
        return img_bytes


def _build_polza_generation_suffix(current_prompt: str, n_images: int, generation_meta: dict | None) -> str:
    """Текстовые инструкции к промпту (дополнение к user-сообщению для image-модели)."""
    generation_meta = generation_meta or {}
    prompt_low = (current_prompt or "").lower()
    cat_from_meta = str(generation_meta.get("category", "") or "")
    model_id = str(generation_meta.get("model_id", "") or "")

    universal_override = (
        " Preserve original details and patterns. "
        "ORIENTATION: Match the front or back view of the uploaded garment naturally. Do not force a front view if a back view is provided."
    )
    universal_negative = "low quality, blurry, text, watermark, signature, letters, deformed, distorted, interface, crop"

    if cat_from_meta == "collage":
        return ""

    if n_images <= 0:
        return (
            "\n\nGenerate exactly ONE photorealistic image from the instructions. "
            "No collage, no split view, no watermarks, no UI overlays."
        )

    if n_images == 1:
        if cat_from_meta == "whitebg":
            return (
                "\n\nCRITICAL OVERRIDE FOR 1 PHOTO: "
                "Put the EXACT item from the photo on a pure white (#FFFFFF) background. "
                "This is strictly a background removal and lighting adjustment task. "
                "Create a 1:1 exact copy of the garment, preserving its current state exactly, just with soft professional studio lighting."
                + universal_override
                + f"\nAvoid: {universal_negative}, added details, extra seams, extra pockets, invented features."
            )
        if cat_from_meta == "storefront":
            return (
                "\n\nSTOREFRONT (1 PRODUCT PHOTO): "
                "Use the uploaded garment image as the ONLY product source — keep its design, colors, logos, and proportions identical. "
                "Build the final storefront / showcase shot described in the prompt text only (scene, lighting, layout). "
                "Do not invent a different garment; integrate THIS exact item into the scene."
                + universal_override
                + f"\nAvoid: {universal_negative}, bad anatomy, humans, hands, distorted product, wrong colors, extra items."
            )
        return (
            "\n\nCRITICAL OVERRIDE: "
            "Place the original garment exactly as it is into the scene described in the prompt. "
            "Generate a realistic background based ONLY on the text prompt. "
            + universal_override
            + f"\nAvoid: {universal_negative}, bad anatomy, logo, writing."
        )

    user_context = prompt_low.split("additional details:")[-1] if "additional details:" in prompt_low else prompt_low[-250:]
    footwear_keywords = [
        "shoes", "footwear", "sneakers", "boots", "обувь", "кроссовки", "кеды", "туфли", "ботинки",
    ]
    storefront_keywords = ["storefront", "showcase", "hanger", "product-in-scene"]
    is_footwear = any(word in user_context for word in footwear_keywords)
    is_storefront = any(word in user_context for word in storefront_keywords)
    is_own_bg = cat_from_meta == "own_variant"

    if is_footwear:
        return (
            "\n\nFOOTWEAR OVERRIDE: Replace the original footwear with the item from Photo 2. Keep the pants and background intact."
            + universal_override
            + f"\nAvoid: {universal_negative}, change pants, change jeans, alter background, morphing."
        )
    if cat_from_meta == "own":
        return (
            "\n\nCRITICAL MAPPING:\n"
            "Photo 1 (SCENE_AND_MODEL_REFERENCE) = first attached image.\n"
            "Photo 2 (CLOTHING_ITEM_TO_WEAR) = second attached image."
            + universal_override
            + f"\nAvoid: {universal_negative}."
        )
    if is_own_bg:
        return (
            "\n\nCRITICAL — TWO INPUT IMAGES (PHOTO 1 = BACKGROUND REFERENCE, PHOTO 2 = PRODUCT — ONLY SOURCE OF TRUTH FOR THE PRODUCT):\n"
            "First attached image = BACKGROUND / scene reference.\n"
            "Second attached image = PRODUCT (garment) — preserve design, colors, logos, texture 1:1.\n"
            "The output MUST be a new composite integrating the product into the background; "
            "it MUST NOT be an unchanged copy of the background photo alone."
            + universal_override
            + f"\nAvoid: {universal_negative}, identical copy of background frame, passthrough, missing garment, wrong product."
        )
    if is_storefront:
        if model_id == "13":
            return (
                "\n\nSTOREFRONT OVERRIDE (MODEL 13): "
                "Replace the main clothing with the garment from Photo 2. "
                "IMPORTANT: Keep the background and the human hand holding the item intact. "
                "Blend the new garment naturally behind the hand."
                + universal_override
                + f"\nAvoid: {universal_negative}, change background, modified environment, missing parts, deformed."
            )
        return (
            "\n\nSTOREFRONT OVERRIDE: "
            "Replace the existing clothing in the scene with the garment from Photo 2. "
            "Keep the original background and props intact. Strictly product only, no humans."
            + universal_override
            + f"\nAvoid: {universal_negative}, bad anatomy, logo, writing, messy scene."
        )
    return (
        "\n\nMODEL OVERRIDE: The new garment (Photo 2) must be worn naturally by the person in Photo 1. "
        "Change the shape of the original clothes to match the new garment. "
        "The background, face, and pose must remain the same as Photo 1."
        + universal_override
        + f"\nAvoid: {universal_negative}, multiple garments, floating items, cloned clothes, split garment, bad anatomy."
    )


POLZA_DEFAULT_BASE_URL = "https://polza.ai/api/v1"
_POLZA_IMAGE_ASPECTS = (
    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
)


def polza_base_url() -> str:
    """Базовый URL Polza.ai (OpenAI-совместимый, с /v1)."""
    base = (os.getenv("POLZA_BASE_URL") or POLZA_DEFAULT_BASE_URL).strip().rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base


def _polza_error_text(data: dict | None) -> str:
    """Склеивает code/message/raw из ошибки Polza (HTTP-ошибка или error внутри статуса)."""
    if not isinstance(data, dict):
        return ""
    err = data.get("error")
    if isinstance(err, str):
        return err
    if not isinstance(err, dict):
        return ""
    parts = [str(err.get("code") or ""), str(err.get("message") or "")]
    meta = err.get("metadata")
    if isinstance(meta, dict):
        for key in ("reason", "raw", "provider_name"):
            if meta.get(key):
                parts.append(f"{key}={meta.get(key)}")
    return " ".join(p for p in parts if p).strip()


def _polza_is_safety_text(text: str) -> bool:
    t = (text or "").lower()
    return any(w in t for w in (
        "safety", "prohibited", "content policy", "moderation", "nsfw", "blocked", "guard_blocked",
    ))


def _polza_extract_image_ref(data: dict | None) -> str | None:
    """Достаёт URL / base64 результата из MediaStatusPresenter (data.url | data.b64_json | output.url)."""
    if not isinstance(data, dict):
        return None
    for container_key in ("data", "output"):
        block = data.get(container_key)
        items = block if isinstance(block, list) else [block]
        for item in reversed(items):
            if not isinstance(item, dict):
                continue
            url = item.get("url")
            if isinstance(url, str) and url:
                return url
            b64 = item.get("b64_json")
            if isinstance(b64, str) and b64:
                return b64 if b64.startswith("data:") else "data:image/png;base64," + b64
    return None


def _polza_no_image_summary(data: dict | None) -> str:
    if not isinstance(data, dict):
        return "payload_type=non_dict"
    content = str(data.get("content") or "")[:240].replace("\n", " ")
    warnings = data.get("warnings")
    return (
        f"id={data.get('id') or 'n/a'}; status={data.get('status') or 'n/a'}; "
        f"error={_polza_error_text(data) or 'n/a'}; content={content or 'n/a'}; "
        f"warnings={warnings if warnings else 'n/a'}"
    )


async def _polza_fetch_image_bytes(session: aiohttp.ClientSession, ref: str) -> bytes:
    if ref.startswith("data:"):
        comma = ref.find(",")
        if comma == -1:
            return b""
        return base64.b64decode(ref[comma + 1:])
    async with session.get(ref, timeout=aiohttp.ClientTimeout(total=120)) as r:
        r.raise_for_status()
        return await r.read()


def _build_polza_provider_preferences() -> dict | None:
    """Опциональный роутинг провайдеров Polza (provider.order/only/ignore)."""
    provider: dict = {}
    only = _env_csv("POLZA_IMAGE_PROVIDER_ONLY")
    order = _env_csv("POLZA_IMAGE_PROVIDER_ORDER")
    ignore = _env_csv("POLZA_IMAGE_PROVIDER_IGNORE")
    if only:
        provider["only"] = only
    if order:
        provider["order"] = order
    if ignore:
        provider["ignore"] = ignore
    return provider or None


async def polza_media_request(
    session: aiohttp.ClientSession,
    api_key: str,
    model: str,
    media_input: dict,
    *,
    timeout_sec: float,
    poll_interval: float = 3.0,
    on_progress=None,
    provider: dict | None = None,
) -> dict:
    """
    POST /v1/media (async=true) + опрос GET /v1/media/{id} до completed/failed.
    Возвращает финальный MediaStatusPresenter. Ошибки HTTP → Exception с префиксом polza_*.
    """
    base = polza_base_url()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload: dict = {"model": model, "input": media_input, "async": True}
    if provider:
        payload["provider"] = provider

    retries_429 = max(0, _env_int("POLZA_RATE_LIMIT_RETRIES", 2))
    attempt = 0
    while True:
        attempt += 1
        async with session.post(
            f"{base}/media", headers=headers, json=payload,
            timeout=aiohttp.ClientTimeout(total=90),
        ) as resp:
            body = await resp.text()
            try:
                data = json.loads(body) if body else {}
            except Exception:
                data = {}
            if resp.status in (200, 201, 202):
                break
            err_text = f"Polza media HTTP {resp.status}: {_polza_error_text(data) or body[:800]}"
            if resp.status == 429 and attempt <= retries_429:
                wait = _env_float("POLZA_RATE_LIMIT_WAIT_SECONDS", 5.0) * attempt
                try:
                    wait = max(wait, float(resp.headers.get("Retry-After") or 0))
                except ValueError:
                    pass
                logger.warning("[POLZA] 429 on submit, retry %s/%s in %.0fs", attempt, retries_429, min(wait, 30))
                await asyncio.sleep(min(wait, 30))
                continue
            if resp.status == 402 or is_polza_insufficient_credits_error(err_text):
                raise Exception(f"polza_insufficient_credits: {err_text}")
            if resp.status == 429:
                raise Exception(f"polza_rate_limited: {err_text}")
            if _polza_is_safety_text(err_text):
                raise Exception(f"polza_safety_blocked: {err_text}")
            raise Exception(err_text)

    status = str(data.get("status") or "pending")
    gen_id = data.get("id")
    if status in ("completed", "failed", "cancelled"):
        return data
    if not gen_id:
        raise Exception(f"Polza media: no id in response: {str(data)[:400]}")

    logger.info("[POLZA] media submitted id=%s model=%s status=%s", gen_id, model, status)
    started = time.time()
    deadline = started + timeout_sec
    polls = 0
    while time.time() < deadline:
        await asyncio.sleep(poll_interval)
        polls += 1
        try:
            async with session.get(
                f"{base}/media/{gen_id}", headers=headers,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    logger.warning("[POLZA] poll id=%s HTTP %s: %s", gen_id, resp.status, text[:200])
                    if resp.status in (401, 403, 404):
                        raise Exception(f"Polza media poll HTTP {resp.status}: {text[:400]}")
                    continue
                data = await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            logger.warning("[POLZA] poll id=%s transient error: %s", gen_id, exc)
            continue
        new_status = str(data.get("status") or status)
        if new_status != status:
            logger.info("[POLZA] media id=%s status=%s", gen_id, new_status)
            status = new_status
        if status in ("completed", "failed", "cancelled"):
            return data
        if on_progress and polls % 2 == 0:
            try:
                await on_progress(time.time() - started, status)
            except Exception:
                pass
    raise Exception(f"polza_timeout: media id={gen_id} not ready in {int(timeout_sec)}s")


async def _polza_image_request(
    session: aiohttp.ClientSession,
    api_key: str,
    model: str,
    user_text: str,
    images_bytes: list[bytes],
    mime_type: str,
    aspect_ratio: str,
    image_labels: list[str] | None = None,
    image_size: str | None = None,
) -> tuple[bytes, dict]:
    """Генерация/редактирование изображения через Polza Media API. Возвращает (bytes, status_payload)."""
    timeout_sec = max(45, _env_int("POLZA_IMAGE_TIMEOUT_SECONDS", 180))
    prompt = user_text
    if image_labels:
        # Media API не поддерживает подписи между картинками — переносим их в текст промпта.
        mapping = "\n".join(f"Image {i + 1}: {lbl}" for i, lbl in enumerate(image_labels))
        prompt = f"{user_text}\n\n{mapping}"

    files: list[dict] = []
    for raw in images_bytes[:8]:
        prep_bytes, pm = prepare_image_for_upload(raw, mime_type)
        b64 = base64.b64encode(prep_bytes).decode("ascii")
        files.append({"type": "base64", "data": f"data:{pm};base64,{b64}"})

    ar = aspect_ratio if aspect_ratio in _POLZA_IMAGE_ASPECTS else "1:1"
    media_input: dict = {
        "prompt": prompt,
        "aspect_ratio": ar,
        "image_resolution": image_size or (os.getenv("POLZA_IMAGE_RESOLUTION") or "1K").strip(),
        "output_format": (os.getenv("POLZA_IMAGE_OUTPUT_FORMAT") or "png").strip(),
    }
    if files:
        media_input["images"] = files

    empty_retries = max(0, _env_int("POLZA_IMAGE_EMPTY_RETRIES", 0))
    retry_delay = max(0.2, _env_float("POLZA_IMAGE_RETRY_DELAY_SECONDS", 1.5))
    attempt = 0
    while True:
        attempt += 1
        data = await polza_media_request(
            session, api_key, model, media_input,
            timeout_sec=timeout_sec,
            poll_interval=max(0.5, _env_float("POLZA_IMAGE_POLL_SECONDS", 1.0)),
            provider=_build_polza_provider_preferences(),
        )
        status = str(data.get("status") or "")
        ref = _polza_extract_image_ref(data) if status == "completed" else None
        if ref:
            break
        summary = _polza_no_image_summary(data)
        err_text = _polza_error_text(data)
        if status in ("failed", "cancelled"):
            if _polza_is_safety_text(err_text):
                raise Exception(f"polza_image_safety_blocked:{summary}")
            if "location" in err_text.lower() and "not supported" in err_text.lower():
                raise Exception(f"polza_location_unsupported:{summary}")
            if is_polza_insufficient_credits_error(err_text):
                raise Exception(f"polza_insufficient_credits:{summary}")
        if attempt <= empty_retries:
            logger.warning("[POLZA] no image, retry %s/%s model=%s summary=%s", attempt, empty_retries, model, summary)
            await asyncio.sleep(retry_delay * attempt)
            continue
        if status in ("failed", "cancelled"):
            raise Exception(f"polza_generation_failed:{summary}")
        if _polza_is_safety_text(str(data.get("content") or "")):
            raise Exception(f"polza_image_safety_blocked:{summary}")
        raise Exception(f"polza_empty_image:{summary}")

    img_bytes = await _polza_fetch_image_bytes(session, ref)
    if not img_bytes:
        raise Exception("Polza image API: empty image payload")
    return img_bytes, data


async def upscale_image_vertex(session: aiohttp.ClientSession, api_key: str, image_bytes: bytes,
                               prompt: str = None, aspect_ratio: str = "1:1") -> bytes | None:
    """Усиление детализации / 4K через второй запрос к Polza image."""
    if not _is_enabled("POLZA_IMAGE_UPSCALE", "1"):
        return None
    logger.info("[POLZA Upscale] Второй проход по изображению...")
    try:
        up_model = _resolve_polza_image_model(
            (
                os.getenv("POLZA_UPSCALE_MODEL")
                or os.getenv("POLZA_IMAGE_MODEL_CHEAP")
                or "google/gemini-nano-banana-2.1"
            )
        )
        if prompt and ("model" in prompt.lower() or "person" in prompt.lower() or "face" in prompt.lower()):
            txt = (
                "Upscale and refine this image to maximum commercial quality and resolution. "
                "Preserve faces, pose, clothing, and background exactly — no new objects, no redesign. "
                "Output one image only."
            )
        else:
            txt = (
                "Upscale this product image to maximum commercial sharpness and resolution. "
                "Preserve the product and composition exactly — no new elements, no people if none were present. "
                "Output one image only."
            )
        up_started = time.time()
        ar = aspect_ratio if aspect_ratio in ("1:1", "16:9", "9:16", "4:3", "3:4", "4:5", "5:4", "3:2", "2:3", "21:9") else "1:1"
        out, up_data = await _polza_image_request(
            session, api_key, up_model, txt, [image_bytes], "image/jpeg", ar,
        )
        up_usage = up_data.get("usage") if isinstance(up_data, dict) else None
        up_cost = _provider_rub_from_polza_payload(up_usage, up_data)
        logger.info(
            "[POLZA Upscale] Готово за %.1fs, model=%s, cost_rub=%s, tokens=%s",
            time.time() - up_started, up_model,
            f"{up_cost:.2f}" if up_cost is not None else "flat",
            up_usage.get("total_tokens") if up_usage else "?",
        )
        try:
            await record_polza_usage(
                api_key=api_key,
                model=up_model,
                usage=up_usage,
                latency_ms=int((time.time() - up_started) * 1000),
                status="ok",
                full_response=up_data if isinstance(up_data, dict) else None,
            )
        except Exception:
            pass
        return out
    except Exception as exc:
        logger.error("[POLZA Upscale] Ошибка: %s", exc)
        return None


def _ensure_4k_resolution(image_bytes: bytes) -> bytes:
    """
    Гарантирует повышенное разрешение для режима 4K.
    Если длинная сторона меньше 3840, делаем high-quality upscale (Lanczos).
    """
    target_long_side = max(2560, _env_int("POLZA_4K_LONG_SIDE", 3840))
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = img.size
        long_side = max(w, h)
        if long_side >= target_long_side:
            return image_bytes

        scale = target_long_side / float(long_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        up = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        up.save(buf, format="JPEG", quality=96, optimize=True)
        return buf.getvalue()
    except Exception as exc:
        logger.warning("[POLZA 4K] local upscale fallback failed: %s", exc)
        return image_bytes


def _local_upscale_4k(image_bytes: bytes) -> bytes:
    """
    Локальный высококачественный upscale до 4K без API-вызовов.
    Использует OpenCV (из rembg) + Pillow для многопроходного улучшения:
      1. Двухступенчатый Lanczos resize (для лучшей интерполяции)
      2. Bilateral filter для деноиза без размытия границ
      3. Unsharp mask для резкости деталей
      4. CLAHE для локального контраста (текстуры ткани)
    """
    target_long_side = max(2560, _env_int("POLZA_4K_LONG_SIDE", 3840))
    jpeg_quality = _env_int("POLZA_4K_JPEG_QUALITY", 96)

    try:
        import cv2
        import numpy as np
    except ImportError:
        logger.warning("[LOCAL_UPSCALE] opencv/numpy not available, falling back to Pillow Lanczos")
        return _ensure_4k_resolution(image_bytes)

    try:
        # Декодируем изображение
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("[LOCAL_UPSCALE] cv2.imdecode вернул None")
            return _ensure_4k_resolution(image_bytes)

        h, w = img.shape[:2]
        long_side = max(w, h)
        if long_side >= target_long_side:
            logger.info("[LOCAL_UPSCALE] Изображение уже %dx%d, upscale не нужен", w, h)
            return image_bytes

        scale = target_long_side / float(long_side)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        logger.info("[LOCAL_UPSCALE] %dx%d → %dx%d (scale %.2fx)", w, h, new_w, new_h, scale)

        # --- Шаг 1: Двухступенчатый resize для лучшего качества ---
        if scale > 2.5:
            mid_w = int(round(w * (scale ** 0.5)))
            mid_h = int(round(h * (scale ** 0.5)))
            img = cv2.resize(img, (mid_w, mid_h), interpolation=cv2.INTER_LANCZOS4)
            logger.info("[LOCAL_UPSCALE] Промежуточный шаг: %dx%d", mid_w, mid_h)

        upscaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # --- Шаг 2: Bilateral filter — убираем шум, сохраняем границы ---
        upscaled = cv2.bilateralFilter(upscaled, d=5, sigmaColor=40, sigmaSpace=40)

        # --- Шаг 3: Unsharp mask — восстанавливаем резкость ---
        gaussian = cv2.GaussianBlur(upscaled, (0, 0), sigmaX=2.0)
        sharpness_amount = 1.5
        upscaled = cv2.addWeighted(upscaled, 1.0 + sharpness_amount, gaussian, -sharpness_amount, 0)
        upscaled = np.clip(upscaled, 0, 255).astype(np.uint8)

        # --- Шаг 4: CLAHE на L-канале (локальный контраст для текстур) ---
        lab = cv2.cvtColor(upscaled, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        lab = cv2.merge([l_channel, a_channel, b_channel])
        upscaled = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        # Кодируем результат
        # PNG, а не JPEG: апскейл и без того интерполирует, добавлять поверх
        # JPEG-артефакты нельзя. Признак ошибки — файл ~1.8–2.5 МБ вместо 10–20 МБ.
        # POLZA_4K_JPEG_QUALITY оставлен для отката на JPEG через POLZA_4K_FORMAT=jpeg.
        if (os.getenv("POLZA_4K_FORMAT") or "png").lower() == "jpeg":
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1]
            success, buf = cv2.imencode(".jpg", upscaled, encode_params)
        else:
            success, buf = cv2.imencode(".png", upscaled, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        if not success:
            logger.warning("[LOCAL_UPSCALE] cv2.imencode не удалось")
            return _ensure_4k_resolution(image_bytes)

        result = buf.tobytes()
        logger.info(
            "[LOCAL_UPSCALE] Готово: %dx%d, размер %.1f KB → %.1f KB",
            new_w, new_h,
            len(image_bytes) / 1024, len(result) / 1024,
        )
        return result

    except Exception as exc:
        logger.error("[LOCAL_UPSCALE] Ошибка: %s", exc, exc_info=True)
        return _ensure_4k_resolution(image_bytes)


async def generate_image(
        api_key: str, prompt: str, images_bytes: Optional[list[bytes]] = None,
        aspect_ratio: str = "1:1", quality: str = "HD", key_id: int = 0,
        db_instance=None, generation_meta: dict = None, mime_type: str = "image/jpeg",
        model_name: str | None = None, ref_image_bytes: bytes | None = None,
) -> Optional[str]:
    """
    Генерация изображения через Polza.ai Media API (POST /v1/media + опрос GET /v1/media/{id}).
    См. https://polza.ai/docs/api-reference/media/create
    """
    z_key = await acquire_polza_key_slot()
    if not z_key:
        raise Exception("polza_keys_busy")

    images = images_bytes if isinstance(images_bytes, list) else [images_bytes] if images_bytes else []
    images = [img for img in images if img]
    if ref_image_bytes:
        images.append(ref_image_bytes)

    asp = str(aspect_ratio).lower().replace("x", ":").strip()
    if asp == "4:5":
        asp = "3:4"
    elif asp == "5:4":
        asp = "4:3"
    elif asp not in ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"]:
        asp = "1:1"

    # 4K → pro-модель (качественная), 1K → flash-модель (дешёвая/быстрая).
    image_model = _pick_polza_image_model(model_name, generation_meta, quality=str(quality))
    polza_user_id: int | None = None
    if isinstance(generation_meta, dict) and generation_meta.get("user_id") is not None:
        try:
            polza_user_id = int(generation_meta.get("user_id"))
        except Exception:
            polza_user_id = None

    async with aiohttp.ClientSession() as session:
        try:
            t0 = time.time()
            category_for_polza = ""
            if isinstance(generation_meta, dict):
                category_for_polza = str(generation_meta.get("category") or "")
            prompt, polza_used = await enrich_prompt_with_polza(
                session=session,
                base_prompt=prompt,
                images_bytes=images,
                aspect_ratio=asp,
                category=category_for_polza,
            )
            if generation_meta is not None:
                generation_meta["polza_prompt_enhanced"] = polza_used
                generation_meta["polza_key_preview"] = _mask_key_preview(z_key)
                generation_meta["kie_key_preview"] = generation_meta["polza_key_preview"]
            if _is_enabled("LOG_PROMPT_PREVIEW", "0"):
                logger.info("[PROMPT] len=%s sha1=%s", len(prompt), hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12])
                logger.info("[PROMPT] preview=%s", prompt[:700].replace("\n", " "))

            # Не добавляем суффиксы — используем только промпт из админки.
            full_user_text = prompt.strip()
            logger.info("[POLZA] Генерация изображения model=%s images=%s", image_model, len(images))

            img_labels = None
            if category_for_polza == "collage" and len(images) >= 3:
                img_labels = [
                    "[Photo 1 — Reference model photo]",
                    "[Photo 2 — Product front view photo]",
                    "[Photo 3 — Product back view photo]",
                ]

            gen_started = time.time()
            try:
                final_image_bytes, z_data = await _polza_image_request(
                    session, z_key, image_model, full_user_text, images, mime_type, asp,
                    image_labels=img_labels,
                    image_size=("4K" if str(quality).lower() == "4k" else None),
                )
            except Exception as first_exc:
                first_err = str(first_exc or "")
                err_l = first_err.lower()
                is_safety = "polza_image_safety_blocked" in err_l
                is_location = "polza_location_unsupported" in err_l or "user location is not supported" in err_l
                is_empty_image = "polza_empty_image" in err_l or "no image in response" in err_l or "empty image payload" in err_l
                if not (is_safety or is_empty_image or is_location):
                    raise
                reason = "LOCATION_UNSUPPORTED" if is_location else ("EMPTY_IMAGE" if is_empty_image else "IMAGE_SAFETY")
                logger.warning(
                    "[POLZA] %s on %s, fallback disabled, ask user to retry: %s",
                    reason,
                    image_model,
                    first_err[:600],
                )
                if is_location:
                    raise Exception("image_generation_provider_location_unsupported") from first_exc
                if is_safety:
                    raise Exception("image_generation_safety_blocked") from first_exc
                raise Exception("image_generation_empty_image") from first_exc
            gen_usage = z_data.get("usage") if isinstance(z_data, dict) else None
            gen_cost = _provider_rub_from_polza_payload(gen_usage, z_data)
            logger.info(
                "[POLZA] Генерация: model=%s, cost_rub=%s, tokens=%s, time=%.1fs",
                image_model,
                f"{gen_cost:.2f}" if gen_cost is not None else "flat",
                gen_usage.get("total_tokens") if gen_usage else "?",
                time.time() - gen_started,
            )
            await record_polza_usage(
                api_key=z_key,
                model=image_model,
                usage=gen_usage,
                latency_ms=int((time.time() - gen_started) * 1000),
                status="ok",
                full_response=z_data if isinstance(z_data, dict) else None,
                user_id=polza_user_id,
            )

            # 4K включается только если UI/хендлеры передали quality=="4k".
            # Глобальный запрет (app_settings.disable_4k) форсит 1k ещё до вызова generate_image.
            is_4k = str(quality).lower() == "4k"
            if is_4k:
                logger.info("[POLZA] 4K: локальный upscale до %spx по длинной стороне...",
                            _env_int("POLZA_4K_LONG_SIDE", 3840))
                final_image_bytes = _local_upscale_4k(final_image_bytes)

            if generation_meta is not None:
                generation_meta["model_used"] = image_model
                generation_meta["upscaled"] = is_4k

            os.makedirs(os.path.join(BASE_DIR, "data", "history"), exist_ok=True)
            # 4K-апскейл отдаёт PNG — расширение определяем по сигнатуре файла
            _ext = "png" if final_image_bytes[:8] == b"\x89PNG\r\n\x1a\n" else "jpg"
            file_name = f"result_{uuid.uuid4().hex}.{_ext}"
            file_path = os.path.join(BASE_DIR, "data", "history", file_name)
            with open(file_path, "wb") as f:
                f.write(final_image_bytes)

            logger.info("[POLZA] Генерация завершена за %.1fs → %s", time.time() - t0, file_path)
            return file_path

        except Exception as e:
            logger.error("[POLZA] Критическая ошибка: %s", e)
            try:
                await record_polza_usage(
                    api_key=z_key,
                    model=image_model,
                    usage=None,
                    latency_ms=0,
                    status="exception",
                    error_text=str(e)[:500],
                    user_id=polza_user_id,
                )
            except Exception:
                pass
            raise e
        finally:
            await release_polza_key_slot(z_key)


def is_fatal_key_error(exc: BaseException) -> bool:
    """Ошибки ключа/доступа — ключ в админке можно деактивировать."""
    sl = str(exc).lower()
    for needle in (
        "401", "403", "invalid api", "incorrect api", "unauthorized", "forbidden",
        "api key", "authentication", "неверный ключ", "недействителен",
    ):
        if needle in sl:
            return True
    return False


def is_proxy_error(exc: BaseException) -> bool:
    """Сетевые/временные сбои (для логирования api_key_errors), не считаем фатальными для ключа."""
    if is_fatal_key_error(exc):
        return False
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, aiohttp.ServerTimeoutError)):
        return True
    if isinstance(exc, aiohttp.ClientError):
        return True
    sl = str(exc).lower()
    for needle in (
        "503", "502", "504", "timeout", "timed out", "connection", "network", "proxy",
        "temporarily unavailable", "eof", "reset", "broken pipe",
    ):
        if needle in sl:
            return True
    return False


def is_polza_insufficient_credits_error(exc: BaseException | str) -> bool:
    sl = str(exc).lower()
    return (
        "insufficient credits" in sl
        or "insufficient_balance" in sl
        or "insufficient funds" in sl
        or "http 402" in sl
        or '"code":402' in sl
        or "code 402" in sl
    )


async def generate_content_with_image(
    prompt: str,
    image_bytes: bytes | None = None,
) -> tuple[str, bytes | None]:
    """
    Утилита/тест: одна картинка по промпту (опционально с входным изображением) через Polza image API.
    """
    z_key = await acquire_polza_key_slot()
    if not z_key:
        raise RuntimeError("polza_keys_busy")
    model = _resolve_polza_image_model(None)
    images = [image_bytes] if image_bytes else []
    try:
        async with aiohttp.ClientSession() as session:
            img_out, data = await _polza_image_request(
                session,
                z_key,
                model,
                (prompt or "").strip(),
                images,
                "image/jpeg",
                "1:1",
            )
            choices = data.get("choices") or []
            msg = (choices[0] or {}).get("message") or {} if choices else {}
            txt = _extract_chat_content_text(msg.get("content"))
            return txt, img_out
    finally:
        await release_polza_key_slot(z_key)


def generate_clothing_mask_sync(image_bytes: bytes) -> bytes:
    """
    Синхронная функция для генерации пустой маски.
    Используется в качестве заглушки, если другие части кода требуют этот метод.
    """
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Создаем полностью белую маску
        mask = Image.new("L", img.size, 255)
        buf = io.BytesIO()
        mask.save(buf, "JPEG")
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Error in sync mask: {e}")
        return image_bytes


async def get_task_info(task_id: str) -> dict:
    """Заглушка: опрос задач Polza выполняется внутри polza_media_request()."""
    return {"taskId": task_id, "status": "n/a", "message": "Polza: polling выполняется внутри генерации."}
