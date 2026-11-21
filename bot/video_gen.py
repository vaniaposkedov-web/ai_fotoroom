"""Video Generation client (Polza.ai Media API)."""
from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
from typing import Optional

import aiohttp
from PIL import Image

logger = logging.getLogger(__name__)


class VideoError(Exception):
    """Base video generation error with category for user-facing messages."""
    def __init__(self, message: str, category: str = "unknown"):
        super().__init__(message)
        self.category = category


class VideoRateLimitError(VideoError):
    def __init__(self, message: str = "rate_limited"):
        super().__init__(message, category="rate_limit")


class VideoSubmitError(VideoError):
    def __init__(self, message: str):
        super().__init__(message, category="submit_failed")


class VideoTimeoutError(VideoError):
    def __init__(self, message: str = "poll_timeout"):
        super().__init__(message, category="timeout")


class VideoGenerationFailed(VideoError):
    def __init__(self, message: str):
        cat = "unknown"
        msg_lower = message.lower()
        if "safety" in msg_lower or "blocked" in msg_lower or "content" in msg_lower:
            cat = "safety"
        elif "429" in message or "rate" in msg_lower:
            cat = "rate_limit"
        super().__init__(message, category=cat)


class VideoBillingError(VideoError):
    """Provider rejected the job for lack of funds (HTTP 402 / insufficient funds)."""
    def __init__(self, message: str):
        super().__init__(message, category="insufficient_funds")


def _is_retryable_video_failure(err_str: str) -> bool:
    """Whether a *failed* video job is worth resubmitting (transient upstream hiccup).

    Excludes permanent rejections (safety/content/invalid input) and billing,
    where a retry would only waste time and credits.
    """
    s = (err_str or "").lower()
    if any(t in s for t in (
        "safety", "blocked", "content policy", "nsfw",
        "invalid", "unsupported", "insufficient",
    )):
        return False
    return any(t in s for t in (
        "429", "rate", "http failed", "timeout", "timed out", "temporar",
        "internal", "500", "502", "503", "504", "unavailable", "try again", "failed",
    ))

# Polza.ai Media API: POST /v1/media (async) -> GET /v1/media/{id} -> data.url (MP4 на CDN, 7 дней).
VIDEO_MODEL = (os.getenv("POLZA_VIDEO_MODEL") or "grok-imagine-video-1-5").strip()
VIDEO_RESOLUTION = os.getenv("POLZA_VIDEO_RESOLUTION", "720p")
VIDEO_POLL_INTERVAL = float(os.getenv("POLZA_VIDEO_POLL_INTERVAL", "8"))
VIDEO_POLL_TIMEOUT = float(os.getenv("POLZA_VIDEO_POLL_TIMEOUT", "600"))
VIDEO_MAX_PROMPT_LEN = 2500


def _polza_base_url() -> str:
    from bot.gemini import polza_base_url
    return polza_base_url()


def _error_message(body: dict, fallback: str = "") -> str:
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        parts = [str(err.get("code") or ""), str(err.get("message") or "")]
        meta = err.get("metadata")
        if isinstance(meta, dict) and meta.get("raw"):
            parts.append(str(meta.get("raw")))
        msg = " ".join(p for p in parts if p).strip()
        if msg:
            return msg
    if err:
        return str(err)
    return fallback


# Форматы, которые модель принимает в aspect_ratio (Grok Imagine Video 1.5: 1:1, 16:9, 9:16, 3:2, 2:3, auto).
# Для остальных (напр. 3:4) входной кадр уже обрезан под формат, а в API уходит "auto" —
# модель берёт пропорции из кадра.
VIDEO_API_ASPECTS = set(
    a.strip() for a in (os.getenv("POLZA_VIDEO_ASPECTS") or "1:1,16:9,9:16,3:2,2:3").split(",") if a.strip()
)


def _api_aspect_ratio(aspect_ratio: str) -> str:
    return aspect_ratio if aspect_ratio in VIDEO_API_ASPECTS else "auto"


async def submit_video_job(
    api_key: str,
    prompt: str,
    duration: int = 5,
    frame_images: list[dict] | None = None,
    aspect_ratio: str = "9:16",
) -> dict:
    """POST /v1/media — submit async video generation job. Returns {id, status, ...}."""
    endpoint = f"{_polza_base_url()}/media"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    media_input: dict = {
        "prompt": prompt[:VIDEO_MAX_PROMPT_LEN] if prompt else "",
        "duration": str(int(duration)),
        "resolution": VIDEO_RESOLUTION,
        "aspect_ratio": _api_aspect_ratio(aspect_ratio),
    }
    if frame_images:
        media_input["images"] = frame_images
    payload = {"model": VIDEO_MODEL, "input": media_input, "async": True}

    logger.info("[VIDEO] submit model=%s duration=%ss resolution=%s aspect=%s frames=%s prompt_len=%d prompt_preview=%.100s",
                VIDEO_MODEL, duration, VIDEO_RESOLUTION, aspect_ratio,
                len(frame_images or []), len(prompt or ""), repr((prompt or "")[:100]))

    timeout = aiohttp.ClientTimeout(total=90)
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(endpoint, headers=headers, json=payload) as resp:
                try:
                    body = await resp.json(content_type=None)
                except Exception:
                    body = {"error": (await resp.text())[:500]}
                if resp.status in (200, 201, 202):
                    logger.info("[VIDEO] job submitted: id=%s status=%s", body.get("id"), body.get("status"))
                    return body

                msg = _error_message(body, f"HTTP {resp.status}")
                logger.error("[VIDEO] submit failed HTTP %s (attempt %s/%s): %s",
                             resp.status, attempt, max_retries, msg)

                if resp.status == 429 and attempt < max_retries:
                    wait = 15 * attempt
                    logger.info("[VIDEO] rate limited, retrying in %ss...", wait)
                    await asyncio.sleep(wait)
                    continue

                if resp.status == 429:
                    raise VideoRateLimitError(msg)
                low = msg.lower()
                if resp.status == 402 or "insufficient" in low:
                    raise VideoBillingError(msg)
                raise VideoSubmitError(f"HTTP {resp.status}: {msg}")
    raise VideoSubmitError("video_submit_failed")


async def poll_video_job(api_key: str, job_id: str, on_progress=None) -> dict:
    """Poll GET /v1/media/{id} until terminal status (completed / failed / cancelled).

    on_progress(elapsed_sec, status) — optional async callback for progress updates.
    """
    endpoint = f"{_polza_base_url()}/media/{job_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    timeout = aiohttp.ClientTimeout(total=30)

    deadline = time.time() + VIDEO_POLL_TIMEOUT
    start_time = time.time()
    last_status = "pending"
    poll_count = 0

    async with aiohttp.ClientSession(timeout=timeout) as session:
        while time.time() < deadline:
            try:
                async with session.get(endpoint, headers=headers) as resp:
                    body = await resp.json(content_type=None)
                    status = body.get("status", "unknown")
                    if status != last_status:
                        logger.info("[VIDEO] poll id=%s status=%s", job_id, status)
                        last_status = status
                    if status in ("completed", "failed", "cancelled"):
                        return body
            except Exception as exc:
                logger.warning("[VIDEO] poll error for %s: %s", job_id, exc)

            poll_count += 1
            elapsed = time.time() - start_time
            if on_progress and poll_count % 2 == 0:
                try:
                    await on_progress(elapsed, last_status)
                except Exception:
                    pass
            await asyncio.sleep(VIDEO_POLL_INTERVAL)

    raise VideoTimeoutError("video_poll_timeout")


def _video_url_from_result(job_data: dict) -> str | None:
    for key in ("data", "output"):
        block = job_data.get(key)
        items = block if isinstance(block, list) else [block]
        for item in items:
            if isinstance(item, dict) and isinstance(item.get("url"), str) and item["url"]:
                return item["url"]
    return None


async def download_video(api_key: str, job_data: dict, save_path: str) -> str:
    """Скачивает MP4 по data.url из результата Polza."""
    url = _video_url_from_result(job_data)
    if not url:
        raise Exception(f"video_download_failed: no url in result {str(job_data)[:300]}")
    timeout = aiohttp.ClientTimeout(total=180)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise Exception(f"video_download_failed: HTTP {resp.status} {text[:300]}")
            with open(save_path, "wb") as f:
                async for chunk in resp.content.iter_chunked(1 << 20):
                    f.write(chunk)
    logger.info("[VIDEO] downloaded %s -> %s", job_data.get("id"), save_path)
    return save_path


VIDEO_MIN_PX = 300
VIDEO_MAX_PX = 2048


def _resize_image_bytes(img_bytes: bytes) -> bytes:
    """Resize image so width and height are within [300, 2048] range. Returns JPEG bytes."""
    img = Image.open(io.BytesIO(img_bytes))
    w, h = img.size

    if VIDEO_MIN_PX <= w <= VIDEO_MAX_PX and VIDEO_MIN_PX <= h <= VIDEO_MAX_PX:
        if img.format == "JPEG":
            return img_bytes
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=92)
        return buf.getvalue()

    scale = 1.0
    if w > VIDEO_MAX_PX or h > VIDEO_MAX_PX:
        scale = min(VIDEO_MAX_PX / w, VIDEO_MAX_PX / h)
    if w * scale < VIDEO_MIN_PX or h * scale < VIDEO_MIN_PX:
        scale = max(VIDEO_MIN_PX / w, VIDEO_MIN_PX / h)

    new_w = max(VIDEO_MIN_PX, min(VIDEO_MAX_PX, int(w * scale)))
    new_h = max(VIDEO_MIN_PX, min(VIDEO_MAX_PX, int(h * scale)))

    img = img.convert("RGB").resize((new_w, new_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    logger.info("[VIDEO] resized image %dx%d -> %dx%d", w, h, new_w, new_h)
    return buf.getvalue()


def _crop_to_aspect(img_bytes: bytes, aspect_ratio: str) -> bytes:
    """Center-crop image to the target W:H aspect.

    Kling image-to-video derives the OUTPUT aspect from the input frame and ignores
    the API ``aspect_ratio`` field, so to honor the user's chosen format (e.g. 9:16,
    4:5) we must conform the input frame itself.
    """
    try:
        tw, th = aspect_ratio.split(":")
        target = float(tw) / float(th)
    except Exception:
        return img_bytes
    if target <= 0:
        return img_bytes

    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    cur = w / h
    if abs(cur - target) < 0.01:
        return img_bytes  # already correct shape

    if cur > target:
        new_w = max(1, int(round(h * target)))
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = max(1, int(round(w / target)))
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    logger.info("[VIDEO] cropped input %dx%d -> %dx%d for aspect %s", w, h, img.size[0], img.size[1], aspect_ratio)
    return buf.getvalue()


def build_frame_images_from_bytes(
    images: list[bytes],
    labels: list[str] | None = None,
) -> list[dict]:
    """Build Polza `images` payload (MediaFileDto) from raw image bytes.

    Порядок важен: первое изображение — стартовый кадр (image-to-video).
    labels оставлен для совместимости вызовов и не передаётся в API.
    """
    result = []
    limit = len(labels) if labels else len(images)
    for img_bytes in images[:limit]:
        resized = _resize_image_bytes(img_bytes)
        b64 = base64.b64encode(resized).decode("ascii")
        result.append({"type": "base64", "data": f"data:image/jpeg;base64,{b64}"})
    return result


async def generate_video(
    api_key: str,
    prompt: str,
    duration: int = 5,
    image_bytes: bytes | None = None,
    aspect_ratio: str = "9:16",
    on_progress=None,
) -> tuple[str, dict]:
    """Full pipeline: submit → poll → download. Returns (video_path, job_data)."""
    frame_images = None
    if image_bytes:
        image_bytes = _crop_to_aspect(image_bytes, aspect_ratio)
        frame_images = build_frame_images_from_bytes([image_bytes], ["first_frame"])

    max_retries_on_429 = int(os.getenv("VIDEO_429_RETRIES", "2"))
    for attempt in range(1, max_retries_on_429 + 1):
        job = await submit_video_job(api_key, prompt, duration, frame_images, aspect_ratio)
        job_id = job.get("id")
        if not job_id:
            raise VideoSubmitError("video_submit_no_id")

        result = await poll_video_job(api_key, job_id, on_progress=on_progress)
        status = result.get("status")
        if status == "completed":
            break

        err_str = _error_message(result, "unknown error")
        if attempt < max_retries_on_429 and _is_retryable_video_failure(err_str):
            wait = 10 * attempt
            logger.warning("[VIDEO] job %s failed (%.160s), retrying in %ss (attempt %s/%s)...",
                           job_id, err_str, wait, attempt, max_retries_on_429)
            await asyncio.sleep(wait)
            continue
        logger.error("[VIDEO] job %s failed permanently: %s | full=%r", job_id, err_str, result)
        raise VideoGenerationFailed(err_str)

    os.makedirs("/tmp/video_gen", exist_ok=True)
    save_path = f"/tmp/video_gen/{job_id}.mp4"
    await download_video(api_key, result, save_path)
    return save_path, result
