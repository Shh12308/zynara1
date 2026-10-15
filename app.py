import os
import re
import json
import uuid
import asyncio
import logging
import time
import base64
import io
import tempfile
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timezone, timedelta
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, Request, Response, HTTPException, UploadFile, File, Query, Form, Header
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from supabase import create_client
from PIL import Image

# =========================
# CONFIG & LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("HeloXAi")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY").strip() if os.getenv("GROQ_API_KEY") else None
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# fal.ai — real video generation
FAL_KEY = os.getenv("FAL_KEY")
FAL_VIDEO_MODEL = os.getenv("FAL_VIDEO_MODEL", "fal-ai/wan")
FAL_VIDEO_MAX_WAIT = int(os.getenv("FAL_VIDEO_MAX_WAIT", "240"))   # seconds (4 min)
FAL_VIDEO_POLL_INTERVAL = float(os.getenv("FAL_VIDEO_POLL_INTERVAL", "2.0"))
FAL_VIDEO_MAX_POLL_ERRORS = 5

MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_TEXT_LENGTH = 100000
MAX_IMAGE_SIZE = 20 * 1024 * 1024

SESSION_DURATION = 365 * 24 * 60 * 60
REFRESH_THRESHOLD = 7 * 24 * 60 * 60

GROQ_MAX_RETRIES = 3


app = FastAPI(
    title="HeloxAI Pro",
    description="Agentic AI Backend with Tools, Code Execution, Projects & Memory",
    version="5.0.0" 
)

# =========================
# MODEL CONFIGURATION
# =========================
OPENAI_CHAT_MODEL = "gpt-4o-mini"
OPENAI_VISION_MODEL = "gpt-4o-mini"
GROQ_STT_MODEL = "whisper-large-v3"
OPENAI_TTS_MODEL = "tts-1"
OPENAI_IMAGE_MODEL = "gpt-image-1"

MODEL_ROUTING = {
    "helox": {
        "chat": OPENAI_CHAT_MODEL,
        "vision": OPENAI_VISION_MODEL,
        "provider": "openai"
    },
    "chatgpt": {
        "chat": "gpt-4o-mini",
        "vision": "gpt-4o-mini",
        "provider": "openai"
    },
    "chatz": {
        "chat": OPENAI_CHAT_MODEL,
        "vision": OPENAI_VISION_MODEL,
        "provider": "openai"
    },
}

# =========================
# CORS CONFIGURATION
# =========================
service_url = os.getenv("RENDER_EXTERNAL_URL") or os.getenv("SERVICE_URL") or "https://heloxai2.onrender.com"
frontend_url = os.getenv("FRONTEND_URL", service_url)

allowed_origins = list({
    frontend_url,
    service_url,
    "https://heloxai.xyz",
    "https://www.heloxai.xyz",
    "capacitor://localhost",
})

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# =========================
# DATABASE & STATE
# =========================
supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
active_streams: Dict[str, asyncio.Task] = {}

_session_cache: Dict[str, Dict[str, Any]] = {}
_session_cache_ttl = 300

_rate_limit_store: Dict[str, List[float]] = {}
_conv_creation_locks: Dict[str, asyncio.Lock] = {}

_new_user_lock = asyncio.Lock()
_pending_new_user_id: Optional[str] = None
_new_user_created_event = asyncio.Event()


def _get_conv_lock(conv_id: str) -> asyncio.Lock:
    if conv_id not in _conv_creation_locks:
        _conv_creation_locks[conv_id] = asyncio.Lock()
    return _conv_creation_locks[conv_id]

# =========================
# RATE LIMITING
# =========================
IP_RATE_LIMIT = 30
IP_RATE_WINDOW = 60

ENDPOINT_LIMITS = {
    "/ask/universal":       {"limit": 20, "window": 60},
    "/tts":                 {"limit": 10, "window": 60},
    "/stt":                 {"limit": 10, "window": 60},
    "/analysis":            {"limit": 15, "window": 60},
}

_RATE_CLEANUP_INTERVAL = 300
_last_rate_cleanup = 0
_rate_store: Dict[str, List[float]] = {}


def _get_rate_key(client_ip: str, path: str) -> str:
    for ep in ENDPOINT_LIMITS:
        if path.startswith(ep):
            return f"{client_ip}:{ep}"
    return f"{client_ip}:__global__"

def _get_limits_for_key(key: str) -> Tuple[int, int]:
    for ep, cfg in ENDPOINT_LIMITS.items():
        if key.endswith(ep):
            return cfg["limit"], cfg["window"]
    return IP_RATE_LIMIT, IP_RATE_WINDOW

def _cleanup_rate_store(now: float):
    global _last_rate_cleanup
    if now - _last_rate_cleanup < _RATE_CLEANUP_INTERVAL:
        return
    _last_rate_cleanup = now
    stale_keys = []
    for key, timestamps in _rate_store.items():
        _, window = _get_limits_for_key(key)
        filtered = [t for t in timestamps if now - t < window]
        if filtered:
            _rate_store[key] = filtered
        else:
            stale_keys.append(key)
    for key in stale_keys:
        del _rate_store[key]

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/" or request.method == "OPTIONS":
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    _cleanup_rate_store(now)

    key = _get_rate_key(client_ip, request.url.path)
    limit, window = _get_limits_for_key(key)

    if key not in _rate_store:
        _rate_store[key] = []

    _rate_store[key] = [t for t in _rate_store[key] if now - t < window]
    current_count = len(_rate_store[key])

    if current_count >= limit:
        oldest = _rate_store[key][0]
        reset_at = int(oldest + window)
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please slow down."},
            headers={
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(reset_at),
                "Retry-After": str(max(1, reset_at - int(now))),
            }
        )

    _rate_store[key].append(now)
    remaining = limit - current_count - 1
    reset_at = int(_rate_store[key][0] + window)

    response = await call_next(request)

    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
    response.headers["X-RateLimit-Reset"] = str(reset_at)

    return response

# =========================
# AUTH SYSTEM
# =========================
PRIMARY_COOKIE = "HeloxAI_Session"
SESSION_TOKEN_COOKIE = "HeloxAI_Token"
SESSION_EXPIRY_COOKIE = "HeloxAI_Expiry"

def get_cookie_settings(remember: bool = True) -> Dict:
    base = {
        "max_age": SESSION_DURATION if remember else 24 * 60 * 60,
        "httponly": True,
        "secure": True,
        "samesite": "none",
        "path": "/"
    }
    cookie_domain = os.getenv("COOKIE_DOMAIN")
    if cookie_domain:
        base["domain"] = cookie_domain
    return base

def generate_session_token() -> str:
    import secrets
    return secrets.token_urlsafe(64)

def set_session_cookies(response: Response, user_id: str, token: str, remember: bool = True):
    settings = get_cookie_settings(remember)
    expiry = int(time.time()) + (SESSION_DURATION if remember else 24 * 60 * 60)
    response.set_cookie(key=PRIMARY_COOKIE, value=user_id, **settings)
    response.set_cookie(key=SESSION_TOKEN_COOKIE, value=token, **settings)
    response.set_cookie(key=SESSION_EXPIRY_COOKIE, value=str(expiry), **settings)

def clear_session_cookies(response: Response):
    cookie_domain = os.getenv("COOKIE_DOMAIN")
    for c in [PRIMARY_COOKIE, SESSION_TOKEN_COOKIE, SESSION_EXPIRY_COOKIE]:
        kwargs = {"key": c, "path": "/", "secure": True, "samesite": "none"}
        if cookie_domain:
            kwargs["domain"] = cookie_domain
        response.delete_cookie(**kwargs)

def is_session_expired(expiry_str: str) -> bool:
    try:
        return time.time() > int(expiry_str)
    except Exception:
        return True

async def validate_session_token(user_id: str, token: str) -> bool:
    try:
        if user_id in _session_cache and _session_cache[user_id].get("token") == token:
            cache_time = _session_cache[user_id].get("time", 0)
            if time.time() - cache_time < _session_cache_ttl:
                return True

        result = await asyncio.to_thread(
            supabase.table("user_sessions")
            .select("token")
            .eq("user_id", user_id)
            .eq("is_valid", True)
            .order("created_at", desc=True)
            .limit(1)
            .execute
        )

        if result.data and result.data[0]["token"] == token:
            _session_cache[user_id] = {"token": token, "time": time.time()}
            return True
        return False
    except Exception as e:
        logger.error(f"Session validation error: {e}")
        return False

async def ensure_user_exists(user_id: str) -> bool:
    try:
        await asyncio.to_thread(
            supabase.table("users")
            .upsert(
                {"id": user_id, "created_at": datetime.now(timezone.utc).isoformat()},
                on_conflict="id"
            ).execute
        )
        return True
    except Exception as e:
        logger.error(f"Failed to ensure user exists: {e}")
        return False

async def create_user_session(user_id: str, remember: bool = True) -> Optional[str]:
    if not await ensure_user_exists(user_id):
        return None

    token = generate_session_token()
    expires_at = datetime.now(timezone.utc) + timedelta(
        seconds=SESSION_DURATION if remember else 24 * 60 * 60
    )
    try:
        await asyncio.to_thread(
            supabase.table("user_sessions").insert({
                "id": str(uuid.uuid4()),
                "user_id": user_id,
                "token": token,
                "expires_at": expires_at.isoformat(),
                "is_valid": True,
                "created_at": datetime.now(timezone.utc).isoformat()
            }).execute
        )
        _session_cache[user_id] = {"token": token, "time": time.time()}
        return token
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        return None

# =========================
# SYSTEM PROMPTS
# =========================
BASE_SYSTEM_PROMPT = """You are HeloxAi, a powerful AI assistant.

**Capabilities:**
1. **Text & Reasoning:** Advanced understanding, reasoning, writing, and conversation.
2. **Coding:** Expert in writing, debugging, and reviewing code across all languages.
3. **Math & Data Analysis:** You can write and execute Python code to solve complex math, analyze data, or generate files.
4. **Research:** You have access to real-time web search. Use it for current events or facts.

**Response Style:**
- Use Markdown for structure (headers, bolding, code blocks, lists, tables).
- Be concise but thorough.
- For code execution, wrap your python code in ```python ... ``` blocks. You have the ability to execute it.
- For math, use LaTeX notation with $...$ for inline and $$...$$ for display math.

**Identity:**
- If asked who created you, say: "I was constructed by GoldYLocks. You can find them on Twitter @HeloxAi" """

FINANCE_SYSTEM_PROMPT = """You are HeloxAi, a financial analysis assistant.

You have access to real-time web search for financial data. When analyzing financial topics:

1. **Always** search for current data before answering
2. Provide specific numbers, percentages, and dates
3. Include relevant context (market conditions, comparisons)
4. **Disclaimer:** Always end with: "*Note: This is not financial advice. Do your own research before making investment decisions.*"
5. Use tables for comparing stocks/metrics when relevant
6. Cite your sources

Be precise with numbers. If you can't find current data, say so clearly."""

def get_system_prompt(user_prompt: str, project_instructions: str = "") -> str:
    prompt = BASE_SYSTEM_PROMPT
    if project_instructions:
        prompt += f"\n\n**Project Context & Instructions:**\n{project_instructions}"
    return prompt

# =========================
# HELPERS
# =========================
def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"

async def _execute_supabase_with_retry(query_builder):
    try:
        return await asyncio.to_thread(query_builder.execute)
    except Exception as e:
        logger.error(f"Supabase Error: {e}")
        raise

async def get_user(req: Request, res: Response, remember: bool = True) -> Dict[str, Any]:
    global _pending_new_user_id, _new_user_created_event

    user_id = req.cookies.get(PRIMARY_COOKIE)
    token = req.cookies.get(SESSION_TOKEN_COOKIE)
    expiry = req.cookies.get(SESSION_EXPIRY_COOKIE)

    if user_id and token:
        if is_session_expired(expiry or "0"):
            clear_session_cookies(res)
        elif await validate_session_token(user_id, token):
            return {"id": user_id, "session_valid": True}

    async with _new_user_lock:
        if _pending_new_user_id is not None:
            await _new_user_created_event.wait()
            _new_user_created_event.clear()

            if _pending_new_user_id != "__failed__":
                candidate_id = _pending_new_user_id
                result = await asyncio.to_thread(
                    supabase.table("user_sessions")
                    .select("token")
                    .eq("user_id", candidate_id)
                    .eq("is_valid", True)
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute
                )
                if result.data:
                    winning_token = result.data[0]["token"]
                    set_session_cookies(res, candidate_id, winning_token, remember)
                    _session_cache[candidate_id] = {"token": winning_token, "time": time.time()}
                    _pending_new_user_id = None
                    return {"id": candidate_id, "session_valid": True}
            _pending_new_user_id = None

        _pending_new_user_id = "creating"
        _new_user_created_event.clear()

    try:
        new_id = str(uuid.uuid4())
        new_token = await create_user_session(new_id, remember)
        if new_token is None:
            _pending_new_user_id = "__failed__"
            _new_user_created_event.set()
            raise HTTPException(500, "Failed to create user session")

        set_session_cookies(res, new_id, new_token, remember)

        _pending_new_user_id = new_id
        _new_user_created_event.set()

        return {"id": new_id, "session_valid": True}
    except HTTPException:
        _pending_new_user_id = "__failed__"
        _new_user_created_event.set()
        raise
    except Exception as e:
        _pending_new_user_id = "__failed__"
        _new_user_created_event.set()
        logger.error(f"Unexpected error in get_user: {e}")
        raise HTTPException(500, "Failed to create user session")

async def get_user_with_auth(req: Request, res: Response, remember: bool = True) -> Dict[str, Any]:
    auth_header = req.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "")

        if SUPABASE_ANON_KEY and token == SUPABASE_ANON_KEY:
            return await get_user(req, res, remember)

        try:
            user = await asyncio.to_thread(supabase.auth.get_user, token)
            if user and user.user:
                user_id = user.user.id
                await ensure_user_exists(user_id)
                return {"id": user_id, "session_valid": True}
        except Exception as e:
            logger.debug(f"Auth header validation failed: {e}")

    return await get_user(req, res, remember)

async def save_message(user_id: str, conv_id: str, role: str, content: str):
    data = {
        "id": str(uuid.uuid4()),
        "conversation_id": conv_id,
        "role": role,
        "content": content,
        "created_at": datetime.now(timezone.utc).isoformat()
    }
    await _execute_supabase_with_retry(supabase.table("messages").insert(data))

async def get_history(conv_id: str, limit: int = 6):
    res = await _execute_supabase_with_retry(
        supabase.table("messages")
        .select("role, content")
        .eq("conversation_id", conv_id)
        .order("created_at", desc=False)
        .limit(limit)
    )
    return [{"role": m["role"], "content": m["content"]} for m in (res.data or [])]

async def get_or_create_conversation(
    user_id: str,
    proposed_id: Optional[str],
    title: str,
    project_id: Optional[str] = None
) -> str:
    lock_key = proposed_id or "__new__"
    lock = _get_conv_lock(lock_key)

    async with lock:
        if proposed_id:
            for _retry in range(3):
                check = await _execute_supabase_with_retry(
                    supabase.table("conversations")
                    .select("id")
                    .eq("id", proposed_id)
                    .limit(1)
                )
                if check.data:
                    _conv_creation_locks.pop(lock_key, None)
                    return proposed_id
                if _retry < 2:
                    await asyncio.sleep(0.2)

        new_id = str(uuid.uuid4())
        logger.info(f"Creating new conversation: {new_id}")
        now = datetime.now(timezone.utc).isoformat()
        insert_data = {
            "id": new_id,
            "user_id": user_id,
            "title": title[:50],
            "created_at": now,
            "updated_at": now,
        }
        if project_id:
            insert_data["project_id"] = project_id
            
        await _execute_supabase_with_retry(
            supabase.table("conversations").insert(insert_data)
        )
        _conv_creation_locks.pop(lock_key, None)
        return new_id

# =========================
# API INTEGRATIONS
# =========================
def get_groq_headers():
    return {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

def get_groq_headers_multipart():
    return {"Authorization": f"Bearer {GROQ_API_KEY}"}

def get_openai_headers():
    return {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}

def _parse_retry_after(error_body: str) -> float:
    match = re.search(r'try again in ([\d\.]+)s', error_body)
    if match:
        return float(match.group(1)) + 0.5
    return 5.0

async def perform_web_search_formatted(query: str) -> Tuple[str, str]:
    """Returns (context_for_ai, html_for_frontend)"""
    if not TAVILY_API_KEY:
        return "[Search API Key missing]", ""

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post("https://api.tavily.com/search", json={
                "api_key": TAVILY_API_KEY,
                "query": query,
                "search_depth": "basic",
                "max_results": 5,
                "include_answer": True,
                "include_images": True,
                "include_raw_content": False,
            })
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            raw_images = data.get("images", [])

            if not results:
                return "[No search results found]", ""

            context = ""
            if data.get("answer"):
                context += f"Answer: {data['answer']}\n"
            for i, r in enumerate(results):
                context += f"[{i+1}] {r['title']}: {r['content']}\nURL: {r['url']}\n\n"

            domain_images = {}
            for img_url in raw_images[:20]:
                try:
                    parsed = urlparse(img_url)
                    domain = parsed.hostname or ""
                    if domain and domain not in domain_images:
                        domain_images[domain] = img_url
                except Exception:
                    pass

            html = '<div class="search-sources-bar">\n'
            html += '<i class="fa-solid fa-globe"></i> Sources:\n'
            for r in results[:5]:
                domain = urlparse(r["url"]).hostname or ""
                html += (
                    f'<a href="{r["url"]}" class="source-chip" '
                    f'target="_blank" rel="noopener">'
                    f'<img class="source-chip-img" '
                    f'src="https://www.google.com/s2/favicons?domain={domain}&sz=32" '
                    f'alt="" onerror="this.style.display=\'none\'">'
                    f'{domain}</a>\n'
                )
            html += '</div>\n\n'

            for i, r in enumerate(results[:4]):
                domain = urlparse(r["url"]).hostname or ""
                thumb_src = domain_images.get(domain, "")
                favicon_src = f"https://www.google.com/s2/favicons?domain={domain}&sz=32"

                if thumb_src:
                    html += (
                        f'<a href="{r["url"]}" class="search-card" '
                        f'target="_blank" rel="noopener">'
                        f'<img class="search-thumb" '
                        f'src="{thumb_src}" '
                        f'alt="" loading="lazy" '
                        f'onerror="this.src=\'{favicon_src}\';this.style.width=\'32px\';this.style.height=\'32px\';this.style.borderRadius=\'6px\';">'
                        f'<div class="search-info">'
                        f'<div class="search-title">{r["title"]}</div>'
                        f'<div class="search-link">'
                        f'<img class="search-link-favicon" '
                        f'src="{favicon_src}" '
                        f'alt="" onerror="this.style.display=\'none\'">'
                        f'{domain}</div>'
                        f'<div class="search-snippet">'
                        f'{r.get("content", "")[:300]}</div>'
                        f'</div></a>\n\n'
                    )
                else:
                    html += (
                        f'<a href="{r["url"]}" class="search-card compact" '
                        f'target="_blank" rel="noopener">'
                        f'<div class="search-info">'
                        f'<div class="search-title">{r["title"]}</div>'
                        f'<div class="search-link">'
                        f'<img class="search-link-favicon" '
                        f'src="{favicon_src}" '
                        f'alt="" onerror="this.style.display=\'none\'">'
                        f'{r["url"][:80]}</div>'
                        f'<div class="search-snippet">'
                        f'{r.get("content", "")[:300]}</div>'
                        f'</div></a>\n\n'
                    )

            return context, html

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return "[Search failed]", ""

async def stream_groq_chat(messages: list, model: str = None):
    use_model = model or "llama-3.3-70b-versatile"
    attempt = 0
    while attempt < GROQ_MAX_RETRIES:
        attempt += 1
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream(
                    "POST",
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=get_groq_headers(),
                    json={
                        "model": use_model,
                        "messages": messages,
                        "stream": True,
                        "max_tokens": 4096
                    }
                ) as resp:
                    if resp.status_code == 429:
                        error_body = (await resp.aread()).decode()
                        retry_delay = _parse_retry_after(error_body)
                        logger.warning(
                            f"Groq 429. Attempt {attempt}/{GROQ_MAX_RETRIES}. "
                            f"Retrying in {retry_delay:.1f}s..."
                        )
                        await asyncio.sleep(retry_delay)
                        continue

                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        raise Exception(
                            f"Groq Error {resp.status_code}: {error_body.decode()}"
                        )

                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            payload = line[6:]
                            if payload == "[DONE]":
                                return
                            try:
                                chunk = json.loads(payload)
                                delta = chunk["choices"][0]["delta"].get("content")
                                if delta:
                                    yield delta
                            except (json.JSONDecodeError, KeyError, IndexError):
                                pass
                    return

            except httpx.RemoteProtocolError:
                if attempt < GROQ_MAX_RETRIES:
                    await asyncio.sleep(2.0)
                    continue
                raise

    raise Exception(f"Groq rate limit exceeded after {GROQ_MAX_RETRIES} retries.")

async def stream_openai_chat(messages: list, model: str = "gpt-4o-mini"):
    if not OPENAI_API_KEY:
        yield "[OpenAI API not configured]"
        return

    async with httpx.AsyncClient(timeout=None) as client:
        try:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                headers=get_openai_headers(),
                json={
                    "model": model,
                    "messages": messages,
                    "stream": True,
                    "max_tokens": 4096
                }
            ) as resp:
                if resp.status_code != 200:
                    error_body = await resp.aread()
                    raise Exception(
                        f"OpenAI Error {resp.status_code}: {error_body.decode()}"
                    )

                async for line in resp.aiter_lines():
                    if line.startswith("data: "):
                        payload = line[6:]
                        if payload == "[DONE]":
                            return
                        try:
                            chunk = json.loads(payload)
                            delta = chunk["choices"][0]["delta"].get("content")
                            if delta:
                                yield delta
                        except (json.JSONDecodeError, KeyError, IndexError):
                                pass
        except httpx.RemoteProtocolError:
            raise

async def groq_chat_sync(messages: list, model: str = None, max_tokens: int = 4096) -> str:
    use_model = model or "llama-3.3-70b-versatile"
    attempt = 0
    while attempt < GROQ_MAX_RETRIES:
        attempt += 1
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=get_groq_headers(),
                json={"model": use_model, "messages": messages, "max_tokens": max_tokens}
            )
            if r.status_code == 429:
                retry_delay = _parse_retry_after(r.text)
                logger.warning(
                    f"Groq 429 (sync). Attempt {attempt}/{GROQ_MAX_RETRIES}. "
                    f"Retrying in {retry_delay:.1f}s..."
                )
                await asyncio.sleep(retry_delay)
                continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

    raise Exception(f"Groq rate limit exceeded after {GROQ_MAX_RETRIES} retries.")

async def openai_chat_sync(messages: list, model: str = "gpt-4o-mini", max_tokens: int = 4096) -> str:
    if not OPENAI_API_KEY:
        raise Exception("OpenAI API not configured")

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=get_openai_headers(),
            json={"model": model, "messages": messages, "max_tokens": max_tokens}
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

async def generate_image_openai_sync(
    prompt: str, size: str = "1024x1024", quality: str = "medium"
) -> str:
    """Generate image and return pure base64 string (Non-streaming)"""
    if not OPENAI_API_KEY:
        raise Exception("OpenAI API Key not configured")

    valid_sizes = ["1024x1024", "1536x1024", "1024x1536"]
    if size not in valid_sizes:
        size = "1024x1024"

    valid_qualities = ["low", "medium", "high"]
    if quality not in valid_qualities:
        quality = "medium"

    payload = {
        "model": OPENAI_IMAGE_MODEL,
        "prompt": prompt,
        "n": 1,
        "size": size,
        "quality": quality,
        "stream": False
    }

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            "https://api.openai.com/v1/images/generations",
            headers=get_openai_headers(),
            json=payload
        )

        if resp.status_code != 200:
            error_msg = resp.text
            logger.error(f"OpenAI Image Error {resp.status_code}: {error_msg}")
            raise Exception(f"Image generation failed: {error_msg}")

        data = resp.json()
        if data.get("data") and len(data["data"]) > 0:
            image_data = data["data"][0]
            if "b64_json" in image_data:
                return image_data["b64_json"]
            elif "url" in image_data:
                async with httpx.AsyncClient(timeout=30) as img_client:
                    img_resp = await img_client.get(image_data["url"])
                    img_resp.raise_for_status()
                    return base64.b64encode(img_resp.content).decode()
        raise Exception("No image data in response")

# =========================
# CODE EXECUTION (Data Analysis Layer)
# =========================
async def execute_python_code(code: str) -> str:
    """
    Executes Python code in a restricted subprocess.
    Allows the LLM to perform calculations and data analysis.
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
            f.write(code)
            temp_path = f.name
            
        process = await asyncio.create_subprocess_exec(
            "python3", temp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10.0)
        
        output = stdout.decode('utf-8', errors='ignore')
        error = stderr.decode('utf-8', errors='ignore')
        
        if process.returncode != 0:
            return f"Error during execution:\n{error}"
        return output if output else "(No output)"
        
    except asyncio.TimeoutError:
        return "Execution timed out after 10 seconds."
    except Exception as e:
        return f"Failed to execute code: {str(e)}"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# =========================
# FAL.AI VIDEO GENERATION
# =========================
def get_fal_headers() -> Dict[str, str]:
    return {
        "Authorization": f"Key {FAL_KEY}",
        "Content-Type": "application/json",
    }

def _extract_video_url(result: Dict[str, Any]) -> Optional[str]:
    if not result:
        return None
    if isinstance(result.get("video"), dict):
        return result["video"].get("url")
    if "videos" in result:
        vids = result["videos"]
        if isinstance(vids, list) and vids:
            return vids[0].get("url") if isinstance(vids[0], dict) else None
        if isinstance(vids, dict):
            return vids.get("url")
    if "output" in result:
        out = result["output"]
        if isinstance(out, str):
            return out
        if isinstance(out, dict):
            return out.get("url")
        if isinstance(out, list) and out:
            item = out[0]
            return item.get("url") if isinstance(item, dict) else item
    if "url" in result:
        return result["url"]
    return None

async def submit_fal_video_request(prompt: str) -> str:
    if not FAL_KEY:
        raise Exception("FAL_KEY is not configured")
    url = f"https://queue.fal.run/{FAL_VIDEO_MODEL}"
    payload = {"prompt": prompt}
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, headers=get_fal_headers(), json=payload)
        if resp.status_code != 200:
            raise Exception(f"fal.ai submit failed ({resp.status_code}): {resp.text}")
        data = resp.json()
        request_id = data.get("request_id")
        if not request_id:
            raise Exception(f"fal.ai submit returned no request_id: {data}")
        return request_id

async def poll_fal_video_status(request_id: str) -> Dict[str, Any]:
    url = f"https://queue.fal.run/{FAL_VIDEO_MODEL}/requests/{request_id}/status"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(url, headers=get_fal_headers())
        if resp.status_code != 200:
            raise Exception(f"fal.ai status check failed ({resp.status_code}): {resp.text}")
        return resp.json()

async def get_fal_video_result(request_id: str) -> Dict[str, Any]:
    url = f"https://queue.fal.run/{FAL_VIDEO_MODEL}/requests/{request_id}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, headers=get_fal_headers())
        if resp.status_code != 200:
            raise Exception(f"fal.ai result fetch failed ({resp.status_code}): {resp.text}")
        return resp.json()

async def cancel_fal_video_request(request_id: str):
    try:
        url = f"https://queue.fal.run/{FAL_VIDEO_MODEL}/requests/{request_id}"
        async with httpx.AsyncClient(timeout=10) as client:
            await client.delete(url, headers=get_fal_headers())
    except Exception as e:
        logger.warning(f"Failed to cancel fal.ai request {request_id}: {e}")

# =========================
# PROGRESSIVE IMAGE STREAMING
# =========================
def generate_progressive_frames(image_b64: str, steps: int = 8) -> list:
    try:
        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes))

        if img.mode in ('RGBA', 'LA', 'P'):
            bg = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            if img.mode in ('RGBA', 'LA'):
                bg.paste(img, mask=img.split()[-1])
            img = bg
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        frames = []
        quality_curve = [3, 8, 15, 28, 45, 65, 82, 95][:steps]

        for i, q in enumerate(quality_curve):
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=q, progressive=True, optimize=True)
            frame_b64 = base64.b64encode(buf.getvalue()).decode()
            frames.append({
                "progress": int((i + 1) / len(quality_curve) * 100),
                "data": frame_b64
            })
            buf.close()

        logger.info(
            f"Progressive frames generated: {len(frames)} levels, "
            f"original={len(image_b64)} chars, "
            f"first_frame={len(frames[0]['data'])} chars, "
            f"last_frame={len(frames[-1]['data'])} chars"
        )

        return frames

    except Exception as e:
        logger.error(f"Progressive frame generation failed: {e}")
        return [{"progress": 100, "data": image_b64}]

# =========================
# ANALYSIS HELPERS
# =========================

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tif", ".tiff"
}

CODE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".html", ".css", ".scss",
    ".json", ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".sh", ".bash", ".zsh", ".sql", ".java", ".kt", ".kts",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".go", ".rs",
    ".rb", ".php", ".swift", ".dart", ".r", ".R", ".m", ".scala",
    ".vue", ".svelte", ".ipynb", ".dockerfile", ".tf", ".graphql",
    ".gql", ".md", ".markdown"
}

TEXT_EXTENSIONS = {
    ".txt", ".log", ".csv", ".tsv", ".env", ".conf", ".properties",
    ".rst", ".tex", ".diff", ".patch"
}


def _is_image_mime(mime_type: Optional[str]) -> bool:
    """Return True when a MIME type identifies an image."""
    if not mime_type:
        return False
    return mime_type.lower().split(";")[0].strip().startswith("image/")


def get_language_from_extension(filename: str) -> str:
    """Get a readable language name for code-analysis prompts."""
    extension = Path(filename).suffix.lower()

    languages = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "jsx",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".json": "json",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".sql": "sql",
        ".java": "java",
        ".kt": "kotlin",
        ".kts": "kotlin",
        ".c": "c",
        ".h": "c",
        ".cpp": "cpp",
        ".hpp": "cpp",
        ".cc": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".swift": "swift",
        ".dart": "dart",
        ".r": "r",
        ".m": "objective-c",
        ".scala": "scala",
        ".vue": "vue",
        ".svelte": "svelte",
        ".ipynb": "json",
        ".tf": "hcl",
        ".graphql": "graphql",
        ".gql": "graphql",
        ".md": "markdown",
        ".markdown": "markdown",
        ".txt": "text",
        ".csv": "csv",
        ".tsv": "tsv",
        ".log": "text",
        ".env": "bash",
    }

    return languages.get(extension, "text")


async def extract_text_safe(content_bytes: bytes) -> str:
    """
    Extract text from common document, text, and code formats.
    Returns a clear message for unsupported or unreadable files.
    """
    if not content_bytes:
        return ""

    try:
        # PDF
        if content_bytes.startswith(b"%PDF"):
            try:
                import fitz  # PyMuPDF

                with fitz.open(stream=content_bytes, filetype="pdf") as pdf:
                    pages = []
                    for page in pdf:
                        text = page.get_text("text")
                        if text:
                            pages.append(text)

                extracted = "\n\n".join(pages).strip()
                if extracted:
                    return extracted[:MAX_TEXT_LENGTH]

                return (
                    "[This PDF contains no extractable text. "
                    "It may be a scanned document requiring OCR.]"
                )
            except ImportError:
                logger.error("PDF extraction requires PyMuPDF.")
                return "[PDF extraction unavailable: install PyMuPDF.]"
            except Exception as e:
                logger.warning("PDF extraction failed: %s", e)
                return "[Could not read this PDF file.]"

        # DOCX
        if content_bytes.startswith(b"PK") and len(content_bytes) > 4:
            try:
                from docx import Document

                with tempfile.NamedTemporaryFile(
                    suffix=".docx", delete=False
                ) as temp_file:
                    temp_file.write(content_bytes)
                    temp_path = temp_file.name

                try:
                    document = Document(temp_path)
                    paragraphs = [
                        paragraph.text
                        for paragraph in document.paragraphs
                        if paragraph.text.strip()
                    ]

                    for table in document.tables:
                        for row in table.rows:
                            cells = [
                                cell.text.strip()
                                for cell in row.cells
                                if cell.text.strip()
                            ]
                            if cells:
                                paragraphs.append(" | ".join(cells))

                    extracted = "\n".join(paragraphs).strip()
                    if extracted:
                        return extracted[:MAX_TEXT_LENGTH]
                finally:
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass

            except ImportError:
                logger.error("DOCX extraction requires python-docx.")
                return "[Word extraction unavailable: install python-docx.]"
            except Exception:
                # The file may be XLSX, PPTX, or another ZIP-based format.
                pass

            # XLSX
            try:
                from openpyxl import load_workbook

                workbook = load_workbook(
                    io.BytesIO(content_bytes),
                    read_only=True,
                    data_only=True,
                )
                rows_text = []

                try:
                    for sheet in workbook.worksheets:
                        rows_text.append(f"Sheet: {sheet.title}")

                        for row in sheet.iter_rows(values_only=True):
                            values = [
                                str(value)
                                for value in row
                                if value is not None
                            ]
                            if values:
                                rows_text.append(" | ".join(values))
                finally:
                    workbook.close()

                extracted = "\n".join(rows_text).strip()
                if extracted:
                    return extracted[:MAX_TEXT_LENGTH]

            except ImportError:
                pass
            except Exception:
                pass

            # PPTX
            try:
                from pptx import Presentation

                presentation = Presentation(io.BytesIO(content_bytes))
                slides_text = []

                for slide_number, slide in enumerate(
                    presentation.slides, start=1
                ):
                    slide_parts = []

                    for shape in slide.shapes:
                        if getattr(shape, "has_text_frame", False):
                            text = shape.text.strip()
                            if text:
                                slide_parts.append(text)

                        if getattr(shape, "has_table", False):
                            for row in shape.table.rows:
                                values = [
                                    cell.text.strip()
                                    for cell in row.cells
                                    if cell.text.strip()
                                ]
                                if values:
                                    slide_parts.append(" | ".join(values))

                    if slide_parts:
                        slides_text.append(
                            f"Slide {slide_number}:\n"
                            + "\n".join(slide_parts)
                        )

                extracted = "\n\n".join(slides_text).strip()
                if extracted:
                    return extracted[:MAX_TEXT_LENGTH]

            except ImportError:
                pass
            except Exception:
                pass

        # Plain text and source code. UTF-8 first, then common fallback.
        try:
            decoded = content_bytes.decode("utf-8-sig")
            if decoded.strip():
                return decoded[:MAX_TEXT_LENGTH]
        except UnicodeDecodeError:
            pass

        try:
            decoded = content_bytes.decode("utf-16")
            if decoded.strip():
                return decoded[:MAX_TEXT_LENGTH]
        except UnicodeDecodeError:
            pass

        # Last resort: accept content only if it looks reasonably textual.
        decoded = content_bytes.decode("utf-8", errors="replace")
        if decoded.strip():
            replacement_ratio = decoded.count("\ufffd") / max(
                len(decoded), 1
            )
            if replacement_ratio < 0.05 and "\x00" not in decoded:
                return decoded[:MAX_TEXT_LENGTH]

        return "[Binary or unreadable content]"

    except Exception as e:
        logger.exception("Unexpected text extraction error: %s", e)
        return "[Binary or unreadable content]"


def _build_image_analysis_messages(
    image_b64: str,
    mime_type: str,
    user_prompt: Optional[str],
) -> list:
    user_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_b64}"
            },
        },
        {
            "type": "text",
            "text": user_prompt or (
                "Analyze this image in detail. Describe what you see, "
                "read any text, explain any code or diagrams, "
                "and provide a comprehensive analysis."
            ),
        },
    ]

    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_code_analysis_messages(
    code_text: str,
    filename: str,
    language: str,
    user_prompt: Optional[str],
) -> list:
    instruction = user_prompt or (
        f"Analyze this {language} code from the file `{filename}`."
    )

    user_content = f"""{instruction}

Filename: {filename}
Language: {language}

```{language.lower()}
{code_text}
```"""

    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_document_analysis_messages(
    document_text: str,
    filename: str,
    user_prompt: Optional[str],
) -> list:
    instruction = user_prompt or (
        f"Analyze the document `{filename}`. "
        "Summarize its key points, explain important details, "
        "and identify significant issues or conclusions."
    )

    user_content = (
        f"{instruction}\n\n"
        f"Document: {filename}\n\n"
        f"Document contents:\n{document_text[:MAX_TEXT_LENGTH]}"
    )

    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

# =========================
# STREAMING GENERATORS
# =========================
async def _stream_image_generation(prompt: str, size: str, quality: str, result: dict):
    yield sse({"type": "image_generating"})
    try:
        image_b64 = await generate_image_openai_sync(prompt, size=size, quality=quality)
        frames = generate_progressive_frames(image_b64, steps=8)
        for frame in frames:
            yield sse({
                "type": "image_progress",
                "progress": frame["progress"],
                "data": frame["data"]
            })
            await asyncio.sleep(0.08)

        is_png = not image_b64.startswith('/9j/')
        mime = "data:image/png;base64," if is_png else "data:image/jpeg;base64,"
        markdown_image = f"![Generated Image]({mime}{image_b64})"
        
        yield sse({
            "type": "image_generated",
            "data": image_b64,
            "size": size,
            "quality": quality
        })

        result["text"] = markdown_image

    except Exception as e:
        error_str = str(e)
        logger.error(f"Image generation stream error: {error_str}")
        yield sse({
            "type": "image_error",
            "error": error_str
        })
        result["text"] = f"[Image generation failed: {error_str}]"

    return

async def _stream_video_generation(prompt: str, result: dict):
    if not FAL_KEY:
        error_msg = "Video generation is not configured (FAL_KEY missing)."
        logger.error(error_msg)
        yield sse({"type": "text_delta", "content": f"\n\n*Error: {error_msg}*"})
        result["text"] = f"[{error_msg}]"
        return

    yield sse({"type": "status", "message": "Submitting video request..."})

    request_id: Optional[str] = None
    try:
        request_id = await submit_fal_video_request(prompt)
        logger.info(f"fal.ai video request submitted: {request_id} (model={FAL_VIDEO_MODEL})")
        yield sse({"type": "status", "message": "Waiting in queue..."})

        elapsed = 0.0
        consecutive_errors = 0

        while elapsed < FAL_VIDEO_MAX_WAIT:
            await asyncio.sleep(FAL_VIDEO_POLL_INTERVAL)
            elapsed += FAL_VIDEO_POLL_INTERVAL

            try:
                status_data = await poll_fal_video_status(request_id)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                logger.warning(f"Poll error {consecutive_errors}/{FAL_VIDEO_MAX_POLL_ERRORS} for {request_id}: {e}")
                if consecutive_errors >= FAL_VIDEO_MAX_POLL_ERRORS:
                    raise Exception(f"Lost connection to video service after {FAL_VIDEO_MAX_POLL_ERRORS} failed polls.")
                continue

            status = status_data.get("status", "IN_PROGRESS")
            queue_position = status_data.get("queue_position")

            if status == "IN_QUEUE":
                msg = f"In queue (position: {queue_position})..." if queue_position else "Waiting in queue..."
                yield sse({"type": "status", "message": msg})

            elif status == "IN_PROGRESS":
                yield sse({"type": "status", "message": "Generating video frames..."})

            elif status == "COMPLETED":
                final_result = await get_fal_video_result(request_id)
                video_url = _extract_video_url(final_result)

                if not video_url:
                    logger.error(f"fal.ai returned no video URL: {final_result}")
                    raise Exception("Video completed but no URL was returned by the service.")

                yield sse({"type": "status", "message": "Video ready!"})
                yield sse({"type": "video", "url": video_url})

                markdown_video = (
                    f'\n\n<video controls autoplay muted loop playsinline '
                    f'style="max-width:100%;border-radius:12px;'
                    f'display:block;margin:8px 0;">'
                    f'<source src="{video_url}" type="video/mp4">'
                    f'Your browser does not support the video tag.'
                    f'</video>\n'
                )

                result["text"] = markdown_video
                return

            elif status == "FAILED":
                error_msg = status_data.get("error") or "Video generation failed on the server side."
                raise Exception(f"fal.ai: {error_msg}")

            else:
                logger.debug(f"Unknown fal.ai status '{status}' for {request_id}")

        raise Exception(f"Video generation timed out after {int(FAL_VIDEO_MAX_WAIT)}s. Please try again.")

    except Exception as e:
        error_str = str(e)
        logger.error(f"Video generation stream error: {error_str}")
        yield sse({"type": "text_delta", "content": f"\n\n*Error: {error_str}*"})
        result["text"] = f"[Video generation failed: {error_str}]"

    return

# =========================
# AGENT LAYER (Tool Calling)
# =========================
async def _stream_agent_response(
    prompt: str,
    conversation_id: str,
    model_key: str,
    mode: Optional[str],
    user_id: str,
    result: dict,
    project_instructions: str = ""
):
    """
    Handle text chat with OpenAI Tool Calling (Agent mode).
    The LLM can autonomously decide to search the web or execute Python code.
    """
    model_config = MODEL_ROUTING.get(model_key, MODEL_ROUTING["helox"])
    chat_model = model_config["chat"]

    # Define tools available to the LLM
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information, news, weather, or facts.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."}
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "execute_python_code",
                "description": "Execute Python code to perform calculations, data analysis, or solve math problems. Print results to stdout.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "The Python code to execute."}
                    },
                    "required": ["code"]
                }
            }
        }
    ]

    system_prompt = get_system_prompt(prompt, project_instructions)
    if mode == "finance":
        system_prompt = FINANCE_SYSTEM_PROMPT

    messages = [{"role": "system", "content": system_prompt}]

    try:
        history = await get_history(conversation_id, limit=6)
        messages.extend(history)
    except Exception as e:
        logger.warning(f"Failed to load history: {e}")

    messages.append({"role": "user", "content": prompt})

    # Step 1: Non-streaming call to decide if tools are needed
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=get_openai_headers(),
            json={
                "model": chat_model,
                "messages": messages,
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": 4096
            }
        )
        
        if resp.status_code != 200:
            error_body = resp.text
            logger.error(f"OpenAI Agent Error {resp.status_code}: {error_body}")
            yield sse({"type": "text_delta", "content": f"\n\n*Error: {error_body}*"})
            result["text"] = f"[Error: {error_body}]"
            return
            
        resp_data = resp.json()
        choice = resp_data["choices"][0]
        message = choice["message"]
        
        # Step 2: If tools are called, execute them and stream the final response
        if message.get("tool_calls"):
            messages.append(message)
            
            for tool_call in message["tool_calls"]:
                func_name = tool_call["function"]["name"]
                try:
                    args = json.loads(tool_call["function"]["arguments"])
                except:
                    args = {}
                
                tool_call_id = tool_call["id"]
                
                if func_name == "web_search":
                    query = args.get("query", prompt)
                    yield sse({"type": "status", "message": f"Searching the web for: {query}..."})
                    
                    search_context, search_html = await perform_web_search_formatted(query)
                    if search_html:
                        yield sse({"type": "search_results", "html": search_html})
                        
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": func_name,
                        "content": search_context
                    })
                    
                elif func_name == "execute_python_code":
                    code = args.get("code", "")
                    yield sse({"type": "status", "message": "Executing Python code..."})
                    
                    output = await execute_python_code(code)
                    yield sse({"type": "status", "message": "Code execution complete."})
                    
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": func_name,
                        "content": output
                    })
                    
        # Step 3: Stream the final response back to the user
        yield sse({"type": "status", "message": ""}) # Clear status
        
        full_response = ""
        async for delta in stream_openai_chat(messages, model=chat_model):
            full_response += delta
            yield sse({"type": "text_delta", "content": delta})
            
        result["text"] = full_response

# =========================
# ENDPOINTS
# =========================
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {
        "status": "running",
        "service": "HeloxAI Pro",
        "version": "5.0.0",
        "features": [
            "chat", "code", "math", "web_search", "tts", "stt",
            "image_generation", "real_video_generation",
            "image_analysis", "code_analysis", "document_analysis",
            "finance", "model_routing", "mode_routing",
            "progressive_image_streaming", "progressive_video_streaming",
            "agent_tool_calling", "python_code_execution", "projects_context"
        ],
        "endpoints": {
            "chat": "POST /ask/universal",
            "new_chat": "POST /newchat",
            "analysis": "POST /analysis",
            "delete_chat": "DELETE /chats/{chat_id}",
            "list_chats": "GET /chats",
            "messages": "GET /chat/{conversation_id}/messages",
            "user_plan": "GET /user/plan",
            "tts": "POST /tts",
            "tts_voices": "GET /tts/voices",
            "stt": "POST /stt",
            "logout": "POST /session/logout",
            "projects": "GET/POST /projects"
        }
    }

# =========================
# USER PLAN ENDPOINT
# =========================
@app.get("/user/plan")
async def get_user_plan(req: Request):
    auth_header = req.headers.get("authorization", "")

    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = auth_header.replace("Bearer ", "")

    try:
        user = await asyncio.to_thread(supabase.auth.get_user, token)
        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid token")

        user_id = user.user.id

        result = await _execute_supabase_with_retry(
            supabase.table("users")
            .select("plan, is_premium, is_lifetime")
            .eq("id", user_id)
            .limit(1)
        )

        if result.data and result.data[0]:
            u = result.data[0]
            plan = "free"
            if u.get("is_lifetime"):
                plan = "lifetime"
            elif u.get("is_premium"):
                plan = u.get("plan", "ultimate_monthly") or "ultimate_monthly"
            else:
                plan = u.get("plan", "free") or "free"

            return {
                "plan": plan,
                "is_premium": bool(u.get("is_premium", False)),
                "is_lifetime": bool(u.get("is_lifetime", False))
            }

        return {"plan": "free", "is_premium": False, "is_lifetime": False}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Plan endpoint error: {e}")
        return {"plan": "free", "is_premium": False, "is_lifetime": False}

# =========================
# ANALYSIS ENDPOINT (MULTIPART)
# =========================
@app.post("/analysis")
async def analyze_file(
    req: Request,
    res: Response,
    file: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
    stream: bool = Form(True),
    remember: bool = Form(True),
    analysis_type: Optional[str] = Form(None),
    image_base64: Optional[str] = Form(None),
    image_mime: Optional[str] = Form("image/png"),
):
    user = await get_user_with_auth(req, res, remember)

    image_data_b64 = None
    image_mime_type = image_mime or "image/png"
    file_text_content = None
    file_filename = "unknown"
    file_category = None

        if image_base64:

        file_category = "IMAGE"

        file_filename = file_filename or "uploaded_image.png"

    elif file and file.filename:

        file_filename = file.filename

        ext = Path(file_filename).suffix.lower()

        requested_type = (analysis_type or "auto").strip().upper()

        if requested_type not in {"AUTO", "IMAGE", "CODE", "DOCUMENT"}:

            raise HTTPException(

                status_code=400,

                detail="Invalid analysis_type. Use AUTO, IMAGE, CODE, or DOCUMENT.",

            )

        if requested_type != "AUTO":

            file_category = requested_type

        elif ext in IMAGE_EXTENSIONS or _is_image_mime(file.content_type):

            if image_base64:

        file_category = "IMAGE"

        elif ext in CODE_EXTENSIONS:

            file_category = "CODE"

        elif ext in DOCUMENT_EXTENSIONS:

            file_category = "DOCUMENT"

        else:

            raise HTTPException(

                status_code=400,

                detail=f"Unsupported file type: {ext or 'unknown'}",

            )

        if file_category == "IMAGE":

            if (

                not _is_image_mime(file.content_type)

                and ext not in IMAGE_EXTENSIONS

            ):

                raise HTTPException(

                    status_code=400,

                    detail="The uploaded file does not appear to be a supported image.",

                )

    else:

        raise HTTPException(

            status_code=400,

            detail="Either 'file' or 'image_base64' must be provided.",
)
    conv_id = await get_or_create_conversation(user["id"], conversation_id, f"Analysis: {file_filename}" if file_filename else "Image Analysis")

    user_msg_content = prompt or f"[Uploaded {file_filename} for analysis]"
    await save_message(user["id"], conv_id, "user", user_msg_content)

    if file_category == "IMAGE":
        analysis_messages = _build_image_analysis_messages(image_data_b64, image_mime_type, prompt)
    else:
        language = get_language_from_extension(file_filename)
        if file_category == "CODE":
            analysis_messages = _build_code_analysis_messages(file_text_content, file_filename, language, prompt)
        else:
            analysis_messages = _build_document_analysis_messages(file_text_content, file_filename, prompt)

    if stream:
        async def analysis_stream():
            full_response = ""
            try:
                async for delta in stream_openai_chat(analysis_messages, model=OPENAI_VISION_MODEL):
                    full_response += delta
                    yield sse({"type": "text_delta", "content": delta})
            except Exception as e:
                error_str = str(e)
                logger.error(f"Analysis stream error: {error_str}")
                if not full_response:
                    full_response = f"[Analysis error: {error_str}]"
                    yield sse({"type": "text_delta", "content": f"*Error: {error_str}*"})

            try:
                await save_message(user["id"], conv_id, "assistant", full_response)
            except Exception as e:
                logger.error(f"Failed to save analysis response: {e}")

            yield sse({"type": "done", "conversation_id": conv_id})

        return StreamingResponse(
            analysis_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )
    else:
        try:
            response_text = await openai_chat_sync(analysis_messages, model=OPENAI_VISION_MODEL)
        except Exception as e:
            response_text = f"[Analysis error: {str(e)}]"

        await save_message(user["id"], conv_id, "assistant", response_text)

        return JSONResponse({
            "response": response_text,
            "conversation_id": conv_id
        })

# =========================
# MAIN CHAT ENDPOINT (Agent Mode)
# =========================
@app.post("/ask/universal")
async def ask_universal(req: Request, res: Response):
    """
    Universal chat endpoint with Agent Layer.
    SSE Event Types:
      - image_generating   : { type: "image_generating" }
      - image_progress     : { type: "image_progress", progress: 0-100, data: "base64..." }
      - image_generated    : { type: "image_generated", data: "base64...", size: "...", quality: "..." }
      - image_error        : { type: "image_error", error: "..." }
      - status             : { type: "status", message: "..." }
      - video              : { type: "video", url: "..." }
      - search_results     : { type: "search_results", html: "..." }
      - text_delta         : { type: "text_delta", content: "..." }
      - done               : { type: "done", conversation_id: "..." }
    """
    try:
        body = await req.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    prompt = body.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(400, "Prompt is required")

    conversation_id = body.get("conversation_id")
    remember = body.get("remember", True)
    image_size = body.get("image_size", "1024x1024")
    image_quality = body.get("image_quality", "medium")
    model_key = body.get("model", "helox") or "helox"
    mode = body.get("mode", "general")
    project_id = body.get("project_id")

    user = await get_user_with_auth(req, res, remember)
    user_id = user["id"]

    # Fetch project context if provided
    project_instructions = ""
    if project_id:
        try:
            proj_res = await _execute_supabase_with_retry(
                supabase.table("projects")
                .select("instructions")
                .eq("id", project_id)
                .eq("user_id", user_id)
                .limit(1)
            )
            if proj_res.data:
                project_instructions = proj_res.data[0].get("instructions", "")
        except Exception as e:
            logger.error(f"Failed to fetch project context: {e}")

    title = prompt[:50] if len(prompt) > 10 else prompt
    conv_id = await get_or_create_conversation(user_id, conversation_id, title, project_id)

    await save_message(user_id, conv_id, "user", prompt)

    # Explicit image/video generation routes
    if "generate image" in prompt.lower() or "create image" in prompt.lower():
        async def image_stream():
            result = {}
            async for event in _stream_image_generation(prompt=prompt, size=image_size, quality=image_quality, result=result):
                yield event
            text_response = f"\n\nHere's the image I generated based on your request: *\"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"*"
            for char in text_response:
                yield sse({"type": "text_delta", "content": char})
            full_text = result.get("text", "") + text_response
            try:
                await save_message(user_id, conv_id, "assistant", full_text)
            except Exception as e:
                logger.error(f"Failed to save image response: {e}")
            yield sse({"type": "done", "conversation_id": conv_id})

        return StreamingResponse(
            image_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no",}
        )

    if "generate video" in prompt.lower() or "create video" in prompt.lower():
        async def video_stream():
            result = {}
            async for event in _stream_video_generation(prompt=prompt, result=result):
                yield event
            text_response = f"\n\nHere's the video I generated based on your request: *\"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"*"
            for char in text_response:
                yield sse({"type": "text_delta", "content": char})
            full_text = result.get("text", "") + text_response
            try:
                await save_message(user_id, conv_id, "assistant", full_text)
            except Exception as e:
                logger.error(f"Failed to save video response: {e}")
            yield sse({"type": "done", "conversation_id": conv_id})

        return StreamingResponse(
            video_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no",}
        )

    # Default: Agent Chat Path (Tool Calling)
    async def chat_stream():
        result = {}
        async for event in _stream_agent_response(
            prompt=prompt,
            conversation_id=conv_id,
            model_key=model_key,
            mode=mode,
            user_id=user_id,
            result=result,
            project_instructions=project_instructions
        ):
            yield event

        full_response = result.get("text", "")
        try:
            await save_message(user_id, conv_id, "assistant", full_response)
        except Exception as e:
            logger.error(f"Failed to save chat response: {e}")

        yield sse({"type": "done", "conversation_id": conv_id})

    return StreamingResponse(
        chat_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

# =========================
# NEW CHAT ENDPOINT
# =========================
@app.post("/newchat")
async def new_chat(req: Request, res: Response):
    user = await get_user_with_auth(req, res)
    try:
        body = await req.json()
    except Exception:
        body = {}

    title = body.get("title", "New Chat")[:50]
    project_id = body.get("project_id")
    new_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    insert_data = {
        "id": new_id,
        "user_id": user["id"],
        "title": title,
        "created_at": now,
        "updated_at": now,
    }
    if project_id:
        insert_data["project_id"] = project_id

    try:
        await _execute_supabase_with_retry(
            supabase.table("conversations").insert(insert_data)
        )
        return JSONResponse({"id": new_id, "title": title})
    except Exception as e:
        logger.error(f"Failed to create chat: {e}")
        raise HTTPException(500, "Failed to create chat")

# =========================
# PROJECTS API (Context Layer)
# =========================
@app.post("/projects")
async def create_project(req: Request, res: Response):
    user = await get_user_with_auth(req, res)
    try:
        body = await req.json()
    except:
        body = {}
        
    title = body.get("title", "New Project")
    instructions = body.get("instructions", "")
    
    new_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    
    try:
        await _execute_supabase_with_retry(
            supabase.table("projects").insert({
                "id": new_id,
                "user_id": user["id"],
                "title": title,
                "instructions": instructions,
                "created_at": now,
                "updated_at": now,
            })
        )
        return JSONResponse({"id": new_id, "title": title})
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        raise HTTPException(500, "Failed to create project")

@app.get("/projects")
async def list_projects(req: Request, res: Response):
    user = await get_user_with_auth(req, res)
    try:
        result = await _execute_supabase_with_retry(
            supabase.table("projects")
            .select("id, title, instructions, created_at, updated_at")
            .eq("user_id", user["id"])
            .order("updated_at", desc=True)
        )
        return JSONResponse({"projects": result.data or []})
    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        raise HTTPException(500, "Failed to list projects")

# =========================
# LIST CHATS
# =========================
@app.get("/chats")
async def list_chats(req: Request, res: Response):
    user = await get_user_with_auth(req, res)
    try:
        result = await _execute_supabase_with_retry(
            supabase.table("conversations")
            .select("id, title, created_at, updated_at")
            .eq("user_id", user["id"])
            .order("updated_at", desc=True)
            .limit(100)
        )
        return JSONResponse({"chats": result.data or []})
    except Exception as e:
        logger.error(f"Failed to list chats: {e}")
        raise HTTPException(500, "Failed to list chats")

# =========================
# DELETE CHAT
# =========================
@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, req: Request, res: Response):
    user = await get_user_with_auth(req, res)

    try:
        check = await _execute_supabase_with_retry(
            supabase.table("conversations")
            .select("id")
            .eq("id", chat_id)
            .eq("user_id", user["id"])
            .limit(1)
        )
        if not check.data:
            raise HTTPException(404, "Chat not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat ownership check failed: {e}")
        raise HTTPException(500, "Failed to verify chat ownership")

    try:
        await _execute_supabase_with_retry(
            supabase.table("messages")
            .delete()
            .eq("conversation_id", chat_id)
        )
        await _execute_supabase_with_retry(
            supabase.table("conversations")
            .delete()
            .eq("id", chat_id)
        )
        return JSONResponse({"deleted": True})
    except Exception as e:
        logger.error(f"Failed to delete chat: {e}")
        raise HTTPException(500, "Failed to delete chat")

# =========================
# GET MESSAGES
# =========================
@app.get("/chat/{conversation_id}/messages")
async def get_messages(conversation_id: str, req: Request, res: Response):
    user = await get_user_with_auth(req, res)

    try:
        check = await _execute_supabase_with_retry(
            supabase.table("conversations")
            .select("id")
            .eq("id", conversation_id)
            .eq("user_id", user["id"])
            .limit(1)
        )
        if not check.data:
            raise HTTPException(404, "Chat not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Message ownership check failed: {e}")
        raise HTTPException(500, "Failed to verify chat")

    try:
        result = await _execute_supabase_with_retry(
            supabase.table("messages")
            .select("id, role, content, created_at")
            .eq("conversation_id", conversation_id)
            .order("created_at", desc=False)
        )
        return JSONResponse({"messages": result.data or []})
    except Exception as e:
        logger.error(f"Failed to get messages: {e}")
        raise HTTPException(500, "Failed to get messages")

# =========================
# TTS ENDPOINT
# =========================
@app.post("/tts")
async def text_to_speech(req: Request, res: Response):
    user = await get_user_with_auth(req, res)

    if not OPENAI_API_KEY:
        raise HTTPException(500, "TTS not configured")

    try:
        body = await req.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    text = body.get("text", "").strip()
    voice = body.get("voice", "alloy")

    if not text:
        raise HTTPException(400, "Text is required")
    if len(text) > 4096:
        raise HTTPException(400, "Text too long (max 4096 chars)")

    valid_voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
    if voice not in valid_voices:
        voice = "alloy"

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/audio/speech",
                headers=get_openai_headers(),
                json={
                    "model": OPENAI_TTS_MODEL,
                    "input": text,
                    "voice": voice,
                    "response_format": "mp3",
                }
            )
            resp.raise_for_status()

            return StreamingResponse(
                io.BytesIO(resp.content),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": "inline; filename=speech.mp3"
                }
            )
    except httpx.HTTPStatusError as e:
        logger.error(f"TTS error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(502, f"TTS API error: {e.response.status_code}")
    except Exception as e:
        logger.error(f"TTS failed: {e}")
        raise HTTPException(500, "TTS generation failed")

@app.get("/tts/voices")
async def list_tts_voices():
    return JSONResponse({
        "voices": [
            {"id": "alloy", "name": "Alloy", "description": "Balanced and neutral"},
            {"id": "shimmer", "name": "Shimmer", "description": "Soft and gentle"},
        ]
    })

# =========================
# STT ENDPOINT
# =========================
@app.post("/stt")
async def speech_to_text(
    req: Request,
    res: Response,
    file: UploadFile = File(...),
):
    user = await get_user_with_auth(req, res)

    if not GROQ_API_KEY:
        raise HTTPException(500, "STT not configured")

    audio_bytes = b""
    while chunk := await file.read(1024 * 1024):
        audio_bytes += chunk
        if len(audio_bytes) > MAX_FILE_SIZE:
            raise HTTPException(413, "Audio file too large")

    if not audio_bytes:
        raise HTTPException(400, "Empty audio file")

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=get_groq_headers_multipart(),
                files={
                    "file": (file.filename or "audio.wav", audio_bytes),
                },
                data={
                    "model": GROQ_STT_MODEL,
                    "response_format": "json",
                    "language": "en",
                }
            )
            resp.raise_for_status()
            data = resp.json()
            return JSONResponse({
                "text": data.get("text", ""),
                "language": data.get("language", "en")
            })
    except httpx.HTTPStatusError as e:
        logger.error(f"STT error: {e.response.status_code} - {e.response.text}")
        raise HTTPException(502, f"STT API error: {e.response.status_code}")
    except Exception as e:
        logger.error(f"STT failed: {e}")
        raise HTTPException(500, "Speech transcription failed")

# =========================
# LOGOUT ENDPOINT
# =========================
@app.post("/session/logout")
async def logout(req: Request, res: Response):
    user_id = req.cookies.get(PRIMARY_COOKIE)
    token = req.cookies.get(SESSION_TOKEN_COOKIE)

    if user_id and token:
        try:
            await _execute_supabase_with_retry(
                supabase.table("user_sessions")
                .update({"is_valid": False})
                .eq("user_id", user_id)
                .eq("token", token)
            )
        except Exception as e:
            logger.error(f"Failed to invalidate session: {e}")

    if user_id and user_id in _session_cache:
        del _session_cache[user_id]

    clear_session_cookies(res)
    return JSONResponse({"logged_out": True})

# =========================
# AUTOMATION ENDPOINT (Placeholder)
# =========================
@app.post("/automations")
async def create_automation(req: Request, res: Response):
    user = await get_user_with_auth(req, res)
    try:
        body = await req.json()
    except:
        body = {}
        
    # In a real implementation, you would save this to a DB and use a scheduler (e.g., Celery, APScheduler)
    # to run the prompt at the specified time.
    
    return JSONResponse({
        "status": "received", 
        "message": "Automation scheduling is not yet fully implemented."
    })
