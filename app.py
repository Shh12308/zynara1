import os
import re
import json
import uuid
import asyncio
import logging
import time
import base64
import io
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

MAX_FILE_SIZE = 20 * 1024 * 1024
MAX_TEXT_LENGTH = 100000
MAX_IMAGE_SIZE = 20 * 1024 * 1024

SESSION_DURATION = 365 * 24 * 60 * 60
REFRESH_THRESHOLD = 7 * 24 * 60 * 60

GROQ_MAX_RETRIES = 3


app = FastAPI(
    title="HeloxAi Lite",
    description="Text, Code, Math, Research, Image/Video Generation & File Analysis Backend",
    version="4.2.0"
)

# =========================
# MODEL CONFIGURATION
# =========================
GROQ_CHAT_MODEL = "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_STT_MODEL = "whisper-large-v3"
OPENAI_TTS_MODEL = "tts-1"
OPENAI_IMAGE_MODEL = "gpt-image-1"

# =========================
# MODEL ROUTING
# =========================
MODEL_ROUTING = {
    "helox": {
        "chat": GROQ_CHAT_MODEL,
        "vision": GROQ_VISION_MODEL,
        "provider": "groq"
    },
    "chatgpt": {
        "chat": "gpt-4o-mini",
        "vision": "gpt-4o-mini",
        "provider": "openai"
    },
    "chatz": {
        "chat": GROQ_CHAT_MODEL,
        "vision": GROQ_VISION_MODEL,
        "provider": "groq"
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

logger.info(f"CORS Allowed Origins: {allowed_origins}")

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

# ══════════════════════════════════════════════
# RATE LIMITING CONFIGURATION
# ══════════════════════════════════════════════

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

    if stale_keys:
        logger.debug(f"Rate store cleanup: pruned {len(stale_keys)} stale keys")


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
        logger.warning(f"Rate limit hit: {key} ({current_count}/{limit})")
        return JSONResponse(
            status_code=429,
            content={
                "detail": "Too many requests. Please slow down.",
                "limit": limit,
                "window": window,
                "reset_at": reset_at,
            },
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

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")


# =========================
# FILE TYPES
# =========================
class FileCategory(Enum):
    CODE = "code"
    DOCUMENT = "document"
    DATA = "data"
    IMAGE = "image"
    UNKNOWN = "unknown"


CODE_EXTENSIONS = {
    '.py', '.js', '.ts', '.jsx', '.tsx', '.html', '.css', '.java',
    '.c', '.cpp', '.go', '.rs', '.php', '.rb', '.swift', '.sql',
    '.json', '.yaml', '.xml', '.h', '.hpp', '.cs', '.kt', '.dart',
    '.lua', '.r', '.m', '.mm', '.sh', '.bash', '.zsh', '.ps1',
    '.scala', '.clj', '.hs', '.ex', '.exs', '.erl', '.zig', '.nim',
    '.v', '.sol', '.move', '.tf', '.hcl', '.dockerfile', '.makefile',
    '.cmake', '.gradle', '.pom', '.csproj', '.sln', '.vue', '.svelte'
}
DOCUMENT_EXTENSIONS = {'.txt', '.md', '.csv', '.pdf', '.doc', '.docx', '.log', '.rtf', '.odt'}
DATA_EXTENSIONS = {'.csv', '.json', '.xml', '.yaml', '.yml', '.tsv', '.ini', '.toml', '.env'}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.svg', '.tiff', '.ico'}


def get_file_category(filename: str) -> FileCategory:
    if not filename:
        return FileCategory.UNKNOWN
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return FileCategory.IMAGE
    if ext in CODE_EXTENSIONS:
        return FileCategory.CODE
    if ext in DOCUMENT_EXTENSIONS:
        return FileCategory.DOCUMENT
    if ext in DATA_EXTENSIONS:
        return FileCategory.DATA
    return FileCategory.UNKNOWN


def get_language_from_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    lang_map = {
        '.py': 'Python', '.js': 'JavaScript', '.ts': 'TypeScript',
        '.jsx': 'React JSX', '.tsx': 'React TSX', '.html': 'HTML',
        '.css': 'CSS', '.java': 'Java', '.c': 'C', '.cpp': 'C++',
        '.go': 'Go', '.rs': 'Rust', '.php': 'PHP', '.rb': 'Ruby',
        '.swift': 'Swift', '.sql': 'SQL', '.json': 'JSON',
        '.yaml': 'YAML', '.yml': 'YAML', '.xml': 'XML',
        '.h': 'C Header', '.hpp': 'C++ Header', '.cs': 'C#',
        '.kt': 'Kotlin', '.dart': 'Dart', '.lua': 'Lua',
        '.r': 'R', '.m': 'Objective-C', '.mm': 'Objective-C++',
        '.sh': 'Shell', '.bash': 'Bash', '.zsh': 'Zsh',
        '.ps1': 'PowerShell', '.scala': 'Scala', '.clj': 'Clojure',
        '.hs': 'Haskell', '.ex': 'Elixir', '.exs': 'Elixir',
        '.erl': 'Erlang', '.zig': 'Zig', '.nim': 'Nim',
        '.v': 'V', '.sol': 'Solidity', '.vue': 'Vue',
        '.svelte': 'Svelte', '.md': 'Markdown', '.txt': 'Plain Text',
        '.csv': 'CSV', '.log': 'Log', '.dockerfile': 'Dockerfile',
        '.makefile': 'Makefile', '.tf': 'Terraform', '.hcl': 'HCL',
    }
    return lang_map.get(ext, 'Unknown')


async def extract_text_safe(content: bytes) -> str:
    for enc in ['utf-8', 'latin-1', 'cp1252']:
        try:
            return content.decode(enc, errors='ignore')[:MAX_TEXT_LENGTH]
        except Exception:
            continue
    return "[Binary or unreadable content]"


def _is_image_mime(mime: str) -> bool:
    return mime and mime.startswith("image/")


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
        logger.error(f"Cannot create session: failed to ensure user {user_id} exists")
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
BASE_SYSTEM_PROMPT = """You are HeloxAi, a powerful AI assistant powered by Llama 3.3 70B.

**Capabilities:**
1. **Text & Reasoning:** Advanced understanding, reasoning, writing, and conversation.
2. **Coding:** Expert in writing, debugging, and reviewing code across all languages.
3. **Math:** Capable of solving mathematical problems and equations.
4. **Research:** You have access to real-time web search. Use it for current events or facts.

**Response Style:**
- Use Markdown for structure (headers, bolding, code blocks with language tags, lists, tables).
- Be concise but thorough.
- If you use web search, cite sources as [1], [2] etc. with a "Sources" section at the bottom with URLs.
- For code, always provide complete, runnable code — never use placeholders.
- For math, use LaTeX notation with $...$ for inline and $$...$$ for display math.

**Identity:**
- If asked who created you, say: "I was constructed by GoldYLocks. You can find them on Twitter @HeloxAi" """

IMAGE_ANALYSIS_SYSTEM_PROMPT = """You are HeloxAi, an expert visual analyst powered by Llama 3.2 90B Vision.

Analyze the provided image thoroughly. Cover:
1. **Description:** What is shown in the image (objects, scene, people, text, etc.)
2. **Details:** Notable colors, layout, style, composition, quality
3. **Context:** What the image might be used for, its likely purpose
4. **Text:** If there is any readable text in the image, transcribe it exactly
5. **Issues:** Any problems, errors, or anomalies visible

Be specific and precise. If the image contains code screenshots, read and explain the code.
If it's a diagram or chart, describe the data/trends shown.
Use Markdown formatting for structure."""

CODE_ANALYSIS_SYSTEM_PROMPT = """You are HeloxAi, a senior software engineer and code reviewer powered by Llama 3.3 70B.

Analyze the provided code thoroughly:

1. **Overview:** What does this code do? What language and purpose?
2. **Architecture:** How is it structured? Patterns used?
3. **Quality Assessment:** Rate code quality (1-10) with justification
4. **Issues Found:**
   - Critical: Bugs, security vulnerabilities, crashes
   - Warnings: Bad practices, performance issues, maintainability
   - Suggestions: Improvements, modernizations, best practices
5. **Security Review:** Any vulnerabilities (XSS, injection, auth issues, etc.)
6. **Performance:** Any bottlenecks or inefficiencies
7. **Refactored Version:** Provide an improved version of the code with fixes applied

Be specific - reference line numbers or code sections. Provide working improved code."""

DOCUMENT_ANALYSIS_SYSTEM_PROMPT = """You are HeloxAi, an expert document analyst powered by Llama 3.3 70B.

Analyze the provided document/file content thoroughly:

1. **Summary:** Concise summary of the content (2-3 sentences)
2. **Key Points:** Bullet points of the main ideas/facts
3. **Structure:** How is the document organized?
4. **Analysis:** Deep analysis of the content, arguments, or data
5. **Issues:** Any errors, inconsistencies, or problems found
6. **Recommendations:** Suggestions for improvement or next steps

Be thorough but well-organized. Use Markdown formatting."""

FINANCE_SYSTEM_PROMPT = """You are HeloxAi, a financial analysis assistant powered by Llama 3.3 70B.

You have access to real-time web search for financial data. When analyzing financial topics:

1. **Always** search for current data before answering
2. Provide specific numbers, percentages, and dates
3. Include relevant context (market conditions, comparisons)
4. **Disclaimer:** Always end with: "*Note: This is not financial advice. Do your own research before making investment decisions.*"
5. Use tables for comparing stocks/metrics when relevant
6. Cite your sources

Be precise with numbers. If you can't find current data, say so clearly."""


def get_system_prompt(user_prompt: str) -> str:
    return BASE_SYSTEM_PROMPT


# =========================
# INTENT DETECTION
# =========================
class IntentCategory(Enum):
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_DEBUG = "code_debug"
    MATHEMATICAL = "mathematical"
    RESEARCH = "research"
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    CONVERSATION = "conversation"


@dataclass
class IntentResult:
    intent: IntentCategory
    confidence: float


class AdvancedIntentDetector:
    def __init__(self):
        self.patterns = {
            IntentCategory.CODE_GENERATION: [
                r'\b(write|create|make)\s+(code|function|script|program)',
                r'\bimplement\s+',
                r'\bhow\s+to\s+code\s+'
            ],
            IntentCategory.CODE_DEBUG: [
                r'\b(fix|debug|solve)\s+(this|my|the)\s+(bug|error)',
                r'\bwhy\s+is\s+(this|it)\s+not\s+working',
                r'\berror\s*:'
            ],
            IntentCategory.CODE_REVIEW: [
                r'\b(review|refactor|improve)\s+(this|my)\s+code',
                r'\b(is\s+this)\s+code\s+(good|clean)'
            ],
            IntentCategory.MATHEMATICAL: [
                r'\b(calculate|solve|compute)\s+',
                r'\b\d+[\+\-\*\/\^]\d+',
                r'\bintegral|derivative|equation\b'
            ],
            IntentCategory.RESEARCH: [
                r'\b(search|find|look\s+up)\s+(for|about)',
                r'\blatest\s+news|current\s+events',
                r'\bwho\s+is\s+(currently|now)'
            ],
            IntentCategory.IMAGE_GENERATION: [
                r'\b(generate|create|make|draw|render)\s+(an?\s+)?(image|picture|photo|illustration|art|drawing|painting|sketch)',
                r'\bimage\s+of\s+',
                r'\bdrawing\s+of\s+',
                r'\billustration\s+of\s+',
                r'\bpicture\s+of\s+',
                r'\bdraw\s+me\s+',
                r'\bvisualize\s+',
                r'\bcreate\s+(an?\s+)?art',
                r'\bmake\s+(me\s+)?(an?\s+)?(image|picture|art)',
                r'\bgenerate\s+(an?\s+)?(image|picture|art|photo)',
                r'\brender\s+(an?\s+)?(image|scene|picture)',
                r'\bpaint\s+(me\s+)?',
                r'\bsketch\s+(me\s+)?',
                r'\bdesign\s+(an?\s+)?(logo|icon|banner|thumbnail)',
                r'\b(\w+\s+){0,3}(image|picture|art|drawing|illustration|photo|painting)\s+(of|for|showing|depicting)',
                r'^\s*(generate|create|make|draw|render)\s+',
            ],
            IntentCategory.VIDEO_GENERATION: [
                r'\b(generate|create|make|render)\s+(a?\s+)?(video|clip|animation|movie|film)',
                r'\bvideo\s+of\s+',
                r'\banimate\s+',
                r'\b(\w+\s+){0,3}(video|clip|animation|movie)\s+(of|for|showing|depicting)',
                r'^\s*(generate|create|make|render)\s+(a?\s+)?video',
            ],
            IntentCategory.CONVERSATION: [
                r'^(hello|hi|hey|thanks)',
                r'^(how\s+are\s+you)'
            ]
        }
        self.compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.patterns.items()
        }

    def detect(self, text: str) -> IntentResult:
        # Check Video first to prevent it falling into Image
        for intent in [IntentCategory.VIDEO_GENERATION]:
            patterns = self.compiled_patterns.get(intent, [])
            matches = sum(1 for p in patterns if p.search(text))
            if matches > 0:
                return IntentResult(intent=intent, confidence=min(0.6 + matches * 0.1, 0.98))

        for intent in [IntentCategory.IMAGE_GENERATION]:
            patterns = self.compiled_patterns.get(intent, [])
            matches = sum(1 for p in patterns if p.search(text))
            if matches > 0:
                return IntentResult(intent=intent, confidence=min(0.6 + matches * 0.1, 0.98))

        for intent, patterns in self.compiled_patterns.items():
            if intent in [IntentCategory.IMAGE_GENERATION, IntentCategory.VIDEO_GENERATION]:
                continue
            matches = sum(1 for p in patterns if p.search(text))
            if matches > 0:
                return IntentResult(intent=intent, confidence=min(0.5 + matches * 0.1, 0.95))
        return IntentResult(intent=IntentCategory.CONVERSATION, confidence=0.5)


_detector = AdvancedIntentDetector()


# =========================
# MODELS
# =========================
class ChatRequest(BaseModel):
    prompt: str
    conversation_id: Optional[str] = None
    stream: bool = True
    remember: bool = True
    image_size: str = "1024x1024"
    image_quality: str = "medium"
    model: Optional[str] = "helox"
    mode: Optional[str] = "general"


class AnalysisRequest(BaseModel):
    prompt: Optional[str] = None
    conversation_id: Optional[str] = None
    stream: bool = True
    remember: bool = True
    analysis_type: Optional[str] = None


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
            logger.debug("Waiting for concurrent user creation to complete...")
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


async def get_history(conv_id: str, limit: int = 4):
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
    title: str
) -> str:
    lock_key = proposed_id or "__new__"
    lock = _get_conv_lock(lock_key)

    async with lock:
        if proposed_id:
            # Retry up to 3 times with 200ms delay to handle
            # Supabase eventual consistency after /newchat
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
            logger.warning(f"Conversation ID {proposed_id} provided but not found in DB after retries.")

        new_id = str(uuid.uuid4())
        logger.info(f"Creating new conversation: {new_id}")
        now = datetime.now(timezone.utc).isoformat()
        await _execute_supabase_with_retry(
            supabase.table("conversations").insert({
                "id": new_id,
                "user_id": user_id,
                "title": title[:50],
                "created_at": now,
                "updated_at": now,
            })
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
    use_model = model or GROQ_CHAT_MODEL
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
    use_model = model or GROQ_CHAT_MODEL
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


# ════════════════════════════════════════════════════════════════
# PROGRESSIVE IMAGE STREAMING
# ════════════════════════════════════════════════════════════════

def generate_progressive_frames(image_b64: str, steps: int = 8) -> list:
    """
    Convert a base64 image into multiple JPEG quality levels
    for progressive streaming to the frontend.
    """
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
def _build_image_analysis_messages(
    image_b64: str, mime_type: str, user_prompt: Optional[str]
) -> list:
    user_content = [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_b64}"
            }
        },
        {
            "type": "text",
            "text": user_prompt or (
                "Analyze this image in detail. Describe what you see, "
                "read any text, explain any code or diagrams, "
                "and provide a comprehensive analysis."
            )
        }
    ]
    return [
        {"role": "system", "content": IMAGE_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


def _build_code_analysis_messages(
    code_text: str, filename: str, language: str, user_prompt: Optional[str]
) -> list:
    instruction = user_prompt or f"Analyze this {language} code from the file `{filename}`."
    user_content = f"""{instruction}

```{language.lower()}
{code_text}
```

Provide a thorough code review covering: bugs, security issues, performance, style, and an improved version if needed."""
    return [
        {"role": "system", "content": CODE_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


def _build_document_analysis_messages(
    doc_text: str, filename: str, user_prompt: Optional[str]
) -> list:
    instruction = user_prompt or f"Analyze the content from the file `{filename}`."
    user_content = f"""{instruction}

--- FILE CONTENT START ---
{doc_text}
--- FILE CONTENT END ---

Provide a thorough analysis: summary, key points, structure, issues, and recommendations."""
    return [
        {"role": "system", "content": DOCUMENT_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]


# ════════════════════════════════════════════════════════════════
# STREAMING GENERATORS
# ════════════════════════════════════════════════════════════════

async def _stream_image_generation(
    prompt: str,
    size: str,
    quality: str,
    result: dict,
):
    """
    Handle the full image generation pipeline with progressive streaming.
    Yields SSE event strings.
    Writes the response text to result["text"] for DB saving.
    """
    yield sse({"type": "image_generating"})

    try:
        image_b64 = await generate_image_openai_sync(
            prompt, size=size, quality=quality
        )

        frames = generate_progressive_frames(image_b64, steps=8)

        for frame in frames:
            yield sse({
                "type": "image_progress",
                "progress": frame["progress"],
                "data": frame["data"]
            })
            await asyncio.sleep(0.08)

        # DETERMINE MIME TYPE
        is_png = not image_b64.startswith('/9j/')
        mime = "data:image/png;base64," if is_png else "data:image/jpeg;base64,"
        
        # CREATE MARKDOWN IMAGE TAG TO SAVE IN DATABASE
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


async def _stream_video_generation(
    prompt: str,
    result: dict,
):
    """
    Handle the video generation pipeline.
    Yields SSE event strings.
    Writes the response text to result["text"] for DB saving.
    """
    yield sse({"type": "video_generating"})

    try:
        # TODO: Replace with actual video generation API call (e.g., Replicate, RunwayML, OpenAI Sora)
        # Simulating API delay for UI demonstration
        await asyncio.sleep(3)
        
        # Placeholder video URL (Big Buck Bunny)
        video_url = "https://www.w3schools.com/html/mov_bbb.mp4"
        
        # CREATE MARKDOWN VIDEO TAG TO SAVE IN DATABASE
        markdown_video = f"[![Generated Video]({video_url})]({video_url})"
        
        yield sse({
            "type": "video_generated",
            "url": video_url,
            "prompt": prompt
        })

        result["text"] = markdown_video

    except Exception as e:
        error_str = str(e)
        logger.error(f"Video generation stream error: {error_str}")
        yield sse({
            "type": "video_error",
            "error": error_str
        })
        result["text"] = f"[Video generation failed: {error_str}]"

    return


async def _stream_chat_response(
    prompt: str,
    conversation_id: str,
    model_key: str,
    mode: Optional[str],
    user_id: str,
    result: dict,
):
    """
    Handle text chat with optional web search, streaming tokens.
    """
    model_config = MODEL_ROUTING.get(model_key, MODEL_ROUTING["helox"])
    chat_model = model_config["chat"]
    provider = model_config["provider"]

    should_search = False
    search_context = ""
    search_html = ""

    intent = _detector.detect(prompt)
    if intent.intent == IntentCategory.RESEARCH:
        should_search = True
    elif mode in ("research", "finance", "web"):
        should_search = True
    else:
        time_keywords = [
            'today', 'now', 'current', 'latest', 'recent',
            '2024', '2025', 'price', 'stock', 'news',
            'weather', 'score', 'update', 'happening'
        ]
        if any(kw in prompt.lower() for kw in time_keywords):
            should_search = True

    if should_search and TAVILY_API_KEY:
        try:
            search_context, search_html = await perform_web_search_formatted(prompt)
            if search_html:
                yield sse({
                    "type": "search_results",
                    "html": search_html
                })
        except Exception as e:
            logger.error(f"Search failed in chat stream: {e}")

    system_prompt = get_system_prompt(prompt)
    if mode == "finance":
        system_prompt = FINANCE_SYSTEM_PROMPT

    messages = [{"role": "system", "content": system_prompt}]

    try:
        history = await get_history(conversation_id, limit=6)
        messages.extend(history)
    except Exception as e:
        logger.warning(f"Failed to load history: {e}")

    if (search_context
            and search_context != "[Search API Key missing]"
            and search_context != "[No search results found]"
            and search_context != "[Search failed]"):
        user_content = f"""Using these search results as context:

{search_context}

User question: {prompt}

Provide a comprehensive answer based on the search results above. Cite sources as [1], [2] etc."""
    else:
        user_content = prompt

    messages.append({"role": "user", "content": user_content})

    full_response = ""
    use_model = chat_model
    stream_fn = stream_groq_chat if provider == "groq" else stream_openai_chat

    try:
        async for delta in stream_fn(messages, model=use_model):
            full_response += delta
            yield sse({
                "type": "text_delta",
                "content": delta
            })
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Chat stream error: {error_msg}")
        if not full_response:
            full_response = f"[Error: {error_msg}]"
            yield sse({
                "type": "text_delta",
                "content": f"\n\n*Error occurred: {error_msg}*"
            })

    result["text"] = full_response
    return


# =========================
# ENDPOINTS
# =========================
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {
        "status": "running",
        "service": "HeloxAi Lite",
        "version": "4.2.0",
        "models": {
            "chat": GROQ_CHAT_MODEL,
            "vision": GROQ_VISION_MODEL,
            "tts": OPENAI_TTS_MODEL,
            "stt": GROQ_STT_MODEL,
            "image": OPENAI_IMAGE_MODEL
        },
        "features": [
            "chat", "code", "math", "web_search", "tts", "stt",
            "image_generation", "video_generation", "image_analysis", "code_analysis",
            "document_analysis", "finance", "model_routing", "mode_routing",
            "progressive_image_streaming"
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
            "logout": "POST /session/logout"
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
    file_category = FileCategory.UNKNOWN

    if image_base64:
        clean_b64 = image_base64
        if "," in image_base64:
            clean_b64 = image_base64.split(",", 1)[1]
        image_data_b64 = clean_b64.strip()
        file_category = FileCategory.IMAGE
        logger.info(f"Analysis: received base64 image ({len(image_data_b64)} chars)")

    elif file and file.filename:
        file_filename = file.filename
        content_bytes = b""
        while chunk := await file.read(1024 * 1024):
            content_bytes += chunk
            if len(content_bytes) > MAX_FILE_SIZE:
                raise HTTPException(
                    413,
                    f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB."
                )

        if len(content_bytes) == 0:
            raise HTTPException(400, "Empty file uploaded.")

        if analysis_type and analysis_type != "auto":
            try:
                file_category = FileCategory(analysis_type)
            except ValueError:
                file_category = get_file_category(file_filename)
        else:
            file_category = get_file_category(file_filename)

        if file.content_type and _is_image_mime(file.content_type):
            file_category = FileCategory.IMAGE

        if file_category == FileCategory.IMAGE:
            image_data_b64 = base64.b64encode(content_bytes).decode()
            image_mime_type = file.content_type or "image/png"
            logger.info(f"Analysis: uploaded image file: {file_filename}")
        else:
            file_text_content = await extract_text_safe(content_bytes)
            if (not file_text_content.strip()
                    or file_text_content.strip() == "[Binary or unreadable content]"):
                raise HTTPException(
                    400,
                    f"Could not extract text from file: {file_filename}. "
                    "For images, ensure the file is a valid image format."
                )
            logger.info(
                f"Analysis: uploaded {file_category.value} file: "
                f"{file_filename} ({len(file_text_content)} chars)"
            )
    else:
        raise HTTPException(400, "Either 'file' or 'image_base64' must be provided.")

    conv_id = await get_or_create_conversation(
        user["id"],
        conversation_id,
        f"Analysis: {file_filename}" if file_filename else "Image Analysis"
    )

    user_msg_content = prompt or f"[Uploaded {file_filename} for analysis]"
    await save_message(user["id"], conv_id, "user", user_msg_content)

    if file_category == FileCategory.IMAGE:
        analysis_messages = _build_image_analysis_messages(
            image_data_b64, image_mime_type, prompt
        )
    else:
        language = get_language_from_extension(file_filename)
        if file_category == FileCategory.CODE:
            analysis_messages = _build_code_analysis_messages(
                file_text_content, file_filename, language, prompt
            )
        else:
            analysis_messages = _build_document_analysis_messages(
                file_text_content, file_filename, prompt
            )

    if stream:
        async def analysis_stream():
            full_response = ""
            try:
                async for delta in stream_groq_chat(
                    analysis_messages, model=GROQ_VISION_MODEL
                ):
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
            response_text = await groq_chat_sync(
                analysis_messages, model=GROQ_VISION_MODEL
            )
        except Exception as e:
            response_text = f"[Analysis error: {str(e)}]"

        await save_message(user["id"], conv_id, "assistant", response_text)

        return JSONResponse({
            "response": response_text,
            "conversation_id": conv_id
        })


# ════════════════════════════════════════════════════════════════
# MAIN CHAT ENDPOINT — WITH PROGRESSIVE IMAGE/VIDEO STREAMING
# ════════════════════════════════════════════════════════════════

@app.post("/ask/universal")
async def ask_universal(req: Request, res: Response):
    """
    Universal chat endpoint.

    SSE Event Types:
      - image_generating   : { type: "image_generating" }
      - image_progress     : { type: "image_progress", progress: 0-100, data: "base64..." }
      - image_generated    : { type: "image_generated", data: "base64...", size: "...", quality: "..." }
      - image_error        : { type: "image_error", error: "..." }
      - video_generating   : { type: "video_generating" }
      - video_generated    : { type: "video_generated", url: "...", prompt: "..." }
      - video_error        : { type: "video_error", error: "..." }
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

    user = await get_user_with_auth(req, res, remember)
    user_id = user["id"]

    title = prompt[:50] if len(prompt) > 10 else prompt
    conv_id = await get_or_create_conversation(user_id, conversation_id, title)

    await save_message(user_id, conv_id, "user", prompt)

    intent = _detector.detect(prompt)

    # ── VIDEO GENERATION PATH ──
    if intent.intent == IntentCategory.VIDEO_GENERATION:
        async def video_stream():
            result = {}

            async for event in _stream_video_generation(
                prompt=prompt,
                result=result,
            ):
                yield event

            text_response = (
                f"\n\nHere's the video I generated based on your request: "
                f"*\"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"*"
            )
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
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    # ── IMAGE GENERATION PATH ──
    if intent.intent == IntentCategory.IMAGE_GENERATION:
        async def image_stream():
            result = {}

            async for event in _stream_image_generation(
                prompt=prompt,
                size=image_size,
                quality=image_quality,
                result=result,
            ):
                yield event

            text_response = (
                f"\n\nHere's the image I generated based on your request: "
                f"*\"{prompt[:100]}{'...' if len(prompt) > 100 else ''}\"*"
            )
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
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    # ── TEXT CHAT PATH ──
    else:
        async def chat_stream():
            result = {}

            async for event in _stream_chat_response(
                prompt=prompt,
                conversation_id=conv_id,
                model_key=model_key,
                mode=mode,
                user_id=user_id,
                result=result,
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
    new_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    try:
        await _execute_supabase_with_retry(
            supabase.table("conversations").insert({
                "id": new_id,
                "user_id": user["id"],
                "title": title,
                "created_at": now,
                "updated_at": now,
            })
        )
        return JSONResponse({"id": new_id, "title": title})
    except Exception as e:
        logger.error(f"Failed to create chat: {e}")
        raise HTTPException(500, "Failed to create chat")


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
            {"id": "echo", "name": "Echo", "description": "Warm and clear"},
            {"id": "fable", "name": "Fable", "description": "Expressive and storytelling"},
            {"id": "onyx", "name": "Onyx", "description": "Deep and authoritative"},
            {"id": "nova", "name": "Nova", "description": "Friendly and upbeat"},
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
# ANALYSIS JSON ENDPOINT (non-streaming)
# =========================
@app.post("/analysis/json")
async def analyze_file_json(
    req: Request,
    res: Response,
    file: Optional[UploadFile] = File(None),
    prompt: Optional[str] = Form(None),
    conversation_id: Optional[str] = Form(None),
    analysis_type: Optional[str] = Form(None),
    image_base64: Optional[str] = Form(None),
    image_mime: Optional[str] = Form("image/png"),
):
    user = await get_user_with_auth(req, res)

    image_data_b64 = None
    image_mime_type = image_mime or "image/png"
    file_text_content = None
    file_filename = "unknown"
    file_category = FileCategory.UNKNOWN

    if image_base64:
        clean_b64 = image_base64
        if "," in image_base64:
            clean_b64 = image_base64.split(",", 1)[1]
        image_data_b64 = clean_b64.strip()
        file_category = FileCategory.IMAGE
    elif file and file.filename:
        file_filename = file.filename
        content_bytes = b""
        while chunk := await file.read(1024 * 1024):
            content_bytes += chunk
            if len(content_bytes) > MAX_FILE_SIZE:
                raise HTTPException(413, "File too large")

        if not content_bytes:
            raise HTTPException(400, "Empty file")

        if analysis_type and analysis_type != "auto":
            try:
                file_category = FileCategory(analysis_type)
            except ValueError:
                file_category = get_file_category(file_filename)
        else:
            file_category = get_file_category(file_filename)

        if file.content_type and _is_image_mime(file.content_type):
            file_category = FileCategory.IMAGE

        if file_category == FileCategory.IMAGE:
            image_data_b64 = base64.b64encode(content_bytes).decode()
            image_mime_type = file.content_type or "image/png"
        else:
            file_text_content = await extract_text_safe(content_bytes)
            if not file_text_content.strip() or file_text_content.strip() == "[Binary or unreadable content]":
                raise HTTPException(400, f"Could not extract text from: {file_filename}")
    else:
        raise HTTPException(400, "Either 'file' or 'image_base64' required")

    conv_id = await get_or_create_conversation(
        user["id"], conversation_id,
        f"Analysis: {file_filename}" if file_filename else "Image Analysis"
    )

    user_msg = prompt or f"[Uploaded {file_filename} for analysis]"
    await save_message(user["id"], conv_id, "user", user_msg)

    if file_category == FileCategory.IMAGE:
        messages = _build_image_analysis_messages(image_data_b64, image_mime_type, prompt)
    else:
        language = get_language_from_extension(file_filename)
        if file_category == FileCategory.CODE:
            messages = _build_code_analysis_messages(file_text_content, file_filename, language, prompt)
        else:
            messages = _build_document_analysis_messages(file_text_content, file_filename, prompt)

    try:
        response_text = await groq_chat_sync(messages, model=GROQ_VISION_MODEL)
    except Exception as e:
        response_text = f"[Analysis error: {str(e)}]"

    await save_message(user["id"], conv_id, "assistant", response_text)

    return JSONResponse({
        "response": response_text,
        "conversation_id": conv_id
    })
