import os
import re
import json
import uuid
import asyncio
import logging
import time
import base64
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

RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW = 60

if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set.")

app = FastAPI(
    title="HeloxAi Lite",
    description="Text, Code, Math, Research, Image Generation & File Analysis Backend",
    version="4.0.0"
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


def _get_conv_lock(conv_id: str) -> asyncio.Lock:
    if conv_id not in _conv_creation_locks:
        _conv_creation_locks[conv_id] = asyncio.Lock()
    return _conv_creation_locks[conv_id]


# =========================
# MIDDLEWARE: IP RATE LIMITER
# =========================
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/" or request.method == "OPTIONS":
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    if client_ip not in _rate_limit_store:
        _rate_limit_store[client_ip] = []

    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if now - t < RATE_LIMIT_WINDOW
    ]

    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        logger.warning(f"Rate limit hit for IP: {client_ip}")
        return JSONResponse(
            status_code=429,
            content={"detail": "Too many requests. Please slow down."}
        )

    _rate_limit_store[client_ip].append(now)
    response = await call_next(request)
    return response


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
            _session_cache[user_id] = {"token": token}
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
        _session_cache[user_id] = {"token": token}
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
        for intent in [IntentCategory.IMAGE_GENERATION]:
            patterns = self.compiled_patterns.get(intent, [])
            matches = sum(1 for p in patterns if p.search(text))
            if matches > 0:
                return IntentResult(intent=intent, confidence=min(0.6 + matches * 0.1, 0.98))

        for intent, patterns in self.compiled_patterns.items():
            if intent == IntentCategory.IMAGE_GENERATION:
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
    user_id = req.cookies.get(PRIMARY_COOKIE)
    token = req.cookies.get(SESSION_TOKEN_COOKIE)
    expiry = req.cookies.get(SESSION_EXPIRY_COOKIE)

    if user_id and token:
        if is_session_expired(expiry or "0"):
            clear_session_cookies(res)
        elif await validate_session_token(user_id, token):
            return {"id": user_id, "session_valid": True}

    new_id = str(uuid.uuid4())
    new_token = await create_user_session(new_id, remember)
    if new_token is None:
        raise HTTPException(500, "Failed to create user session")

    set_session_cookies(res, new_id, new_token, remember)
    return {"id": new_id, "session_valid": True}


async def get_user_with_auth(req: Request, res: Response, remember: bool = True) -> Dict[str, Any]:
    """Extended get_user that also checks Authorization header."""
    auth_header = req.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "")
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


async def get_history(conv_id: str, limit: int = 20):
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
            check = await _execute_supabase_with_retry(
                supabase.table("conversations")
                .select("id")
                .eq("id", proposed_id)
                .limit(1)
            )
            if check.data:
                _conv_creation_locks.pop(lock_key, None)
                return proposed_id
            logger.warning(f"Conversation ID {proposed_id} provided but not found in DB.")

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
                "include_images": False
            })
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])

            if not results:
                return "[No search results found]", ""

            # Build plain context for AI
            context = ""
            if data.get("answer"):
                context += f"Answer: {data['answer']}\n"
            for i, r in enumerate(results):
                context += f"[{i+1}] {r['title']}: {r['content']}\nURL: {r['url']}\n\n"

            # Build HTML matching frontend CSS classes
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

            for i, r in enumerate(results[:3]):
                domain = urlparse(r["url"]).hostname or ""
                html += (
                    f'<a href="{r["url"]}" class="search-card" '
                    f'target="_blank" rel="noopener">'
                    f'<img class="search-thumb" '
                    f'src="https://www.google.com/s2/favicons?domain={domain}&sz=64" '
                    f'alt="" onerror="this.style.display=\'none\'">'
                    f'<div class="search-info">'
                    f'<div class="search-title">{r["title"]}</div>'
                    f'<div class="search-link">'
                    f'<img class="search-link-favicon" '
                    f'src="https://www.google.com/s2/favicons?domain={domain}&sz=16" '
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
    """Stream from OpenAI API — mirrors stream_groq_chat interface."""
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
    """Non-streaming Groq chat completion with retry."""
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
    """Non-streaming OpenAI chat completion."""
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


# =========================
# ENDPOINTS
# =========================
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {
        "status": "running",
        "service": "HeloxAi Lite",
        "version": "4.0.0",
        "models": {
            "chat": GROQ_CHAT_MODEL,
            "vision": GROQ_VISION_MODEL,
            "tts": OPENAI_TTS_MODEL,
            "stt": GROQ_STT_MODEL,
            "image": OPENAI_IMAGE_MODEL
        },
        "features": [
            "chat", "code", "math", "web_search", "tts", "stt",
            "image_generation", "image_analysis", "code_analysis",
            "document_analysis", "finance", "model_routing", "mode_routing"
        ],
        "endpoints": {
            "chat": "POST /ask/universal",
            "new_chat": "POST /newchat",
            "analysis": "POST /analysis",
            "analysis_json": "POST /analysis/json",
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
    """Fallback plan endpoint when Supabase RLS fails on the client."""
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

    display_prompt = prompt or f"Analysis of {file_filename}"
    conv_id = await get_or_create_conversation(
        user_id=user["id"], proposed_id=conversation_id, title=display_prompt
    )

    if file_category == FileCategory.IMAGE:
        user_msg = (
            f"[Image Analysis] "
            f"{file_filename if file and file.filename else 'Base64 image'}"
            + (f"\n{prompt}" if prompt else "")
        )
    else:
        language = get_language_from_extension(file_filename)
        user_msg = (
            f"[{file_category.value.title()} Analysis] "
            f"{file_filename} ({language})"
            + (f"\n{prompt}" if prompt else "")
        )
    await save_message(user["id"], conv_id, "user", user_msg)

    if file_category == FileCategory.IMAGE:
        if not GROQ_API_KEY:
            raise HTTPException(500, "Groq API Key required for image analysis")
        messages = _build_image_analysis_messages(image_data_b64, image_mime_type, prompt)
        use_model = GROQ_VISION_MODEL
    elif file_category == FileCategory.CODE:
        language = get_language_from_extension(file_filename)
        messages = _build_code_analysis_messages(
            file_text_content, file_filename, language, prompt
        )
        use_model = GROQ_CHAT_MODEL
    else:
        messages = _build_document_analysis_messages(
            file_text_content, file_filename, prompt
        )
        use_model = GROQ_CHAT_MODEL

    if stream:
        async def analysis_event_gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                yield sse({"type": "conversation_id", "conversation_id": conv_id})
                full_text = ""
                async for token in stream_groq_chat(messages, model=use_model):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})
                await save_message(user["id"], conv_id, "assistant", full_text)
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Analysis stream error: {e}")
                yield sse({"type": "error", "message": str(e)})
            finally:
                active_streams.pop(user["id"], None)
        return StreamingResponse(analysis_event_gen(), media_type="text/event-stream")
    else:
        try:
            reply = await groq_chat_sync(messages, model=use_model)
            await save_message(user["id"], conv_id, "assistant", reply)
            return {
                "reply": reply,
                "conversation_id": conv_id,
                "analysis_type": file_category.value
            }
        except Exception as e:
            logger.error(f"Analysis error: {e}")
            raise HTTPException(500, str(e))


# =========================
# ANALYSIS ENDPOINT (JSON BODY)
# =========================
@app.post("/analysis/json")
async def analyze_file_json(req: Request, res: Response):
    body = await req.json()

    image_b64 = body.get("image_base64")
    text_content = body.get("content")
    filename = body.get("filename", "unknown")
    prompt = body.get("prompt")
    conv_id_proposed = body.get("conversation_id")
    do_stream = body.get("stream", True)
    remember = body.get("remember", True)
    analysis_type_str = body.get("analysis_type", "auto")

    user = await get_user_with_auth(req, res, remember)

    file_category = FileCategory.UNKNOWN
    image_data_b64 = None
    image_mime_type = body.get("image_mime", "image/png")
    file_text = None

    if image_b64:
        clean_b64 = image_b64
        if "," in image_b64:
            clean_b64 = image_b64.split(",", 1)[1]
        image_data_b64 = clean_b64.strip()
        file_category = (
            FileCategory.IMAGE
            if analysis_type_str in ("auto", "image")
            else FileCategory(analysis_type_str)
        )
    elif text_content:
        file_text = text_content[:MAX_TEXT_LENGTH]
        if analysis_type_str and analysis_type_str != "auto":
            try:
                file_category = FileCategory(analysis_type_str)
            except ValueError:
                file_category = get_file_category(filename)
        else:
            file_category = get_file_category(filename)
    else:
        raise HTTPException(400, "Either 'image_base64' or 'content' must be provided.")

    display_prompt = prompt or f"Analysis of {filename}"
    conv_id = await get_or_create_conversation(
        user_id=user["id"], proposed_id=conv_id_proposed, title=display_prompt
    )

    if file_category == FileCategory.IMAGE:
        user_msg = f"[Image Analysis] {filename}" + (f"\n{prompt}" if prompt else "")
    else:
        language = get_language_from_extension(filename)
        user_msg = (
            f"[{file_category.value.title()} Analysis] {filename} ({language})"
            + (f"\n{prompt}" if prompt else "")
        )
    await save_message(user["id"], conv_id, "user", user_msg)

    if file_category == FileCategory.IMAGE:
        if not GROQ_API_KEY:
            raise HTTPException(500, "Groq API Key required for image analysis")
        messages = _build_image_analysis_messages(image_data_b64, image_mime_type, prompt)
        use_model = GROQ_VISION_MODEL
    elif file_category == FileCategory.CODE:
        language = get_language_from_extension(filename)
        messages = _build_code_analysis_messages(file_text, filename, language, prompt)
        use_model = GROQ_CHAT_MODEL
    else:
        messages = _build_document_analysis_messages(file_text, filename, prompt)
        use_model = GROQ_CHAT_MODEL

    if do_stream:
        async def analysis_json_event_gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                yield sse({"type": "conversation_id", "conversation_id": conv_id})
                full_text = ""
                async for token in stream_groq_chat(messages, model=use_model):
                    if task.cancelled():
                        break
                    full_text += token
                    yield sse({"type": "token", "text": token})
                await save_message(user["id"], conv_id, "assistant", full_text)
                yield sse({"type": "done"})
            except Exception as e:
                logger.error(f"Analysis JSON stream error: {e}")
                yield sse({"type": "error", "message": str(e)})
            finally:
                active_streams.pop(user["id"], None)
        return StreamingResponse(analysis_json_event_gen(), media_type="text/event-stream")
    else:
        try:
            reply = await groq_chat_sync(messages, model=use_model)
            await save_message(user["id"], conv_id, "assistant", reply)
            return {
                "reply": reply,
                "conversation_id": conv_id,
                "analysis_type": file_category.value
            }
        except Exception as e:
            logger.error(f"Analysis JSON error: {e}")
            raise HTTPException(500, str(e))


# =========================
# UNIVERSAL CHAT ENDPOINT
# =========================
@app.post("/ask/universal")
async def ask_universal(req: Request, res: Response):
    content_type = req.headers.get("content-type", "")
    body = {}

    if "application/json" in content_type:
        body = await req.json()
    elif "multipart/form-data" in content_type:
        form = await req.form()
        body = dict(form)
        if "file" in form:
            file: UploadFile = form["file"]
            content_bytes = b""
            while chunk := await file.read(1024 * 1024):
                content_bytes += chunk
                if len(content_bytes) > MAX_FILE_SIZE:
                    raise HTTPException(413, "File too large")
            text_content = await extract_text_safe(content_bytes)
            file_prefix = (
                f"\n\n[FILE CONTENT: {file.filename}]\n"
                f"{text_content}\n[END FILE]\n"
            )
            body["prompt"] = body.get("prompt", "") + file_prefix

    prompt = body.get("prompt", "")
    conv_id = body.get("conversation_id")
    stream = body.get("stream", True)
    remember = body.get("remember", True)
    image_size = body.get("image_size", "1024x1024")
    image_quality = body.get("image_quality", "medium")
    mode = body.get("mode", "general")
    requested_model = body.get("model", "helox")

    if not prompt:
        raise HTTPException(400, "Prompt required")

    # Auth
    user = await get_user_with_auth(req, res, remember)

    # Model routing
    model_config = MODEL_ROUTING.get(requested_model, MODEL_ROUTING["helox"])

    # Intent detection
    intent = _detector.detect(prompt)
    is_image_request = intent.intent == IntentCategory.IMAGE_GENERATION

    # Mode-aware behavior
    needs_search = False
    force_code_prompt = False
    force_finance_prompt = False

    if mode == "search":
        needs_search = True
    elif mode == "code":
        force_code_prompt = True
    elif mode == "finance":
        needs_search = True
        force_finance_prompt = True
    elif mode == "general":
        search_keywords = [
            "latest", "news", "current", "price", "weather",
            "stock", "who is", "how much", "market", "score"
        ]
        if any(kw in prompt.lower() for kw in search_keywords):
            needs_search = True

    # Create/get conversation
    conv_id = await get_or_create_conversation(
        user_id=user["id"], proposed_id=conv_id, title=prompt
    )
    await save_message(user["id"], conv_id, "user", prompt)

    # ===========================
    # IMAGE GENERATION PATH
    # ===========================
    if is_image_request:
        async def image_event_gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                yield sse({"type": "status", "message": "Generating image..."})

                image_b64 = await generate_image_openai_sync(
                    prompt=prompt, size=image_size, quality=image_quality
                )

                data_url = f"data:image/png;base64,{image_b64}"

                # Structured media event — frontend's onMedia() handles this
                yield sse({
                    "type": "media",
                    "media_type": "image",
                    "data": [{"url": data_url, "name": "Generated Image"}]
                })

                # Save as markdown for history
                md_image = f"![Generated Image]({data_url})"
                await save_message(user["id"], conv_id, "assistant", md_image)
                yield sse({"type": "done"})

            except Exception as e:
                logger.error(f"Image stream error: {e}")
                yield sse({"type": "error", "message": str(e)})
            finally:
                active_streams.pop(user["id"], None)

        if stream:
            return StreamingResponse(
                image_event_gen(), media_type="text/event-stream"
            )
        else:
            try:
                image_b64 = await generate_image_openai_sync(
                    prompt=prompt, size=image_size, quality=image_quality
                )
                data_url = f"data:image/png;base64,{image_b64}"
                md_image = f"![Generated Image]({data_url})"
                await save_message(user["id"], conv_id, "assistant", md_image)
                return {"reply": md_image, "conversation_id": conv_id}
            except Exception as e:
                logger.error(f"Image generation error: {e}")
                raise HTTPException(500, str(e))

    # ===========================
    # TEXT/CHAT PATH
    # ===========================
    if stream:
        async def event_gen():
            task = asyncio.current_task()
            active_streams[user["id"]] = task
            try:
                full_text = ""
                search_context = ""
                search_html = ""

                if needs_search:
                    yield sse({"type": "status", "message": "Searching the web..."})
                    search_context, search_html = await perform_web_search_formatted(prompt)

                    if search_html:
                        yield sse({"type": "token", "text": search_html})

                    yield sse({"type": "status", "message": "Analyzing results..."})
                else:
                    yield sse({"type": "status", "message": "Thinking..."})

                history = await get_history(conv_id)

                # Select system prompt based on mode
                if force_code_prompt:
                    system_prompt = CODE_ANALYSIS_SYSTEM_PROMPT
                elif force_finance_prompt:
                    system_prompt = FINANCE_SYSTEM_PROMPT
                else:
                    system_prompt = get_system_prompt(prompt)

                if search_context:
                    system_prompt += (
                        f"\n\nWEB SEARCH RESULTS:\n{search_context}\n\n"
                        "Use these results to answer. "
                        "Cite sources as [1], [2] etc. "
                        "Include a 'Sources' section at the bottom with full URLs."
                    )

                messages = [{"role": "system", "content": system_prompt}] + history

                # Route to correct provider based on model selection
                if model_config["provider"] == "openai" and OPENAI_API_KEY:
                    async for token in stream_openai_chat(
                        messages, model_config["chat"]
                    ):
                        if task.cancelled():
                            break
                        full_text += token
                        yield sse({"type": "token", "text": token})
                else:
                    async for token in stream_groq_chat(
                        messages, model=model_config["chat"]
                    ):
                        if task.cancelled():
                            break
                        full_text += token
                        yield sse({"type": "token", "text": token})

                await save_message(user["id"], conv_id, "assistant", full_text)
                yield sse({"type": "done"})

            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield sse({"type": "error", "message": str(e)})
            finally:
                active_streams.pop(user["id"], None)

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    else:
        # Non-streaming path
        search_context = ""
        if needs_search:
            search_context, _ = await perform_web_search_formatted(prompt)

        history = await get_history(conv_id)

        if force_code_prompt:
            system_prompt = CODE_ANALYSIS_SYSTEM_PROMPT
        elif force_finance_prompt:
            system_prompt = FINANCE_SYSTEM_PROMPT
        else:
            system_prompt = get_system_prompt(prompt)

        if search_context:
            system_prompt += (
                f"\n\nWEB SEARCH RESULTS:\n{search_context}\n\n"
                "Use these results to answer. Cite sources as [1], [2] etc."
            )

        messages = [{"role": "system", "content": system_prompt}] + history

        try:
            if model_config["provider"] == "openai" and OPENAI_API_KEY:
                reply = await openai_chat_sync(
                    messages, model_config["chat"]
                )
            else:
                reply = await groq_chat_sync(
                    messages, model=model_config["chat"]
                )

            await save_message(user["id"], conv_id, "assistant", reply)
            return {"reply": reply, "conversation_id": conv_id}
        except Exception as e:
            logger.error(f"Non-stream chat error: {e}")
            raise HTTPException(
                429, "Rate limit exceeded. Please wait a minute before trying again."
            )


# =========================
# CHAT MANAGEMENT
# =========================
@app.post("/newchat")
async def new_chat(req: Request, res: Response):
    user = await get_user_with_auth(req, res)
    new_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    await _execute_supabase_with_retry(
        supabase.table("conversations").insert({
            "id": new_id,
            "user_id": user["id"],
            "title": "New Chat",
            "created_at": now,
            "updated_at": now,
        })
    )
    return {"conversation_id": new_id, "status": "created"}


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, req: Request, res: Response):
    user = await get_user_with_auth(req, res)

    check = await _execute_supabase_with_retry(
        supabase.table("conversations")
        .select("id, user_id")
        .eq("id", chat_id)
        .eq("user_id", user["id"])
        .limit(1)
    )

    if not check.data:
        logger.warning(
            f"DELETE /chats/{chat_id}: "
            f"Chat not found or not owned by user {user['id']}"
        )
        raise HTTPException(404, "Chat not found")

    try:
        await _execute_supabase_with_retry(
            supabase.table("messages")
            .delete()
            .eq("conversation_id", chat_id)
        )
    except Exception as e:
        logger.error(f"Failed to delete messages for chat {chat_id}: {e}")

    try:
        await _execute_supabase_with_retry(
            supabase.table("conversations")
            .delete()
            .eq("id", chat_id)
            .eq("user_id", user["id"])
        )
    except Exception as e:
        logger.error(f"Failed to delete chat {chat_id}: {e}")
        raise HTTPException(500, "Failed to delete chat")

    logger.info(f"Deleted chat {chat_id} and its messages for user {user['id']}")
    return {"status": "deleted", "conversation_id": chat_id}


@app.get("/chats")
async def list_chats(
    req: Request,
    res: Response,
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0)
):
    user = await get_user_with_auth(req, res)
    result = await _execute_supabase_with_retry(
        supabase.table("conversations")
        .select("*")
        .eq("user_id", user["id"])
        .order("updated_at", desc=True)
        .range(offset, offset + limit - 1)
    )
    return {"chats": result.data}


@app.get("/chat/{conversation_id}/messages")
async def get_messages(conversation_id: str):
    msgs = await _execute_supabase_with_retry(
        supabase.table("messages")
        .select("role, content, created_at")
        .eq("conversation_id", conversation_id)
        .order("created_at", desc=False)
    )
    return {"messages": msgs.data}


# =========================
# TTS / STT
# =========================
@app.post("/tts")
async def text_to_speech(req: Request):
    data = await req.json()
    text = data.get("text")
    voice = data.get("voice", "alloy")

    allowed_voices = ["alloy", "onyx", "nova", "shimmer", "echo", "fable"]
    if voice not in allowed_voices:
        voice = "alloy"

    if not text:
        raise HTTPException(400, "text required")
    if not OPENAI_API_KEY:
        raise HTTPException(500, "Missing OpenAI API Key")

    async def stream_audio():
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                "https://api.openai.com/v1/audio/speech",
                headers=get_openai_headers(),
                json={
                    "model": OPENAI_TTS_MODEL,
                    "voice": voice,
                    "input": text,
                    "response_format": "mp3"
                }
            ) as response:
                if response.status_code != 200:
                    error_body = await response.aread()
                    logger.error(
                        f"OpenAI TTS Error {response.status_code}: "
                        f"{error_body.decode()}"
                    )
                    return
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream_audio(), media_type="audio/mpeg")


@app.get("/tts/voices")
async def get_voices():
    return {
        "voices": [
            {"id": "alloy", "name": "Alloy"},
            {"id": "fable", "name": "Fable"},
            {"id": "onyx", "name": "Onyx"},
            {"id": "nova", "name": "Nova"},
            {"id": "shimmer", "name": "Shimmer"},
            {"id": "echo", "name": "Echo"}
        ]
    }


@app.post("/stt")
async def speech_to_text(file: UploadFile = File(...)):
    if not GROQ_API_KEY:
        raise HTTPException(500, "Missing Groq API Key")

    allowed_types = [
        "audio/mpeg", "audio/mp4", "audio/wav", "audio/x-wav",
        "audio/webm", "audio/ogg", "audio/flac", "audio/m4a",
        "video/mp4", "video/webm"
    ]
    if file.content_type and file.content_type not in allowed_types:
        logger.warning(f"STT: Unexpected content type: {file.content_type}")

    content = b""
    while chunk := await file.read(1024 * 1024):
        content += chunk
        if len(content) > 25 * 1024 * 1024:
            raise HTTPException(400, "Audio file too large. Maximum size is 25MB.")

    if len(content) == 0:
        raise HTTPException(400, "Empty audio file")

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            files = {
                "file": (
                    file.filename or "audio.mp3",
                    content,
                    file.content_type or "audio/mpeg"
                )
            }
            data = {"model": GROQ_STT_MODEL, "response_format": "json"}

            r = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=get_groq_headers_multipart(),
                files=files,
                data=data
            )
            if r.status_code != 200:
                error_detail = r.text
                logger.error(f"Groq STT Error {r.status_code}: {error_detail}")
                raise HTTPException(
                    status_code=r.status_code,
                    detail=f"Groq STT failed: {error_detail}"
                )

            result = r.json()
            return {
                "text": result.get("text", ""),
                "model": GROQ_STT_MODEL,
                "provider": "groq"
            }
    except httpx.TimeoutException:
        raise HTTPException(504, "Speech transcription timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT Error: {e}")
        raise HTTPException(500, f"Speech to text failed: {str(e)}")


# =========================
# SESSION
# =========================
@app.post("/session/logout")
async def logout(req: Request, res: Response):
    clear_session_cookies(res)
    return {"status": "logged_out"}


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
