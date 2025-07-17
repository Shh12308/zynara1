  # === ZYNARA AI API: GPT-Grade Assistant ===

import os, io, json, tempfile, base64, torch, requests
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    AutoProcessor, AutoModelForVision2Seq,
    AutoModelForSeq2SeqLM
)
from duckduckgo_search import ddg
import whisper
import soundfile as sf
import chromadb

# === Optional Supabase ===
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
SUPABASE_ENABLED = bool(SUPABASE_URL and SUPABASE_KEY)

if SUPABASE_ENABLED:
    from supabase import create_client
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# === API KEYS ===
OPENWEATHER_KEY = os.getenv("OPENWEATHER_KEY", "")
WOLFRAM_KEY = os.getenv("WOLFRAM_KEY", "")
SIGHTENGINE_API = os.getenv("SIGHTENGINE_API", "")
SIGHTENGINE_USER = os.getenv("SIGHTENGINE_USER", "")
TAVILY_KEY = os.getenv("TAVILY_API_KEY", "")

# === Lazy Model Loader ===
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_IDS = {
    "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "codellama": "codellama/CodeLlama-70b-Instruct-hf",
    "deepseek": "deepseek-ai/deepseek-coder-33b-instruct"
}

loaded_models = {}

def load_model(name):
    if name in loaded_models:
        return loaded_models[name]
    
    model_id = MODEL_IDS.get(name, MODEL_IDS["mixtral"])  # fallback to mixtral
    print(f"🔄 Loading {name} model...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    loaded_models[name] = (tokenizer, model)
    return tokenizer, model

# === Load Whisper for STT ===
whisper_model = whisper.load_model("base")

# === Load Coqui TTS ===
from TTS.api import TTS as CoquiTTS
tts_model = CoquiTTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

def text_to_speech(text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tts_model.tts_to_file(text=text, file_path=f.name)
        with open(f.name, "rb") as audio_file:
            return audio_file.read()

# === Load Vision ===
image_model = AutoModelForVision2Seq.from_pretrained("bakLLaVA/BakLLaVA-v1-mixtral")
image_processor = AutoProcessor.from_pretrained("bakLLaVA/BakLLaVA-v1-mixtral")

# === Translation ===
from transformers import AutoTokenizer as TransTokenizer
translate_tokenizer = TransTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
translate_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
lang = detect(text) 

from langdetect import detect
lang = detect(text)  # e.g., 'fr'

# === RAG Memory ===
chroma = chromadb.Client()
chroma.create_collection("zynara_facts")

# === App Init ===
app = FastAPI()

# === Request Schema ===
class PromptRequest(BaseModel):
    prompt: str
    user_id: str = "anonymous"
    stream: bool = False
    model: str = "mixtral"  # "mixtral", "codellama", "deepseek"
# === Helpers ===

def tavily_search(query):
    if not TAVILY_KEY:
        return ["Tavily API key not set."]
    
    headers = {
        "Authorization": f"Bearer {TAVILY_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "query": query,
        "include_links": True,
        "include_answers": True
    }

    try:
        res = requests.post("https://api.tavily.com/search", headers=headers, json=payload)
        if res.status_code == 200:
            data = res.json()
            results = [f"{src['title']} - {src['url']}" for src in data.get("sources", [])]
            answer = data.get("answer", "")
            return [answer] + results if answer else results
        else:
            return [f"❌ Tavily failed: {res.status_code}"]
    except Exception as e:
        return [f"⚠️ Error: {str(e)}"]

def moderate_text(text):
    if not SIGHTENGINE_USER or not SIGHTENGINE_API:
        return True
    r = requests.post("https://api.sightengine.com/1.0/text/check.json", data={
        "text": text,
        "mode": "standard",
        "api_user": SIGHTENGINE_USER,
        "api_secret": SIGHTENGINE_API
    })
    result = r.json()
    return result.get("profanity", {}).get("matches") == []

def log_usage(user_id, prompt, reply):
    if SUPABASE_ENABLED:
        supabase.table("chat_logs").insert({"user_id": user_id, "prompt": prompt, "response": reply}).execute()

def remember(user_id, key, value):
    if SUPABASE_ENABLED:
        supabase.table("memory").insert({"user_id": user_id, "key": key, "value": value}).execute()

def recall(user_id, key):
    if not SUPABASE_ENABLED:
        return None
    result = supabase.table("memory").select("value").eq("user_id", user_id).eq("key", key).execute()
    return result.data[0]["value"] if result.data else None

def web_search(query):
    return [f"{r['title']} - {r['href']}" for r in ddg(query, max_results=3)]

def get_weather(city):
    if not OPENWEATHER_KEY:
        return "Weather API key not set."
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_KEY}&units=metric"
    r = requests.get(url).json()
    return f"{r['weather'][0]['description']} in {city}, {r['main']['temp']}°C" if "weather" in r else "Not found"

def wolfram_query(q):
    if not WOLFRAM_KEY:
        return "Wolfram key not set."
    url = f"https://api.wolframalpha.com/v1/result?appid={WOLFRAM_KEY}&i={requests.utils.quote(q)}"
    r = requests.get(url)
    return r.text if r.status_code == 200 else "Not found"

def describe_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    inputs = image_processor(prompt="What's in this image?", images=image, return_tensors="pt")
    output = image_model.generate(**inputs)
    return image_processor.decode(output[0], skip_special_tokens=True)

def translate_text(text, to_lang="fra_Latn"):
    translate_tokenizer.src_lang = "eng_Latn"
    inputs = translate_tokenizer(text, return_tensors="pt")
    translated = translate_model.generate(**inputs, forced_bos_token_id=translate_tokenizer.lang_code_to_id[to_lang])
    return translate_tokenizer.decode(translated[0], skip_special_tokens=True)

def store_context(prompt, answer):
    chroma.get_collection("zynara_facts").add(
        documents=[f"{prompt} => {answer}"],
        metadatas=[{"source": "chat"}],
        ids=[f"id_{hash(prompt)}"]
    )

def retrieve_context(prompt):
    results = chroma.get_collection("zynara_facts").query(query_texts=[prompt], n_results=3)
    return results["documents"][0] if results["documents"] else []

def generate_response(prompt, stream=False, model="mixtral"):
    tokenizer, lm_model = load_model(model)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(lm_model.device)

    if stream:
        def gen():
            for i in range(1, 150):
                out = lm_model.generate(input_ids, max_new_tokens=i, do_sample=True)
                yield tokenizer.decode(out[0], skip_special_tokens=True) + "\n"
        return StreamingResponse(gen(), media_type="text/plain")
    else:
        out = lm_model.generate(input_ids, max_new_tokens=512, do_sample=True)
        return tokenizer.decode(out[0], skip_special_tokens=True)

def fetch_history(user_id: str, limit: int = 6):
    if not SUPABASE_ENABLED:
        return []
    result = supabase.table("chat_logs").select("role, message").eq("user_id", user_id).order("timestamp", desc=True).limit(limit).execute()
    history = result.data[::-1]  # Reverse to get oldest → newest
    return history

# === API Routes ===

@app.post("/chat")
def chat(req: PromptRequest):
    if not moderate_text(req.prompt):
        return {"error": "Prompt contains unsafe content."}
    history = fetch_history(req.user_id)
context_prompt = ""

for turn in history:
    role = turn["role"]
    msg = turn["message"]
    context_prompt += f"{role}: {msg}\n"

context_prompt += f"user: {req.prompt}\nassistant:"

reply = generate_response(context_prompt, stream=req.stream, model=req.model)
   if not req.stream:
    supabase.table("chat_logs").insert([
        {"user_id": req.user_id, "role": "user", "message": req.prompt},
        {"user_id": req.user_id, "role": "assistant", "message": reply}
    ]).execute()

    remember(req.user_id, "last_prompt", req.prompt)
    store_context(req.prompt, reply)

@app.post("/tts")
def tts(req: PromptRequest):
    audio_bytes = text_to_speech(req.prompt)
    return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/wav")

@app.post("/image")
def vision(prompt: str = Form(...), image: UploadFile = File(...)):
    image_bytes = image.file.read()
    caption = describe_image(image_bytes)
    final_prompt = f"{prompt}\nImage description: {caption}"
    response = generate_response(final_prompt)
    return {"caption": caption, "response": response}

@app.post("/voice")
def transcribe(file: UploadFile = File(...)):
    with open("temp.wav", "wb") as f:
        f.write(file.file.read())
    result = whisper_model.transcribe("temp.wav")
    return {"transcription": result["text"]}

@app.get("/search")
def search(query: str):
    return {"results": web_search(query)}

@app.get("/tavily")
def tavily(query: str):
    return {"results": tavily_search(query)}

@app.get("/weather")
def weather(city: str):
    return {"weather": get_weather(city)}

@app.get("/wolfram")
def wolfram(q: str):
    return {"result": wolfram_query(q)}

@app.post("/translate")
def translate(text: str, lang: str = "fra_Latn"):
    return {"translated": translate_text(text, lang)}

@app.post("/docqa")
def document_qa(file: UploadFile = File(...), query: str = Form(...)):
    text = extract_text(file)  # use PyMuPDF or unstructured
    context = get_answer_from_text(query, text)  # using RAG or direct prompt
    return {"answer": context}

@app.post("/exec")
def exec_code(req: PromptRequest):
    try:
        exec_globals = {}
        exec(req.prompt, exec_globals)
        return {"output": exec_globals}
    except Exception as e:
        return {"error": str(e)}

from sympy import simplify, solve

@app.post("/math")
def math_solver(expr: str):
    try:
        result = simplify(expr)
        return {"result": str(result)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/explain")
def explain_code(req: PromptRequest):
    prompt = f"Explain this code clearly:\n\n{req.prompt}"
    return generate_response(prompt)

@app.post("/generate-code")
def gen_code(req: PromptRequest):
    prompt = f"Write {req.prompt} in Python."
    return generate_response(prompt, model="codellama")

@app.post("/generate-image")
async def generate_image(data: TextPrompt):
    image = text2img_pipe(data.prompt).images[0]
    filename = f"{uuid.uuid4().hex}.png"
    filepath = f"./{filename}"
    image.save(filepath)
    return FileResponse(filepath, media_type="image/png", filename=filename)

@app.post("/clear-memory")
def clear_memory(req: PromptRequest):
    supabase.table("chat_logs").delete().eq("user_id", req.user_id).execute()
    return {"status": "cleared"}

# === 2. Image Captioning ===
@app.post("/caption-image")
async def caption_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    inputs = caption_processor(image, return_tensors="pt").to(device)
    out = caption_model.generate(**inputs)
    caption = caption_processor.decode(out[0], skip_special_tokens=True)
    return JSONResponse(content={"caption": caption})

@app.get("/")
def root():
    return {"message": "✅ Zynara AI is ready to Go!"} 
