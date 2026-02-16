"""
FastAPI inference server with streaming support.

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
    MODEL_PATH=./outputs/mistral-hospitality-qlora uvicorn api.main:app
"""

import os
import asyncio
from contextlib import asynccontextmanager
from threading import Thread

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from transformers import TextIteratorStreamer

from src.inference import generate, load_model

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
_state: dict = {}

MODEL_PATH = os.getenv("MODEL_PATH", "Hadix10/mistral-hospitality-qlora")
BASE_MODEL = os.getenv("BASE_MODEL", "mistralai/Mistral-7B-Instruct-v0.3")


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Loading model from {MODEL_PATH}...")
    model, tokenizer = load_model(MODEL_PATH, BASE_MODEL)
    _state["model"] = model
    _state["tokenizer"] = tokenizer
    _state["ready"] = True
    print("Model loaded — server ready.")
    yield
    _state.clear()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Mistral Hospitality API",
    description="Inference API for fine-tuned Mistral-7B hospitality model",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt")
    max_tokens: int = Field(256, ge=1, le=1024, description="Max tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    stream: bool = Field(False, description="Stream response via SSE")


class GenerateResponse(BaseModel):
    response: str
    prompt: str
    tokens_generated: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    mem = ""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        mem = f"{alloc:.1f}/{total:.1f} GB"
    return {
        "status": "ready" if _state.get("ready") else "loading",
        "model": MODEL_PATH,
        "device": gpu,
        "gpu_memory": mem or "N/A",
    }


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    if not _state.get("ready"):
        raise HTTPException(503, "Model is still loading")

    if req.stream:
        return _stream_response(req)

    model = _state["model"]
    tokenizer = _state["tokenizer"]

    response = generate(
        model, tokenizer, req.prompt,
        max_new_tokens=req.max_tokens,
        temperature=req.temperature,
        do_sample=req.temperature > 0,
        top_p=req.top_p,
    )

    return GenerateResponse(
        response=response,
        prompt=req.prompt,
        tokens_generated=len(tokenizer.encode(response)),
    )


# ---------------------------------------------------------------------------
# SSE streaming
# ---------------------------------------------------------------------------
def _stream_response(req: GenerateRequest) -> EventSourceResponse:
    model = _state["model"]
    tokenizer = _state["tokenizer"]

    prompt = req.prompt
    if "[INST]" not in prompt:
        prompt = f"[INST] {prompt} [/INST]"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    gen_kwargs = {
        **inputs,
        "max_new_tokens": req.max_tokens,
        "temperature": req.temperature if req.temperature > 0 else 1.0,
        "do_sample": req.temperature > 0,
        "top_p": req.top_p,
        "pad_token_id": tokenizer.eos_token_id,
        "streamer": streamer,
    }

    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()

    async def event_generator():
        for token in streamer:
            yield {"data": token}
            await asyncio.sleep(0)
        yield {"data": "[DONE]"}

    return EventSourceResponse(event_generator())


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def run():
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
