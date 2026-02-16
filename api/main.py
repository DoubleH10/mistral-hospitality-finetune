"""
FastAPI inference server with streaming support.

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

# TODO: Implement FastAPI server
# This will be completed after training and evaluation are working

from fastapi import FastAPI

app = FastAPI(
    title="Mistral Hospitality API",
    description="Inference API for fine-tuned Mistral-7B hospitality model",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def generate(prompt: str, max_tokens: int = 256):
    # TODO: Load model and generate
    return {"error": "Not yet implemented. Run training first."}


def run():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    run()
