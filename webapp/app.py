"""
Empathetic Chat API Server
==========================
FastAPI backend for the empathetic chatbot webapp.

Run with: uvicorn app:app --reload --port 8000
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from contextlib import asynccontextmanager

# Global model variables
model = None
tokenizer = None

# ============================================================
# Configuration
# ============================================================
MODEL_PATH = os.environ.get("MODEL_PATH", "../trained_model/lora_adapter")
USE_UNSLOTH = True  # Set to False if you have the merged model

SYSTEM_PROMPT = """You are a warm, supportive, and empathetic friend. You listen carefully to what people share and respond with genuine care and understanding.

When someone shares their feelings:
1. Acknowledge and validate their emotions first
2. Show that you understand their situation
3. Ask thoughtful follow-up questions
4. Offer support without being preachy or giving unsolicited advice
5. Never minimize their feelings with phrases like "just" or "at least"

You're not a therapist - you're a caring friend who's always there to listen."""


# ============================================================
# Model Loading
# ============================================================
def load_model():
    """Load the empathetic model."""
    global model, tokenizer
    
    print(f"🔄 Loading model from: {MODEL_PATH}")
    
    if USE_UNSLOTH:
        from unsloth import FastLanguageModel
        
        # Check if it's a LoRA adapter or merged model
        if os.path.exists(os.path.join(MODEL_PATH, "adapter_config.json")):
            # It's a LoRA adapter - load base model first
            print("  📦 Loading LoRA adapter...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name="unsloth/Qwen3-8B-bnb-4bit",
                max_seq_length=1024,
                load_in_4bit=True,
            )
            # Load adapter
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, MODEL_PATH)
        else:
            # It's a merged model
            print("  📦 Loading merged model...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=MODEL_PATH,
                max_seq_length=1024,
                load_in_4bit=True,
            )
        
        FastLanguageModel.for_inference(model)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded successfully!")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    load_model()
    yield
    # Cleanup on shutdown
    global model, tokenizer
    del model, tokenizer
    torch.cuda.empty_cache()


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="Empathetic Chat API",
    description="A supportive AI friend that listens and cares",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


# ============================================================
# API Models
# ============================================================
class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.7

class ChatResponse(BaseModel):
    response: str
    emotion: Optional[str] = None


# ============================================================
# API Endpoints
# ============================================================
@app.get("/")
async def root():
    """Serve the chat interface."""
    return FileResponse("static/index.html")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Generate an empathetic response."""
    global model, tokenizer
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Build conversation with system prompt
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})
    
    # Apply chat template
    try:
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except:
        # Fallback for models without chat template
        text = f"System: {SYSTEM_PROMPT}\n\n"
        for msg in request.messages:
            role = "User" if msg.role == "user" else "Assistant"
            text += f"{role}: {msg.content}\n\n"
        text += "Assistant:"
    
    # Tokenize
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    ).strip()
    
    # Remove thinking tokens if present (Qwen3)
    if "<think>" in response:
        response = response.split("</think>")[-1].strip()
    
    return ChatResponse(response=response)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "gpu_available": torch.cuda.is_available(),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

