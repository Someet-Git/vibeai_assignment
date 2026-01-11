# 🌿 Empathetic Friend Chat Webapp

A beautiful, warm chat interface for the empathetic AI chatbot.

![Chat Interface](screenshot.png)

## Quick Start

### 1. Install Dependencies

```bash
cd webapp
pip install -r requirements.txt
```

### 2. Extract Your Model

Make sure your model is extracted in the right location:

```bash
# If you have empathetic_model.zip
cd ../trained_model
unzip empathetic_model.zip
```

Your folder structure should look like:
```
trained_model/
├── lora_adapter/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── tokenizer files...
```

### 3. Run the Server

```bash
cd webapp
python app.py
```

Or with uvicorn (recommended for development):
```bash
uvicorn app:app --reload --port 8000
```

### 4. Open in Browser

Visit: **http://localhost:8000**

---

## Configuration

### Using a Different Model Path

Set the `MODEL_PATH` environment variable:

```bash
# Windows PowerShell
$env:MODEL_PATH = "C:\path\to\your\model"
python app.py

# Linux/Mac
MODEL_PATH=/path/to/your/model python app.py
```

### Using Merged Model (No Base Model Needed)

If you have a merged model instead of LoRA adapter:

1. Point `MODEL_PATH` to the merged model folder
2. In `app.py`, the code automatically detects if it's a LoRA adapter or merged model

### Disabling Unsloth

If you don't have Unsloth installed, edit `app.py`:

```python
USE_UNSLOTH = False  # Change to False
```

This will use standard transformers instead (slightly slower).

---

## API Endpoints

### POST /api/chat

Send a message and get a response.

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am feeling anxious today"}
    ],
    "temperature": 0.7,
    "max_tokens": 256
  }'
```

Response:
```json
{
  "response": "I hear you, and I want you to know that anxiety is a really common experience..."
}
```

### GET /api/health

Check if the server and model are ready.

```bash
curl http://localhost:8000/api/health
```

---

## Deployment Options

### Local (Development)

```bash
uvicorn app:app --reload --port 8000
```

### Production (Gunicorn)

```bash
pip install gunicorn
gunicorn app:app -w 1 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

# Copy your model
COPY trained_model /app/trained_model

ENV MODEL_PATH=/app/trained_model/lora_adapter

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Cloud Deployment

**Hugging Face Spaces (Free GPU)**:
1. Create a new Space with "Docker" SDK
2. Upload your code and model
3. Set `MODEL_PATH` in Space settings

**RunPod / Vast.ai**:
1. Rent a GPU instance
2. Clone your repo
3. Run with uvicorn

---

## Hardware Requirements

| Setup | VRAM | Notes |
|-------|------|-------|
| 4-bit quantized | 6-8 GB | Recommended for most users |
| 8-bit quantized | 10-12 GB | Better quality |
| Full precision | 16+ GB | Best quality |

---

## Customization

### Change the System Prompt

Edit `SYSTEM_PROMPT` in `app.py`:

```python
SYSTEM_PROMPT = """Your custom prompt here..."""
```

### Modify the UI

Edit `static/index.html` to customize:
- Colors (CSS variables at the top)
- Avatar emoji
- Welcome message
- Starter prompts

---

## Troubleshooting

### "Model not loaded" error
- Check that `MODEL_PATH` points to the correct folder
- Ensure the model files exist

### Out of memory
- Use 4-bit quantization (default with Unsloth)
- Reduce `max_tokens` in requests
- Use a GPU with more VRAM

### Slow responses
- Enable Unsloth (`USE_UNSLOTH = True`)
- Use a faster GPU
- Reduce `max_tokens`

---

## License

Apache 2.0

