"""
🌿 Empathetic Friend - Hugging Face Spaces
==========================================
A supportive AI companion that listens with care.

Deploy to: https://huggingface.co/spaces
"""

import gradio as gr
import torch
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

# ============================================================
# Configuration - UPDATE THESE!
# ============================================================
MODEL_ID = "Someet24/empathetic-qwen3-8b-11-01"  # <-- Replace with your HF model ID

SYSTEM_PROMPT = """You are a warm, supportive, and empathetic friend. You listen carefully to what people share and respond with genuine care and understanding.

When someone shares their feelings:
1. Acknowledge and validate their emotions first
2. Show that you understand their situation  
3. Ask thoughtful follow-up questions
4. Offer support without being preachy or giving unsolicited advice
5. Never minimize their feelings with phrases like "just" or "at least"

You're not a therapist - you're a caring friend who's always there to listen."""

# ============================================================
# Load Model
# ============================================================
print(f"🔄 Loading model: {MODEL_ID}")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("✅ Model loaded!")

# ============================================================
# Chat Function
# ============================================================
def respond(message, history, temperature, max_tokens):
    """Generate empathetic response with streaming."""
    
    # Build conversation
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for user_msg, assistant_msg in history:
        messages.append({"role": "user", "content": user_msg})
        if assistant_msg:
            messages.append({"role": "assistant", "content": assistant_msg})
    
    messages.append({"role": "user", "content": message})
    
    # Apply chat template
    try:
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except:
        text = f"System: {SYSTEM_PROMPT}\n\n"
        for user_msg, assistant_msg in history:
            text += f"User: {user_msg}\n\nAssistant: {assistant_msg}\n\n"
        text += f"User: {message}\n\nAssistant:"
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # Setup streamer for real-time output
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = {
        **inputs,
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }
    
    # Generate in separate thread for streaming
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream response
    response = ""
    for new_text in streamer:
        # Handle Qwen3 thinking tokens
        if "<think>" in new_text:
            continue
        if "</think>" in new_text:
            new_text = new_text.split("</think>")[-1]
        
        response += new_text
        yield response
    
    thread.join()


# ============================================================
# Gradio Interface
# ============================================================
DESCRIPTION = """
# 🌿 Empathetic Friend

**A safe space to share what's on your mind.**

I'm here to listen without judgment. Share your thoughts, feelings, or whatever is weighing on you, and I'll do my best to understand and support you.

---

*This is an AI companion, not a replacement for professional mental health support. If you're in crisis, please reach out to a [helpline](https://findahelpline.com/).*
"""

CSS = """
.gradio-container {
    font-family: 'Segoe UI', system-ui, sans-serif !important;
}

#chatbot {
    height: 500px !important;
}

.message.user {
    background: #f0e6d3 !important;
}

.message.bot {
    background: #ffffff !important;
    border: 1px solid #e8e2d9 !important;
}

footer {
    display: none !important;
}
"""

EXAMPLES = [
    ["I'm feeling really overwhelmed with work lately and don't know how to cope."],
    ["My friend hasn't talked to me in weeks and I don't know why."],
    ["I just got some really exciting news about a job promotion!"],
    ["I've been feeling lonely since moving to a new city."],
    ["I had an argument with my family and I feel terrible about it."],
]

with gr.Blocks(css=CSS, title="Empathetic Friend", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)
    
    chatbot = gr.Chatbot(
        elem_id="chatbot",
        bubble_full_width=False,
        avatar_images=(None, "https://em-content.zobj.net/source/twitter/376/herb_1f33f.png"),
        height=500,
    )
    
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Share what's on your mind...",
            container=False,
            scale=7,
            autofocus=True,
        )
        submit = gr.Button("Send", variant="primary", scale=1)
    
    with gr.Accordion("⚙️ Settings", open=False):
        with gr.Row():
            temperature = gr.Slider(
                minimum=0.1, 
                maximum=1.0, 
                value=0.7, 
                step=0.1, 
                label="Temperature",
                info="Higher = more creative, Lower = more focused"
            )
            max_tokens = gr.Slider(
                minimum=50, 
                maximum=512, 
                value=256, 
                step=32, 
                label="Max Length",
                info="Maximum response length"
            )
    
    gr.Examples(
        examples=EXAMPLES,
        inputs=msg,
        label="💬 Try these conversation starters:",
    )
    
    # Event handlers
    def user_message(message, history):
        return "", history + [[message, None]]
    
    def bot_response(history, temperature, max_tokens):
        message = history[-1][0]
        history_pairs = history[:-1]
        
        response = ""
        for chunk in respond(message, history_pairs, temperature, max_tokens):
            response = chunk
            history[-1][1] = response
            yield history
    
    msg.submit(
        user_message, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        bot_response, [chatbot, temperature, max_tokens], chatbot
    )
    
    submit.click(
        user_message, [msg, chatbot], [msg, chatbot], queue=False
    ).then(
        bot_response, [chatbot, temperature, max_tokens], chatbot
    )
    
    gr.Markdown("""
    ---
    <center>
    Made with 💚 | <a href="https://github.com/YOUR_USERNAME">GitHub</a> | 
    Model: <a href="https://huggingface.co/YOUR_USERNAME/empathetic-qwen3-8b">empathetic-qwen3-8b</a>
    </center>
    """)

# Launch
if __name__ == "__main__":
    demo.queue().launch()

