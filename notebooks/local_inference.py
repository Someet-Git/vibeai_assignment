"""
Local Inference Script for Empathetic LLM
==========================================

Usage:
    python local_inference.py --model_path ./empathetic_model_output/lora_adapter

Requirements:
    pip install unsloth torch transformers peft bitsandbytes accelerate
"""

import argparse
import torch
from unsloth import FastLanguageModel

# Base model that was fine-tuned
BASE_MODEL = "unsloth/Qwen3-8B-bnb-4bit"

def load_model(adapter_path: str):
    """Load the fine-tuned model with LoRA adapter."""
    print(f"🚀 Loading base model: {BASE_MODEL}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=512,
        dtype=None,
        load_in_4bit=True,
    )
    
    print(f"📦 Loading LoRA adapter from: {adapter_path}")
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("✅ Model loaded successfully!")
    return model, tokenizer


def generate_response(model, tokenizer, user_input: str, max_new_tokens: int = 256):
    """Generate an empathetic response."""
    
    system_prompt = """You are a supportive, empathetic friend who listens carefully and responds with genuine care and understanding. 
When someone shares their feelings or problems:
1. Acknowledge and validate their emotions
2. Show that you understand their situation
3. Offer support without being preachy or dismissive
4. Never say "just" or minimize their feelings"""
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input}
    ]
    
    # Switch to inference mode
    FastLanguageModel.for_inference(model)
    
    # Prepare inputs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except:
        text = f"System: {system_prompt}\n\nUser: {user_input}\n\nAssistant:"
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def interactive_chat(model, tokenizer):
    """Run interactive chat session."""
    print("\n" + "="*60)
    print("🤗 Empathetic Chatbot - Interactive Mode")
    print("="*60)
    print("Type your message and press Enter. Type 'quit' to exit.\n")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye! Take care of yourself. 💙")
            break
        
        if not user_input:
            continue
        
        print("\nThinking...")
        response = generate_response(model, tokenizer, user_input)
        print(f"\nAssistant: {response}\n")
        print("-" * 40)


def run_evaluation(model, tokenizer):
    """Run EQ-Bench style evaluation."""
    scenarios = [
        {"context": "My grandmother passed away last week. We were very close.", "category": "grief"},
        {"context": "I have a big job interview tomorrow and I can't stop worrying.", "category": "anxiety"},
        {"context": "My coworker took credit for my project in front of our boss!", "category": "anger"},
        {"context": "I just got accepted to my dream school!", "category": "joy"},
        {"context": "I feel so alone since moving to this new city.", "category": "loneliness"},
    ]
    
    print("\n" + "="*60)
    print("📊 EQ-Bench Evaluation")
    print("="*60)
    
    for scenario in scenarios:
        print(f"\n📍 Category: {scenario['category'].upper()}")
        print(f"User: {scenario['context']}")
        response = generate_response(model, tokenizer, scenario['context'])
        print(f"\nAssistant: {response}")
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Run empathetic chatbot locally")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to LoRA adapter (e.g., ./empathetic_model_output/lora_adapter)")
    parser.add_argument("--mode", type=str, choices=["chat", "eval"], default="chat",
                        help="Mode: 'chat' for interactive, 'eval' for evaluation")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path)
    
    if args.mode == "chat":
        interactive_chat(model, tokenizer)
    else:
        run_evaluation(model, tokenizer)


if __name__ == "__main__":
    main()

