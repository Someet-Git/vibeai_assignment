"""
Multi-Head Model Architecture
=============================
Implements the empathetic LLM with auxiliary classification heads
for emotion and strategy prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)
from typing import Dict, Optional, Tuple
import yaml


class EmotionClassificationHead(nn.Module):
    """
    Auxiliary head for emotion classification.
    
    Input: Hidden state at the end of user's last utterance
    Output: Logits over 27 emotion classes (GoEmotions)
    """
    
    def __init__(
        self,
        hidden_size: int = 3584,
        num_classes: int = 27,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, hidden_size] - hidden state at emotion position
            
        Returns:
            logits: [batch_size, num_classes]
        """
        return self.classifier(hidden_states)


class StrategyClassificationHead(nn.Module):
    """
    Auxiliary head for support strategy classification.
    
    Input: Hidden state at the start of assistant's response
    Output: Logits over 8 strategy classes (ESConv)
    """
    
    def __init__(
        self,
        hidden_size: int = 3584,
        num_classes: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, hidden_size] - hidden state at strategy position
            
        Returns:
            logits: [batch_size, num_classes]
        """
        return self.classifier(hidden_states)


class EmpatheticLLM(nn.Module):
    """
    Multi-task Empathetic Language Model.
    
    Architecture:
    - Base: Qwen2.5-7B-Instruct (4-bit quantized)
    - Adapters: LoRA on attention projections
    - Auxiliary Heads:
        - Emotion classification (27 classes)
        - Strategy classification (8 classes)
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        config_path: Optional[str] = None,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.device = device
        self.model_name = model_name
        
        # Load config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._default_config()
        
        # Initialize components
        self._setup_quantization()
        self._setup_base_model()
        self._setup_lora()
        self._setup_auxiliary_heads()
        
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            "model": {"hidden_size": 3584},
            "quantization": {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True
            },
            "lora": {
                "r": 16,
                "lora_alpha": 32,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "lora_dropout": 0.05,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            },
            "heads": {
                "emotion": {"num_classes": 27, "hidden_dim": 512, "dropout": 0.1},
                "strategy": {"num_classes": 8, "hidden_dim": 256, "dropout": 0.1}
            }
        }
    
    def _setup_quantization(self):
        """Setup 4-bit quantization config."""
        quant_config = self.config.get("quantization", {})
        
        compute_dtype = getattr(torch, quant_config.get("bnb_4bit_compute_dtype", "bfloat16"))
        
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True)
        )
        
    def _setup_base_model(self):
        """Load and prepare base model."""
        print(f"Loading base model: {self.model_name}")
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        
        # Prepare for k-bit training
        self.base_model = prepare_model_for_kbit_training(
            self.base_model,
            use_gradient_checkpointing=True
        )
        
        # Get tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Get hidden size from model config
        self.hidden_size = self.base_model.config.hidden_size
        print(f"Model hidden size: {self.hidden_size}")
        
    def _setup_lora(self):
        """Setup LoRA adapters."""
        lora_config = self.config.get("lora", {})
        
        self.lora_config = LoraConfig(
            r=lora_config.get("r", 16),
            lora_alpha=lora_config.get("lora_alpha", 32),
            target_modules=lora_config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            lora_dropout=lora_config.get("lora_dropout", 0.05),
            bias=lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM
        )
        
        self.base_model = get_peft_model(self.base_model, self.lora_config)
        self.base_model.print_trainable_parameters()
        
    def _setup_auxiliary_heads(self):
        """Initialize emotion and strategy classification heads."""
        heads_config = self.config.get("heads", {})
        emo_config = heads_config.get("emotion", {})
        strat_config = heads_config.get("strategy", {})
        
        self.emotion_head = EmotionClassificationHead(
            hidden_size=self.hidden_size,
            num_classes=emo_config.get("num_classes", 27),
            hidden_dim=emo_config.get("hidden_dim", 512),
            dropout=emo_config.get("dropout", 0.1)
        ).to(self.device)
        
        self.strategy_head = StrategyClassificationHead(
            hidden_size=self.hidden_size,
            num_classes=strat_config.get("num_classes", 8),
            hidden_dim=strat_config.get("hidden_dim", 256),
            dropout=strat_config.get("dropout", 0.1)
        ).to(self.device)
        
        print(f"Initialized emotion head: {emo_config.get('num_classes', 27)} classes")
        print(f"Initialized strategy head: {strat_config.get('num_classes', 8)} classes")
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        strategy_labels: Optional[torch.Tensor] = None,
        has_emotion: Optional[torch.Tensor] = None,
        has_strategy: Optional[torch.Tensor] = None,
        return_hidden_states: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-task outputs.
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            labels: [batch_size, seq_len] - for LM loss
            emotion_labels: [batch_size] - emotion class indices
            strategy_labels: [batch_size] - strategy class indices
            has_emotion: [batch_size] - mask for samples with emotion labels
            has_strategy: [batch_size] - mask for samples with strategy labels
            
        Returns:
            Dict with 'lm_loss', 'emotion_logits', 'strategy_logits', 'hidden_states'
        """
        # Forward through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get last hidden states
        hidden_states = outputs.hidden_states[-1]  # [batch, seq, hidden]
        
        # Get hidden state at last non-padding position for classification
        # This is a simplification - ideally we'd find the exact user/assistant positions
        seq_lengths = attention_mask.sum(dim=1) - 1  # Last non-padding index
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        
        # Use last position hidden state for both heads
        classification_hidden = hidden_states[batch_indices, seq_lengths]  # [batch, hidden]
        
        # Emotion classification
        emotion_logits = self.emotion_head(classification_hidden)
        
        # Strategy classification
        strategy_logits = self.strategy_head(classification_hidden)
        
        result = {
            "lm_loss": outputs.loss,
            "lm_logits": outputs.logits,
            "emotion_logits": emotion_logits,
            "strategy_logits": strategy_logits,
        }
        
        if return_hidden_states:
            result["hidden_states"] = hidden_states
            
        return result
    
    def generate(
        self,
        input_text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """
        Generate empathetic response.
        
        Args:
            input_text: User input or conversation context
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            Generated response text
        """
        # Format as chat
        messages = [
            {"role": "system", "content": "You are a supportive, empathetic friend who listens carefully and responds with genuine care and understanding."},
            {"role": "user", "content": input_text}
        ]
        
        # Tokenize
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        
        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def predict_emotion(self, input_text: str) -> Tuple[int, torch.Tensor]:
        """Predict emotion class for input text."""
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model(
                **inputs,
                output_hidden_states=True
            )
            hidden = outputs.hidden_states[-1][:, -1, :]
            logits = self.emotion_head(hidden)
            probs = F.softmax(logits, dim=-1)
            
        predicted_class = logits.argmax(dim=-1).item()
        return predicted_class, probs.squeeze()
    
    def predict_strategy(self, input_text: str) -> Tuple[int, torch.Tensor]:
        """Predict support strategy for input text."""
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.base_model(
                **inputs,
                output_hidden_states=True
            )
            hidden = outputs.hidden_states[-1][:, -1, :]
            logits = self.strategy_head(hidden)
            probs = F.softmax(logits, dim=-1)
            
        predicted_class = logits.argmax(dim=-1).item()
        return predicted_class, probs.squeeze()
    
    def save_pretrained(self, output_dir: str):
        """Save model adapters and heads."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save LoRA adapters
        self.base_model.save_pretrained(output_dir)
        
        # Save auxiliary heads
        torch.save(
            self.emotion_head.state_dict(),
            os.path.join(output_dir, "emotion_head.pt")
        )
        torch.save(
            self.strategy_head.state_dict(),
            os.path.join(output_dir, "strategy_head.pt")
        )
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Model saved to {output_dir}")
    
    @classmethod
    def from_pretrained(cls, model_path: str, base_model_name: str = None, device: str = "cuda"):
        """Load saved model."""
        import os
        from peft import PeftModel
        
        # Determine base model
        if base_model_name is None:
            base_model_name = "Qwen/Qwen2.5-7B-Instruct"
        
        # Create instance (will load base model)
        model = cls(model_name=base_model_name, device=device)
        
        # Load LoRA adapters
        model.base_model = PeftModel.from_pretrained(
            model.base_model,
            model_path
        )
        
        # Load auxiliary heads
        emotion_path = os.path.join(model_path, "emotion_head.pt")
        strategy_path = os.path.join(model_path, "strategy_head.pt")
        
        if os.path.exists(emotion_path):
            model.emotion_head.load_state_dict(torch.load(emotion_path))
        if os.path.exists(strategy_path):
            model.strategy_head.load_state_dict(torch.load(strategy_path))
            
        return model


def get_trainable_parameters(model: EmpatheticLLM) -> Dict[str, int]:
    """Get count of trainable parameters."""
    # LoRA parameters
    lora_params = sum(p.numel() for p in model.base_model.parameters() if p.requires_grad)
    
    # Emotion head parameters
    emotion_params = sum(p.numel() for p in model.emotion_head.parameters())
    
    # Strategy head parameters
    strategy_params = sum(p.numel() for p in model.strategy_head.parameters())
    
    return {
        "lora": lora_params,
        "emotion_head": emotion_params,
        "strategy_head": strategy_params,
        "total": lora_params + emotion_params + strategy_params
    }


if __name__ == "__main__":
    # Test model initialization
    print("Testing EmpatheticLLM initialization...")
    
    # This would require GPU - just show structure
    print("\nModel architecture:")
    print("- Base: Qwen2.5-7B-Instruct (4-bit quantized)")
    print("- LoRA: r=16, alpha=32")
    print("- Emotion Head: Linear(hidden, 512) -> ReLU -> Linear(512, 27)")
    print("- Strategy Head: Linear(hidden, 256) -> ReLU -> Linear(256, 8)")

