"""
Data Loading and Preprocessing Module
=====================================
Handles loading EmpatheticDialogues, ESConv, and GoEmotions datasets
with temperature-based sampling for multi-task learning.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
import numpy as np
from typing import Dict, List, Optional, Tuple
import random


# Strategy label mapping for ESConv
STRATEGY_LABELS = {
    "Question": 0,
    "Restatement or Paraphrasing": 1,
    "Reflection of Feelings": 2,
    "Self-disclosure": 3,
    "Affirmation and Reassurance": 4,
    "Providing Suggestions": 5,
    "Information": 6,
    "Others": 7,
}

# Emotion labels for GoEmotions (simplified - 27 classes)
EMOTION_LABELS = {
    "admiration": 0, "amusement": 1, "anger": 2, "annoyance": 3,
    "approval": 4, "caring": 5, "confusion": 6, "curiosity": 7,
    "desire": 8, "disappointment": 9, "disapproval": 10, "disgust": 11,
    "embarrassment": 12, "excitement": 13, "fear": 14, "gratitude": 15,
    "grief": 16, "joy": 17, "love": 18, "nervousness": 19,
    "optimism": 20, "pride": 21, "realization": 22, "relief": 23,
    "remorse": 24, "sadness": 25, "surprise": 26,
}


def compute_sampling_weights(dataset_sizes: List[int], alpha: float = 0.5) -> List[float]:
    """
    Compute temperature-based sampling weights.
    
    Formula: p_i = n_i^α / Σn_j^α
    
    Args:
        dataset_sizes: List of dataset sizes [n_1, n_2, ...]
        alpha: Temperature parameter (0, 1]. Lower = more uniform
        
    Returns:
        List of sampling probabilities for each dataset
    """
    # Apply temperature
    weighted = [n ** alpha for n in dataset_sizes]
    total = sum(weighted)
    
    # Normalize to probabilities
    probs = [w / total for w in weighted]
    return probs


class EmpatheticDialoguesProcessor:
    """Process EmpatheticDialogues dataset."""
    
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_and_process(self, split: str = "train") -> List[Dict]:
        """Load and process EmpatheticDialogues."""
        print(f"Loading EmpatheticDialogues ({split})...")
        dataset = load_dataset("empathetic_dialogues", split=split, trust_remote_code=True)
        
        processed = []
        current_conv = []
        current_conv_id = None
        
        for item in dataset:
            conv_id = item["conv_id"]
            
            # New conversation
            if conv_id != current_conv_id:
                if current_conv and len(current_conv) >= 2:
                    processed.extend(self._create_examples(current_conv, item.get("context", "")))
                current_conv = []
                current_conv_id = conv_id
            
            current_conv.append({
                "speaker": "user" if item["speaker_idx"] == 0 else "assistant",
                "text": item["utterance"],
                "emotion": item.get("context", "neutral"),  # Context is the emotion
            })
        
        # Don't forget last conversation
        if current_conv and len(current_conv) >= 2:
            processed.extend(self._create_examples(current_conv, ""))
            
        print(f"  Processed {len(processed)} examples from EmpatheticDialogues")
        return processed
    
    def _create_examples(self, conversation: List[Dict], emotion_context: str) -> List[Dict]:
        """Create training examples from a conversation."""
        examples = []
        
        for i in range(1, len(conversation)):
            if conversation[i]["speaker"] == "assistant":
                # Build context from previous turns
                context_turns = conversation[max(0, i-5):i]
                context = self._format_context(context_turns)
                response = conversation[i]["text"]
                
                # Get emotion from user's last turn or context
                user_emotion = None
                for turn in reversed(context_turns):
                    if turn["speaker"] == "user" and turn.get("emotion"):
                        # Try to map emotion to our label set
                        emotion_str = turn["emotion"].lower()
                        if emotion_str in EMOTION_LABELS:
                            user_emotion = EMOTION_LABELS[emotion_str]
                        break
                
                examples.append({
                    "input": context,
                    "output": response,
                    "emotion_label": user_emotion,
                    "strategy_label": None,
                    "has_emotion": user_emotion is not None,
                    "has_strategy": False,
                    "source": "empathetic_dialogues"
                })
        
        return examples
    
    def _format_context(self, turns: List[Dict]) -> str:
        """Format conversation turns into a string."""
        formatted = []
        for turn in turns:
            role = "User" if turn["speaker"] == "user" else "Assistant"
            formatted.append(f"{role}: {turn['text']}")
        return "\n".join(formatted)


class ESConvProcessor:
    """Process ESConv dataset with strategy labels."""
    
    def __init__(self, tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_and_process(self, split: str = "train") -> List[Dict]:
        """Load and process ESConv dataset."""
        print(f"Loading ESConv ({split})...")
        
        try:
            dataset = load_dataset("thu-coai/esconv", split=split, trust_remote_code=True)
        except Exception as e:
            print(f"  Warning: Could not load ESConv directly, trying alternative...")
            # Fallback: load from specific config
            try:
                dataset = load_dataset("thu-coai/esconv", "default", split=split, trust_remote_code=True)
            except:
                print(f"  Error loading ESConv: {e}")
                return []
        
        processed = []
        
        for item in dataset:
            # ESConv has dialog as list of utterances with strategies
            dialog = item.get("dialog", [])
            
            if not dialog:
                continue
                
            # Process each assistant turn
            for i, turn in enumerate(dialog):
                if turn.get("speaker", "").lower() in ["sys", "system", "supporter"]:
                    # Build context from previous turns
                    context_turns = dialog[max(0, i-5):i]
                    context = self._format_context(context_turns)
                    response = turn.get("content", turn.get("text", ""))
                    
                    # Get strategy label
                    strategy_str = turn.get("strategy", "Others")
                    strategy_label = STRATEGY_LABELS.get(strategy_str, 7)  # Default to "Others"
                    
                    # Try to get emotion from user's context
                    emotion_label = None
                    if item.get("emotion_type"):
                        emotion_str = item["emotion_type"].lower()
                        if emotion_str in EMOTION_LABELS:
                            emotion_label = EMOTION_LABELS[emotion_str]
                    
                    if response.strip():
                        processed.append({
                            "input": context,
                            "output": response,
                            "emotion_label": emotion_label,
                            "strategy_label": strategy_label,
                            "has_emotion": emotion_label is not None,
                            "has_strategy": True,
                            "source": "esconv"
                        })
        
        print(f"  Processed {len(processed)} examples from ESConv")
        return processed
    
    def _format_context(self, turns: List[Dict]) -> str:
        """Format conversation turns into a string."""
        formatted = []
        for turn in turns:
            speaker = turn.get("speaker", "unknown").lower()
            role = "User" if speaker in ["usr", "user", "seeker"] else "Assistant"
            text = turn.get("content", turn.get("text", ""))
            if text.strip():
                formatted.append(f"{role}: {text}")
        return "\n".join(formatted)


class GoEmotionsProcessor:
    """Process GoEmotions for emotion classification."""
    
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def load_and_process(self, split: str = "train") -> List[Dict]:
        """Load and process GoEmotions dataset."""
        print(f"Loading GoEmotions ({split})...")
        
        try:
            dataset = load_dataset(
                "google-research-datasets/go_emotions", 
                "simplified", 
                split=split,
                trust_remote_code=True
            )
        except Exception as e:
            print(f"  Error loading GoEmotions: {e}")
            return []
        
        processed = []
        
        for item in dataset:
            text = item["text"]
            labels = item["labels"]
            
            # Take first label if multiple (simplified)
            if labels:
                emotion_label = labels[0] if isinstance(labels, list) else labels
                
                # Format as a user message seeking response
                context = f"User: {text}"
                
                # Create a placeholder response (model will learn to respond empathetically)
                response = "[Generate empathetic response]"
                
                processed.append({
                    "input": context,
                    "output": response,
                    "emotion_label": emotion_label,
                    "strategy_label": None,
                    "has_emotion": True,
                    "has_strategy": False,
                    "source": "goemotions"
                })
        
        print(f"  Processed {len(processed)} examples from GoEmotions")
        return processed


class MultiTaskDataset(Dataset):
    """
    Combined dataset for multi-task empathetic fine-tuning.
    """
    
    def __init__(
        self,
        tokenizer,
        max_length: int = 1024,
        split: str = "train",
        alpha: float = 0.5,
        limit_per_dataset: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        
        # Load all datasets
        ed_processor = EmpatheticDialoguesProcessor(tokenizer, max_length)
        esconv_processor = ESConvProcessor(tokenizer, max_length)
        goemotions_processor = GoEmotionsProcessor(tokenizer, max_length)
        
        ed_data = ed_processor.load_and_process(split)
        esconv_data = esconv_processor.load_and_process(split)
        goemotions_data = goemotions_processor.load_and_process(split)
        
        # Apply limit if specified (for faster iteration)
        if limit_per_dataset:
            ed_data = ed_data[:limit_per_dataset]
            esconv_data = esconv_data[:limit_per_dataset]
            goemotions_data = goemotions_data[:limit_per_dataset]
        
        # Compute sampling weights
        sizes = [len(ed_data), len(esconv_data), len(goemotions_data)]
        weights = compute_sampling_weights(sizes, alpha)
        
        print(f"\nDataset sizes: ED={sizes[0]}, ESConv={sizes[1]}, GoEmotions={sizes[2]}")
        print(f"Sampling weights (α={alpha}): ED={weights[0]:.3f}, ESConv={weights[1]:.3f}, GoEmotions={weights[2]:.3f}")
        
        # Combine with temperature-based sampling
        self.data = self._sample_and_combine(
            [ed_data, esconv_data, goemotions_data],
            weights
        )
        
        print(f"Total combined dataset size: {len(self.data)}")
        
    def _sample_and_combine(
        self, 
        datasets: List[List[Dict]], 
        weights: List[float]
    ) -> List[Dict]:
        """Combine datasets with temperature-based sampling."""
        combined = []
        
        # Calculate target samples per dataset based on weights
        total_available = sum(len(d) for d in datasets)
        target_total = min(total_available, 50000)  # Cap for memory
        
        for dataset, weight in zip(datasets, weights):
            if not dataset:
                continue
            target_samples = int(target_total * weight)
            
            # Sample with replacement if needed
            if target_samples <= len(dataset):
                sampled = random.sample(dataset, target_samples)
            else:
                sampled = dataset.copy()
                # Oversample to reach target
                while len(sampled) < target_samples:
                    sampled.extend(random.sample(dataset, min(len(dataset), target_samples - len(sampled))))
            
            combined.extend(sampled)
        
        random.shuffle(combined)
        return combined
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        item = self.data[idx]
        return self._tokenize(item)
    
    def _tokenize(self, item: Dict) -> Dict:
        """Tokenize a single example for training."""
        # Format as chat template
        messages = [
            {"role": "system", "content": "You are a supportive, empathetic friend who listens carefully and responds with genuine care and understanding."},
            {"role": "user", "content": item["input"]},
        ]
        
        # Add assistant response if not placeholder
        if item["output"] != "[Generate empathetic response]":
            messages.append({"role": "assistant", "content": item["output"]})
        
        # Tokenize
        try:
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=item["output"] == "[Generate empathetic response]"
            )
        except:
            # Fallback for models without chat template
            text = f"System: You are a supportive, empathetic friend.\n\n{item['input']}\n\nAssistant: {item['output']}"
        
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Squeeze batch dimension
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        
        # Create labels (same as input_ids for causal LM)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "emotion_label": torch.tensor(item["emotion_label"]) if item["has_emotion"] else torch.tensor(-1),
            "strategy_label": torch.tensor(item["strategy_label"]) if item["has_strategy"] else torch.tensor(-1),
            "has_emotion": torch.tensor(item["has_emotion"]),
            "has_strategy": torch.tensor(item["has_strategy"]),
        }


def create_dataloaders(
    tokenizer,
    batch_size: int = 4,
    max_length: int = 1024,
    alpha: float = 0.5,
    limit: Optional[int] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.
    
    Args:
        tokenizer: HuggingFace tokenizer
        batch_size: Batch size for training
        max_length: Maximum sequence length
        alpha: Temperature for dataset mixing
        limit: Optional limit per dataset (for debugging)
        
    Returns:
        (train_dataloader, val_dataloader)
    """
    train_dataset = MultiTaskDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        split="train",
        alpha=alpha,
        limit_per_dataset=limit
    )
    
    val_dataset = MultiTaskDataset(
        tokenizer=tokenizer,
        max_length=max_length,
        split="validation",
        alpha=alpha,
        limit_per_dataset=limit // 10 if limit else None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loading
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    train_loader, val_loader = create_dataloaders(
        tokenizer,
        batch_size=2,
        max_length=512,
        alpha=0.5,
        limit=100  # Small limit for testing
    )
    
    # Check a batch
    batch = next(iter(train_loader))
    print("\nBatch keys:", batch.keys())
    print("Input shape:", batch["input_ids"].shape)
    print("Has emotion:", batch["has_emotion"])
    print("Has strategy:", batch["has_strategy"])

