"""
Training Module
===============
Multi-task supervised fine-tuning with combined loss function.
Implements emotion classification, strategy classification, and safety KL losses.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import os
import json
import yaml
from datetime import datetime
from typing import Dict, Optional, Tuple, List
import matplotlib.pyplot as plt


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss function.
    
    L_SFT = λ_LM * L_NLL + λ_emo * L_emo + λ_strat * L_strat + λ_safe * L_safe
    
    Where:
    - L_NLL: Negative log-likelihood (language modeling)
    - L_emo: Emotion classification cross-entropy
    - L_strat: Strategy classification cross-entropy
    - L_safe: Safety KL divergence (simplified)
    """
    
    def __init__(
        self,
        lambda_lm: float = 1.0,
        lambda_emo: float = 0.2,
        lambda_strat: float = 0.2,
        lambda_safe: float = 0.1,
        num_emotion_classes: int = 27,
        num_strategy_classes: int = 8
    ):
        super().__init__()
        
        self.lambda_lm = lambda_lm
        self.lambda_emo = lambda_emo
        self.lambda_strat = lambda_strat
        self.lambda_safe = lambda_safe
        
        self.emotion_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.strategy_criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
    def forward(
        self,
        lm_loss: torch.Tensor,
        emotion_logits: torch.Tensor,
        strategy_logits: torch.Tensor,
        emotion_labels: torch.Tensor,
        strategy_labels: torch.Tensor,
        has_emotion: torch.Tensor,
        has_strategy: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        student_logits: Optional[torch.Tensor] = None,
        safety_temperature: float = 2.0
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined multi-task loss.
        
        Args:
            lm_loss: Language modeling loss from base model
            emotion_logits: [batch, num_emotion_classes]
            strategy_logits: [batch, num_strategy_classes]
            emotion_labels: [batch] - class indices, -1 for no label
            strategy_labels: [batch] - class indices, -1 for no label
            has_emotion: [batch] - boolean mask
            has_strategy: [batch] - boolean mask
            teacher_logits: Optional safety teacher logits
            student_logits: Optional student logits for safety KL
            safety_temperature: Temperature for KL softmax
            
        Returns:
            Dict with 'total_loss' and individual loss components
        """
        losses = {}
        
        # 1. Language Modeling Loss (primary)
        losses["lm_loss"] = lm_loss
        
        # 2. Emotion Classification Loss
        if has_emotion.any():
            # Only compute for samples with emotion labels
            emotion_mask = has_emotion.bool()
            if emotion_mask.sum() > 0:
                emo_loss = self.emotion_criterion(
                    emotion_logits[emotion_mask],
                    emotion_labels[emotion_mask]
                )
                losses["emotion_loss"] = emo_loss
            else:
                losses["emotion_loss"] = torch.tensor(0.0, device=lm_loss.device)
        else:
            losses["emotion_loss"] = torch.tensor(0.0, device=lm_loss.device)
        
        # 3. Strategy Classification Loss
        if has_strategy.any():
            strategy_mask = has_strategy.bool()
            if strategy_mask.sum() > 0:
                strat_loss = self.strategy_criterion(
                    strategy_logits[strategy_mask],
                    strategy_labels[strategy_mask]
                )
                losses["strategy_loss"] = strat_loss
            else:
                losses["strategy_loss"] = torch.tensor(0.0, device=lm_loss.device)
        else:
            losses["strategy_loss"] = torch.tensor(0.0, device=lm_loss.device)
        
        # 4. Safety KL Divergence (if teacher logits provided)
        if teacher_logits is not None and student_logits is not None:
            # Softmax with temperature
            teacher_probs = F.softmax(teacher_logits / safety_temperature, dim=-1)
            student_log_probs = F.log_softmax(student_logits / safety_temperature, dim=-1)
            
            # KL divergence
            safety_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction="batchmean"
            ) * (safety_temperature ** 2)
            
            losses["safety_loss"] = safety_loss
        else:
            losses["safety_loss"] = torch.tensor(0.0, device=lm_loss.device)
        
        # Combine losses
        total_loss = (
            self.lambda_lm * losses["lm_loss"] +
            self.lambda_emo * losses["emotion_loss"] +
            self.lambda_strat * losses["strategy_loss"] +
            self.lambda_safe * losses["safety_loss"]
        )
        
        losses["total_loss"] = total_loss
        
        return losses


class Trainer:
    """
    Multi-task trainer for EmpatheticLLM.
    """
    
    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        config: Dict,
        output_dir: str = "./checkpoints",
        device: str = "cuda"
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.output_dir = output_dir
        self.device = device
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Training config
        train_config = config.get("training", {})
        loss_config = config.get("loss_weights", {})
        
        # Initialize loss function
        self.loss_fn = MultiTaskLoss(
            lambda_lm=loss_config.get("lambda_lm", 1.0),
            lambda_emo=loss_config.get("lambda_emo", 0.2),
            lambda_strat=loss_config.get("lambda_strat", 0.2),
            lambda_safe=loss_config.get("lambda_safe", 0.1)
        )
        
        # Optimizer
        self.optimizer = AdamW(
            self._get_trainable_params(),
            lr=train_config.get("learning_rate", 2e-4),
            weight_decay=train_config.get("weight_decay", 0.01)
        )
        
        # Scheduler
        num_training_steps = len(train_dataloader) * train_config.get("num_epochs", 2)
        num_warmup_steps = int(num_training_steps * train_config.get("warmup_ratio", 0.1))
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "lm_loss": [],
            "emotion_loss": [],
            "strategy_loss": [],
            "learning_rate": []
        }
        
        # Config values
        self.num_epochs = train_config.get("num_epochs", 2)
        self.gradient_accumulation_steps = train_config.get("gradient_accumulation_steps", 8)
        self.max_grad_norm = train_config.get("max_grad_norm", 1.0)
        self.save_steps = train_config.get("save_steps", 500)
        self.eval_steps = train_config.get("eval_steps", 250)
        self.logging_steps = train_config.get("logging_steps", 50)
        
    def _get_trainable_params(self):
        """Get all trainable parameters."""
        params = []
        
        # LoRA parameters
        for param in self.model.base_model.parameters():
            if param.requires_grad:
                params.append(param)
        
        # Emotion head
        params.extend(self.model.emotion_head.parameters())
        
        # Strategy head
        params.extend(self.model.strategy_head.parameters())
        
        return params
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print("Starting Multi-Task Empathetic LLM Training")
        print(f"{'='*60}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Training samples: {len(self.train_dataloader.dataset)}")
        print(f"Batch size: {self.train_dataloader.batch_size}")
        print(f"Gradient accumulation: {self.gradient_accumulation_steps}")
        print(f"Effective batch size: {self.train_dataloader.batch_size * self.gradient_accumulation_steps}")
        print(f"{'='*60}\n")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")
            print("-" * 40)
            
            epoch_loss = self._train_epoch(epoch)
            
            print(f"\nEpoch {epoch + 1} completed. Average loss: {epoch_loss:.4f}")
            
            # Validation
            val_loss = self._validate()
            print(f"Validation loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint("best_model")
                print("Saved best model!")
        
        # Final save
        self._save_checkpoint("final_model")
        self._save_training_history()
        self._plot_training_curves()
        
        print(f"\n{'='*60}")
        print("Training completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}")
        
    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.base_model.train()
        self.model.emotion_head.train()
        self.model.strategy_head.train()
        
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training",
            leave=True
        )
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    return_hidden_states=True
                )
                
                # Compute multi-task loss
                losses = self.loss_fn(
                    lm_loss=outputs["lm_loss"],
                    emotion_logits=outputs["emotion_logits"],
                    strategy_logits=outputs["strategy_logits"],
                    emotion_labels=batch["emotion_label"],
                    strategy_labels=batch["strategy_label"],
                    has_emotion=batch["has_emotion"],
                    has_strategy=batch["has_strategy"]
                )
                
                loss = losses["total_loss"] / self.gradient_accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._get_trainable_params(),
                    self.max_grad_norm
                )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    self.history["train_loss"].append(losses["total_loss"].item())
                    self.history["lm_loss"].append(losses["lm_loss"].item())
                    self.history["emotion_loss"].append(losses["emotion_loss"].item())
                    self.history["strategy_loss"].append(losses["strategy_loss"].item())
                    self.history["learning_rate"].append(self.scheduler.get_last_lr()[0])
                
                # Evaluation
                if self.global_step % self.eval_steps == 0:
                    val_loss = self._validate()
                    self.history["val_loss"].append(val_loss)
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self._save_checkpoint("best_model")
                
                # Checkpointing
                if self.global_step % self.save_steps == 0:
                    self._save_checkpoint(f"checkpoint-{self.global_step}")
            
            total_loss += losses["total_loss"].item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": f"{losses['total_loss'].item():.4f}",
                "lm": f"{losses['lm_loss'].item():.4f}",
                "emo": f"{losses['emotion_loss'].item():.4f}",
                "strat": f"{losses['strategy_loss'].item():.4f}",
                "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation."""
        self.model.base_model.eval()
        self.model.emotion_head.eval()
        self.model.strategy_head.eval()
        
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_dataloader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"]
            )
            
            losses = self.loss_fn(
                lm_loss=outputs["lm_loss"],
                emotion_logits=outputs["emotion_logits"],
                strategy_logits=outputs["strategy_logits"],
                emotion_labels=batch["emotion_label"],
                strategy_labels=batch["strategy_label"],
                has_emotion=batch["has_emotion"],
                has_strategy=batch["has_strategy"]
            )
            
            total_loss += losses["total_loss"].item()
            num_batches += 1
        
        self.model.base_model.train()
        self.model.emotion_head.train()
        self.model.strategy_head.train()
        
        return total_loss / max(num_batches, 1)
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join(self.output_dir, name)
        self.model.save_pretrained(checkpoint_dir)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        torch.save(state, os.path.join(checkpoint_dir, "training_state.pt"))
        
    def _save_training_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.output_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"Training history saved to {history_path}")
        
    def _plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Total loss
        ax = axes[0, 0]
        if self.history["train_loss"]:
            ax.plot(self.history["train_loss"], label="Train Loss", alpha=0.7)
        if self.history["val_loss"]:
            val_steps = [i * (self.eval_steps // self.logging_steps) 
                        for i in range(len(self.history["val_loss"]))]
            ax.plot(val_steps, self.history["val_loss"], label="Val Loss", marker="o")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Total Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Component losses
        ax = axes[0, 1]
        if self.history["lm_loss"]:
            ax.plot(self.history["lm_loss"], label="LM Loss", alpha=0.7)
        if self.history["emotion_loss"]:
            ax.plot(self.history["emotion_loss"], label="Emotion Loss", alpha=0.7)
        if self.history["strategy_loss"]:
            ax.plot(self.history["strategy_loss"], label="Strategy Loss", alpha=0.7)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Component Losses")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[1, 0]
        if self.history["learning_rate"]:
            ax.plot(self.history["learning_rate"])
        ax.set_xlabel("Steps")
        ax.set_ylabel("Learning Rate")
        ax.set_title("Learning Rate Schedule")
        ax.grid(True, alpha=0.3)
        
        # Summary stats
        ax = axes[1, 1]
        ax.axis("off")
        summary_text = f"""
Training Summary
================
Total Steps: {self.global_step}
Best Val Loss: {self.best_val_loss:.4f}

Loss Weights:
  λ_LM: {self.loss_fn.lambda_lm}
  λ_emo: {self.loss_fn.lambda_emo}
  λ_strat: {self.loss_fn.lambda_strat}
  λ_safe: {self.loss_fn.lambda_safe}

Final Metrics:
  Train Loss: {self.history['train_loss'][-1] if self.history['train_loss'] else 'N/A':.4f}
  Val Loss: {self.history['val_loss'][-1] if self.history['val_loss'] else 'N/A':.4f}
"""
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, 
                fontfamily="monospace", fontsize=10, verticalalignment="top")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "training_curves.png"), dpi=150)
        plt.close()
        print(f"Training curves saved to {self.output_dir}/training_curves.png")


def train_with_ablation(
    model_class,
    train_dataloader,
    val_dataloader,
    config: Dict,
    ablation_name: str = "full",
    output_dir: str = "./checkpoints"
) -> Dict:
    """
    Train model with specific ablation configuration.
    
    Args:
        model_class: EmpatheticLLM class
        train_dataloader: Training data
        val_dataloader: Validation data
        config: Training configuration
        ablation_name: One of ["full", "no_emotion", "no_strategy", "no_safety"]
        output_dir: Output directory
        
    Returns:
        Dict with training results
    """
    # Modify config based on ablation
    ablation_config = config.copy()
    
    if ablation_name == "no_emotion":
        ablation_config["loss_weights"]["lambda_emo"] = 0.0
    elif ablation_name == "no_strategy":
        ablation_config["loss_weights"]["lambda_strat"] = 0.0
    elif ablation_name == "no_safety":
        ablation_config["loss_weights"]["lambda_safe"] = 0.0
    
    print(f"\n{'='*60}")
    print(f"Training with ablation: {ablation_name}")
    print(f"Loss weights: {ablation_config['loss_weights']}")
    print(f"{'='*60}")
    
    # Initialize model
    model = model_class(
        model_name=config["model"]["name"],
        config_path=None  # Use config dict directly
    )
    model.config = ablation_config
    
    # Train
    ablation_output = os.path.join(output_dir, f"ablation_{ablation_name}")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=ablation_config,
        output_dir=ablation_output
    )
    
    trainer.train()
    
    return {
        "ablation": ablation_name,
        "best_val_loss": trainer.best_val_loss,
        "history": trainer.history,
        "output_dir": ablation_output
    }


if __name__ == "__main__":
    print("Training module loaded successfully.")
    print("\nUsage:")
    print("  from train import Trainer, MultiTaskLoss")
    print("  trainer = Trainer(model, train_loader, val_loader, config)")
    print("  trainer.train()")

