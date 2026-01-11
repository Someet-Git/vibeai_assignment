"""
Empathetic LLM Fine-Tuning Package
==================================
Multi-task supervised fine-tuning for empathetic chatbot development.

Modules:
    data: Dataset loading and preprocessing
    model: Multi-head architecture with emotion and strategy heads
    train: Training loop with combined multi-task loss
    eval: EQ-Bench evaluation and ablation studies
"""

from .data import (
    MultiTaskDataset,
    create_dataloaders,
    compute_sampling_weights,
    EMOTION_LABELS,
    STRATEGY_LABELS,
)

from .model import (
    EmpatheticLLM,
    EmotionClassificationHead,
    StrategyClassificationHead,
    get_trainable_parameters,
)

from .train import (
    MultiTaskLoss,
    Trainer,
    train_with_ablation,
)

from .eval import (
    EQBenchEvaluator,
    AblationEvaluator,
    QualitativeEvaluator,
    evaluate_emotion_head,
    evaluate_strategy_head,
    save_evaluation_results,
)

__version__ = "1.0.0"
__author__ = "Vibe AI Assignment"

