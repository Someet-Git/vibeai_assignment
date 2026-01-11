# 🚀 Empathetic LLM Fine-Tuning with Unsloth

## Vibe AI Assignment: Multi-Task SFT with Auxiliary Heads

Fine-tune an open-weights LLM to behave as an empathetic, best-friend chatbot using multi-objective supervised fine-tuning with auxiliary classification heads for emotion recognition and support strategy prediction.

## ⚡ Why Unsloth?

| Feature | Standard Training | With Unsloth |
|---------|------------------|--------------|
| Training Speed | 1x | **2x faster** |
| Memory Usage | High | **60% less** |
| Model Size on T4 | 7B max | **8-14B** |
| Setup Complexity | Manual | **Automatic** |

## 🎯 Objective

Train an 8B parameter LLM (Qwen3-8B) to:
1. Generate empathetic, supportive responses
2. Recognize user emotions (27 classes from GoEmotions)
3. Predict appropriate support strategies (8 classes from ESConv)
4. Maintain safety boundaries

## 📊 Architecture

```
┌─────────────────────────────────────────────────────┐
│      Qwen3-8B via Unsloth (4-bit quantized)         │
│          LoRA Adapters (Unsloth optimized)          │
└─────────────────────────────────────────────────────┘
                         │
                  Hidden States
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌─────────┐     ┌───────────┐    ┌───────────┐
   │ LM Head │     │  Emotion  │    │ Strategy  │
   │(tokens) │     │   Head    │    │   Head    │
   └─────────┘     │(27 class) │    │ (8 class) │
                   └───────────┘    └───────────┘
```

## 📁 Project Structure

```
vibe_ai_assignment/
├── notebooks/
│   └── empathetic_sft.ipynb     # Main Colab notebook
├── src/
│   ├── __init__.py              # Package exports
│   ├── data.py                  # Dataset loading + mixing
│   ├── model.py                 # Multi-head architecture
│   ├── train.py                 # Training loop
│   └── eval.py                  # EQ-Bench evaluation
├── configs/
│   └── training_config.yaml     # All hyperparameters
├── results/
│   ├── training_curves.png      # Loss plots
│   ├── eq_bench_results.json    # Evaluation results
│   └── ablation_results.json    # Ablation comparison
├── safety/
│   └── red_team_sheet.md        # Safety evaluation
├── report.md                    # Final write-up
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Open in Google Colab

Upload `notebooks/empathetic_sft.ipynb` to Google Colab or click:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

### 2. Install Unsloth (2x faster training!)

```python
# Install Unsloth - handles all dependencies automatically
!pip install unsloth
!pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 3. Run Training

```python
from unsloth import FastLanguageModel

# Load Qwen3 with Unsloth (2x faster!)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

# Apply LoRA with Unsloth optimizations
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    use_gradient_checkpointing="unsloth",
)

# See notebook for complete training code with auxiliary heads
```

## 📚 Datasets

| Dataset | Purpose | Size | Source |
|---------|---------|------|--------|
| EmpatheticDialogues | Empathetic conversation | ~25K | Facebook Research |
| ESConv | Support strategies | ~1.3K | THU-COAI |
| GoEmotions | Emotion labels | ~58K | Google Research |

### Temperature-Based Sampling

Datasets are mixed using temperature sampling to prevent larger datasets from dominating:

$$p_i = \frac{n_i^{\alpha}}{\sum_j n_j^{\alpha}}, \quad \alpha \in (0, 1]$$

With α = 0.5:
- EmpatheticDialogues: 36%
- ESConv: 8%
- GoEmotions: 56%

## 🧮 Loss Function

Multi-objective combined loss:

$$\mathcal{L}_{SFT} = \lambda_{LM} \mathcal{L}_{NLL} + \lambda_{emo} \mathcal{L}_{emo} + \lambda_{strat} \mathcal{L}_{strat} + \lambda_{safe} \mathcal{L}_{safe}$$

| Component | Weight | Description |
|-----------|--------|-------------|
| L_NLL | 1.0 | Language modeling (next token) |
| L_emo | 0.2 | Emotion classification (CE) |
| L_strat | 0.2 | Strategy classification (CE) |
| L_safe | 0.1 | Safety KL divergence |

## 📈 Evaluation

### EQ-Bench 3
Primary metric for measuring emotional intelligence and empathy.

### Ablation Studies
1. **No Emotion Head**: λ_emo = 0
2. **No Strategy Head**: λ_strat = 0

### Metrics Collected
- EQ-Bench score (raw and Elo-normalized)
- Emotion classification accuracy (GoEmotions test)
- Strategy classification accuracy (ESConv test)
- Validation loss curves

## 🛡️ Safety

See `safety/red_team_sheet.md` for:
- 3 red-team prompts with expected behaviors
- Model responses and safety ratings
- Crisis resource integration

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Base Model | Qwen3-8B via Unsloth |
| Quantization | 4-bit (Unsloth optimized) |
| LoRA Rank | 16 |
| LoRA Alpha | 16 |
| Target Modules | 7 (all projections) |
| Learning Rate | 2e-4 |
| Batch Size | 2 (effective: 8) |
| Epochs | 2 |
| Max Seq Length | 2048 |
| Training Speed | **2x faster with Unsloth** |

## 📋 Requirements

- Python 3.10+
- CUDA-compatible GPU (16GB+ VRAM recommended)
- Google Colab (T4/A100) or equivalent

## 📄 Deliverables

- [x] Training config and hyperparameters
- [x] Data cards and preprocessing documentation
- [x] Training logs (loss curves)
- [x] EQ-Bench results (Base vs SFT)
- [x] 2 ablation studies
- [x] 3-5 qualitative examples
- [x] Safety sheet (3 red-team prompts)
- [x] Model weights (LoRA adapters)

## 📝 Report

See `report.md` for the full technical write-up including:
- Approach and architecture decisions
- Implementation details
- Results and analysis
- Error taxonomy
- Safety considerations

## 🔗 References

1. [Unsloth](https://github.com/unslothai/unsloth) - 2x faster LLM fine-tuning
2. [Qwen3](https://huggingface.co/Qwen) - Base model
3. [EQ-Bench](https://eqbench.com/) - Emotional Intelligence Benchmark
4. [EmpatheticDialogues](https://github.com/facebookresearch/EmpatheticDialogues)
5. [ESConv](https://github.com/thu-coai/Emotional-Support-Conversation)
6. [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions)
7. [QLoRA](https://arxiv.org/abs/2305.14314) - Efficient Fine-tuning

## 📜 License

This project is for educational and evaluation purposes only.

---

*Vibe AI Assignment - Empathetic LLM Fine-Tuning*

