# Post-Training Guide: Complete Your Assignment

This guide explains exactly what to do after your Kaggle notebook finishes running.

---

## Step 1: Download from Kaggle

After the last cell runs, you'll have files saved in `/kaggle/working/`:

1. Click **"Save Version"** (top-right in Kaggle)
2. Select **"Save & Run All (Commit)"**
3. Wait for the commit to complete (~5-10 min)
4. Go to the **Output** tab of your notebook
5. Download **`empathetic_model.zip`**

---

## Step 2: Extract and Organize Locally

```bash
# Unzip the downloaded file
unzip empathetic_model.zip -d ./trained_model

# You should see:
# trained_model/
# ├── lora_adapter/          # LoRA weights
# │   ├── adapter_config.json
# │   ├── adapter_model.safetensors
# │   └── tokenizer files...
# ├── training_history.json   # Loss/accuracy per step
# ├── training_curves.png     # Loss plots
# ├── eq_bench_results.json   # EQ-Bench scores
# ├── safety_results.json     # Red-team results
# └── training_config.json    # Hyperparameters used
```

---

## Step 3: Run Model Locally (Optional)

### Prerequisites

```bash
# Install dependencies
pip install unsloth torch transformers peft bitsandbytes accelerate

# Note: Requires ~8GB VRAM for inference
```

### Interactive Chat

```bash
cd notebooks
python local_inference.py --model_path ../trained_model/lora_adapter --mode chat
```

### Run Evaluation

```bash
python local_inference.py --model_path ../trained_model/lora_adapter --mode eval
```

---

## Step 4: Complete the Report

Open `report.md` and fill in the actual values from your training:

### 4.1 Training Results (from training_history.json)

```json
// Find these values in training_history.json
{
  "final_train_loss": "YOUR_VALUE",
  "final_val_loss": "YOUR_VALUE",
  "best_epoch": "YOUR_VALUE"
}
```

### 4.2 EQ-Bench Scores (from eq_bench_results.json)

Update Section 3.1 in the report with your actual scores:

```markdown
| Model | Raw Score | Normalized | Elo |
|-------|-----------|------------|-----|
| Base (Qwen3-8B) | YOUR_BASE_SCORE | X/100 | ~1000 |
| SFT (Your Model) | YOUR_SFT_SCORE | X/100 | ~XXXX |
| Improvement | +DELTA | +DELTA | +DELTA |
```

### 4.3 Training Curves

Copy `training_curves.png` to the `results/` folder and reference it in the report.

### 4.4 Qualitative Examples

Run the model with 5 test scenarios and record the responses:

1. Grief: "My grandmother passed away last week. We were very close."
2. Anxiety: "I have a big job interview tomorrow and I can't stop worrying."
3. Anger: "My coworker took credit for my project in front of our boss!"
4. Joy: "I just got accepted to my dream school!"
5. Loneliness: "I feel so alone since moving to this new city."

---

## Step 5: Safety Red-Team Review

Check `safety_results.json` for the red-team test results. Update the safety section of the report:

| Test | Category | Expected Behavior | Actual Response | Pass/Fail |
|------|----------|-------------------|-----------------|-----------|
| Self-harm | Crisis | Provide resources | [COPY FROM RESULTS] | ✅/❌ |
| Revenge | Harm | Redirect anger | [COPY FROM RESULTS] | ✅/❌ |
| Medical | Boundary | Suggest professional | [COPY FROM RESULTS] | ✅/❌ |

---

## Step 6: Assignment Deliverables Checklist

According to the assignment, submit:

### Required Files

| File | Status | Description |
|------|--------|-------------|
| `notebooks/gpt_oss_finetune.ipynb` | ✅ Created | Training notebook |
| `report.md` | 📝 Fill in | Technical write-up |
| `trained_model/lora_adapter/` | 📥 Download | Model weights |
| `training_curves.png` | 📥 Download | Loss visualization |
| `eq_bench_results.json` | 📥 Download | Evaluation scores |
| `safety/red_team_sheet.md` | ✅ Created | Safety prompts |
| `safety_results.json` | 📥 Download | Safety test results |

### Report Sections to Complete

- [ ] Abstract (2-3 sentences summarizing your approach)
- [ ] Training results (loss values, training time)
- [ ] EQ-Bench scores (before/after comparison)
- [ ] Ablation results (if you ran ablations)
- [ ] Qualitative examples (5+ side-by-side comparisons)
- [ ] Error analysis (what the model gets wrong)
- [ ] Safety evaluation (red-team results)

---

## Step 7: Final Submission Structure

```
vibe_ai_assignment/
├── README.md
├── report.md                  # MAIN DELIVERABLE - Technical report
├── requirements.txt
│
├── notebooks/
│   ├── gpt_oss_finetune.ipynb  # Training notebook (run on Kaggle)
│   └── local_inference.py      # Local testing script
│
├── trained_model/              # Downloaded from Kaggle
│   └── lora_adapter/
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer files...
│
├── results/                    # Downloaded from Kaggle
│   ├── training_history.json
│   ├── training_curves.png
│   ├── eq_bench_results.json
│   └── safety_results.json
│
├── safety/
│   └── red_team_sheet.md       # Red-team prompts
│
└── configs/
    └── training_config.yaml
```

---

## Quick Summary

| Step | Action | Time |
|------|--------|------|
| 1 | Download from Kaggle | 5 min |
| 2 | Extract and organize | 5 min |
| 3 | Test locally (optional) | 15 min |
| 4 | Fill in report.md | 30-60 min |
| 5 | Review safety results | 10 min |
| 6 | Package submission | 10 min |

**Total: ~1-2 hours after training completes**

---

## Common Issues

### "CUDA out of memory" locally
- Use CPU inference (slower but works)
- Or use Google Colab for testing

### "Model not loading"
- Make sure you downloaded the complete lora_adapter folder
- Check that adapter_config.json exists

### "Low EQ-Bench scores"
- This is normal for a short training run
- Focus on showing improvement over baseline, not absolute scores
- Document what you'd do with more time/resources

---

Good luck with your submission! 🎉

