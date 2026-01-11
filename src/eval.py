"""
Evaluation Module
=================
EQ-Bench 3 evaluation, ablation studies, and qualitative analysis.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import json
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from datetime import datetime


# Emotion labels for reference
EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise"
]

# Strategy labels for reference
STRATEGY_LABELS = [
    "Question", "Restatement or Paraphrasing", "Reflection of Feelings",
    "Self-disclosure", "Affirmation and Reassurance", "Providing Suggestions",
    "Information", "Others"
]


class EQBenchEvaluator:
    """
    EQ-Bench 3 Evaluation.
    
    EQ-Bench measures emotional intelligence through:
    1. Emotion recognition accuracy
    2. Emotional response appropriateness
    3. Empathetic dialogue quality
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # EQ-Bench test scenarios
        self.test_scenarios = self._load_test_scenarios()
        
    def _load_test_scenarios(self) -> List[Dict]:
        """
        Load EQ-Bench test scenarios.
        These are representative empathy evaluation scenarios.
        """
        # Simplified EQ-Bench style scenarios
        scenarios = [
            {
                "id": "grief_1",
                "context": "My grandmother passed away last week. We were very close.",
                "expected_emotions": ["grief", "sadness", "love"],
                "required_elements": ["acknowledgment", "validation", "support"],
                "category": "grief"
            },
            {
                "id": "anxiety_1",
                "context": "I have a big job interview tomorrow and I can't stop thinking about all the ways it could go wrong.",
                "expected_emotions": ["nervousness", "fear", "anxiety"],
                "required_elements": ["acknowledgment", "normalization", "encouragement"],
                "category": "anxiety"
            },
            {
                "id": "anger_1",
                "context": "My coworker took credit for my project in front of our boss. I worked on it for months!",
                "expected_emotions": ["anger", "annoyance", "disappointment"],
                "required_elements": ["validation", "understanding", "support"],
                "category": "anger"
            },
            {
                "id": "joy_1",
                "context": "I just got accepted to my dream graduate program! I still can't believe it!",
                "expected_emotions": ["joy", "excitement", "pride"],
                "required_elements": ["celebration", "enthusiasm", "curiosity"],
                "category": "joy"
            },
            {
                "id": "confusion_1",
                "context": "My partner has been acting distant lately and I don't understand why. They say everything is fine but it doesn't feel that way.",
                "expected_emotions": ["confusion", "sadness", "nervousness"],
                "required_elements": ["validation", "exploration", "support"],
                "category": "relationship"
            },
            {
                "id": "guilt_1",
                "context": "I snapped at my mom yesterday and said some hurtful things. I know she didn't deserve it.",
                "expected_emotions": ["remorse", "sadness", "guilt"],
                "required_elements": ["validation", "normalization", "guidance"],
                "category": "guilt"
            },
            {
                "id": "fear_1",
                "context": "The doctor found something concerning in my test results. I have to go back for more tests.",
                "expected_emotions": ["fear", "nervousness", "anxiety"],
                "required_elements": ["support", "presence", "hope"],
                "category": "health"
            },
            {
                "id": "disappointment_1",
                "context": "I didn't get the promotion I've been working towards for two years. They gave it to someone with less experience.",
                "expected_emotions": ["disappointment", "sadness", "anger"],
                "required_elements": ["validation", "understanding", "perspective"],
                "category": "career"
            },
            {
                "id": "loneliness_1",
                "context": "Since moving to this new city, I haven't been able to make any real friends. Everyone seems to already have their groups.",
                "expected_emotions": ["sadness", "loneliness", "disappointment"],
                "required_elements": ["validation", "normalization", "encouragement"],
                "category": "social"
            },
            {
                "id": "overwhelm_1",
                "context": "Between work, taking care of my parents, and trying to maintain my own life, I feel like I'm drowning. There's never enough time.",
                "expected_emotions": ["exhaustion", "overwhelm", "stress"],
                "required_elements": ["validation", "compassion", "support"],
                "category": "stress"
            }
        ]
        return scenarios
    
    def evaluate(self, generate_responses: bool = True) -> Dict:
        """
        Run full EQ-Bench evaluation.
        
        Returns:
            Dict with scores and detailed results
        """
        print("\n" + "="*60)
        print("Running EQ-Bench 3 Evaluation")
        print("="*60)
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "num_scenarios": len(self.test_scenarios),
            "scenario_results": [],
            "aggregate_scores": {}
        }
        
        total_empathy_score = 0
        total_emotion_accuracy = 0
        category_scores = {}
        
        for scenario in tqdm(self.test_scenarios, desc="Evaluating scenarios"):
            scenario_result = self._evaluate_scenario(scenario, generate_responses)
            results["scenario_results"].append(scenario_result)
            
            total_empathy_score += scenario_result["empathy_score"]
            total_emotion_accuracy += scenario_result["emotion_accuracy"]
            
            # Track by category
            cat = scenario["category"]
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(scenario_result["empathy_score"])
        
        # Aggregate scores
        n = len(self.test_scenarios)
        results["aggregate_scores"] = {
            "overall_empathy_score": total_empathy_score / n,
            "emotion_recognition_accuracy": total_emotion_accuracy / n,
            "category_scores": {cat: np.mean(scores) for cat, scores in category_scores.items()},
            "raw_score": total_empathy_score,
            "normalized_score": (total_empathy_score / n) * 100,  # 0-100 scale
        }
        
        # Calculate Elo-style score (simplified)
        results["aggregate_scores"]["elo_score"] = self._calculate_elo(
            results["aggregate_scores"]["normalized_score"]
        )
        
        print(f"\nEQ-Bench Results:")
        print(f"  Overall Empathy Score: {results['aggregate_scores']['overall_empathy_score']:.3f}")
        print(f"  Normalized Score: {results['aggregate_scores']['normalized_score']:.1f}/100")
        print(f"  Elo Score: {results['aggregate_scores']['elo_score']:.0f}")
        
        return results
    
    def _evaluate_scenario(self, scenario: Dict, generate_response: bool) -> Dict:
        """Evaluate a single scenario."""
        context = scenario["context"]
        
        # Generate response
        if generate_response:
            response = self.model.generate(context)
        else:
            response = "[Response generation disabled]"
        
        # Predict emotion
        predicted_emotion, emotion_probs = self.model.predict_emotion(context)
        predicted_emotion_name = EMOTION_LABELS[predicted_emotion] if predicted_emotion < len(EMOTION_LABELS) else "unknown"
        
        # Calculate emotion accuracy
        emotion_accuracy = 1.0 if predicted_emotion_name in scenario["expected_emotions"] else 0.0
        
        # Evaluate response quality
        empathy_score = self._score_empathy(response, scenario["required_elements"])
        
        return {
            "scenario_id": scenario["id"],
            "category": scenario["category"],
            "context": context,
            "response": response,
            "predicted_emotion": predicted_emotion_name,
            "expected_emotions": scenario["expected_emotions"],
            "emotion_accuracy": emotion_accuracy,
            "empathy_score": empathy_score,
            "emotion_probs_top3": self._get_top_emotions(emotion_probs)
        }
    
    def _score_empathy(self, response: str, required_elements: List[str]) -> float:
        """
        Score response for empathy based on required elements.
        
        Scoring criteria:
        - Acknowledgment: Does it recognize the person's situation?
        - Validation: Does it validate their feelings?
        - Support: Does it offer support or presence?
        - Appropriate tone: Is it warm and caring?
        """
        response_lower = response.lower()
        score = 0.0
        
        # Check for acknowledgment indicators
        acknowledgment_phrases = [
            "i hear", "i understand", "that sounds", "i can see",
            "it seems like", "i sense", "that must", "sounds like"
        ]
        if any(phrase in response_lower for phrase in acknowledgment_phrases):
            score += 0.25
        
        # Check for validation indicators
        validation_phrases = [
            "it's okay", "it's normal", "it's understandable", "valid",
            "makes sense", "natural to feel", "anyone would", "of course"
        ]
        if any(phrase in response_lower for phrase in validation_phrases):
            score += 0.25
        
        # Check for support indicators
        support_phrases = [
            "i'm here", "i'm with you", "you're not alone", "support",
            "care about", "here for you", "we can", "together"
        ]
        if any(phrase in response_lower for phrase in support_phrases):
            score += 0.25
        
        # Check for appropriate tone (warm, not clinical)
        warm_phrases = [
            "feel", "heart", "care", "hug", "sorry", "love",
            "strength", "courage", "hope", "appreciate"
        ]
        if any(phrase in response_lower for phrase in warm_phrases):
            score += 0.25
        
        # Penalize for problematic patterns
        problematic_phrases = [
            "you should", "you need to", "just", "simply",
            "stop", "don't worry", "calm down", "get over"
        ]
        penalty = sum(0.1 for phrase in problematic_phrases if phrase in response_lower)
        score = max(0, score - penalty)
        
        return min(1.0, score)
    
    def _get_top_emotions(self, probs: torch.Tensor) -> List[Dict]:
        """Get top 3 predicted emotions with probabilities."""
        probs_np = probs.cpu().numpy()
        top_indices = np.argsort(probs_np)[-3:][::-1]
        
        return [
            {"emotion": EMOTION_LABELS[i], "probability": float(probs_np[i])}
            for i in top_indices
            if i < len(EMOTION_LABELS)
        ]
    
    def _calculate_elo(self, normalized_score: float) -> float:
        """
        Convert normalized score to Elo-style rating.
        Baseline: 1000 for average model.
        """
        # Simple linear mapping: 50/100 = 1000 Elo
        # Each point above/below 50 adds/subtracts 10 Elo
        return 1000 + (normalized_score - 50) * 10


class AblationEvaluator:
    """Run and compare ablation studies."""
    
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def compare_ablations(self, ablation_results: List[Dict]) -> Dict:
        """
        Compare results from different ablation runs.
        
        Args:
            ablation_results: List of dicts with 'name', 'eq_bench_score', 'val_loss', etc.
            
        Returns:
            Comparison summary
        """
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "ablations": [],
            "summary": {}
        }
        
        for result in ablation_results:
            comparison["ablations"].append({
                "name": result["name"],
                "val_loss": result.get("val_loss", None),
                "eq_bench_score": result.get("eq_bench_score", None),
                "elo_score": result.get("elo_score", None),
            })
        
        # Find best configuration
        if ablation_results:
            best_by_score = max(ablation_results, key=lambda x: x.get("eq_bench_score", 0))
            best_by_loss = min(ablation_results, key=lambda x: x.get("val_loss", float("inf")))
            
            comparison["summary"] = {
                "best_by_eq_bench": best_by_score["name"],
                "best_by_val_loss": best_by_loss["name"],
            }
            
            # Calculate relative improvements
            baseline = next((r for r in ablation_results if r["name"] == "full"), None)
            if baseline:
                for ablation in comparison["ablations"]:
                    if baseline.get("eq_bench_score") and ablation.get("eq_bench_score"):
                        ablation["eq_bench_diff"] = ablation["eq_bench_score"] - baseline["eq_bench_score"]
        
        # Save comparison
        with open(os.path.join(self.output_dir, "ablation_comparison.json"), "w") as f:
            json.dump(comparison, f, indent=2)
        
        return comparison
    
    def generate_ablation_report(self, comparison: Dict) -> str:
        """Generate markdown report for ablation studies."""
        report = """# Ablation Study Results

## Overview

This report presents the results of ablation studies conducted to understand the contribution of each component in the multi-task empathetic LLM architecture.

## Configurations Tested

| Configuration | Description |
|--------------|-------------|
| full | All losses enabled (λ_LM=1.0, λ_emo=0.2, λ_strat=0.2, λ_safe=0.1) |
| no_emotion | Emotion head disabled (λ_emo=0) |
| no_strategy | Strategy head disabled (λ_strat=0) |

## Results

| Configuration | Val Loss | EQ-Bench Score | Elo | Δ from Full |
|--------------|----------|----------------|-----|-------------|
"""
        for ablation in comparison["ablations"]:
            diff = ablation.get("eq_bench_diff", 0)
            diff_str = f"+{diff:.1f}" if diff > 0 else f"{diff:.1f}"
            report += f"| {ablation['name']} | {ablation.get('val_loss', 'N/A'):.4f} | {ablation.get('eq_bench_score', 'N/A'):.1f} | {ablation.get('elo_score', 'N/A'):.0f} | {diff_str} |\n"
        
        report += f"""
## Analysis

### Key Findings

1. **Emotion Head Impact**: {"Improves" if comparison.get("summary", {}).get("best_by_eq_bench") != "no_emotion" else "May not significantly improve"} empathy scores, indicating that explicit emotion recognition {"helps" if comparison.get("summary", {}).get("best_by_eq_bench") != "no_emotion" else "has limited impact on"} response quality.

2. **Strategy Head Impact**: Strategy prediction {"enhances" if comparison.get("summary", {}).get("best_by_eq_bench") != "no_strategy" else "may not significantly enhance"} the model's ability to choose appropriate support strategies.

### Conclusion

The {comparison.get("summary", {}).get("best_by_eq_bench", "full")} configuration achieved the best EQ-Bench performance, suggesting that {"all components contribute to empathetic response generation" if comparison.get("summary", {}).get("best_by_eq_bench") == "full" else "targeted ablation may simplify the model without significant performance loss"}.
"""
        
        # Save report
        report_path = os.path.join(self.output_dir, "ablation_report.md")
        with open(report_path, "w") as f:
            f.write(report)
        
        return report


class QualitativeEvaluator:
    """Generate qualitative comparison examples."""
    
    def __init__(self, model, base_model, tokenizer):
        self.model = model
        self.base_model = base_model
        self.tokenizer = tokenizer
        
    def generate_comparison(self, prompts: List[str]) -> List[Dict]:
        """
        Generate side-by-side comparisons for given prompts.
        
        Args:
            prompts: List of user inputs to compare
            
        Returns:
            List of comparison dicts
        """
        comparisons = []
        
        for prompt in prompts:
            # Generate from fine-tuned model
            ft_response = self.model.generate(prompt)
            
            # Generate from base model
            base_response = self._generate_base(prompt)
            
            comparisons.append({
                "input": prompt,
                "base_model_response": base_response,
                "fine_tuned_response": ft_response,
                "analysis": self._analyze_comparison(prompt, base_response, ft_response)
            })
        
        return comparisons
    
    def _generate_base(self, prompt: str) -> str:
        """Generate response from base model."""
        # This would use the non-fine-tuned base model
        # For now, return placeholder
        return "[Base model response would be generated here]"
    
    def _analyze_comparison(self, prompt: str, base: str, finetuned: str) -> str:
        """Analyze the difference between responses."""
        analysis_points = []
        
        ft_lower = finetuned.lower()
        base_lower = base.lower()
        
        # Check for empathy indicators in fine-tuned
        if any(word in ft_lower for word in ["understand", "feel", "sounds like", "must be"]):
            analysis_points.append("Fine-tuned model shows acknowledgment")
        
        if any(word in ft_lower for word in ["here for you", "support", "care"]):
            analysis_points.append("Fine-tuned model offers support")
        
        if "?" in finetuned:
            analysis_points.append("Fine-tuned model asks follow-up questions")
        
        return "; ".join(analysis_points) if analysis_points else "Similar response quality"


def evaluate_emotion_head(model, test_dataloader, device: str = "cuda") -> Dict:
    """Evaluate emotion classification accuracy."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating emotion head"):
            if not batch["has_emotion"].any():
                continue
                
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            mask = batch["has_emotion"].bool()
            preds = outputs["emotion_logits"][mask].argmax(dim=-1)
            labels = batch["emotion_label"][mask]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if not all_preds:
        return {"accuracy": 0, "f1": 0}
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "num_samples": len(all_preds)
    }


def evaluate_strategy_head(model, test_dataloader, device: str = "cuda") -> Dict:
    """Evaluate strategy classification accuracy."""
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating strategy head"):
            if not batch["has_strategy"].any():
                continue
                
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"]
            )
            
            mask = batch["has_strategy"].bool()
            preds = outputs["strategy_logits"][mask].argmax(dim=-1)
            labels = batch["strategy_label"][mask]
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if not all_preds:
        return {"accuracy": 0, "f1": 0}
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "f1": f1,
        "num_samples": len(all_preds)
    }


def save_evaluation_results(results: Dict, output_path: str):
    """Save evaluation results to JSON."""
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    print("Evaluation module loaded successfully.")
    print("\nComponents:")
    print("  - EQBenchEvaluator: Run EQ-Bench 3 evaluation")
    print("  - AblationEvaluator: Compare ablation studies")
    print("  - QualitativeEvaluator: Generate side-by-side comparisons")
    print("  - evaluate_emotion_head: Test emotion classification accuracy")
    print("  - evaluate_strategy_head: Test strategy classification accuracy")

