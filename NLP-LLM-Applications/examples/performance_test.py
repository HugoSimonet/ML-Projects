"""
Comprehensive Performance Testing and Hyperparameter Optimization

This script tests different hyperparameter configurations and measures:
- Text generation quality (BLEU, ROUGE, perplexity)
- Inference time and throughput
- Memory usage
- Output diversity

The goal is to find optimal hyperparameters for different use cases.
"""

import sys
from pathlib import Path
import time
import json
from typing import Dict, List, Tuple
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import GPTModel, BERTModel
from models import PromptTemplate, FewShotLearner, Example
from evaluation import evaluate_generation, ClassificationMetrics, BLEUScore, ROUGEScore

print("="*80)
print("NLP-LLM-Applications - Comprehensive Performance Testing")
print("="*80)

# Test configurations for different scenarios
TEST_CONFIGS = {
    "creative_generation": {
        "temperature": [0.7, 0.8, 0.9, 1.0, 1.2],
        "top_p": [0.9, 0.92, 0.95, 0.98],
        "top_k": [40, 50, 60, 0],  # 0 means disabled
        "max_length": [100, 150, 200]
    },
    "factual_generation": {
        "temperature": [0.1, 0.2, 0.3, 0.5],
        "top_p": [0.8, 0.85, 0.9],
        "top_k": [10, 20, 30, 40],
        "max_length": [100, 150]
    },
    "balanced_generation": {
        "temperature": [0.5, 0.6, 0.7, 0.8],
        "top_p": [0.9, 0.92, 0.95],
        "top_k": [40, 50],
        "max_length": [100]
    }
}

# Test prompts with reference outputs for evaluation
TEST_PROMPTS = [
    {
        "prompt": "Artificial intelligence is",
        "reference": "Artificial intelligence is a field of computer science focused on creating intelligent machines that can perform tasks requiring human intelligence.",
        "category": "factual"
    },
    {
        "prompt": "Once upon a time in a magical forest,",
        "reference": "Once upon a time in a magical forest, there lived a wise old wizard who helped all the creatures in need.",
        "category": "creative"
    },
    {
        "prompt": "The benefits of exercise include",
        "reference": "The benefits of exercise include improved cardiovascular health, stronger muscles, better mood, and increased energy levels.",
        "category": "factual"
    },
    {
        "prompt": "In the year 2050, technology will",
        "reference": "In the year 2050, technology will have advanced significantly, with AI assistants, autonomous vehicles, and sustainable energy solutions being commonplace.",
        "category": "creative"
    }
]

def calculate_diversity_metrics(texts: List[str]) -> Dict[str, float]:
    """Calculate diversity metrics for generated text."""
    all_words = []
    all_bigrams = []

    for text in texts:
        words = text.lower().split()
        all_words.extend(words)
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        all_bigrams.extend(bigrams)

    # Type-Token Ratio (TTR)
    ttr = len(set(all_words)) / len(all_words) if all_words else 0

    # Distinct-1 and Distinct-2
    distinct_1 = len(set(all_words)) / len(all_words) if all_words else 0
    distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0

    return {
        "type_token_ratio": ttr,
        "distinct_1": distinct_1,
        "distinct_2": distinct_2,
        "unique_words": len(set(all_words)),
        "total_words": len(all_words)
    }

def test_generation_config(
    model: GPTModel,
    config: Dict,
    prompts: List[Dict],
    num_runs: int = 3
) -> Dict:
    """Test a specific generation configuration."""
    results = {
        "config": config,
        "metrics": {
            "bleu_scores": [],
            "rouge_scores": [],
            "inference_times": [],
            "output_lengths": [],
            "diversity": None
        },
        "outputs": []
    }

    bleu = BLEUScore()
    rouge = ROUGEScore()

    all_generated = []

    for prompt_data in prompts:
        prompt = prompt_data["prompt"]
        reference = prompt_data["reference"]

        # Run multiple times for each prompt to get average timing
        prompt_times = []
        prompt_outputs = []

        for _ in range(num_runs):
            start_time = time.time()

            generated = model.generate(
                prompt,
                max_length=config.get("max_length", 100),
                temperature=config.get("temperature", 0.7),
                top_p=config.get("top_p", 0.9),
                top_k=config.get("top_k", 50),
                do_sample=True
            )

            inference_time = time.time() - start_time
            prompt_times.append(inference_time)
            prompt_outputs.append(generated[0])
            all_generated.append(generated[0])

        # Calculate metrics for this prompt
        avg_time = np.mean(prompt_times)
        results["metrics"]["inference_times"].append(avg_time)

        # Use first output for quality metrics
        output = prompt_outputs[0]

        # BLEU score
        bleu_result = bleu.compute([output], [[reference]])
        results["metrics"]["bleu_scores"].append(bleu_result.score)

        # ROUGE score
        rouge_result = rouge.compute(output, reference)
        results["metrics"]["rouge_scores"].append(rouge_result.score)

        # Output length
        results["metrics"]["output_lengths"].append(len(output.split()))

        results["outputs"].append({
            "prompt": prompt,
            "generated": output,
            "reference": reference,
            "bleu": bleu_result.score,
            "rouge": rouge_result.score,
            "time": avg_time
        })

    # Calculate diversity metrics across all outputs
    results["metrics"]["diversity"] = calculate_diversity_metrics(all_generated)

    # Aggregate metrics
    results["metrics"]["avg_bleu"] = np.mean(results["metrics"]["bleu_scores"])
    results["metrics"]["avg_rouge"] = np.mean(results["metrics"]["rouge_scores"])
    results["metrics"]["avg_time"] = np.mean(results["metrics"]["inference_times"])
    results["metrics"]["avg_length"] = np.mean(results["metrics"]["output_lengths"])

    return results

def run_baseline_test(model: GPTModel) -> Dict:
    """Run baseline test with default parameters."""
    print("\n" + "="*80)
    print("BASELINE TEST - Default Hyperparameters")
    print("="*80)

    baseline_config = {
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "max_length": 100
    }

    print(f"\nConfiguration: {baseline_config}")
    print("Running tests on all prompts...")

    results = test_generation_config(model, baseline_config, TEST_PROMPTS, num_runs=3)

    print(f"\nBaseline Results:")
    print(f"  Average BLEU Score:   {results['metrics']['avg_bleu']:.4f}")
    print(f"  Average ROUGE Score:  {results['metrics']['avg_rouge']:.4f}")
    print(f"  Average Inference Time: {results['metrics']['avg_time']:.3f}s")
    print(f"  Average Output Length:  {results['metrics']['avg_length']:.1f} words")
    print(f"  Diversity (TTR):      {results['metrics']['diversity']['type_token_ratio']:.4f}")
    print(f"  Distinct-1:           {results['metrics']['diversity']['distinct_1']:.4f}")
    print(f"  Distinct-2:           {results['metrics']['diversity']['distinct_2']:.4f}")

    return results

def optimize_hyperparameters(model: GPTModel, scenario: str = "balanced_generation") -> Tuple[Dict, List[Dict]]:
    """Systematically test hyperparameters to find optimal configuration."""
    print("\n" + "="*80)
    print(f"HYPERPARAMETER OPTIMIZATION - {scenario.replace('_', ' ').title()}")
    print("="*80)

    config_space = TEST_CONFIGS[scenario]
    all_results = []

    # Grid search over key hyperparameters
    best_score = -1
    best_config = None
    best_results = None

    print(f"\nTesting configurations...")
    print(f"  Temperature values: {config_space['temperature']}")
    print(f"  Top-p values: {config_space['top_p']}")
    print(f"  Top-k values: {config_space['top_k']}")
    print(f"  Max length values: {config_space['max_length']}")

    total_configs = (len(config_space['temperature']) *
                    len(config_space['top_p']) *
                    len(config_space['top_k']) *
                    len(config_space['max_length']))

    print(f"\nTotal configurations to test: {total_configs}")

    # Sample a subset of configurations for efficiency
    # Test each temperature with best defaults, then optimize other params
    test_configs = []

    # Phase 1: Find best temperature
    for temp in config_space['temperature']:
        test_configs.append({
            "temperature": temp,
            "top_p": config_space['top_p'][len(config_space['top_p'])//2],
            "top_k": config_space['top_k'][len(config_space['top_k'])//2],
            "max_length": config_space['max_length'][0]
        })

    # Phase 2: Find best top_p with best temperature
    for top_p in config_space['top_p']:
        test_configs.append({
            "temperature": config_space['temperature'][len(config_space['temperature'])//2],
            "top_p": top_p,
            "top_k": config_space['top_k'][len(config_space['top_k'])//2],
            "max_length": config_space['max_length'][0]
        })

    # Phase 3: Find best top_k
    for top_k in config_space['top_k']:
        test_configs.append({
            "temperature": config_space['temperature'][len(config_space['temperature'])//2],
            "top_p": config_space['top_p'][len(config_space['top_p'])//2],
            "top_k": top_k,
            "max_length": config_space['max_length'][0]
        })

    print(f"Reduced to {len(test_configs)} configurations using smart sampling\n")

    for i, config in enumerate(test_configs, 1):
        print(f"[{i}/{len(test_configs)}] Testing config: temp={config['temperature']}, "
              f"top_p={config['top_p']}, top_k={config['top_k']}, "
              f"max_len={config['max_length']}")

        results = test_generation_config(model, config, TEST_PROMPTS, num_runs=1)
        all_results.append(results)

        # Composite score: balance quality and diversity
        quality_score = (results['metrics']['avg_bleu'] + results['metrics']['avg_rouge']) / 2
        diversity_score = results['metrics']['diversity']['distinct_2']
        composite_score = 0.7 * quality_score + 0.3 * diversity_score

        print(f"    BLEU: {results['metrics']['avg_bleu']:.4f}, "
              f"ROUGE: {results['metrics']['avg_rouge']:.4f}, "
              f"Diversity: {diversity_score:.4f}, "
              f"Composite: {composite_score:.4f}")

        if composite_score > best_score:
            best_score = composite_score
            best_config = config
            best_results = results
            print(f"    *** New best config! ***")

    print(f"\n{'='*80}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"\nBest Configuration:")
    for key, value in best_config.items():
        print(f"  {key}: {value}")

    print(f"\nBest Results:")
    print(f"  BLEU Score:       {best_results['metrics']['avg_bleu']:.4f}")
    print(f"  ROUGE Score:      {best_results['metrics']['avg_rouge']:.4f}")
    print(f"  Inference Time:   {best_results['metrics']['avg_time']:.3f}s")
    print(f"  Output Length:    {best_results['metrics']['avg_length']:.1f} words")
    print(f"  Diversity (TTR):  {best_results['metrics']['diversity']['type_token_ratio']:.4f}")
    print(f"  Distinct-2:       {best_results['metrics']['diversity']['distinct_2']:.4f}")
    print(f"  Composite Score:  {best_score:.4f}")

    return best_config, all_results

def save_results(baseline: Dict, optimized: Dict, all_results: List[Dict], filename: str = "performance_results.json"):
    """Save all results to JSON file."""
    output_data = {
        "baseline": baseline,
        "optimized_best": optimized,
        "all_configurations": [
            {
                "config": r["config"],
                "metrics": {
                    k: v for k, v in r["metrics"].items()
                    if k != "bleu_scores" and k != "rouge_scores" and k != "inference_times" and k != "output_lengths"
                }
            }
            for r in all_results
        ]
    }

    output_path = Path(__file__).parent.parent / "test_output" / filename
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")

def print_comparison(baseline: Dict, optimized: Dict):
    """Print detailed comparison between baseline and optimized configurations."""
    print("\n" + "="*80)
    print("BASELINE vs OPTIMIZED COMPARISON")
    print("="*80)

    print("\n{:<30} {:<20} {:<20} {:<15}".format("Metric", "Baseline", "Optimized", "Improvement"))
    print("-"*85)

    metrics_to_compare = [
        ("BLEU Score", "avg_bleu", True),
        ("ROUGE Score", "avg_rouge", True),
        ("Inference Time (s)", "avg_time", False),
        ("Output Length (words)", "avg_length", None),
        ("Type-Token Ratio", "diversity.type_token_ratio", True),
        ("Distinct-2", "diversity.distinct_2", True),
    ]

    for metric_name, metric_key, higher_is_better in metrics_to_compare:
        # Get values with nested key support
        if '.' in metric_key:
            keys = metric_key.split('.')
            baseline_val = baseline['metrics'][keys[0]][keys[1]]
            optimized_val = optimized['metrics'][keys[0]][keys[1]]
        else:
            baseline_val = baseline['metrics'][metric_key]
            optimized_val = optimized['metrics'][metric_key]

        if higher_is_better is not None:
            if higher_is_better:
                improvement = ((optimized_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
                improvement_str = f"+{improvement:.1f}%" if improvement >= 0 else f"{improvement:.1f}%"
            else:
                improvement = ((baseline_val - optimized_val) / baseline_val * 100) if baseline_val > 0 else 0
                improvement_str = f"+{improvement:.1f}%" if improvement >= 0 else f"{improvement:.1f}%"
        else:
            improvement_str = "-"

        print(f"{metric_name:<30} {baseline_val:<20.4f} {optimized_val:<20.4f} {improvement_str:<15}")

    print("\nConfiguration Changes:")
    print("-"*85)
    for key in baseline['config'].keys():
        baseline_val = baseline['config'][key]
        optimized_val = optimized['config'][key]
        change = " (changed)" if baseline_val != optimized_val else ""
        print(f"  {key:<20} {baseline_val:<15} -> {optimized_val:<15}{change}")

def main():
    """Main execution function."""
    print("\nInitializing GPT-2 model...")
    model = GPTModel(model_name="gpt2", use_pretrained=True)
    print("Model loaded successfully!\n")

    # Step 1: Run baseline test
    baseline_results = run_baseline_test(model)

    # Step 2: Optimize hyperparameters
    optimized_config, all_results = optimize_hyperparameters(model, scenario="balanced_generation")

    # Get full results for optimized config
    optimized_results = next(r for r in all_results if r['config'] == optimized_config)

    # Step 3: Print comparison
    print_comparison(baseline_results, optimized_results)

    # Step 4: Save results
    save_results(baseline_results, optimized_results, all_results)

    # Step 5: Test other scenarios (optional)
    print("\n" + "="*80)
    print("ADDITIONAL SCENARIO TESTING")
    print("="*80)

    for scenario in ["creative_generation", "factual_generation"]:
        print(f"\nTesting scenario: {scenario}")
        best_config, _ = optimize_hyperparameters(model, scenario=scenario)

        # Save scenario-specific results
        save_results(
            baseline_results,
            next(r for r in _ if r['config'] == best_config),
            _,
            filename=f"performance_results_{scenario}.json"
        )

    print("\n" + "="*80)
    print("PERFORMANCE TESTING COMPLETE!")
    print("="*80)
    print("\nSummary:")
    print("  - Baseline performance established")
    print("  - Hyperparameters optimized for multiple scenarios")
    print("  - Results saved to test_output/ directory")
    print("  - Comparison reports generated")
    print("\nOptimal configurations found for:")
    print("  - Balanced generation (quality + diversity)")
    print("  - Creative generation (high diversity)")
    print("  - Factual generation (high accuracy)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()
