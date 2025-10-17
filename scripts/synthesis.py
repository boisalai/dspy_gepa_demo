"""
Synthesis script - Complete end-to-end DSPy pipeline.

This script demonstrates a complete workflow combining:
1. Configuration with Ollama
2. Data loading
3. Module creation
4. Optimization
5. Evaluation and comparison

This is the final synthesis of all concepts from the tutorial.
"""

import dspy
from config import configure_ollama
from modules import SimpleTicketClassifier, ValidatedClassifier, SequentialClassifier
from data import get_train_examples, get_val_examples
from metrics import exact_match_metric, partial_match_metric
from optimizers import optimize_with_bootstrap, inspect_optimized_demos
from evaluation import evaluate_module


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(title.upper())
    print("=" * 80)


def run_complete_pipeline():
    """
    Run a complete end-to-end DSPy pipeline for IT ticket classification.

    Steps:
    1. Configure DSPy with Ollama
    2. Load training and validation data
    3. Create and evaluate baseline modules
    4. Optimize the best baseline
    5. Compare baseline vs optimized
    6. Show final results and recommendations
    """

    print_section("DSPy Complete Pipeline - IT Ticket Classification")

    # =========================================================================
    # STEP 1: Configuration
    # =========================================================================
    print_section("Step 1: Configuration")
    lm = configure_ollama()
    print(f"✅ Language model configured: {lm.model}")

    # =========================================================================
    # STEP 2: Data Loading
    # =========================================================================
    print_section("Step 2: Data Loading")
    trainset = get_train_examples()
    valset = get_val_examples()

    print(f"📊 Training examples: {len(trainset)}")
    print(f"📊 Validation examples: {len(valset)}")

    # Show sample
    print(f"\n📝 Sample training example:")
    sample = trainset[0]
    print(f"   Ticket: {sample.ticket[:60]}...")
    print(f"   Category: {sample.category}")
    print(f"   Priority: {sample.priority}")

    # =========================================================================
    # STEP 3: Baseline Evaluation
    # =========================================================================
    print_section("Step 3: Baseline Module Comparison")

    # Test different baseline modules
    modules = {
        "SimpleTicketClassifier": SimpleTicketClassifier(),
        "ValidatedClassifier": ValidatedClassifier(),
        "SequentialClassifier": SequentialClassifier()
    }

    baseline_scores = {}
    print("\nEvaluating baseline modules on validation set...\n")

    for name, module in modules.items():
        print(f"Testing {name}...")
        score = evaluate_module(module, valset, exact_match_metric)
        baseline_scores[name] = score
        print(f"   Score: {score:.2%}\n")

    # Select best baseline
    best_baseline_name = max(baseline_scores, key=baseline_scores.get)
    best_baseline_score = baseline_scores[best_baseline_name]

    print(f"🏆 Best baseline: {best_baseline_name} ({best_baseline_score:.2%})")

    # =========================================================================
    # STEP 4: Optimization
    # =========================================================================
    print_section("Step 4: Optimization with BootstrapFewShot")

    # Create fresh instance of best baseline
    if best_baseline_name == "SimpleTicketClassifier":
        module_to_optimize = SimpleTicketClassifier()
    elif best_baseline_name == "ValidatedClassifier":
        module_to_optimize = ValidatedClassifier()
    else:
        module_to_optimize = SequentialClassifier()

    print(f"\n🔧 Optimizing {best_baseline_name}...")
    print(f"⏱️ This will take ~30 seconds...\n")

    optimized = optimize_with_bootstrap(
        module_to_optimize,
        trainset,
        exact_match_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=3
    )

    # =========================================================================
    # STEP 5: Evaluation of Optimized Module
    # =========================================================================
    print_section("Step 5: Optimized Module Evaluation")

    optimized_score = evaluate_module(optimized, valset, exact_match_metric)

    print(f"\n📊 Results:")
    print(f"   Baseline ({best_baseline_name}): {best_baseline_score:.2%}")
    print(f"   Optimized: {optimized_score:.2%}")

    improvement = ((optimized_score - best_baseline_score) / best_baseline_score * 100) if best_baseline_score > 0 else 0
    print(f"   Improvement: {improvement:+.1f}%")

    # Show demonstration examples
    print("\n--- Demonstration Examples Generated ---")
    inspect_optimized_demos(optimized, max_demos=3)

    # =========================================================================
    # STEP 6: Detailed Comparison
    # =========================================================================
    print_section("Step 6: Detailed Comparison on Validation Set")

    # Recreate baseline for comparison
    if best_baseline_name == "SimpleTicketClassifier":
        baseline_fresh = SimpleTicketClassifier()
    elif best_baseline_name == "ValidatedClassifier":
        baseline_fresh = ValidatedClassifier()
    else:
        baseline_fresh = SequentialClassifier()

    print(f"\nComparing predictions on each validation example:\n")

    baseline_correct = 0
    optimized_correct = 0

    for i, example in enumerate(valset, 1):
        print(f"--- Example {i} ---")
        print(f"Ticket: {example.ticket[:60]}...")
        print(f"Expected: {example.category} / {example.priority}")

        # Baseline prediction
        baseline_pred = baseline_fresh(ticket=example.ticket)
        baseline_match = exact_match_metric(example, baseline_pred)
        baseline_status = "✅" if baseline_match == 1.0 else "❌"
        print(f"Baseline:  {baseline_pred.category} / {baseline_pred.priority} {baseline_status}")

        # Optimized prediction
        optimized_pred = optimized(ticket=example.ticket)
        optimized_match = exact_match_metric(example, optimized_pred)
        optimized_status = "✅" if optimized_match == 1.0 else "❌"
        print(f"Optimized: {optimized_pred.category} / {optimized_pred.priority} {optimized_status}")

        # Track scores
        if baseline_match == 1.0:
            baseline_correct += 1
        if optimized_match == 1.0:
            optimized_correct += 1

        # Highlight changes
        if baseline_pred.category != optimized_pred.category or baseline_pred.priority != optimized_pred.priority:
            if baseline_match == 1.0 and optimized_match == 0.0:
                print("⚠️ REGRESSION: Baseline was correct, optimized is wrong!")
            elif baseline_match == 0.0 and optimized_match == 1.0:
                print("✨ IMPROVEMENT: Baseline was wrong, optimized is correct!")
            else:
                print("↔️ Changed but both wrong")

        print()

    # =========================================================================
    # STEP 7: Final Summary
    # =========================================================================
    print_section("Step 7: Final Summary")

    print("\n📈 PERFORMANCE SUMMARY")
    print("-" * 80)
    print(f"Baseline ({best_baseline_name}):")
    print(f"  Correct: {baseline_correct}/{len(valset)}")
    print(f"  Score: {best_baseline_score:.2%}")
    print()
    print(f"Optimized (with BootstrapFewShot):")
    print(f"  Correct: {optimized_correct}/{len(valset)}")
    print(f"  Score: {optimized_score:.2%}")
    print()
    print(f"Improvement: {improvement:+.1f}%")

    # Interpretation
    print("\n💡 INTERPRETATION")
    print("-" * 80)

    if improvement > 10:
        print("✅ Optimization was successful!")
        print("   The few-shot examples significantly improved performance.")
    elif improvement > 0:
        print("➕ Optimization provided modest improvement.")
        print("   Consider using more training data or MIPRO for better results.")
    elif improvement == 0:
        print("➖ Optimization had no effect.")
        print("   The baseline and optimized models perform equally.")
    else:
        print("⚠️ Optimization decreased performance.")
        print("   Possible causes:")
        print("   - Dataset too small (need 50+ examples)")
        print("   - Few-shot examples not representative")
        print("   - Try partial_match_metric for more diversity")

    # Recommendations
    print("\n🎯 RECOMMENDATIONS")
    print("-" * 80)
    print("To improve results:")
    print("1. Increase dataset size (aim for 50-100 training examples)")
    print("2. Ensure balanced distribution of categories and priorities")
    print("3. Try partial_match_metric for less strict evaluation")
    print("4. Test MIPROv2 optimizer for instruction optimization")
    print("5. Experiment with different models (e.g., larger Llama models)")

    print("\n" + "=" * 80)
    print("🎉 PIPELINE COMPLETE")
    print("=" * 80)


def run_quick_demo():
    """
    Run a quick demonstration of the pipeline with minimal output.

    Good for testing or when you just want to see the final results.
    """
    print("🚀 Quick Demo - DSPy IT Ticket Classification\n")

    # Configure
    configure_ollama()

    # Load data
    trainset = get_train_examples()
    valset = get_val_examples()

    # Baseline
    baseline = SimpleTicketClassifier()
    baseline_score = evaluate_module(baseline, valset, exact_match_metric)

    # Optimize
    optimized = optimize_with_bootstrap(
        baseline,
        trainset,
        exact_match_metric,
        max_bootstrapped_demos=3
    )

    optimized_score = evaluate_module(optimized, valset, exact_match_metric)

    # Results
    print(f"\n{'='*50}")
    print(f"Baseline:  {baseline_score:.2%}")
    print(f"Optimized: {optimized_score:.2%}")
    improvement = ((optimized_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
    print(f"Change:    {improvement:+.1f}%")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        run_quick_demo()
    else:
        run_complete_pipeline()
