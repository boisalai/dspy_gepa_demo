"""
DSPy Optimizers for prompt optimization.

This module provides utilities to optimize DSPy modules using various optimizers:
- BootstrapFewShot: Generate demonstration examples
- MIPRO: Optimize instructions and examples
- Other optimizers as needed
"""

from dspy.teleprompt import BootstrapFewShot, MIPROv2
import dspy
from typing import Callable, List


def optimize_with_bootstrap(
    module: dspy.Module,
    trainset: List[dspy.Example],
    metric: Callable,
    max_bootstrapped_demos: int = 4,
    max_labeled_demos: int = 4
) -> dspy.Module:
    """
    Optimize a module using BootstrapFewShot.

    BootstrapFewShot generates demonstration examples by running the module
    on training data and keeping successful predictions.

    Args:
        module: The DSPy module to optimize
        trainset: Training examples in DSPy format
        metric: Metric function for evaluation
        max_bootstrapped_demos: Number of examples to generate
        max_labeled_demos: Maximum examples to use in prompts

    Returns:
        Optimized module with demonstration examples
    """
    print("🔧 Optimizing with BootstrapFewShot...")

    optimizer = BootstrapFewShot(
        metric=metric,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos
    )

    optimized = optimizer.compile(
        student=module,
        trainset=trainset
    )

    print("✅ BootstrapFewShot optimization complete")
    return optimized


def optimize_with_mipro(
    module: dspy.Module,
    trainset: List[dspy.Example],
    metric: Callable,
    auto: str = "light",
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 3
) -> dspy.Module:
    """
    Optimize a module using MIPROv2.

    MIPRO optimizes both instructions and demonstration examples by
    generating multiple variants and testing combinations.

    Args:
        module: The DSPy module to optimize
        trainset: Training examples
        metric: Metric function
        auto: Optimization intensity: "light", "medium", or "heavy"
        max_bootstrapped_demos: Number of demo examples to generate
        max_labeled_demos: Maximum demos to use

    Returns:
        Optimized module with improved instructions and examples
    """
    print("🚀 Optimizing with MIPROv2...")
    print(f"⏱️ Mode: {auto} (This may take 10-20 minutes with Ollama)\n")

    # MIPROv2 avec le paramètre auto (simplifié)
    optimizer = MIPROv2(
        metric=metric,
        auto=auto  # "light", "medium", or "heavy"
    )

    try:
        optimized = optimizer.compile(
            student=module,
            trainset=trainset,
            max_bootstrapped_demos=max_bootstrapped_demos,
            max_labeled_demos=max_labeled_demos,
            requires_permission_to_run=False
        )

        print("\n✅ MIPRO optimization complete")
        return optimized

    except Exception as e:
        print(f"⚠️ MIPRO optimization error: {e}")
        print("   Returning original module")
        return module


def inspect_optimized_demos(optimized_module: dspy.Module, max_demos: int = 3):
    """
    Inspect the demonstration examples added by an optimizer.

    Args:
        optimized_module: The optimized module
        max_demos: Maximum number of demos to display
    """
    # Iterate through named predictors to find demos
    demos_found = False

    for name, predictor in optimized_module.named_predictors():
        if hasattr(predictor, 'demos') and predictor.demos:
            demos_found = True
            print(f"📚 Found {len(predictor.demos)} demonstration examples in '{name}':\n")

            for i, demo in enumerate(predictor.demos[:max_demos], 1):
                print(f"Example {i}:")
                print(f"  Ticket: {demo.ticket[:80]}...")
                if hasattr(demo, 'category'):
                    print(f"  Category: {demo.category}")
                if hasattr(demo, 'priority'):
                    print(f"  Priority: {demo.priority}")
                print()

            # Only show first predictor with demos
            break

    if not demos_found:
        print("ℹ️ No demonstration examples found")


if __name__ == "__main__":
    # Example usage - Compare both optimizers
    from config import configure_ollama
    from modules import SimpleTicketClassifier
    from data import get_train_examples, get_val_examples
    from metrics import exact_match_metric
    from evaluation import evaluate_module

    print("=" * 80)
    print("COMPARISON OF DSPY OPTIMIZERS")
    print("=" * 80)
    print("\nThis script compares BootstrapFewShot and MIPROv2 optimizers.\n")

    # Configure DSPy
    configure_ollama()

    # Get data
    train_examples = get_train_examples()
    val_examples = get_val_examples()

    print(f"📊 Dataset: {len(train_examples)} training, {len(val_examples)} validation examples\n")

    # Create baseline module
    baseline = SimpleTicketClassifier()

    # Evaluate baseline
    print("=" * 80)
    print("BASELINE (No Optimization)")
    print("=" * 80)
    score_baseline = evaluate_module(baseline, val_examples, exact_match_metric)
    print(f"✅ Baseline score: {score_baseline:.2%}\n")

    # =========================================================================
    # TEST 1: BootstrapFewShot
    # =========================================================================
    print("=" * 80)
    print("OPTIMIZER 1: BootstrapFewShot")
    print("=" * 80)
    print("⏱️ Expected time: ~10-30 seconds\n")

    baseline_bootstrap = SimpleTicketClassifier()  # Fresh instance
    optimized_bootstrap = optimize_with_bootstrap(
        baseline_bootstrap,
        train_examples,
        exact_match_metric,
        max_bootstrapped_demos=4,
        max_labeled_demos=3
    )

    score_bootstrap = evaluate_module(optimized_bootstrap, val_examples, exact_match_metric)
    improvement_bootstrap = ((score_bootstrap - score_baseline) / score_baseline * 100) if score_baseline > 0 else 0
    print(f"\n📊 BootstrapFewShot score: {score_bootstrap:.2%}")
    print(f"   Improvement: {improvement_bootstrap:+.1f}%\n")

    print("--- Demonstration Examples (BootstrapFewShot) ---")
    inspect_optimized_demos(optimized_bootstrap, max_demos=3)

    # =========================================================================
    # TEST 2: MIPROv2
    # =========================================================================
    print("\n" + "=" * 80)
    print("OPTIMIZER 2: MIPROv2")
    print("=" * 80)
    print("⏱️ Expected time: 10-20 minutes with Ollama")
    print("⚠️ This is a long process - MIPRO tests multiple instruction variants\n")

    response = input("Do you want to run MIPRO optimization? (y/n): ").strip().lower()

    if response == 'y':
        baseline_mipro = SimpleTicketClassifier()  # Fresh instance
        optimized_mipro = optimize_with_mipro(
            baseline_mipro,
            train_examples,
            exact_match_metric,
            auto="light",  # "light", "medium", or "heavy"
            max_bootstrapped_demos=4,
            max_labeled_demos=3
        )

        score_mipro = evaluate_module(optimized_mipro, val_examples, exact_match_metric)
        improvement_mipro = ((score_mipro - score_baseline) / score_baseline * 100) if score_baseline > 0 else 0
        print(f"\n📊 MIPROv2 score: {score_mipro:.2%}")
        print(f"   Improvement: {improvement_mipro:+.1f}%\n")

        print("--- Demonstration Examples (MIPROv2) ---")
        inspect_optimized_demos(optimized_mipro, max_demos=3)

        # =========================================================================
        # FINAL COMPARISON
        # =========================================================================
        print("\n" + "=" * 80)
        print("FINAL COMPARISON")
        print("=" * 80)
        print(f"\n{'Method':<20} {'Score':<10} {'Improvement':<15}")
        print("-" * 45)
        print(f"{'Baseline':<20} {score_baseline:<10.2%} {'-':<15}")
        print(f"{'BootstrapFewShot':<20} {score_bootstrap:<10.2%} {improvement_bootstrap:+.1f}%")
        print(f"{'MIPROv2':<20} {score_mipro:<10.2%} {improvement_mipro:+.1f}%")
        print("=" * 80)

        # Determine winner
        if score_mipro > score_bootstrap and score_mipro > score_baseline:
            print("\n🏆 Winner: MIPROv2")
        elif score_bootstrap > score_mipro and score_bootstrap > score_baseline:
            print("\n🏆 Winner: BootstrapFewShot")
        elif score_baseline >= max(score_bootstrap, score_mipro):
            print("\n🏆 Winner: Baseline (optimization didn't help)")
        else:
            print("\n🤝 Tie between optimizers")

    else:
        print("\n⏭️ Skipping MIPRO optimization")
        print("\n" + "=" * 80)
        print("RESULTS (without MIPRO)")
        print("=" * 80)
        print(f"\n{'Method':<20} {'Score':<10} {'Improvement':<15}")
        print("-" * 45)
        print(f"{'Baseline':<20} {score_baseline:<10.2%} {'-':<15}")
        print(f"{'BootstrapFewShot':<20} {score_bootstrap:<10.2%} {improvement_bootstrap:+.1f}%")
        print("=" * 80)
