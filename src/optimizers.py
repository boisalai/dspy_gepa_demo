"""
DSPy Optimizers for prompt optimization.

This module provides utilities to optimize DSPy modules using various optimizers:
- BootstrapFewShot: Generate demonstration examples
- MIPRO: Optimize instructions and examples
- Other optimizers as needed
"""

from dspy.teleprompt import BootstrapFewShot, MIPRO
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
    num_candidates: int = 5,
    init_temperature: float = 1.0,
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 3
) -> dspy.Module:
    """
    Optimize a module using MIPRO.

    MIPRO optimizes both instructions and demonstration examples by
    generating multiple variants and testing combinations.

    Args:
        module: The DSPy module to optimize
        trainset: Training examples
        metric: Metric function
        num_candidates: Number of instruction variants to generate
        init_temperature: Temperature for instruction generation
        max_bootstrapped_demos: Number of demo examples to generate
        max_labeled_demos: Maximum demos to use

    Returns:
        Optimized module with improved instructions and examples
    """
    print("🚀 Optimizing with MIPRO...")
    print("⏰ This may take 10-20 minutes with Ollama\n")

    optimizer = MIPRO(
        metric=metric,
        num_candidates=num_candidates,
        init_temperature=init_temperature
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
    if hasattr(optimized_module, 'classifier'):
        predictor = optimized_module.classifier

        if hasattr(predictor, 'demos') and predictor.demos:
            print(f"📚 Found {len(predictor.demos)} demonstration examples:\n")

            for i, demo in enumerate(predictor.demos[:max_demos], 1):
                print(f"Example {i}:")
                print(f"  Ticket: {demo.ticket[:80]}...")
                if hasattr(demo, 'category'):
                    print(f"  Category: {demo.category}")
                if hasattr(demo, 'priority'):
                    print(f"  Priority: {demo.priority}")
                print()
        else:
            print("ℹ️ No demonstration examples found")
    else:
        print("ℹ️ Module structure doesn't match expected format")


if __name__ == "__main__":
    # Example usage
    from config import configure_ollama
    from modules import SimpleTicketClassifier
    from data import get_train_examples, get_val_examples
    from metrics import exact_match_metric
    from evaluation import evaluate_module

    print("Testing optimizers...\n")

    # Configure DSPy
    configure_ollama()

    # Get data
    train_examples = get_train_examples()
    val_examples = get_val_examples()

    # Create baseline module
    baseline = SimpleTicketClassifier()

    # Evaluate before optimization
    print("📊 Evaluating baseline...")
    score_before = evaluate_module(baseline, val_examples[:5], exact_match_metric)
    print(f"   Baseline score: {score_before:.2%}\n")

    # Optimize with BootstrapFewShot
    optimized = optimize_with_bootstrap(
        baseline,
        train_examples[:10],
        exact_match_metric,
        max_bootstrapped_demos=3
    )

    # Evaluate after optimization
    print("\n📊 Evaluating optimized module...")
    score_after = evaluate_module(optimized, val_examples[:5], exact_match_metric)
    print(f"   Optimized score: {score_after:.2%}")

    # Calculate improvement
    improvement = ((score_after - score_before) / score_before * 100) if score_before > 0 else 0
    print(f"   Improvement: {improvement:+.1f}%\n")

    # Inspect demos
    inspect_optimized_demos(optimized)
