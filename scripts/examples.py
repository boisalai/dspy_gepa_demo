"""
Usage examples for the DSPy IT ticket classification system.

This module demonstrates how to use the various components
to build complete workflows.
"""

import dspy
from config import configure_ollama, configure_reflection_lm
from data import get_train_examples, get_val_examples, valset
from modules import SimpleTicketClassifier
from metrics import exact_match_metric, partial_match_metric
from evaluation import evaluate_module, compare_modules, print_comparison_table
from optimizers import optimize_with_bootstrap, inspect_optimized_demos
from gepa_utils import optimize_with_gepa, inspect_gepa_prompts


def example_1_basic_usage():
    """
    Example 1: Basic usage - Configure, create module, make predictions
    """
    print("=" * 70)
    print("EXAMPLE 1: BASIC USAGE")
    print("=" * 70 + "\n")

    # Step 1: Configure DSPy
    print("Step 1: Configuring DSPy with Ollama...")
    lm = configure_ollama()

    # Step 2: Create a classifier
    print("\nStep 2: Creating classifier...")
    classifier = SimpleTicketClassifier()

    # Step 3: Make predictions
    print("\nStep 3: Making predictions...\n")

    test_tickets = [
        "Mon ordinateur ne démarre plus, j'ai une présentation dans 1 heure",
        "Je voudrais accès au VPN pour le télétravail",
        "Toutes les imprimantes de l'étage sont hors ligne"
    ]

    for i, ticket in enumerate(test_tickets, 1):
        result = classifier(ticket=ticket)
        print(f"{i}. {ticket[:60]}...")
        print(f"   → Category: {result.category}")
        print(f"   → Priority: {result.priority}\n")


def example_2_evaluation():
    """
    Example 2: Evaluate a module on validation data
    """
    print("=" * 70)
    print("EXAMPLE 2: EVALUATION")
    print("=" * 70 + "\n")

    # Configure
    configure_ollama()

    # Create classifier
    classifier = SimpleTicketClassifier()
    val_examples = get_val_examples()

    # Evaluate with different metrics
    print("Evaluating on validation set...\n")

    score_exact = evaluate_module(classifier, val_examples, exact_match_metric)
    print(f"Exact match score: {score_exact:.2%}")

    score_partial = evaluate_module(classifier, val_examples, partial_match_metric)
    print(f"Partial match score: {score_partial:.2%}")


def example_3_optimization():
    """
    Example 3: Optimize a module with BootstrapFewShot
    """
    print("=" * 70)
    print("EXAMPLE 3: OPTIMIZATION WITH BOOTSTRAPFEWSHOT")
    print("=" * 70 + "\n")

    # Configure
    configure_ollama()

    # Get data
    train_examples = get_train_examples()
    val_examples = get_val_examples()

    # Create baseline
    baseline = SimpleTicketClassifier()

    # Evaluate baseline
    print("📊 Baseline performance:")
    score_before = evaluate_module(baseline, val_examples, exact_match_metric)
    print(f"   Score: {score_before:.2%}\n")

    # Optimize
    print("🔧 Optimizing with BootstrapFewShot...\n")
    optimized = optimize_with_bootstrap(
        baseline,
        train_examples,
        exact_match_metric,
        max_bootstrapped_demos=4
    )

    # Evaluate optimized
    print("\n📊 Optimized performance:")
    score_after = evaluate_module(optimized, val_examples, exact_match_metric)
    print(f"   Score: {score_after:.2%}")

    # Show improvement
    improvement = ((score_after - score_before) / score_before * 100) if score_before > 0 else 0
    print(f"   Improvement: {improvement:+.1f}%\n")

    # Inspect demos
    inspect_optimized_demos(optimized)


def example_4_gepa_optimization():
    """
    Example 4: Advanced optimization with GEPA
    """
    print("=" * 70)
    print("EXAMPLE 4: ADVANCED OPTIMIZATION WITH GEPA")
    print("=" * 70 + "\n")

    # Configure models
    lm_main = configure_ollama()
    reflection_lm = configure_reflection_lm()

    # Get data
    train_examples = get_train_examples()
    val_examples = get_val_examples()

    # Create baseline
    baseline = SimpleTicketClassifier()

    # Evaluate baseline
    print("📊 Baseline performance:")
    score_before = evaluate_module(baseline, val_examples[:5], exact_match_metric)
    print(f"   Score: {score_before:.2%}\n")

    # Optimize with GEPA (light mode for speed)
    print("🧬 Optimizing with GEPA (light mode)...\n")
    optimized = optimize_with_gepa(
        baseline,
        train_examples[:10],  # Use subset for speed
        val_examples[:5],
        exact_match_metric,
        reflection_lm,
        auto='light'
    )

    # Evaluate optimized
    print("\n📊 GEPA-optimized performance:")
    score_after = evaluate_module(optimized, val_examples[:5], exact_match_metric)
    print(f"   Score: {score_after:.2%}")

    # Show improvement
    improvement = ((score_after - score_before) / score_before * 100) if score_before > 0 else 0
    print(f"   Improvement: {improvement:+.1f}%\n")

    # Inspect GEPA optimizations
    inspect_gepa_prompts(optimized)


def example_5_module_comparison():
    """
    Example 5: Compare different modules
    """
    print("=" * 70)
    print("EXAMPLE 5: MODULE COMPARISON")
    print("=" * 70 + "\n")

    # Configure
    configure_ollama()

    # Import different modules
    from modules import SimpleTicketClassifier, SequentialClassifier, ValidatedClassifier

    # Create modules
    modules_to_compare = [
        ("Simple", SimpleTicketClassifier()),
        ("Sequential", SequentialClassifier()),
        ("Validated", ValidatedClassifier())
    ]

    # Define metrics
    metrics = {
        'exact_match': exact_match_metric,
        'partial_match': partial_match_metric
    }

    # Compare
    val_examples = get_val_examples()
    results = compare_modules(
        modules_to_compare,
        val_examples[:5],  # Use subset for speed
        metrics,
        verbose=True
    )

    # Print comparison table
    print_comparison_table(results, list(metrics.keys()))


def run_all_examples():
    """
    Run all examples in sequence.
    """
    examples = [
        ("Basic Usage", example_1_basic_usage),
        ("Evaluation", example_2_evaluation),
        ("Optimization", example_3_optimization),
        ("GEPA Optimization", example_4_gepa_optimization),
        ("Module Comparison", example_5_module_comparison)
    ]

    print("\n" + "=" * 70)
    print("RUNNING ALL EXAMPLES")
    print("=" * 70 + "\n")

    for i, (name, func) in enumerate(examples, 1):
        print(f"\n{'='*70}")
        print(f"Running Example {i}: {name}")
        print(f"{'='*70}\n")

        try:
            func()
        except Exception as e:
            print(f"\n⚠️ Example {i} failed: {e}")
            print("Continuing with next example...\n")

        print(f"\n✅ Example {i} complete\n")

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Run specific example
        example_num = sys.argv[1]

        examples_map = {
            '1': example_1_basic_usage,
            '2': example_2_evaluation,
            '3': example_3_optimization,
            '4': example_4_gepa_optimization,
            '5': example_5_module_comparison
        }

        if example_num in examples_map:
            examples_map[example_num]()
        elif example_num == 'all':
            run_all_examples()
        else:
            print(f"Unknown example: {example_num}")
            print("Available: 1, 2, 3, 4, 5, all")
    else:
        # Run basic example by default
        print("Running basic example...")
        print("To run other examples: python examples.py [1-5|all]\n")
        example_1_basic_usage()
