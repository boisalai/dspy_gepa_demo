"""
Evaluation functions for DSPy modules.

This module provides utilities to evaluate and compare DSPy modules
on datasets using specified metrics.
"""

from typing import Callable, List, Union, Dict
import dspy


def evaluate_module(
    module: dspy.Module,
    dataset: Union[List[dict], List[dspy.Example]],
    metric: Callable,
    verbose: bool = False
) -> float:
    """
    Evaluate a DSPy module on a complete dataset.

    Args:
        module: The DSPy module to evaluate
        dataset: List of examples (dict or dspy.Example format)
        metric: Metric function that takes (example, prediction) and returns a score
        verbose: If True, print details for each example

    Returns:
        float: Average score across all examples (between 0 and 1)
    """
    total_score = 0
    n_examples = len(dataset)

    for i, example in enumerate(dataset):
        # Handle both dict and dspy.Example formats
        if isinstance(example, dict):
            ticket = example['ticket']
        else:
            ticket = example.ticket

        # Get prediction
        prediction = module(ticket=ticket)

        # Calculate score
        score = metric(example, prediction)
        total_score += score

        # Optional verbose output
        if verbose:
            print(f"Example {i+1}/{n_examples}")
            if isinstance(example, dict):
                print(f"  Ticket: {example['ticket'][:50]}...")
                print(f"  Expected: {example['category']} | {example['priority']}")
            else:
                print(f"  Ticket: {example.ticket[:50]}...")
                print(f"  Expected: {example.category} | {example.priority}")
            print(f"  Predicted: {prediction.category} | {prediction.priority}")
            print(f"  Score: {score}\n")

    # Average score
    avg_score = total_score / n_examples
    return avg_score


def compare_modules(
    modules: List[tuple],
    dataset: Union[List[dict], List[dspy.Example]],
    metrics: Dict[str, Callable],
    verbose: bool = False
) -> List[Dict]:
    """
    Compare multiple modules on a dataset using multiple metrics.

    Args:
        modules: List of (name, module) tuples
        dataset: Dataset to evaluate on
        metrics: Dictionary of {metric_name: metric_function}
        verbose: If True, print progress

    Returns:
        List of dictionaries containing results for each module
    """
    results = []

    for name, module in modules:
        if verbose:
            print(f"Evaluating {name}...")

        module_results = {'module': name}

        for metric_name, metric_func in metrics.items():
            score = evaluate_module(module, dataset, metric_func, verbose=False)
            module_results[metric_name] = score

        results.append(module_results)

    return results


def print_comparison_table(results: List[Dict], metric_names: List[str]):
    """
    Print a formatted comparison table.

    Args:
        results: List of result dictionaries from compare_modules
        metric_names: Names of metrics to display
    """
    # Header
    print("=" * 70)
    print("Module Comparison Results")
    print("=" * 70)

    # Column widths
    name_width = max(len(r['module']) for r in results) + 2
    metric_width = 12

    # Print header row
    header = f"{'Module':<{name_width}}"
    for metric_name in metric_names:
        header += f" | {metric_name:<{metric_width}}"
    print(header)
    print("-" * len(header))

    # Print data rows
    for result in results:
        row = f"{result['module']:<{name_width}}"
        for metric_name in metric_names:
            score = result.get(metric_name, 0)
            row += f" | {score:<{metric_width}.1%}"
        print(row)

    print("=" * 70)

    # Find best module for each metric
    print("\nBest performers:")
    for metric_name in metric_names:
        best = max(results, key=lambda x: x.get(metric_name, 0))
        print(f"  {metric_name}: {best['module']} ({best[metric_name]:.1%})")


if __name__ == "__main__":
    # Example usage
    from config import configure_ollama
    from modules import SimpleTicketClassifier
    from data import valset
    from metrics import exact_match_metric, partial_match_metric

    print("Testing evaluation functions...\n")

    # Configure DSPy
    configure_ollama()

    # Create a module
    classifier = SimpleTicketClassifier()

    # Evaluate on a small subset
    print("Evaluating on 3 examples:")
    score = evaluate_module(
        classifier,
        valset[:3],
        exact_match_metric,
        verbose=True
    )
    print(f"\nAverage score: {score:.2%}")
