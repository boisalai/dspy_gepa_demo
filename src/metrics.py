"""
Evaluation metrics for IT ticket classification.

This module provides various metrics to evaluate model performance:
- Exact match: Both category and priority must be correct
- Partial match: Gives partial credit if one field is correct
"""


def exact_match_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Strict metric: 1 if both category AND priority are correct, 0 otherwise.

    Args:
        example: Ground truth example with 'category' and 'priority'
        prediction: Model prediction with category and priority
        trace: Optional trace information (for GEPA compatibility)
        pred_name: Optional predictor name (for GEPA compatibility)
        pred_trace: Optional prediction trace (for GEPA compatibility)

    Returns:
        float: 1.0 if exact match, 0.0 otherwise
    """
    # Normalize strings (lowercase, no spaces)
    pred_category = prediction.category.strip().lower()
    true_category = example['category'].strip().lower() if isinstance(example, dict) else example.category.strip().lower()

    pred_priority = prediction.priority.strip().lower()
    true_priority = example['priority'].strip().lower() if isinstance(example, dict) else example.priority.strip().lower()

    # Both must be correct
    if pred_category == true_category and pred_priority == true_priority:
        return 1.0
    else:
        return 0.0


def partial_match_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Nuanced metric with partial credit:
    - 1.0: Both correct
    - 0.7: Category correct only (category is more important)
    - 0.5: Priority correct only
    - 0.0: Neither correct

    Args:
        example: Ground truth example
        prediction: Model prediction
        trace: Optional trace information (for GEPA compatibility)
        pred_name: Optional predictor name (for GEPA compatibility)
        pred_trace: Optional prediction trace (for GEPA compatibility)

    Returns:
        float: Score between 0.0 and 1.0
    """
    # Normalize strings
    pred_category = prediction.category.strip().lower()
    true_category = example['category'].strip().lower() if isinstance(example, dict) else example.category.strip().lower()

    pred_priority = prediction.priority.strip().lower()
    true_priority = example['priority'].strip().lower() if isinstance(example, dict) else example.priority.strip().lower()

    category_match = (pred_category == true_category)
    priority_match = (pred_priority == true_priority)

    if category_match and priority_match:
        return 1.0
    elif category_match:
        return 0.7  # Category is more important
    elif priority_match:
        return 0.5
    else:
        return 0.0


def category_only_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Metric that only considers category accuracy.

    Args:
        example: Ground truth example
        prediction: Model prediction
        trace: Optional trace information
        pred_name: Optional predictor name
        pred_trace: Optional prediction trace

    Returns:
        float: 1.0 if category correct, 0.0 otherwise
    """
    pred_category = prediction.category.strip().lower()
    true_category = example['category'].strip().lower() if isinstance(example, dict) else example.category.strip().lower()

    return 1.0 if pred_category == true_category else 0.0


def priority_only_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Metric that only considers priority accuracy.

    Args:
        example: Ground truth example
        prediction: Model prediction
        trace: Optional trace information
        pred_name: Optional predictor name
        pred_trace: Optional prediction trace

    Returns:
        float: 1.0 if priority correct, 0.0 otherwise
    """
    pred_priority = prediction.priority.strip().lower()
    true_priority = example['priority'].strip().lower() if isinstance(example, dict) else example.priority.strip().lower()

    return 1.0 if pred_priority == true_priority else 0.0


if __name__ == "__main__":
    # Test metrics
    import dspy

    print("Testing metrics...\n")

    # Mock example and predictions
    example = {"ticket": "Test", "category": "Hardware", "priority": "High"}

    pred_perfect = dspy.Prediction(category="Hardware", priority="High")
    pred_cat_only = dspy.Prediction(category="Hardware", priority="Low")
    pred_pri_only = dspy.Prediction(category="Software", priority="High")
    pred_wrong = dspy.Prediction(category="Software", priority="Low")

    print("Test 1: Perfect match")
    print(f"  Exact match: {exact_match_metric(example, pred_perfect)}")
    print(f"  Partial match: {partial_match_metric(example, pred_perfect)}")

    print("\nTest 2: Category correct only")
    print(f"  Exact match: {exact_match_metric(example, pred_cat_only)}")
    print(f"  Partial match: {partial_match_metric(example, pred_cat_only)}")

    print("\nTest 3: Priority correct only")
    print(f"  Exact match: {exact_match_metric(example, pred_pri_only)}")
    print(f"  Partial match: {partial_match_metric(example, pred_pri_only)}")

    print("\nTest 4: Both wrong")
    print(f"  Exact match: {exact_match_metric(example, pred_wrong)}")
    print(f"  Partial match: {partial_match_metric(example, pred_wrong)}")
