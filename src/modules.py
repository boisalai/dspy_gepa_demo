"""
DSPy Modules for IT ticket classification.

This module contains various DSPy module implementations:
- Simple modules (Predict, ChainOfThought)
- Composed modules (Sequential, Validated, Ensemble)
"""

import dspy
from typing import List
from collections import Counter
from signatures import TicketClassifier, CategoryClassifier, PriorityClassifier
from data import CATEGORIES, PRIORITIES


class SimpleTicketClassifier(dspy.Module):
    """
    Simple ticket classifier using ChainOfThought.

    This is the baseline module used throughout the tutorial.
    """

    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(TicketClassifier)

    def forward(self, ticket):
        result = self.classifier(ticket=ticket)
        return dspy.Prediction(
            category=result.category,
            priority=result.priority
        )


class SequentialClassifier(dspy.Module):
    """
    Sequential classifier that first determines category, then priority.

    This demonstrates module composition where the output of one module
    feeds into the input of another.
    """

    def __init__(self):
        super().__init__()
        self.category_predictor = dspy.ChainOfThought(CategoryClassifier)
        self.priority_predictor = dspy.ChainOfThought(PriorityClassifier)

    def forward(self, ticket):
        # Step 1: Predict category
        category_result = self.category_predictor(ticket=ticket)

        # Step 2: Predict priority using the category
        priority_result = self.priority_predictor(
            ticket=ticket,
            category=category_result.category
        )

        return dspy.Prediction(
            category=category_result.category,
            priority=priority_result.priority
        )


class ValidatedClassifier(dspy.Module):
    """
    Classifier with output validation.

    Validates that outputs are within the expected categories and priorities.
    Corrects invalid outputs automatically.
    """

    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(TicketClassifier)
        self.valid_categories = set(cat.lower() for cat in CATEGORIES)
        self.valid_priorities = set(pri.lower() for pri in PRIORITIES)

    def forward(self, ticket):
        # Get prediction
        result = self.classifier(ticket=ticket)

        # Validate category
        category = result.category.strip()
        if category.lower() not in self.valid_categories:
            print(f"⚠️ Invalid category '{category}', correcting...")
            category = "Application"  # Default fallback

        # Validate priority
        priority = result.priority.strip()
        if priority.lower() not in self.valid_priorities:
            print(f"⚠️ Invalid priority '{priority}', correcting...")
            priority = "Medium"  # Default fallback

        return dspy.Prediction(
            category=category,
            priority=priority
        )


class EnsembleClassifier(dspy.Module):
    """
    Ensemble classifier using majority voting.

    Combines predictions from multiple classifiers and returns
    the most common prediction (majority vote).
    """

    def __init__(self, n_models: int = 3):
        """
        Args:
            n_models: Number of classifier instances to use
        """
        super().__init__()
        self.classifiers = [
            dspy.ChainOfThought(TicketClassifier)
            for _ in range(n_models)
        ]

    def forward(self, ticket):
        # Collect predictions from all models
        categories = []
        priorities = []

        for classifier in self.classifiers:
            result = classifier(ticket=ticket)
            categories.append(result.category)
            priorities.append(result.priority)

        # Majority vote
        category_vote = Counter(categories).most_common(1)[0][0]
        priority_vote = Counter(priorities).most_common(1)[0][0]

        return dspy.Prediction(
            category=category_vote,
            priority=priority_vote
        )


if __name__ == "__main__":
    # Example usage
    from config import configure_ollama

    print("Testing modules...")

    # Configure DSPy
    configure_ollama()

    # Test ticket
    test_ticket = "Mon ordinateur ne démarre plus. J'ai une présentation dans 1 heure."

    # Test SimpleTicketClassifier
    print("\n1. SimpleTicketClassifier:")
    simple = SimpleTicketClassifier()
    result = simple(ticket=test_ticket)
    print(f"   Category: {result.category}")
    print(f"   Priority: {result.priority}")

    # Test SequentialClassifier
    print("\n2. SequentialClassifier:")
    sequential = SequentialClassifier()
    result = sequential(ticket=test_ticket)
    print(f"   Category: {result.category}")
    print(f"   Priority: {result.priority}")

    # Test ValidatedClassifier
    print("\n3. ValidatedClassifier:")
    validated = ValidatedClassifier()
    result = validated(ticket=test_ticket)
    print(f"   Category: {result.category}")
    print(f"   Priority: {result.priority}")
