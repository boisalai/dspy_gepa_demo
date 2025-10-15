"""
Advanced patterns for production-ready DSPy modules.

This module implements several patterns to make DSPy modules more robust:
- Validation: Verify and correct invalid outputs
- Retry: Automatically retry on errors
- Fallback: Use backup model if primary fails
- Ensemble: Combine multiple predictions
"""

import dspy
import time
from typing import List, Tuple
from collections import Counter
from signatures import TicketClassifier
from data import CATEGORIES, PRIORITIES


class ValidatedTicketClassifier(dspy.Module):
    """
    Classifier with output validation.

    Validates that outputs are within expected categories and priorities.
    Automatically corrects invalid outputs.
    """

    def __init__(self):
        super().__init__()
        self.classifier = dspy.ChainOfThought(TicketClassifier)
        self.valid_categories = set(cat.lower() for cat in CATEGORIES)
        self.valid_priorities = set(pri.lower() for pri in PRIORITIES)

    def validate_and_correct(self, category: str, priority: str) -> Tuple[str, str, bool]:
        """
        Validate and correct outputs if necessary.

        Returns:
            (category, priority, is_valid): Corrected values and validation status
        """
        category_lower = category.strip().lower()
        priority_lower = priority.strip().lower()

        is_valid = True

        # Validate category
        if category_lower not in self.valid_categories:
            # Try fuzzy matching
            if 'hard' in category_lower or 'matér' in category_lower:
                category = 'Hardware'
            elif 'soft' in category_lower or 'logic' in category_lower:
                category = 'Software'
            elif 'réseau' in category_lower or 'network' in category_lower:
                category = 'Network'
            elif 'compte' in category_lower or 'account' in category_lower:
                category = 'Account'
            else:
                category = 'Application'  # Default
                is_valid = False
        else:
            # Normalize casing
            category = next(c for c in CATEGORIES if c.lower() == category_lower)

        # Validate priority
        if priority_lower not in self.valid_priorities:
            # Try fuzzy matching
            if 'critic' in priority_lower or 'critique' in priority_lower:
                priority = 'Critical'
            elif 'urgent' in priority_lower:
                priority = 'Urgent'
            elif 'high' in priority_lower or 'haut' in priority_lower:
                priority = 'High'
            elif 'medium' in priority_lower or 'moyen' in priority_lower:
                priority = 'Medium'
            else:
                priority = 'Low'  # Default
                is_valid = False
        else:
            # Normalize casing
            priority = next(p for p in PRIORITIES if p.lower() == priority_lower)

        return category, priority, is_valid

    def forward(self, ticket):
        # Get prediction
        result = self.classifier(ticket=ticket)

        # Validate and correct
        category, priority, is_valid = self.validate_and_correct(
            result.category,
            result.priority
        )

        return dspy.Prediction(
            category=category,
            priority=priority,
            is_valid=is_valid
        )


class RetryTicketClassifier(dspy.Module):
    """
    Classifier with retry logic for handling errors.

    Automatically retries on failure with exponential backoff.
    """

    def __init__(self, max_retries: int = 3, initial_delay: float = 1.0):
        """
        Args:
            max_retries: Maximum number of retry attempts
            initial_delay: Initial delay in seconds (doubles on each retry)
        """
        super().__init__()
        self.classifier = dspy.ChainOfThought(TicketClassifier)
        self.max_retries = max_retries
        self.initial_delay = initial_delay

    def forward(self, ticket):
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Attempt prediction
                result = self.classifier(ticket=ticket)

                # Success - return result
                return dspy.Prediction(
                    category=result.category,
                    priority=result.priority,
                    attempts=attempt + 1
                )

            except Exception as e:
                last_error = e

                # If last attempt, raise error
                if attempt == self.max_retries - 1:
                    break

                # Calculate delay with exponential backoff
                delay = self.initial_delay * (2 ** attempt)

                print(f"⚠️ Attempt {attempt + 1} failed: {e}")
                print(f"   Retrying in {delay:.1f}s...")

                time.sleep(delay)

        # All attempts failed
        raise Exception(f"Failed after {self.max_retries} attempts: {last_error}")


class FallbackTicketClassifier(dspy.Module):
    """
    Classifier with fallback to a backup model.

    If the primary model fails, automatically switches to a fallback model.
    """

    def __init__(self, primary_lm: dspy.LM, fallback_lm: dspy.LM):
        """
        Args:
            primary_lm: Primary language model
            fallback_lm: Fallback language model
        """
        super().__init__()
        self.primary_lm = primary_lm
        self.fallback_lm = fallback_lm
        self.signature = TicketClassifier

    def forward(self, ticket):
        # Try primary model
        try:
            with dspy.settings.context(lm=self.primary_lm):
                predictor = dspy.ChainOfThought(self.signature)
                result = predictor(ticket=ticket)

                return dspy.Prediction(
                    category=result.category,
                    priority=result.priority,
                    model_used='primary'
                )

        except Exception as e:
            print(f"⚠️ Primary model failed: {e}")
            print(f"   Switching to fallback model...")

            # Fallback to backup model
            try:
                with dspy.settings.context(lm=self.fallback_lm):
                    predictor = dspy.ChainOfThought(self.signature)
                    result = predictor(ticket=ticket)

                    return dspy.Prediction(
                        category=result.category,
                        priority=result.priority,
                        model_used='fallback'
                    )

            except Exception as fallback_error:
                # Both models failed
                raise Exception(f"Both models failed. Fallback error: {fallback_error}")


class EnsembleTicketClassifier(dspy.Module):
    """
    Ensemble classifier using majority voting.

    Combines predictions from multiple models and returns the most common prediction.
    """

    def __init__(self, models: List[Tuple[dspy.LM, int]]):
        """
        Args:
            models: List of (language_model, weight) tuples
        """
        super().__init__()
        self.models = models
        self.signature = TicketClassifier

    def forward(self, ticket):
        predictions = []

        # Get predictions from each model
        for lm, weight in self.models:
            try:
                with dspy.settings.context(lm=lm):
                    predictor = dspy.ChainOfThought(self.signature)
                    result = predictor(ticket=ticket)

                    # Add prediction with its weight
                    for _ in range(weight):
                        predictions.append({
                            'category': result.category.strip().lower(),
                            'priority': result.priority.strip().lower()
                        })

            except Exception as e:
                print(f"⚠️ Error with a model: {e}")
                continue

        if not predictions:
            raise Exception("No model could make a prediction")

        # Majority vote for category
        categories = [p['category'] for p in predictions]
        category_counts = Counter(categories)
        winning_category = category_counts.most_common(1)[0][0]

        # Majority vote for priority
        priorities = [p['priority'] for p in predictions]
        priority_counts = Counter(priorities)
        winning_priority = priority_counts.most_common(1)[0][0]

        # Normalize results
        category = next((c for c in CATEGORIES if c.lower() == winning_category), winning_category)
        priority = next((p for p in PRIORITIES if p.lower() == winning_priority), winning_priority)

        # Calculate confidence (percentage of agreement)
        category_confidence = category_counts[winning_category] / len(predictions)
        priority_confidence = priority_counts[winning_priority] / len(predictions)

        return dspy.Prediction(
            category=category,
            priority=priority,
            category_confidence=category_confidence,
            priority_confidence=priority_confidence,
            num_models=len(self.models)
        )


if __name__ == "__main__":
    # Example usage
    from config import configure_ollama

    print("Testing advanced patterns...\n")

    # Configure DSPy
    lm = configure_ollama()

    # Test ValidatedTicketClassifier
    print("1. Testing ValidatedTicketClassifier:")
    validated = ValidatedTicketClassifier()
    test_ticket = "Mon imprimante ne fonctionne pas"
    result = validated(ticket=test_ticket)
    print(f"   Category: {result.category}")
    print(f"   Priority: {result.priority}")
    print(f"   Valid: {result.is_valid}")

    # Test RetryTicketClassifier
    print("\n2. Testing RetryTicketClassifier:")
    retry = RetryTicketClassifier(max_retries=2)
    result = retry(ticket=test_ticket)
    print(f"   Category: {result.category}")
    print(f"   Priority: {result.priority}")
    print(f"   Attempts: {result.attempts}")
