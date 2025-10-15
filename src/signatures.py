"""
DSPy Signatures module.

This module contains all signature definitions for the IT ticket classification task.
Signatures define the input-output contract for DSPy modules.
"""

import dspy
from data import CATEGORIES, PRIORITIES


class BasicSignature(dspy.Signature):
    """Classify an IT ticket."""

    ticket = dspy.InputField()
    category = dspy.OutputField()


class DescriptiveSignature(dspy.Signature):
    """Classify an IT support ticket by category."""

    ticket = dspy.InputField(desc="Description of the issue reported by the user")
    category = dspy.OutputField(desc="Technical category of the issue")


class ConstrainedSignature(dspy.Signature):
    """Classify an IT ticket by category and priority."""

    ticket = dspy.InputField(desc="IT support ticket description")
    category = dspy.OutputField(desc=f"Category among: {', '.join(CATEGORIES)}")
    priority = dspy.OutputField(desc=f"Priority among: {', '.join(PRIORITIES)}")


class ContextualSignature(dspy.Signature):
    """Classify a ticket considering user history."""

    ticket = dspy.InputField(desc="Current issue description")
    user_history = dspy.InputField(desc="User's previous ticket history")
    category = dspy.OutputField(desc=f"Category among: {', '.join(CATEGORIES)}")
    priority = dspy.OutputField(desc=f"Priority among: {', '.join(PRIORITIES)}")
    reasoning = dspy.OutputField(desc="Explanation of the decision")


class StructuredOutputSignature(dspy.Signature):
    """Analyze a ticket and produce a structured report."""

    ticket = dspy.InputField(desc="Ticket description")
    category = dspy.OutputField(desc="Technical category")
    priority = dspy.OutputField(desc="Priority level")
    estimated_time = dspy.OutputField(desc="Estimated resolution time in hours")
    required_skills = dspy.OutputField(desc="Required skills (comma-separated list)")


class TicketClassifier(dspy.Signature):
    """Classify an IT support ticket by category and priority."""

    ticket = dspy.InputField(desc="IT support ticket description")
    category = dspy.OutputField(desc=f"Category among: {', '.join(CATEGORIES)}")
    priority = dspy.OutputField(desc=f"Priority among: {', '.join(PRIORITIES)}")


class CategoryClassifier(dspy.Signature):
    """Determine the technical category of an IT ticket."""

    ticket = dspy.InputField(desc="Ticket description")
    category = dspy.OutputField(desc=f"Category among: {', '.join(CATEGORIES)}")


class PriorityClassifier(dspy.Signature):
    """Determine the priority of a ticket based on its category."""

    ticket = dspy.InputField(desc="Ticket description")
    category = dspy.InputField(desc="Technical category already identified")
    priority = dspy.OutputField(desc=f"Priority among: {', '.join(PRIORITIES)}")


if __name__ == "__main__":
    # Example usage of signatures
    print("Available signatures:")
    print("1. BasicSignature - Simple classification")
    print("2. DescriptiveSignature - With field descriptions")
    print("3. ConstrainedSignature - With output constraints")
    print("4. ContextualSignature - With additional context")
    print("5. StructuredOutputSignature - Multiple outputs")
    print("6. TicketClassifier - Main signature (recommended)")
    print("7. CategoryClassifier - Category only")
    print("8. PriorityClassifier - Priority with category context")
