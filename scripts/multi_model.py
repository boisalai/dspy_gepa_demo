"""
Multi-model configuration and hybrid architectures.

This module provides utilities for:
- Configuring multiple language models (Ollama, OpenAI, Anthropic)
- Benchmarking different models
- Creating hybrid architectures that use different models for different tasks
"""

import dspy
import time
import os
from typing import List, Callable, Dict
from signatures import CategoryClassifier, PriorityClassifier


def configure_ollama_models() -> Dict[str, dspy.LM]:
    """
    Configure multiple Ollama models.

    Returns:
        Dictionary of {model_name: language_model}
    """
    models = {}

    # Llama 3.1 (8B)
    models['llama3.1:8b'] = dspy.LM(
        model='ollama_chat/llama3.1:8b',
        api_base='http://localhost:11434',
        temperature=0.3
    )

    # Mistral (7B)
    models['mistral:7b'] = dspy.LM(
        model='ollama_chat/mistral:7b',
        api_base='http://localhost:11434',
        temperature=0.3
    )

    # Qwen 2.5 (7B)
    models['qwen2.5:7b'] = dspy.LM(
        model='ollama_chat/qwen2.5:7b',
        api_base='http://localhost:11434',
        temperature=0.3
    )

    print(f"✅ Configured {len(models)} Ollama models:")
    for name in models.keys():
        print(f"   - {name}")

    return models


def configure_openai_models() -> Dict[str, dspy.LM]:
    """
    Configure OpenAI models.

    Requires OPENAI_API_KEY environment variable.

    Returns:
        Dictionary of {model_name: language_model}
    """
    if not os.getenv('OPENAI_API_KEY'):
        print("⚠️ OPENAI_API_KEY not set - OpenAI models unavailable")
        return {}

    models = {}

    # GPT-4o-mini
    models['gpt-4o-mini'] = dspy.LM(
        model='openai/gpt-4o-mini',
        temperature=0.3
    )

    # GPT-4o
    models['gpt-4o'] = dspy.LM(
        model='openai/gpt-4o',
        temperature=0.3
    )

    print(f"✅ Configured {len(models)} OpenAI models:")
    for name in models.keys():
        print(f"   - {name}")

    return models


def configure_anthropic_models() -> Dict[str, dspy.LM]:
    """
    Configure Anthropic Claude models.

    Requires ANTHROPIC_API_KEY environment variable.

    Returns:
        Dictionary of {model_name: language_model}
    """
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("⚠️ ANTHROPIC_API_KEY not set - Anthropic models unavailable")
        return {}

    models = {}

    # Claude 3.5 Haiku
    models['claude-3-5-haiku'] = dspy.LM(
        model='anthropic/claude-3-5-haiku-20241022',
        temperature=0.3
    )

    # Claude 3.5 Sonnet
    models['claude-3-5-sonnet'] = dspy.LM(
        model='anthropic/claude-3-5-sonnet-20241022',
        temperature=0.3
    )

    print(f"✅ Configured {len(models)} Anthropic models:")
    for name in models.keys():
        print(f"   - {name}")

    return models


def benchmark_model(
    lm: dspy.LM,
    model_name: str,
    examples: List,
    metric: Callable
) -> Dict:
    """
    Benchmark a model on a set of examples.

    Args:
        lm: The language model to test
        model_name: Name of the model (for display)
        examples: Test examples
        metric: Evaluation metric

    Returns:
        Dictionary with score and execution time
    """
    from modules import SimpleTicketClassifier

    # Configure DSPy with this model
    dspy.configure(lm=lm)

    # Create classifier
    classifier = SimpleTicketClassifier()

    # Measure time
    start_time = time.time()

    # Evaluate
    total_score = 0
    for example in examples:
        ticket = example['ticket'] if isinstance(example, dict) else example.ticket
        prediction = classifier(ticket=ticket)
        score = metric(example, prediction)
        total_score += score

    end_time = time.time()

    # Calculate results
    avg_score = total_score / len(examples)
    elapsed_time = end_time - start_time

    return {
        'model': model_name,
        'score': avg_score,
        'time': elapsed_time
    }


def benchmark_multiple_models(
    models: Dict[str, dspy.LM],
    examples: List,
    metric: Callable
) -> List[Dict]:
    """
    Benchmark multiple models and compare results.

    Args:
        models: Dictionary of {model_name: language_model}
        examples: Test examples
        metric: Evaluation metric

    Returns:
        List of result dictionaries, sorted by score (descending)
    """
    results = []

    print(f"🔍 Benchmarking {len(models)} models...")
    print(f"⏰ This will take a few minutes\n")

    for i, (name, lm) in enumerate(models.items(), 1):
        print(f"{i}/{len(models)} Evaluating {name}...")

        result = benchmark_model(lm, name, examples, metric)
        results.append(result)

        print(f"   Score: {result['score']:.2%} | Time: {result['time']:.1f}s\n")

    # Sort by score (descending)
    results.sort(key=lambda x: x['score'], reverse=True)

    # Print summary
    print("=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    for r in results:
        print(f"{r['model']:20} | Score: {r['score']:6.2%} | Time: {r['time']:5.1f}s")
    print("=" * 60)

    return results


class HybridTicketClassifier(dspy.Module):
    """
    Hybrid classifier using different models for different tasks.

    Example: Fast model for category, accurate model for priority
    """

    def __init__(self, fast_lm: dspy.LM, accurate_lm: dspy.LM):
        """
        Args:
            fast_lm: Fast model for category prediction
            accurate_lm: Accurate model for priority prediction
        """
        super().__init__()
        self.fast_lm = fast_lm
        self.accurate_lm = accurate_lm
        self.category_signature = CategoryClassifier
        self.priority_signature = PriorityClassifier

    def forward(self, ticket):
        # Step 1: Categorization with fast model
        with dspy.settings.context(lm=self.fast_lm):
            category_predictor = dspy.ChainOfThought(self.category_signature)
            category_result = category_predictor(ticket=ticket)

        # Step 2: Prioritization with accurate model
        with dspy.settings.context(lm=self.accurate_lm):
            priority_predictor = dspy.ChainOfThought(self.priority_signature)
            priority_result = priority_predictor(
                ticket=ticket,
                category=category_result.category
            )

        return dspy.Prediction(
            category=category_result.category,
            priority=priority_result.priority
        )


if __name__ == "__main__":
    # Example usage
    from data import valset
    from metrics import exact_match_metric

    print("Testing multi-model configuration...\n")

    # Configure Ollama models
    ollama_models = configure_ollama_models()

    # Benchmark on a few examples
    if ollama_models:
        print("\n" + "=" * 60)
        print("Running benchmarks on 3 validation examples")
        print("=" * 60 + "\n")

        results = benchmark_multiple_models(
            ollama_models,
            valset[:3],
            exact_match_metric
        )

        # Test hybrid classifier
        if len(ollama_models) >= 2:
            print("\n🔀 Testing hybrid classifier")
            models_list = list(ollama_models.values())
            hybrid = HybridTicketClassifier(
                fast_lm=models_list[1],  # Second model as "fast"
                accurate_lm=models_list[0]  # First model as "accurate"
            )

            test_ticket = "Mon ordinateur ne démarre plus"
            result = hybrid(ticket=test_ticket)
            print(f"   Ticket: {test_ticket}")
            print(f"   Category: {result.category}")
            print(f"   Priority: {result.priority}")
