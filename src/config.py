"""
Configuration module for DSPy with Ollama.

This module handles the initial setup of DSPy, including:
- Library imports
- Language model configuration
- Global DSPy settings
"""

import dspy
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def configure_ollama(
    model: str = 'ollama_chat/llama3.1:8b',
    api_base: str = 'http://localhost:11434',
    temperature: float = 0.3
) -> dspy.LM:
    """
    Configure DSPy to use Ollama with a specified model.

    Args:
        model: The Ollama model to use (e.g., 'ollama_chat/llama3.1:8b')
        api_base: The base URL for the Ollama API
        temperature: Temperature for generation (0.0-2.0, lower is more deterministic)

    Returns:
        dspy.LM: Configured language model
    """
    print(f"🚀 Configuring DSPy with Ollama...")

    # Configure the language model
    lm = dspy.LM(
        model=model,
        api_base=api_base,
        temperature=temperature
    )

    # Set DSPy global configuration
    dspy.configure(lm=lm)

    print(f"✅ DSPy configured with {model} (temp={temperature})")
    return lm


def configure_openai(
    model: str = 'openai/gpt-4o-mini',
    temperature: float = 0.3
) -> dspy.LM:
    """
    Configure DSPy to use OpenAI models.

    Requires OPENAI_API_KEY environment variable to be set.

    Args:
        model: The OpenAI model to use
        temperature: Temperature for generation

    Returns:
        dspy.LM: Configured language model
    """
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY environment variable not set")

    lm = dspy.LM(
        model=model,
        temperature=temperature
    )

    dspy.configure(lm=lm)
    print(f"✅ DSPy configured with {model}")
    return lm


def configure_anthropic(
    model: str = 'anthropic/claude-3-5-haiku-20241022',
    temperature: float = 0.3
) -> dspy.LM:
    """
    Configure DSPy to use Anthropic Claude models.

    Requires ANTHROPIC_API_KEY environment variable to be set.

    Args:
        model: The Anthropic model to use
        temperature: Temperature for generation

    Returns:
        dspy.LM: Configured language model
    """
    if not os.getenv('ANTHROPIC_API_KEY'):
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    lm = dspy.LM(
        model=model,
        temperature=temperature
    )

    dspy.configure(lm=lm)
    print(f"✅ DSPy configured with {model}")
    return lm


def configure_reflection_lm(
    model: str = 'ollama_chat/llama3.1:8b',
    api_base: str = 'http://localhost:11434',
    temperature: float = 1.0,
    max_tokens: int = 8000
) -> dspy.LM:
    """
    Configure a reflection language model for GEPA optimization.

    The reflection LM is used by GEPA to analyze errors and propose improvements.
    It should have:
    - High temperature (0.8-1.2) for creativity
    - High max_tokens (6000-10000) for detailed analysis

    Args:
        model: The model to use for reflection
        api_base: API base URL (for Ollama)
        temperature: Temperature (higher = more creative)
        max_tokens: Maximum tokens for analysis

    Returns:
        dspy.LM: Configured reflection language model
    """
    reflection_lm = dspy.LM(
        model=model,
        api_base=api_base,
        temperature=temperature,
        max_tokens=max_tokens
    )

    print(f"✅ Reflection LM configured: {model} (temp={temperature}, max_tokens={max_tokens})")
    return reflection_lm


if __name__ == "__main__":
    # Example usage
    print("Testing configuration...")

    # Configure Ollama
    lm = configure_ollama()

    # Test simple completion
    response = lm("Say hello in French:")
    print(f"\nTest response: {response}")
