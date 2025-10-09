# DSPy and GEPA Tutorial: Automated IT Ticket Classification

This tutorial demonstrates how to build and optimize an automated IT ticket classification system using DSPy (Declarative Self-improving Language Programs) and GEPA (Genetic-Pareto Algorithm) with local language models via Ollama.

## Overview

This project implements a complete workflow for:
- Classifying IT support tickets by category (Hardware, Software, Network, Account)
- Assigning priority levels (Low, Medium, High, Urgent, Critical)
- Optimizing prompt performance automatically using GEPA
- Switching between different language model providers

All components can run entirely locally using open-source models through Ollama, requiring no API keys or external services.

## Prerequisites

- macOS, Linux, or Windows with WSL
- Python 3.9 or higher
- At least 8GB of RAM (16GB recommended for larger models)
- Ollama installed and running

## Installation

### Step 1: Install Ollama

```bash
# On macOS
brew install ollama

# Start Ollama service
ollama serve
```

In a separate terminal, download a language model:

```bash
# Recommended: Llama 3.1 8B (4.7 GB)
ollama pull llama3.1:8b

# Alternatives:
ollama pull mistral:7b      # Faster, 4.1 GB
ollama pull qwen2.5:7b      # Higher quality, 4.7 GB
```

### Step 2: Install Python Dependencies

```bash
pip install dspy-ai
```

Note: GEPA is now integrated into DSPy 3.0+ and requires no separate installation.

## Tutorial Structure

This tutorial uses four main scripts, each building on previous concepts:

1. **main_simple.py** - Introduction to DSPy fundamentals
2. **main.py** - Advanced features with ChainOfThought
3. **advanced_examples.py** - Multi-model workflows
4. **gepa_guide.py** - Automatic prompt optimization

## Part 1: DSPy Fundamentals (main_simple.py)

### Concepts Covered

- DSPy configuration with Ollama
- Signature definitions (input/output specification)
- Module creation and execution
- Basic evaluation metrics

### Running the Example

```bash
python main_simple.py
```

### What This Script Demonstrates

1. **Configuration**: Connecting DSPy to a local Ollama instance
2. **Signature**: Defining task inputs and outputs declaratively
3. **Module**: Creating a reusable prediction component
4. **Evaluation**: Measuring accuracy on validation data

### Expected Output

```
Classification Results:
  Category accuracy: 70-85%
  Priority accuracy: 60-75%
  Combined accuracy: 50-65%
```

### Key Code Patterns

The script demonstrates the minimal DSPy workflow:

```python
# 1. Configure language model
lm = dspy.LM(model='ollama_chat/llama3.1:8b',
             api_base='http://localhost:11434')
dspy.configure(lm=lm)

# 2. Define signature
class TicketClassifier(dspy.Signature):
    """Classify an IT ticket by category and priority"""
    ticket = dspy.InputField()
    category = dspy.OutputField()
    priority = dspy.OutputField()

# 3. Create module
classifier = dspy.Predict(TicketClassifier)

# 4. Execute
result = classifier(ticket="Printer not working")
```

## Part 2: Advanced Features (main.py)

### Concepts Covered

- ChainOfThought reasoning
- Accessing intermediate reasoning steps
- Improved accuracy through explicit reasoning

### Running the Example

```bash
python main.py
```

### What ChainOfThought Adds

ChainOfThought instructs the model to explain its reasoning before providing an answer, typically improving accuracy by 5-15%.

### Key Differences from main_simple.py

```python
# Instead of:
classifier = dspy.Predict(TicketClassifier)

# Use:
classifier = dspy.ChainOfThought(TicketClassifier)
```

The model now generates intermediate reasoning that can be inspected for debugging or explanation purposes.

## Part 3: Multi-Model Workflows (advanced_examples.py)

### Concepts Covered

- Switching between different LLM providers
- Comparing model performance
- Multi-model architectures
- Cost-performance tradeoffs

### Running the Example

```bash
python advanced_examples.py
```

### Supported Providers

The script demonstrates integration with:

- **Ollama** (local, free): llama3.1, mistral, qwen2.5, and others
- **OpenAI** (API, paid): gpt-4, gpt-4o-mini
- **Anthropic** (API, paid): claude-3-5-sonnet, claude-3-haiku

### Switching Models

Changing models requires only modifying the LM configuration:

```python
# Local Ollama
lm = dspy.LM('ollama_chat/llama3.1:8b',
             api_base='http://localhost:11434')

# OpenAI
lm = dspy.LM('openai/gpt-4o-mini')

# Anthropic
lm = dspy.LM('anthropic/claude-3-5-sonnet-20241022')
```

All other code remains unchanged.

### Benchmarking Multiple Models

The script includes a benchmark function to compare models:

```python
models = [
    {'type': 'ollama', 'name': 'llama3.1:8b'},
    {'type': 'ollama', 'name': 'mistral:7b'},
    {'type': 'ollama', 'name': 'qwen2.5:7b'},
]
benchmark_models(models)
```

Expected results show accuracy and execution time for each model, enabling data-driven model selection.

## Part 4: Automatic Optimization with GEPA (gepa_guide.py)

### Concepts Covered

- Automatic prompt optimization
- Genetic algorithm-based improvement
- Reflection-based error analysis
- Performance measurement

### Running the Example

```bash
uv run gepa_guide.py
```

### What GEPA Does

GEPA (Genetic-Pareto Algorithm) automatically improves prompt performance through:

1. **Variation Generation**: Creating multiple prompt variants
2. **Evaluation**: Testing each variant on training data
3. **Reflection**: Using an LLM to analyze failures and suggest improvements
4. **Iteration**: Repeating until convergence or budget exhaustion

### Menu Options

```
1. Basic optimization (recommended)
   - Quick optimization with auto='light'
   - 5-10 minutes execution time
   - Typical improvement: 10-20%

2. Advanced optimization
   - Deeper optimization with auto='medium'
   - 10-20 minutes execution time
   - Maximum performance gains

3. Display optimization tips
```

### API Changes in DSPy 3.0+

GEPA now requires a reflection language model:

```python
# Required reflection model configuration
reflection_lm = dspy.LM(
    model='ollama_chat/llama3.1:8b',
    api_base='http://localhost:11434',
    temperature=1.0,
    max_tokens=8000
)

# GEPA optimizer configuration
optimizer = GEPA(
    metric=your_metric_function,
    auto='light',  # or 'medium', 'heavy'
    reflection_lm=reflection_lm  # Required
)
```

### Metric Function Requirements

Metrics must accept the updated signature:

```python
def metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    """
    Args:
        example: Ground truth example
        prediction: Model prediction
        trace: Optional execution trace
        pred_name: Optional predictor name (GEPA-specific)
        pred_trace: Optional predictor trace (GEPA-specific)

    Returns:
        float: Score between 0.0 and 1.0
    """
    category_match = prediction.category == example.category
    priority_match = prediction.priority == example.priority
    return 1.0 if (category_match and priority_match) else 0.0
```

### Expected Results

```
Before optimization:  42.86%
After optimization:   57.14%
Improvement:         +33.3%
```

Actual improvements vary based on task complexity and data quality.

## Diagnostics and Troubleshooting

### GEPA Diagnostics

If you encounter errors with GEPA:

```bash
uv run diagnose_gepa.py
```

This script:
- Verifies DSPy installation and version
- Detects GEPA API parameters
- Tests different configurations
- Provides specific recommendations

### Common Issues

**Issue**: `TypeError: GEPA.__init__() got an unexpected keyword argument 'breadth'`

**Cause**: Using deprecated API parameters with DSPy 3.0+

**Solution**: Update to use `auto` parameter instead of `breadth/depth`. See [GEPA_API_CHANGES.md](GEPA_API_CHANGES.md) for migration guide.

## Project Structure

```
dspy_gepa_demo/
├── main_simple.py          # Tutorial Part 1: DSPy basics
├── main.py                 # Tutorial Part 2: ChainOfThought
├── advanced_examples.py    # Tutorial Part 3: Multi-model workflows
├── gepa_guide.py          # Tutorial Part 4: GEPA optimization
├── diagnose_gepa.py       # GEPA diagnostics tool
├── data.py                # Training and validation datasets
├── GEPA_API_CHANGES.md    # GEPA API migration guide
├── GEPA_SOLUTION.md       # GEPA troubleshooting guide
└── README.md              # This file
```

## Adapting to Your Use Case

### Step 1: Prepare Your Data

Edit `data.py`:

```python
trainset = [
    {
        "input_field": "Your input text 1",
        "output_field": "Expected output 1"
    },
    # Minimum 15-20 examples recommended
]

valset = [
    # 5-10 validation examples
]
```

### Step 2: Define Your Signature

In your script:

```python
class YourSignature(dspy.Signature):
    """Clear description of your task"""
    input_field = dspy.InputField(desc="Input description")
    output_field = dspy.OutputField(desc="Output description")
```

### Step 3: Implement Your Metric

```python
def your_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    # Your evaluation logic
    return 1.0 if correct else 0.0
```

### Step 4: Run and Iterate

```bash
python main_simple.py  # Test basic functionality
python gepa_guide.py   # Optimize performance
```

## Technical Details

### DSPy Framework

DSPy is a framework for algorithmically optimizing language model prompts and weights. Key features:

- **Declarative programming**: Define what you want, not how to prompt
- **Automatic optimization**: GEPA and other optimizers improve performance automatically
- **Modular architecture**: Compose complex pipelines from simple components
- **Provider abstraction**: Switch between LLM providers without code changes

### GEPA Algorithm

GEPA uses genetic algorithms and LLM reflection to optimize prompts:

1. **Initial Population**: Generate prompt variants
2. **Evaluation**: Measure performance on training data
3. **Selection**: Keep high-performing variants (Pareto frontier)
4. **Reflection**: Analyze errors with LLM feedback
5. **Mutation**: Generate new variants based on feedback
6. **Iteration**: Repeat until convergence

GEPA typically requires 400-800 LLM calls, taking 5-20 minutes with Ollama.

### Performance Benchmarks

With Llama 3.1 8B on IT ticket classification:

- **Baseline (Predict)**: ~45-55% combined accuracy
- **ChainOfThought**: ~55-65% combined accuracy (+10%)
- **GEPA optimized**: ~65-75% combined accuracy (+10-20%)

Results vary significantly based on:
- Task complexity
- Training data quality and quantity
- Model selection
- Metric definition

## Advanced Configurations

### Custom GEPA Budget

Control optimization depth:

```python
optimizer = GEPA(
    metric=your_metric,
    auto='light',           # Quick optimization
    # or
    max_full_evals=20,      # Precise control
    reflection_lm=reflection_lm
)
```

### Multi-Stage Pipelines

Combine multiple modules:

```python
class Pipeline(dspy.Module):
    def __init__(self):
        self.categorize = dspy.ChainOfThought(CategorySignature)
        self.prioritize = dspy.ChainOfThought(PrioritySignature)

    def forward(self, ticket):
        category = self.categorize(ticket=ticket)
        priority = self.prioritize(ticket=ticket, category=category.category)
        return dspy.Prediction(category=category.category, priority=priority.priority)
```

### Hybrid Model Architectures

Use different models for different tasks:

```python
# Fast model for categorization
fast_lm = dspy.LM('ollama_chat/mistral:7b')

# Accurate model for priority
accurate_lm = dspy.LM('ollama_chat/llama3.1:8b')

# Configure separately in your pipeline
```

## Resources

### Documentation
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Ollama Documentation](https://ollama.ai/docs)
- [GEPA Paper (arXiv)](https://arxiv.org/abs/2507.19457)

### Model Libraries
- [Ollama Model Library](https://ollama.ai/library)
- [Hugging Face Model Hub](https://huggingface.co/models)

### Related Projects
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [GEPA GitHub Repository](https://github.com/gepa-ai/gepa)

## License and Usage

All components are open source and commercially usable:

- DSPy: MIT License
- GEPA: Apache 2.0 License
- Ollama: MIT License
- Llama 3.1: Meta Community License
- Mistral: Apache 2.0 License

This tutorial code is provided as-is for educational and commercial use.

## Next Steps

After completing this tutorial:

1. Experiment with different models to find optimal cost-performance tradeoffs
2. Apply GEPA to your own tasks and datasets
3. Explore DSPy's other optimizers (BootstrapFewShot, MIPRO, etc.)
4. Build production pipelines with multi-model architectures
5. Contribute improvements back to the DSPy community

For questions or issues, consult the DSPy documentation or GEPA troubleshooting guide included in this repository.
