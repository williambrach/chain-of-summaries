# Chain-of-Summaries

[![PyPI version](https://badge.fury.io/py/chain-of-summaries.svg)](https://badge.fury.io/py/chain-of-summaries)

> A Python package improving /llms.txt format via iterative summarization.

## Features

- **LLM Integration**: Works with various language models through LiteLLM integration (gpt-4o-mini, gwen-2.5:7b, llama3.2:3b, llama3.3:70b, qwq)
- **Data Processing**: Tools for creating and manipulating llms.txt format
- **Summary Generation with Iterative Improvement**: Create concise summaries using various models. Refine summaries through multiple training iterations
- **QA Generation**: Automatically generate question-answer pairs for evaluation


## Installation

```bash
pip install chain-of-summaries
```

## Quick Start

```python
from chain_of_summaries import LLMSProcessor, build_llms_txt_file, load_llms_txt_file

# Initialize the processor with your preferred LLM
processor = LLMSProcessor(
    model="azure/gpt-4o-mini",  # Supports various models via LiteLLM
    api_key="your_api_key",     # Your API key
    base_url="your_base_url"    # Optional base URL
)

# Or create new data for llms.txt
sites = [
    {"url": "https://example.com", "content": "Example content", "title": "Example Site", "file_name": "example.txt"}
]

data = processor.create_llms_txt_data(
    title="My Knowledge Base",
    sites=sites,
    summary_model="azure/gpt-4o-mini"  # Model for generating summaries
)

# Generate llms.txt file
llms_txt_content = build_llms_txt_file(
    data=data,
    item_max_tokens=512  # Optional: limit tokens per item
)

# Generate QA pairs for evaluation
questions = processor.generate_qa_pairs(
    sites=data["sites"],
    num_questions=50
)

# Improve summaries through iterative training
improved_data, data_iterations, train_results, eval_results = processor.improve_llms_txt(
    data=data,
    train_questions=questions,
    eval_questions=questions,
    iterations=10
)

# Generate updated llms.txt with improved summaries
updated_llms_txt = build_llms_txt_file(data=improved_data)
```

## Detailed Usage

### LLMSProcessor

The main class that handles processing of language model data.

```python
processor = LLMSProcessor(
    model="azure/gpt-4o-mini",  # Model to use (supports LiteLLM format)
    api_key="your_api_key",     # API key for the model
    base_url="your_base_url",   # Base URL for API (optional)
    temperature=0,              # Temperature for generation (optional)
    max_tokens=8192,            # Maximum tokens for generation (optional)
    cache=True,                 # Enable caching (optional)
    num_threads=20,             # Number of threads for parallel processing (optional)
    display_progress=True,      # Display progress bar (optional)
    device="cpu",               # Device to use (cpu/gpu) (optional)
    suprisal_model="HuggingFaceTB/SmolLM2-135M"  # Model for surprisal calculation (optional)
)
```

### Creating llms.txt Data

```python
data = processor.create_llms_txt_data(
    title="My Knowledge Base",
    sites=sites,                  # List of site dictionaries
    summary_model="azure/gpt-4o-mini",  # Model for summarization
    iteration=0,                  # Iteration number (optional)
    bert_score=False,             # Calculate BERT score (optional)
    suprisal_score=False          # Calculate surprisal score (optional)
)
```

### Building llms.txt File

```python
llms_txt_content = build_llms_txt_file(
    data=data,
    item_max_tokens=None,        # Maximum tokens per item (optional)
    description=True,            # Include descriptions (optional)
    output_file="llms.txt"       # Output file path (optional)
)
```

### Generating QA Pairs

```python
questions = processor.generate_qa_pairs(
    sites=data["sites"],
    num_questions=50,
    model="azure/gpt-4o-mini"  # Optional: specify model
)
```

### Improving llms.txt Through Iterations

```python
improved_data, data_iterations, train_results, eval_results = processor.improve_llms_txt(
    data=data,
    train_questions=questions,
    eval_questions=questions,
    iterations=10,
    model="azure/gpt-4o-mini"  # Optional: specify model
)
```

## Analyzing Results

```python
# View training performance by iteration
train_performance = train_results.groupby("iteration")[["correct", "correct_f1"]].mean()

# View evaluation performance by iteration
eval_performance = eval_results.groupby("iteration")[["correct", "correct_f1"]].mean()
```

## File Format

The llms.txt format follows this structure:

```
# Title

> Optional description/summary

## Site 1 Title

- [Site URL](https://example.com): Site description/summary

## Site 2 Title

- [Site URL](https://another-example.com): Another site description/summary
```

## License

TODO

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

```bibtex
```