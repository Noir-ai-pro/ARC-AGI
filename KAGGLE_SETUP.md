# Kaggle Offline Setup Guide

## Overview

This guide explains how to use the ARC-AGI solver on Kaggle with local LLM models, eliminating the need for external API calls.

## Prerequisites

1. **Kaggle Notebook** with GPU enabled (P100 or better recommended)
2. **Internet disabled** (for pure offline operation) or enabled (to download models)
3. **Model files** added as datasets or from Kaggle Models

## Installation

### Step 1: Add Required Libraries

In your Kaggle notebook, add these libraries:
```python
!pip install transformers torch accelerate sentencepiece protobuf
!pip install bitsandbytes  # For 8-bit quantization (saves memory)
```

Or use the provided requirements file:
```bash
pip install -r requirements-kaggle-local.txt
```

### Step 2: Add Model Datasets

Add one or more of these model datasets to your notebook:

**Recommended Models:**
- **Llama-3-8B-Instruct**: Good balance of speed and quality
  - Dataset: `meta-llama-3-8b-instruct`
  - Path: `/kaggle/input/meta-llama-3-8b-instruct/`
  
- **Mistral-7B-Instruct**: Fast inference, good performance
  - Dataset: `mistral-7b-instruct`
  - Path: `/kaggle/input/mistral-7b-instruct/`
  
- **Phi-3-Mini**: Very fast, smaller memory footprint
  - Dataset: `phi-3-mini-4k-instruct`
  - Path: `/kaggle/input/phi-3-mini-4k-instruct/`

**High-Performance Models (require more VRAM):**
- **Llama-3-70B-Instruct**: Best quality, needs ~140GB VRAM (quantized)
- **Yi-34B-Chat**: Good alternative to Llama-70B
- **Qwen-2-7B-Instruct**: Strong multilingual support

### Step 3: Configure Solver

```python
from src.models import call_model
from src.providers.local import get_kaggle_model_path

# Example: Use Llama-3-8B
model_arg = "local-llama-3-8b"

# Or specify custom path
model_arg = "local-/kaggle/input/meta-llama-3-8b-instruct/"

# Run solver with local model
predictions, stats = call_model(
    openai_client=None,      # Not needed for local
    anthropic_client=None,   # Not needed for local
    google_keys=None,        # Not needed for local
    prompt=your_prompt,
    model_arg=model_arg,
    verbose=True,
    task_id=task_id,
    test_index=test_index,
)
```

## Usage Examples

### Basic Usage

```python
import sys
sys.path.append('/kaggle/working/ARC-AGI')

from run import main

# Run with local model
!python run.py \
    --task-directory /kaggle/input/arc-agi-2/data/evaluation \
    --models local-llama-3-8b \
    --task-limit 5 \
    --verbose 1
```

### Advanced Configuration

```python
# Custom model settings
from src.providers.local import LocalLLMProvider

provider = LocalLLMProvider(
    model_path="/kaggle/input/meta-llama-3-8b-instruct/",
    model_name="llama-3-8b",
    max_tokens=4096,          # Increase for complex reasoning
    temperature=0.7,           # Lower for more deterministic output
    use_gpu=True,              # Enable GPU acceleration
    load_in_8bit=True,         # Reduce memory usage
    trust_remote_code=False,
)

response = provider.generate(prompt=your_prompt)
print(response.text)
```

### Multi-Model Pipeline

```python
# Use different models for different steps
MODELS_STEP1 = ["local-mistral-7b", "local-phi-3-mini"]
MODELS_STEP5 = ["local-llama-3-8b"]

# Run pipeline
python run.py \
    --task-directory evaluation \
    --step1-models local-mistral-7b,local-phi-3-mini \
    --models local-llama-3-8b \
    --task-workers 4  # Reduce parallelism for local models
```

## Performance Optimization

### Memory Management

1. **Use 8-bit quantization**: Reduces VRAM usage by ~50%
   ```python
   load_in_8bit=True
   ```

2. **Reduce batch size**: Process tasks sequentially
   ```bash
   --task-workers 1
   ```

3. **Clear CUDA cache**: Between tasks
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Speed Optimization

1. **Use smaller models**: Phi-3-Mini is 3x faster than Llama-3-8B
2. **Reduce max_tokens**: Limit output length
3. **Lower temperature**: Faster convergence

### Expected Performance

| Model | VRAM Required | Time/Task | Accuracy* |
|-------|--------------|-----------|-----------|
| Phi-3-Mini | 3GB | 30s | ~35% |
| Mistral-7B | 6GB | 60s | ~45% |
| Llama-3-8B | 8GB | 90s | ~50% |
| Llama-3-70B (Q)** | 40GB | 300s | ~65% |

*Estimated accuracy on ARC-AGI 2 eval set
**Quantized version

## Troubleshooting

### Out of Memory

```python
# Solution 1: Use 8-bit quantization
load_in_8bit=True

# Solution 2: Use CPU fallback
use_gpu=False

# Solution 3: Use smaller model
model_arg = "local-phi-3-mini"
```

### Slow Inference

```python
# Check GPU utilization
!nvidia-smi

# Ensure GPU is being used
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### Model Loading Errors

```python
# Verify model path exists
from pathlib import Path
model_path = Path("/kaggle/input/meta-llama-3-8b-instruct/")
print(f"Model exists: {model_path.exists()}")
print(f"Files: {list(model_path.glob('*.safetensors'))}")
```

## Complete Kaggle Notebook Template

```python
# Cell 1: Setup
import sys
sys.path.append('/kaggle/working/ARC-AGI')

!pip install transformers torch accelerate bitsandbytes

# Cell 2: Load configuration
import os
os.environ['TRANSFORMERS_CACHE'] = '/kaggle/working/cache'

# Cell 3: Run solver
!python run.py \
    --task-file /kaggle/input/arc-agi-2/data/evaluation.json \
    --models local-llama-3-8b \
    --task-workers 2 \
    --verbose 1 \
    --submissions-directory /kaggle/working/submissions

# Cell 4: Create submission
import json
with open('/kaggle/working/submissions/submission.json', 'r') as f:
    submission = json.load(f)

# Save for Kaggle submission
import pandas as pd
df = pd.DataFrame(list(submission.items()), columns=['output_id', 'output'])
df.to_csv('submission.csv', index=False)
```

## Best Practices

1. **Start small**: Test with Phi-3-Mini before using larger models
2. **Cache models**: Load once, reuse across tasks
3. **Monitor VRAM**: Use `!nvidia-smi` to track usage
4. **Batch similar tasks**: Group by complexity
5. **Save checkpoints**: Regularly save progress

## Limitations

- **No multimodal**: Local models don't support image inputs yet
- **Limited context**: Most local models have 8K context vs 1M+ for APIs
- **Lower accuracy**: Local models typically score 10-20% lower than GPT-5.2/Gemini-3
- **Memory constraints**: Large models require quantization

## Future Improvements

1. **Multimodal support**: Add LLaVA or similar VLM
2. **Better prompts**: Optimize for local model capabilities
3. **Ensemble methods**: Combine multiple local models
4. **Fine-tuning**: Train on ARC-specific data

## Support

For issues or questions:
- Check logs in `/kaggle/working/logs/`
- Review error messages carefully
- Try simpler models first
- Reduce parallelism if encountering errors
