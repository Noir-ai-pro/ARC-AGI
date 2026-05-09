"""
Local LLM Provider for Kaggle (Offline Mode)
Supports loading models from:
- Kaggle Models (pre-installed)
- Kaggle Datasets (downloaded as datasets)
- HuggingFace cache (if available offline)

Uses transformers library with CPU/GPU inference.
No external API calls required.
"""
import sys
import os
import time
import warnings
from typing import Optional, List, Dict, Any
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from PIL import Image

from src.types import ModelConfig, ModelResponse
from src.llm_utils import run_with_retry
from src.logging import get_logger
from src.errors import RetryableProviderError, NonRetryableProviderError

logger = get_logger("providers.local")

# Cache for loaded models to avoid reloading
_MODEL_CACHE = {}

class LocalLLMProvider:
    """
    Local LLM provider for offline inference on Kaggle.
    Supports various model formats and quantization options.
    """
    
    def __init__(
        self,
        model_path: str,
        model_name: str = "local-model",
        max_tokens: int = 8192,
        temperature: float = 0.7,
        use_gpu: bool = True,
        load_in_8bit: bool = False,
        trust_remote_code: bool = False,
    ):
        """
        Initialize local LLM provider.
        
        Args:
            model_path: Path to model directory or HuggingFace model ID
            model_name: Display name for the model
            max_tokens: Maximum output tokens
            temperature: Sampling temperature
            use_gpu: Whether to use GPU if available
            load_in_8bit: Load model in 8-bit quantization (saves memory)
            trust_remote_code: Trust remote code in model
        """
        self.model_path = model_path
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.load_in_8bit = load_in_8bit
        self.trust_remote_code = trust_remote_code
        
        logger.info(f"Initializing local model: {model_path}")
        logger.info(f"GPU available: {self.use_gpu}")
        
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer with caching."""
        cache_key = f"{self.model_path}_{self.load_in_8bit}"
        
        if cache_key in _MODEL_CACHE:
            logger.info(f"Loading model from cache: {self.model_path}")
            self.tokenizer, self.model = _MODEL_CACHE[cache_key]
            return
        
        logger.info(f"Loading model from scratch: {self.model_path}")
        start_time = time.time()
        
        # Determine device
        device_map = "auto" if self.use_gpu else "cpu"
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=self.trust_remote_code,
                padding_side="left",
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            raise NonRetryableProviderError(f"Failed to load tokenizer: {e}")
        
        # Load model with appropriate settings
        model_kwargs = {
            "trust_remote_code": self.trust_remote_code,
            "device_map": device_map,
            "torch_dtype": torch.float16 if self.use_gpu else torch.float32,
        }
        
        if self.load_in_8bit and self.use_gpu:
            try:
                import bitsandbytes
                model_kwargs["load_in_8bit"] = True
                model_kwargs["device_map"] = "auto"
            except ImportError:
                logger.warning("bitsandbytes not available, loading without 8-bit quantization")
        
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **model_kwargs
            )
            self.model.eval()
        except Exception as e:
            raise NonRetryableProviderError(f"Failed to load model: {e}")
        
        # Cache the model
        _MODEL_CACHE[cache_key] = (self.tokenizer, self.model)
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f}s")
    
    def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> ModelResponse:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt text
            image_path: Optional path to image (for multimodal models)
            max_tokens: Override default max tokens
            temperature: Override default temperature
            system_prompt: Optional system prompt
            
        Returns:
            ModelResponse with generated text and metadata
        """
        start_time = time.perf_counter()
        
        # Prepare input
        if system_prompt:
            full_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
        else:
            full_prompt = prompt
        
        # Tokenize
        inputs = self.tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=16384,
        )
        
        if self.use_gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generation config
        gen_config = GenerationConfig(
            max_new_tokens=max_tokens or self.max_tokens,
            temperature=temperature or self.temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        # Generate
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=gen_config,
                )
            
            # Decode output
            generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
            response_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            
            # Calculate timing
            duration = time.perf_counter() - start_time
            
            # Estimate token counts (approximate)
            prompt_tokens = inputs['input_ids'].shape[1]
            completion_tokens = len(generated_ids)
            
            return ModelResponse(
                text=response_text.strip(),
                prompt_tokens=prompt_tokens,
                cached_tokens=0,
                completion_tokens=completion_tokens,
                thought_tokens=0,
                detailed_logs=[
                    {"type": "text", "content": response_text},
                    {"type": "metadata", "duration": duration}
                ]
            )
            
        except Exception as e:
            raise RetryableProviderError(f"Generation failed: {e}")


def call_local_llm(
    model_path: str,
    prompt: str,
    config: ModelConfig,
    image_path: str = None,
    return_strategy: bool = False,
    verbose: bool = False,
    task_id: str = None,
    test_index: int = None,
    run_timestamp: str = None,
    model_alias: str = None,
    timing_tracker: list = None,
    enable_code_execution: bool = False,
    model_settings: Dict[str, Any] = None,
) -> ModelResponse:
    """
    Main entry point for local LLM provider.
    
    Args:
        model_path: Path to model (Kaggle dataset or model)
        prompt: Input prompt
        config: Model configuration
        image_path: Optional image path
        return_strategy: Whether to return strategy explanation
        verbose: Enable verbose logging
        task_id: Task identifier
        test_index: Test case index
        run_timestamp: Timestamp for this run
        model_alias: Alias name for the model
        timing_tracker: List to track timing information
        enable_code_execution: Enable code execution capability
        model_settings: Additional model-specific settings
        
    Returns:
        ModelResponse with generated solution
    """
    model_name = model_alias or f"local-{Path(model_path).name}"
    
    if verbose:
        logger.info(f"Using local model: {model_name} at {model_path}")
    
    # Default settings
    settings = model_settings or {}
    max_tokens = settings.get("max_tokens", 4096)
    temperature = settings.get("temperature", 0.7)
    use_gpu = settings.get("use_gpu", True)
    load_in_8bit = settings.get("load_in_8bit", True)
    
    # Initialize provider
    try:
        provider = LocalLLMProvider(
            model_path=model_path,
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            use_gpu=use_gpu,
            load_in_8bit=load_in_8bit,
        )
    except Exception as e:
        logger.error(f"Failed to initialize local model: {e}")
        raise
    
    # Generate response
    def _generate():
        return provider.generate(
            prompt=prompt,
            image_path=image_path,
        )
    
    response = run_with_retry(
        _generate,
        task_id=task_id,
        test_index=test_index,
        run_timestamp=run_timestamp,
        model_name=model_name,
        timing_tracker=timing_tracker,
    )
    
    # Handle strategy extraction if requested
    if return_strategy and response and response.text:
        # Simple strategy extraction (can be enhanced)
        strategy_prompt = "\n\nExplain the strategy you used in broad terms such that it can be applied on other similar examples."
        
        def _extract_strategy():
            return provider.generate(
                prompt=response.text + strategy_prompt,
                max_tokens=1024,
                temperature=0.5,
            )
        
        try:
            strategy_response = _extract_strategy()
            if strategy_response:
                response.strategy = strategy_response.text
        except Exception as e:
            logger.warning(f"Strategy extraction failed: {e}")
    
    return response


# Convenience function for common Kaggle model paths
def get_kaggle_model_path(model_type: str) -> str:
    """
    Get standard Kaggle model paths for common model types.
    
    Args:
        model_type: Type of model (e.g., 'llama-3-8b', 'mistral-7b', 'phi-3')
        
    Returns:
        Path to model on Kaggle
    """
    kaggle_models = {
        "llama-3-8b": "/kaggle/input/meta-llama-3-8b-instruct/",
        "llama-3-70b": "/kaggle/input/meta-llama-3-70b-instruct/",
        "mistral-7b": "/kaggle/input/mistral-7b-instruct/",
        "phi-3-mini": "/kaggle/input/phi-3-mini-4k-instruct/",
        "gemma-2b": "/kaggle/input/gemma-2b-it/",
        "gemma-7b": "/kaggle/input/gemma-7b-it/",
        "qwen-2-7b": "/kaggle/input/qwen-2-7b-instruct/",
        "yi-34b": "/kaggle/input/yi-34b-chat/",
    }
    
    if model_type not in kaggle_models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(kaggle_models.keys())}")
    
    return kaggle_models[model_type]
