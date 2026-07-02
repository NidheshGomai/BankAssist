"""
BankAssist RAG — Qwen3-4B Model Loader
========================================
Loads the fine-tuned Qwen3-4B model with:
  - 4-bit BitsAndBytes quantization (Q4_K_M equivalent via bitsandbytes NF4)
  - PEFT LoRA adapter from `fine tuned qwen/QWEN3 QA final/`
  - FP16 compute dtype for maximum RTX 3050 6GB throughput
  - Greedy decoding (do_sample=False) for reproducible, factual outputs

Model Loading Architecture
--------------------------
1. BitsAndBytesConfig: NF4 4-bit quantization with BF16 compute dtype
2. Base model: AutoModelForCausalLM from "Qwen/Qwen3-4B" (HF cache)
3. LoRA merge: PeftModel.from_pretrained() loads adapter weights
4. Tokenizer: AutoTokenizer with padding_side="left" for batch inference

Windows / CUDA Notes
--------------------
- RTX 3050 6GB with CUDA 13.3 driver: 4-bit NF4 quantization + LoRA
  adapter fits within 6 GB VRAM (~2.5 GB for 4B model + overhead).
- `device_map="auto"` with `accelerate` handles layer placement.
- FP16 is used instead of BF16 on the RTX 3050 (Ampere supports BF16 but
  fp16 is more stable with bitsandbytes on Windows).
- Falls back to CPU (FP32, no quantization) when CUDA is unavailable.

Singleton Pattern
-----------------
The model is loaded once per process via `get_qwen3_model()`. Use this
function everywhere — never instantiate Qwen3Model directly in hot paths.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Generator

import torch

from app.config.settings import get_settings
from app.utils.exceptions import LLMLoadError, LLMInferenceError, LLMUnavailableError
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Qwen3Model:
    """
    Thread-safe singleton wrapper for Qwen3-4B + LoRA adapter.

    Exposes:
      - `generate_text()` — synchronous text generation
      - `stream_generate()` — token-streaming generator
    """

    _instance: Qwen3Model | None = None
    _class_lock = threading.Lock()

    def __new__(cls) -> Qwen3Model:
        if not cls._instance:
            with cls._class_lock:
                if not cls._instance:
                    inst = super().__new__(cls)
                    inst._initialized = False
                    cls._instance = inst
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        self.settings = get_settings()
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_lock = threading.Lock()
        self._initialized = True
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(
            "qwen3_wrapper_initialized",
            base_model=self.settings.qwen3_base_model,
            lora_path=self.settings.lora_adapter_path,
            quantization=self.settings.llm_quantization,
            device=self._device,
        )

    # -----------------------------------------------------------------------
    # Model loading
    # -----------------------------------------------------------------------
    def load_model(self) -> None:
        """
        Load the quantized Qwen3 base model and attach the LoRA adapter.
        Thread-safe via double-checked locking.
        """
        if self._model is not None:
            return

        with self._model_lock:
            if self._model is not None:
                return

            try:
                from transformers import (  # noqa: PLC0415
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    BitsAndBytesConfig,
                )

                settings = self.settings
                base_model_id = settings.qwen3_base_model
                lora_path = Path(settings.lora_adapter_path)

                logger.info(
                    "loading_qwen3_model",
                    base_model=base_model_id,
                    lora_path=str(lora_path),
                    quantization=settings.llm_quantization,
                )

                # Quantization config (only if CUDA is available — CPU doesn't support bitsandbytes)
                bnb_config = None
                if settings.llm_quantization == "4bit" and torch.cuda.is_available():
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                elif settings.llm_quantization == "4bit" and not torch.cuda.is_available():
                    logger.warning("CUDA not available — falling back to CPU without quantization for Qwen3")
                    bnb_config = None
                elif settings.llm_quantization == "8bit" and torch.cuda.is_available():
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

                # Load tokenizer
                logger.info("loading_qwen3_tokenizer")
                try:
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        base_model_id,
                        trust_remote_code=True,
                        padding_side="left",
                    )
                except Exception as e:
                    logger.warning("tokenizer_remote_load_failed_trying_local", error=str(e))
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        base_model_id,
                        trust_remote_code=True,
                        padding_side="left",
                        local_files_only=True,
                    )
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token

                # Load base model
                logger.info("loading_qwen3_base_weights")
                if torch.cuda.is_available():
                    model_kwargs: dict[str, Any] = {
                        "trust_remote_code": True,
                        "device_map": "auto",
                        "torch_dtype": torch.float16,
                        "low_cpu_mem_usage": True,
                    }
                else:
                    model_kwargs = {
                        "trust_remote_code": True,
                        "torch_dtype": torch.float32,
                        "low_cpu_mem_usage": True,
                    }
                if bnb_config:
                    model_kwargs["quantization_config"] = bnb_config

                try:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_id,
                        **model_kwargs,
                    )
                except Exception as e:
                    logger.warning("model_remote_load_failed_trying_local", error=str(e))
                    model_kwargs["local_files_only"] = True
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_id,
                        **model_kwargs,
                    )

                # Attach LoRA adapter if it exists
                if lora_path.exists():
                    from peft import PeftModel  # noqa: PLC0415
                    logger.info("loading_lora_adapter", path=str(lora_path))
                    self._model = PeftModel.from_pretrained(base_model, str(lora_path))
                    logger.info("lora_adapter_loaded_successfully")
                else:
                    logger.warning(
                        "lora_adapter_not_found_using_base_model",
                        path=str(lora_path),
                    )
                    self._model = base_model

                self._model.eval()
                logger.info(
                    "qwen3_model_loaded_successfully",
                    base_model=base_model_id,
                    lora_loaded=lora_path.exists(),
                )

            except Exception as exc:
                logger.error("qwen3_model_load_failed", error=str(exc))
                raise LLMLoadError(f"Failed to load Qwen3 model: {exc}") from exc

    # -----------------------------------------------------------------------
    # Text generation (synchronous)
    # -----------------------------------------------------------------------
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        do_sample: bool | None = None,
        top_p: float | None = None,
        stop_strings: list[str] | None = None,
    ) -> str:
        """
        Generate a complete text response (non-streaming).

        Args:
            prompt: Full formatted prompt string.
            max_new_tokens: Override max tokens.
            temperature: Override temperature.
            do_sample: Override sampling flag.
            top_p: Override top-p.
            stop_strings: Token sequences to stop at.

        Returns:
            Generated text (stripped, with stop sequences removed).
        """
        self.load_model()

        if self._model is None:
            raise LLMUnavailableError("Qwen3 model is not loaded.")

        settings = self.settings
        _max_tokens = max_new_tokens or settings.llm_max_new_tokens
        _temp = temperature if temperature is not None else settings.llm_temperature
        _sample = do_sample if do_sample is not None else settings.llm_do_sample
        _top_p = top_p or settings.llm_top_p

        # If thinking mode is disabled, and prompt ends with assistant start tag, bypass it by prefilling empty think tags
        if not settings.llm_thinking_mode:
            assistant_tag = "<|im_start|>assistant\n"
            if prompt.endswith(assistant_tag):
                prompt = prompt + "<think>\n\n</think>\n\n"
            elif prompt.endswith("<|im_start|>assistant"):
                prompt = prompt + "\n<think>\n\n</think>\n\n"

        try:
            # truncation_side="left" preserves the assistant start tag at the end of prompt
            self._tokenizer.truncation_side = "left"
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=settings.llm_max_length,
            ).to(self._model.device)

            with torch.no_grad():
                # Qwen models can end with either <|im_end|> (151645) or <|endoftext|> (151643)
                eos_ids = [self._tokenizer.eos_token_id, 151643]
                gen_kwargs: dict[str, Any] = {
                    "max_new_tokens": _max_tokens,
                    "do_sample": _sample,
                    "repetition_penalty": settings.llm_repetition_penalty,
                    "pad_token_id": self._tokenizer.pad_token_id,
                    "eos_token_id": eos_ids,
                }
                if _sample:
                    gen_kwargs["temperature"] = _temp
                    gen_kwargs["top_p"] = _top_p

                if _sample:
                    self._model.generation_config.temperature = _temp
                    self._model.generation_config.top_p = _top_p
                else:
                    # Reset to defaults to avoid conflicting config warnings
                    self._model.generation_config.temperature = None
                    self._model.generation_config.top_p = None

                output_ids = self._model.generate(
                    **inputs,
                    **gen_kwargs,
                )

            # Decode only the newly generated tokens
            input_length = inputs["input_ids"].shape[1]
            new_tokens = output_ids[0][input_length:]
            generated = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

            # Remove stop sequences if requested
            if stop_strings:
                for stop in stop_strings:
                    if stop in generated:
                        generated = generated[: generated.index(stop)]

            return generated.strip()

        except Exception as exc:
            logger.error("qwen3_generate_failed", error=str(exc))
            raise LLMInferenceError(f"Qwen3 inference failed: {exc}") from exc

    # -----------------------------------------------------------------------
    # Streaming generation
    # -----------------------------------------------------------------------
    def stream_generate(
        self,
        prompt: str,
        max_new_tokens: int | None = None,
    ) -> Generator[str, None, None]:
        """
        Token-streaming generator using HuggingFace TextIteratorStreamer.

        Usage::

            for token in model.stream_generate(prompt):
                print(token, end="", flush=True)

        Yields:
            Individual decoded token strings as they are generated.
        """
        self.load_model()

        if self._model is None:
            raise LLMUnavailableError("Qwen3 model is not loaded.")

        settings = self.settings
        _max_tokens = max_new_tokens or settings.llm_max_new_tokens

        # If thinking mode is disabled, and prompt ends with assistant start tag, bypass it by prefilling empty think tags
        if not settings.llm_thinking_mode:
            assistant_tag = "<|im_start|>assistant\n"
            if prompt.endswith(assistant_tag):
                prompt = prompt + "<think>\n\n</think>\n\n"
            elif prompt.endswith("<|im_start|>assistant"):
                prompt = prompt + "\n<think>\n\n</think>\n\n"

        try:
            from transformers import TextIteratorStreamer  # noqa: PLC0415
 
            self._tokenizer.truncation_side = "left"
            inputs = self._tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=settings.llm_max_length,
            ).to(self._model.device)
 
            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
 
            # Qwen models can end with either <|im_end|> (151645) or <|endoftext|> (151643)
            eos_ids = [self._tokenizer.eos_token_id, 151643]
            gen_kwargs: dict[str, Any] = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": _max_tokens,
                "do_sample": settings.llm_do_sample,
                "repetition_penalty": settings.llm_repetition_penalty,
                "pad_token_id": self._tokenizer.pad_token_id,
                "eos_token_id": eos_ids,
            }

            # Run generation in a background thread
            import threading as _threading  # noqa: PLC0415
            gen_thread = _threading.Thread(
                target=lambda: self._model.generate(**gen_kwargs),
                daemon=True,
            )
            gen_thread.start()

            # Yield tokens as they stream
            for token_text in streamer:
                yield token_text

            gen_thread.join(timeout=60)

        except Exception as exc:
            logger.error("qwen3_stream_generate_failed", error=str(exc))
            raise LLMInferenceError(f"Qwen3 streaming failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Module-level singleton accessor
# ---------------------------------------------------------------------------
_qwen3_instance: Qwen3Model | None = None
_qwen3_lock = threading.Lock()


def get_qwen3_model() -> Qwen3Model:
    """
    Return the singleton Qwen3Model instance.
    The model is NOT loaded until `generate_text()` or `stream_generate()` is first called.
    Use this function everywhere instead of instantiating directly.
    """
    global _qwen3_instance
    if _qwen3_instance is None:
        with _qwen3_lock:
            if _qwen3_instance is None:
                _qwen3_instance = Qwen3Model()
    return _qwen3_instance
