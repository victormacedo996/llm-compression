from core.compression.engine.techniques.quantization.transformers_lib_interface import (
    IQuantization,
)
from loguru import logger
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig


class TransformersQuantizationPipeline(IQuantization):
    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    def quantize_model_bnb(self, load_in_4bit: bool = True):
        """Quantize model using BitsAndBytes"""
        logger.info(f"Loading model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(
            f"Using {'4-bit' if load_in_4bit else '8-bit'} BitsAndBytes quantization"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.compute_dtype,
        )

        logger.info("Loading model with BitsAndBytes quantization...")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )

        logger.info("Model loaded and quantized successfully")

        logger.info(f"Saving quantized model to: {self.output_dir}")
        model.save_pretrained(self.output_dir)
        tokenizer.save_pretrained(self.output_dir)

        logger.info("Quantized model and tokenizer saved")

        return model, tokenizer

    def measure_quantization_impact(self):
        """Measure quantization impact"""
        logger.info("Measuring quantization impact...")

        import os

        if not os.path.exists(self.output_dir):
            logger.warning(f"Output directory does not exist: {self.output_dir}")
            return {}

        quantized_size = sum(
            os.path.getsize(os.path.join(self.output_dir, f))
            for f in os.listdir(self.output_dir)
            if os.path.isfile(os.path.join(self.output_dir, f))
        )

        metrics = {
            "quantized_model_size_mb": quantized_size / (1024 * 1024),
            "quantization_method": "BitsAndBytes-NF4",
        }

        logger.info(f"Quantization metrics: {metrics}")

        return metrics
