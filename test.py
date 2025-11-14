import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging
import torch.nn as nn
from typing import Dict, List, Tuple
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QuantizationPipeline:
    def __init__(self, model_name: str, output_dir: str):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.is_available() else torch.float32
            ),
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


class PruningPipeline:
    def __init__(self, model, pruning_ratio: float = 0.3):
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.device = next(model.parameters()).device

    def get_prunable_layers(self) -> List[Tuple[str, nn.Module]]:
        """Get prunable layers, excluding BitsAndBytes quantized layers"""
        prunable_layers = []

        for name, module in self.model.named_modules():
            if "quantized" in module.__class__.__name__.lower():
                logger.info(
                    f"Skipping quantized layer: {name} ({module.__class__.__name__})"
                )
                continue

            if isinstance(module, (nn.Linear, nn.Conv2d)):
                prunable_layers.append((name, module))

        return prunable_layers

    def structured_pruning_ffn(self):
        """Prune entire neurons from FFN layers"""
        logger.info("Starting structured pruning on FFN layers...")

        pruned_count = 0
        total_params = 0
        layers_skipped = 0

        for name, module in self.model.named_modules():
            if "mlp" in name.lower() or "feed_forward" in name.lower():
                if isinstance(module, nn.Linear):
                    if hasattr(module, "weight") and hasattr(module.weight, "data"):
                        try:
                            weight_data = module.weight.data

                            # Skip quantized layers
                            if weight_data.dtype not in [
                                torch.float32,
                                torch.float16,
                                torch.bfloat16,
                            ]:
                                logger.info(
                                    f"Skipping quantized layer {name} (dtype: {weight_data.dtype})"
                                )
                                layers_skipped += 1
                                continue

                            total_params += weight_data.numel()

                            importance = torch.norm(weight_data, p=2, dim=1)
                            threshold = torch.quantile(importance, self.pruning_ratio)
                            mask = importance > threshold

                            with torch.no_grad():
                                module.weight.data[~mask] = 0

                            pruned_count += (~mask).sum().item()
                            logger.info(
                                f"Pruned {name}: {(~mask).sum().item()} neurons "
                                f"out of {len(importance)}"
                            )

                        except Exception as e:
                            logger.warning(f"Could not prune layer {name}: {e}")
                            layers_skipped += 1
                            continue

        pruning_percentage = (
            (pruned_count / total_params) * 100 if total_params > 0 else 0
        )
        logger.info(
            f"Total structured pruning: {pruning_percentage:.2f}% of parameters "
            f"({layers_skipped} quantized layers skipped)"
        )

        return self.model

    def unstructured_pruning_attention(self):
        """Prune individual weights from attention layers"""
        logger.info("Starting unstructured pruning on attention layers...")

        pruned_count = 0
        total_params = 0
        layers_skipped = 0

        for name, module in self.model.named_modules():
            if "self_attn" in name.lower() or "attention" in name.lower():
                if isinstance(module, nn.Linear):
                    try:
                        weight_data = module.weight.data

                        if weight_data.dtype not in [
                            torch.float32,
                            torch.float16,
                            torch.bfloat16,
                        ]:
                            logger.info(
                                f"Skipping quantized attention layer {name} "
                                f"(dtype: {weight_data.dtype})"
                            )
                            layers_skipped += 1
                            continue

                        total_params += weight_data.numel()
                        importance = torch.abs(weight_data)
                        threshold = torch.quantile(
                            importance.flatten(), self.pruning_ratio
                        )
                        mask = importance > threshold

                        with torch.no_grad():
                            module.weight.data[~mask] = 0

                        pruned_count += (~mask).sum().item()
                        logger.info(f"Pruned {name}: {(~mask).sum().item()} weights")

                    except Exception as e:
                        logger.warning(f"Could not prune attention layer {name}: {e}")
                        layers_skipped += 1
                        continue

        pruning_percentage = (
            (pruned_count / total_params) * 100 if total_params > 0 else 0
        )
        logger.info(
            f"Total unstructured pruning: {pruning_percentage:.2f}% of parameters "
            f"({layers_skipped} quantized layers skipped)"
        )

        return self.model

    def combined_pruning(self):
        """Apply both structured and unstructured pruning"""
        logger.info("Applying combined pruning strategy...")
        logger.info("Note: BitsAndBytes quantized layers will be skipped")

        self.pruning_ratio = 0.3
        self.structured_pruning_ffn()

        self.pruning_ratio = 0.15
        self.unstructured_pruning_attention()

        return self.model

    def measure_compression(self) -> Dict[str, float]:
        """Measure compression achieved by pruning"""
        total_params = sum(p.numel() for p in self.model.parameters())
        zero_params = sum((p == 0).sum().item() for p in self.model.parameters())

        compression_ratio = (
            (zero_params / total_params) * 100 if total_params > 0 else 0
        )

        metrics = {
            "total_parameters": total_params,
            "zero_parameters": zero_params,
            "compression_ratio_percent": compression_ratio,
            "remaining_parameters": total_params - zero_params,
        }

        logger.info(f"Pruning compression metrics: {metrics}")

        return metrics


class LoRAFineTuner:
    def __init__(
        self,
        model,
        tokenizer,
        output_dir: str,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.device = next(model.parameters()).device

    def setup_lora(self):
        logger.info("Setting up LoRA configuration...")

        self.model = prepare_model_for_kbit_training(self.model)

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            modules_to_save=["lm_head"],
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()

        logger.info("LoRA applied successfully")

        return self.model

    def prepare_dataset(self, dataset_name: str = "wikitext", split: str = "train"):
        logger.info(f"Loading dataset: {dataset_name}")

        if dataset_name == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], padding="max_length", truncation=True, max_length=512
            )

        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset",
        )

        logger.info(f"Dataset prepared with {len(tokenized_dataset)} samples")

        return tokenized_dataset

    def fine_tune(
        self,
        train_dataset,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        warmup_steps: int = 100,
    ):
        logger.info("Starting LoRA fine-tuning...")

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            save_steps=500,
            save_total_limit=3,
            logging_steps=100,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=2,
            max_grad_norm=1.0,
            optim="paged_adamw_32bit" if torch.cuda.is_available() else "adamw_torch",
            report_to=[],
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=lambda data: {
                "input_ids": torch.stack([torch.tensor(f["input_ids"]) for f in data]),
                "attention_mask": torch.stack(
                    [torch.tensor(f["attention_mask"]) for f in data]
                ),
                "labels": torch.stack([torch.tensor(f["input_ids"]) for f in data]),
            },
        )

        trainer.train()

        logger.info("Fine-tuning completed")

        return self.model

    def save_model(self):
        logger.info(f"Saving fine-tuned model to {self.output_dir}")

        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        logger.info("Model saved successfully")


class CompressionPipeline:
    def __init__(self, model_name: str, output_base_dir: str):
        self.model_name = model_name
        self.output_base_dir = output_base_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run_full_pipeline(
        self,
        quantize: bool = True,
        prune: bool = True,
        fine_tune: bool = True,
        pruning_ratio: float = 0.3,
        lora_rank: int = 8,
        num_epochs: int = 3,
    ):
        logger.info("=" * 80)
        logger.info("COMPRESSION PIPELINE STARTED (BitsAndBytes + PRUNING + LoRA)")
        logger.info("=" * 80)

        if quantize:
            logger.info("\n[STAGE 1] QUANTIZATION (BitsAndBytes)")
            logger.info("-" * 80)

            quantizer = QuantizationPipeline(
                model_name=self.model_name,
                output_dir=f"{self.output_base_dir}/stage1-quantized",
            )
            model, tokenizer = quantizer.quantize_model_bnb(load_in_4bit=True)
            quant_metrics = quantizer.measure_quantization_impact()
            logger.info(f"Metrics: {quant_metrics}")
        else:
            logger.info("Loading base model (skipping quantization)...")
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )

        if prune:
            logger.info("\n[STAGE 2] PRUNING")
            logger.info("-" * 80)

            pruner = PruningPipeline(model, pruning_ratio=pruning_ratio)
            model = pruner.combined_pruning()
            prune_metrics = pruner.measure_compression()

            logger.info(
                f"Compression: {prune_metrics['compression_ratio_percent']:.2f}%"
            )

        if fine_tune:
            logger.info("\n[STAGE 3] LoRA FINE-TUNING")
            logger.info("-" * 80)

            fine_tuner = LoRAFineTuner(
                model=model,
                tokenizer=tokenizer,
                output_dir=f"{self.output_base_dir}/stage3-lora-finetuned",
                lora_rank=lora_rank,
            )

            model = fine_tuner.setup_lora()
            train_dataset = fine_tuner.prepare_dataset(
                dataset_name="wikitext", split="train[:2%]"
            )
            model = fine_tuner.fine_tune(
                train_dataset=train_dataset,
                num_epochs=num_epochs,
                batch_size=4,
                learning_rate=1e-4,
            )
            fine_tuner.save_model()

        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        return model, tokenizer


# CPU optimizations
if not torch.cuda.is_available():
    torch.set_num_threads(8)
    logger.info("CPU optimizations applied")


# Execute pipeline
pipeline = CompressionPipeline(
    model_name="Qwen/Qwen3-0.6B", output_base_dir="./models/qwen-compressed"
)

model_final, tokenizer_final = pipeline.run_full_pipeline(
    quantize=True,
    prune=True,
    fine_tune=False,
    pruning_ratio=0.3,
    lora_rank=8,
    num_epochs=1,
)

logger.info("Pipeline execution completed!")
