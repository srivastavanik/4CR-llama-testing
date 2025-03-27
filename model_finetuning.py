#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model fine-tuning script for LLM Q&A tasks."""

import os
import json
import argparse
import torch
import numpy as np
try:
    from datasets import Dataset
except ImportError:
    raise ImportError("The 'datasets' package is missing. Please install it by running: pip install datasets")
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from typing import Dict, List, Optional
from config import Config

def load_processed_data(data_dir: str):
    """Load preprocessed data from JSON files."""
    train_path = os.path.join(data_dir, "train.json")
    val_path = os.path.join(data_dir, "validation.json")
    
    with open(train_path, 'r') as f:
        train_data = json.load(f)
    
    with open(val_path, 'r') as f:
        val_data = json.load(f)
    
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset

def tokenize_function(examples, tokenizer):
    """Tokenize the inputs and targets."""
    # Tokenize inputs
    inputs = tokenizer(
        examples["input_text"],
        padding="max_length",
        truncation=True,
        max_length=Config.MAX_LENGTH,
        return_tensors="pt"
    )
    
    # Tokenize targets/answers
    targets = tokenizer(
        examples["answer"],
        padding="max_length",
        truncation=True,
        max_length=Config.MAX_LENGTH,  # Changed to match input length
        return_tensors="pt"
    )
    
    # Create input_ids for decoder input (shifted targets)
    inputs["labels"] = targets["input_ids"].clone()
    
    # Replace padding token id with -100 in labels to ignore loss
    inputs["labels"] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label] 
        for label in inputs["labels"]
    ]
    
    return inputs

def load_model_and_tokenizer(model_path: str, base_model_name: str = Config.MODEL_NAME):
    """Load the model and tokenizer, using HF_TOKEN if available."""
    # Get HF token from environment
    hf_token = os.environ.get("HF_TOKEN")

    # Load tokenizer with authentication token if available
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_auth_token=hf_token if hf_token else None,
            trust_remote_code=True
        )
        print(f"Tokenizer loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading tokenizer from {model_path}: {e}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                base_model_name,
                use_auth_token=hf_token if hf_token else None,
                trust_remote_code=True
            )
            print(f"Fallback: Loaded tokenizer from base model {base_model_name}")
        except Exception as e:
            raise ImportError(f"Failed to load tokenizer: {e}")

    # Make sure pad token exists, or add it
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load base model with authentication token if available
    print(f"Loading base model {base_model_name} on {Config.DEVICE}")
    try:
        if torch.cuda.is_available():
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                use_auth_token=hf_token if hf_token else None,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )
            base_model = prepare_model_for_kbit_training(base_model)
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                use_auth_token=hf_token if hf_token else None,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to(Config.DEVICE)
    except Exception as e:
        raise RuntimeError(f"Error loading base model: {e}")

    # Resize token embeddings if necessary
    original_vocab_size = base_model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    if original_vocab_size != tokenizer_vocab_size:
        print(f"Resizing token embeddings: {original_vocab_size} -> {tokenizer_vocab_size}")
        base_model.resize_token_embeddings(tokenizer_vocab_size)

    # Ensure vocab sizes match for both input and output embeddings
    assert base_model.get_input_embeddings().weight.size(0) == tokenizer_vocab_size, "Input embedding size does not match tokenizer vocab size"
    assert base_model.get_output_embeddings().weight.size(0) == tokenizer_vocab_size, "Output embedding size does not match tokenizer vocab size"

    # Load PEFT model components
    try:
        model = get_peft_model(base_model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            target_modules=Config.LORA_TARGET_MODULES,
        ))
        print("Successfully loaded PEFT model.")
    except Exception as e:
        print(f"Error loading PEFT model: {e}")
        print("Attempting fallback: loading model without quantization...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            use_auth_token=hf_token if hf_token else None,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float32,
            trust_remote_code=True
        ).to(Config.DEVICE)
        base_model.resize_token_embeddings(tokenizer_vocab_size)
        model = get_peft_model(base_model, LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=Config.LORA_R,
            lora_alpha=Config.LORA_ALPHA,
            lora_dropout=Config.LORA_DROPOUT,
            target_modules=Config.LORA_TARGET_MODULES,
        ))

    model.eval()
    return model, tokenizer

def setup_lora(model):
    """Set up LoRA for parameter-efficient fine-tuning."""
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=Config.LORA_R,
        lora_alpha=Config.LORA_ALPHA,
        lora_dropout=Config.LORA_DROPOUT,
        target_modules=Config.LORA_TARGET_MODULES,
    )
    
    # Apply LoRA to model
    lora_model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    print("\nTrainable parameters:")
    trainable_params = 0
    all_params = 0
    for _, param in lora_model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable params: {trainable_params} / {all_params} = {100 * trainable_params / all_params:.2f}%\n")
    
    return lora_model

def train_model(model, tokenizer, train_dataset, val_dataset, output_dir: str):
    """Train the model on the dataset."""
    # Tokenize datasets
    print("Tokenizing datasets...")
    
    def tokenize_batch(batch):
        return tokenize_function(batch, tokenizer)
    
    tokenized_train = train_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["input_text", "answer", "prompt", "context"]
    )
    
    tokenized_val = val_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["input_text", "answer", "prompt", "context"]
    )
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt"
    )
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        num_train_epochs=Config.NUM_EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,  # Use same batch size for evaluation
        gradient_accumulation_steps=Config.GRADIENT_ACCUMULATION_STEPS,
        evaluation_strategy="epoch",
        save_strategy=Config.SAVE_STRATEGY,
        logging_steps=Config.LOGGING_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        push_to_hub=False,
        remove_unused_columns=True,
        report_to="none",  # Set to "wandb" if using Weights & Biases
        fp16=torch.cuda.is_available(),  # Use FP16 if GPU is available
        dataloader_drop_last=True,  # Drop last incomplete batch
        dataloader_num_workers=2  # Use multiple workers for data loading
    )
    
    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    print("\nStarting fine-tuning...")
    trainer.train()
    
    # Save the model
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")
    
    return trainer

def main():
    """Main function to run the model finetuning pipeline."""
    print("Starting model finetuning pipeline...")
    # Parse arguments
    parser = argparse.ArgumentParser(description="Fine-tune LLM on a preprocessed dataset")
    parser.add_argument(
        "--data_dir", 
        type=str, 
        default=Config.PROCESSED_DATA_DIR,
        help="Directory containing preprocessed data"
    )
    parser.add_argument(
        "--model_name", 
        type=str, 
        default=Config.MODEL_NAME,
        help="Pre-trained model name or path"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=Config.OUTPUT_DIR,
        help="Directory to save fine-tuned model"
    )
    parser.add_argument(
        "--use_8bit", 
        action="store_true",
        help="Use 8-bit quantization for training"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    train_dataset, val_dataset = load_processed_data(args.data_dir)
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Train model
    trainer = train_model(model, tokenizer, train_dataset, val_dataset, args.output_dir)
    
    return trainer

if __name__ == "__main__":
    main()
