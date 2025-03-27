#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Model fine-tuning script for LLM Q&A tasks."""

import os
import json
import argparse
import torch
import numpy as np
from datasets import Dataset
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

def load_model_and_tokenizer(model_name: str, use_8bit: bool = False):
    """Load the pre-trained model and tokenizer."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Make sure pad token exists, or add it
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # Load model with quantization if specified, otherwise on CPU
    if use_8bit and torch.cuda.is_available():
        print("Using 8-bit quantization on GPU")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_8bit=True,
            device_map="auto",
            trust_remote_code=True
        )
        model = prepare_model_for_kbit_training(model)
    else:
        print(f"Loading model on {Config.DEVICE}")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,  # Helpful for CPU training
            torch_dtype=torch.float32,  # Use float32 for CPU
            trust_remote_code=True
        ).to(Config.DEVICE)
    
    # Resize token embeddings to match tokenizer
    original_vocab_size = model.get_input_embeddings().weight.size(0)
    tokenizer_vocab_size = len(tokenizer)
    
    if original_vocab_size != tokenizer_vocab_size:
        print(f"Resizing token embeddings: {original_vocab_size} -> {tokenizer_vocab_size}")
        model.resize_token_embeddings(tokenizer_vocab_size)
    
    # Ensure the model's vocab size matches the tokenizer's
    assert model.get_input_embeddings().weight.size(0) == len(tokenizer)
    assert model.get_output_embeddings().weight.size(0) == len(tokenizer)
    
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
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.use_8bit)
    
    # Setup LoRA
    lora_model = setup_lora(model)
    
    # Train model
    trainer = train_model(lora_model, tokenizer, train_dataset, val_dataset, args.output_dir)
    
    return trainer

if __name__ == "__main__":
    main()
