#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Utility functions for LLM fine-tuning."""

import os
import json
import torch
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from config import Config

def set_random_seed(seed: int = Config.SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_prompt(context: str, question: str) -> str:
    """Generate a formatted prompt for the model."""
    return Config.PROMPT_TEMPLATE.format(context=context, prompt=question)

def load_json_data(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_data(data: Any, file_path: str):
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def load_fine_tuned_model(model_path: str, base_model_name: str = Config.MODEL_NAME):
    """Load a fine-tuned PEFT model for inference."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_8bit=True,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load PEFT model with LoRA adapters
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    return model, tokenizer

def generate_answer_from_model(
    model, 
    tokenizer, 
    context: str, 
    question: str, 
    max_length: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True
) -> str:
    """Generate an answer from the model for a given context and question."""
    # Generate prompt
    prompt = generate_prompt(context, question)
    
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the answer part (after the prompt)
    answer_start = len(prompt)
    answer = generated_text[answer_start:].strip()
    
    return answer

def get_device() -> torch.device:
    """Get the appropriate device for PyTorch."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_evaluation_results(results: Dict[str, Any]) -> str:
    """Format evaluation results as a readable string."""
    output = "Evaluation Results:\n"
    
    # Format basic metrics
    output += f"BLEU Score: {results.get('bleu', 'N/A'):.2f}\n"
    output += f"ROUGE-1 F1: {results.get('rouge1', 'N/A'):.4f}\n"
    output += f"ROUGE-2 F1: {results.get('rouge2', 'N/A'):.4f}\n"
    output += f"ROUGE-L F1: {results.get('rougeL', 'N/A'):.4f}\n"
    output += f"Exact Match: {results.get('exact_match', 'N/A'):.4f}\n"
    
    # Format comparison with baseline if available
    if "baseline_comparison" in results:
        output += "\nComparison with Baseline:\n"
        for metric, comp_data in results["baseline_comparison"].items():
            output += f"{metric}:\n"
            output += f"  Baseline: {comp_data['baseline']}\n"
            output += f"  Fine-tuned: {comp_data['fine_tuned']}\n"
            output += f"  Improvement: {comp_data['improvement']} "
            output += f"({comp_data['improvement_percent']:.2f}%)\n"
    
    return output

def postprocess_answer(answer: str) -> str:
    """Clean up generated answer text."""
    # Remove any extra newlines or spaces
    answer = answer.strip()
    
    # Remove any potential model signature or generation artifacts
    if "\n\n" in answer:
        # Keep only the first paragraph in most cases
        first_para = answer.split("\n\n")[0]
        if len(first_para) > 10:  # Only use first para if it's substantial
            answer = first_para
    
    return answer
