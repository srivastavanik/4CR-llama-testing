#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Evaluation script for fine-tuned LLM on Q&A tasks."""

import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, prepare_model_for_kbit_training
import evaluate
from typing import Dict, List, Any
from config import Config
from bitsandbytes.configs import BitsAndBytesConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load evaluation metrics
bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

def load_test_data(test_file: str) -> Dataset:
    """Load test data from JSON file."""
    with open(test_file, 'r') as f:
        test_data = json.load(f)
    
    test_dataset = Dataset.from_list(test_data)
    return test_dataset

def load_model_and_tokenizer(model_path: str, base_model_name: str = Config.MODEL_NAME):
    """Load the fine-tuned model and tokenizer with comprehensive error handling."""
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        logging.info(f"Tokenizer loaded successfully from {model_path}")
    except Exception as e:
        logging.error(f"Error loading tokenizer: {e}")
        try:
            # Try loading from base model if fine-tuned tokenizer fails
            tokenizer = AutoTokenizer.from_pretrained(base_model_name)
            logging.info(f"Fallback: Using tokenizer from base model {base_model_name}")
        except Exception as e:
            logging.error(f"Error loading tokenizer from base model: {e}")
            raise RuntimeError("Failed to load tokenizer")
    
    # Make sure pad token exists, or add it
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    
    # Load base model with proper quantization
    logging.info(f"Loading model on {Config.DEVICE}")
    model = None
    
    # Try different loading methods in order of preference
    loading_methods = [
        ("8-bit quantization", True),
        ("4-bit quantization", False),
        ("No quantization", None)
    ]
    
    for method_name, quantize in loading_methods:
        try:
            logging.info(f"\nAttempting to load model with {method_name}...")
            
            if quantize is not None:
                # Create quantization config
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=quantize,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
                
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                # CPU-friendly loading
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    low_cpu_mem_usage=True,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                ).to(Config.DEVICE)
            
            # Prepare model for k-bit training if needed
            if quantize:
                base_model = prepare_model_for_kbit_training(base_model)
            
            # Resize token embeddings to match tokenizer
            original_vocab_size = base_model.get_input_embeddings().weight.size(0)
            tokenizer_vocab_size = len(tokenizer)
            
            if original_vocab_size != tokenizer_vocab_size:
                logging.info(f"Resizing token embeddings: {original_vocab_size} -> {tokenizer_vocab_size}")
                base_model.resize_token_embeddings(tokenizer_vocab_size)
            
            # Ensure the model's vocab size matches the tokenizer's
            assert base_model.get_input_embeddings().weight.size(0) == len(tokenizer)
            assert base_model.get_output_embeddings().weight.size(0) == len(tokenizer)
            
            # Load PEFT model with LoRA adapters
            model = PeftModel.from_pretrained(base_model, model_path)
            logging.info(f"Successfully loaded model with {method_name}")
            break
            
        except Exception as e:
            logging.error(f"Failed to load model with {method_name}: {e}")
            continue
    
    if model is None:
        raise RuntimeError("Failed to load model using all available methods")
    
    model.eval()
    return model, tokenizer

def generate_answer(model, tokenizer, prompt: str, max_length: int = 512):
    """Generate an answer from the model for a given prompt."""
    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the answer part - since our prompt format is different, 
    # we need a different approach to extract the answer
    prompt_end = "Context:" if "Context:" in prompt else prompt
    answer = generated_text
    
    # If the generated text still contains the prompt, try to trim it
    if prompt_end in generated_text:
        answer_parts = generated_text.split(prompt_end, 1)
        if len(answer_parts) > 1:
            answer = answer_parts[1].strip()
    
    return answer

def calculate_exact_match(predictions: List[str], references: List[str]) -> float:
    """Calculate exact match score between predictions and references."""
    exact_matches = sum(1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip())
    return exact_matches / len(predictions) if predictions else 0

def evaluate_model(model, tokenizer, test_dataset: Dataset):
    """Evaluate the model on test data with error handling."""
    predictions = []
    references = []
    failed_predictions = 0
    
    logging.info("Generating predictions...")
    for example in tqdm(test_dataset):
        prompt = example["input_text"]
        reference = example["answer"]
        
        try:
            prediction = generate_answer(model, tokenizer, prompt)
            predictions.append(prediction)
            references.append(reference)
        except Exception as e:
            logging.warning(f"Failed to generate prediction for prompt: {prompt[:50]}... Error: {e}")
            failed_predictions += 1
            continue
    
    if failed_predictions > 0:
        logging.warning(f"Failed to generate predictions for {failed_predictions} examples")
    
    if not predictions:
        raise RuntimeError("No successful predictions generated")
    
    # Calculate metrics
    logging.info("\nCalculating evaluation metrics...")
    
    try:
        # BLEU score
        bleu_results = bleu.compute(predictions=predictions, references=[[r] for r in references])
        
        # ROUGE score
        rouge_results = rouge.compute(predictions=predictions, references=references)
        
        # Exact match score
        exact_match_score = calculate_exact_match(predictions, references)
        
        # Compile results
        evaluation_results = {
            "bleu": bleu_results["score"],
            "rouge1": rouge_results["rouge1"],
            "rouge2": rouge_results["rouge2"],
            "rougeL": rouge_results["rougeL"],
            "exact_match": exact_match_score,
            "failed_predictions": failed_predictions,
            "total_predictions": len(predictions)
        }
        
        return evaluation_results, predictions
        
    except Exception as e:
        logging.error(f"Error calculating metrics: {e}")
        raise

def compare_with_baseline(eval_results: Dict[str, Any], baseline_results: Dict[str, Any]):
    """Compare fine-tuned model results with baseline model results."""
    comparison = {}
    for metric in eval_results:
        if metric in baseline_results:
            baseline_val = baseline_results[metric]
            current_val = eval_results[metric]
            improvement = current_val - baseline_val
            improvement_percent = (improvement / baseline_val * 100) if baseline_val > 0 else float('inf')
            
            comparison[metric] = {
                "baseline": baseline_val,
                "fine_tuned": current_val,
                "improvement": improvement,
                "improvement_percent": improvement_percent
            }
    
    return comparison

def save_evaluation_results(
    results: Dict[str, Any], 
    predictions: List[str], 
    test_dataset: Dataset, 
    output_dir: str
):
    """Save evaluation results and predictions to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Save predictions alongside references
    prediction_data = []
    for i, example in enumerate(test_dataset):
        prediction_data.append({
            "prompt": example["prompt"],
            "context": example["context"],
            "reference": example["answer"],
            "prediction": predictions[i]
        })
    
    with open(os.path.join(output_dir, "predictions.json"), "w") as f:
        json.dump(prediction_data, f, indent=2)
    
    logging.info(f"Evaluation results saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LLM on Q&A tasks")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to fine-tuned model"
    )
    parser.add_argument(
        "--base_model", 
        type=str, 
        default=Config.MODEL_NAME,
        help="Name of the base model used for fine-tuning"
    )
    parser.add_argument(
        "--test_data", 
        type=str, 
        required=True,
        help="Path to test data JSON file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--baseline_results", 
        type=str, 
        default=None,
        help="Path to baseline evaluation results JSON for comparison"
    )
    
    args = parser.parse_args()
    
    # Load test data
    logging.info(f"Loading test data from {args.test_data}...")
    test_dataset = load_test_data(args.test_data)
    
    # Load model and tokenizer
    logging.info(f"Loading fine-tuned model from {args.model_path}...")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model)
    
    # Evaluate model
    logging.info("Evaluating model...")
    evaluation_results, predictions = evaluate_model(model, tokenizer, test_dataset)
    
    # Display results
    logging.info("\nEvaluation Results:")
    for metric, value in evaluation_results.items():
        logging.info(f"{metric}: {value}")
    
    # Compare with baseline if provided
    if args.baseline_results:
        logging.info(f"\nComparing with baseline results from {args.baseline_results}...")
        with open(args.baseline_results, 'r') as f:
            baseline_results = json.load(f)
        
        comparison = compare_with_baseline(evaluation_results, baseline_results)
        
        logging.info("\nComparison with Baseline:")
        for metric, comp_data in comparison.items():
            logging.info(f"{metric}:")
            logging.info(f"  Baseline: {comp_data['baseline']}")
            logging.info(f"  Fine-tuned: {comp_data['fine_tuned']}")
            logging.info(f"  Improvement: {comp_data['improvement']} ({comp_data['improvement_percent']:.2f}%)")
        
        # Add comparison to evaluation results
        evaluation_results["baseline_comparison"] = comparison
    
    # Save results
    save_evaluation_results(evaluation_results, predictions, test_dataset, args.output_dir)
    
if __name__ == "__main__":
    main()
