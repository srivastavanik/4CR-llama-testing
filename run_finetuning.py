#!/usr/bin/env python
# -*- coding: utf-8 -*-

try:
    import torch
    import transformers
    import datasets
    import peft
    import pandas
except ImportError as e:
    raise ImportError(f"Missing dependency: {e}. Please run: pip install -r requirements.txt")

"""Run the complete fine-tuning pipeline on qa_dataset.csv."""

import os
import argparse
from config import Config
from data_preprocessing import load_data, preprocess_data, split_data, save_data
from model_finetuning import load_processed_data, load_model_and_tokenizer, setup_lora, train_model
from evaluation import load_test_data, load_model_and_tokenizer as load_eval_model, evaluate_model, save_evaluation_results
from utils import set_random_seed
import sys
import getpass
from huggingface_hub import login

def login_to_hugging_face():
    """Login to Hugging Face and set the HF_TOKEN environment variable."""
    import os
    import getpass
    from huggingface_hub import login
    
    # Check if token is already set
    hf_token = os.environ.get("HF_TOKEN")
    
    if not hf_token:
        print("Hugging Face login required to access the Llama model.")
        print("Please enter your Hugging Face token:")
        hf_token = getpass.getpass(prompt="Password: ")
        
        # Set environment variable for the current process
        os.environ["HF_TOKEN"] = hf_token
    
    # Login to Hugging Face
    try:
        login(token=hf_token)
        print("Successfully logged into Hugging Face!")
        return True
    except Exception as e:
        print(f"Error logging into Hugging Face: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on qa_dataset.csv")
    parser.add_argument(
        "--input_file", 
        type=str, 
        default="qa_dataset.csv",
        help="Path to input data file (default: qa_dataset.csv)"
    )
    parser.add_argument(
        "--processed_data_dir", 
        type=str, 
        default=Config.PROCESSED_DATA_DIR,
        help="Directory to save preprocessed data"
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
        "--eval_dir", 
        type=str, 
        default="./evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--use_8bit", 
        action="store_true",
        help="Use 8-bit quantization for training"
    )
    parser.add_argument(
        "--skip_preprocessing", 
        action="store_true",
        help="Skip preprocessing if data is already processed"
    )
    parser.add_argument(
        "--skip_training", 
        action="store_true",
        help="Skip training if model is already fine-tuned"
    )
    parser.add_argument(
        "--skip_evaluation", 
        action="store_true",
        help="Skip evaluation step"
    )
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_random_seed(Config.SEED)
    
    # Create necessary directories
    os.makedirs(args.processed_data_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.eval_dir, exist_ok=True)
    
    # Login to Hugging Face
    if not login_to_hugging_face():
        print("ERROR: Hugging Face login failed. Please check your token and try again.")
        sys.exit(1)
    
    # Step 1: Preprocess data
    if not args.skip_preprocessing:
        print(f"\n=== Step 1: Preprocessing data from {args.input_file} ===")
        print(f"Loading data from {args.input_file}...")
        df = load_data(args.input_file)
        
        print("Preprocessing data...")
        preprocessed_df = preprocess_data(df)
        
        print("Splitting data into train, validation, and test sets...")
        train_df, val_df, test_df = split_data(preprocessed_df)
        
        print(f"Saving preprocessed data to {args.processed_data_dir}...")
        save_data(train_df, val_df, test_df, args.processed_data_dir)
    else:
        print("\n=== Skipping preprocessing step ===")
    
    # Step 2: Fine-tune model
    if not args.skip_training:
        print(f"\n=== Step 2: Fine-tuning model ===")
        print(f"Loading preprocessed data from {args.processed_data_dir}...")
        train_dataset, val_dataset = load_processed_data(args.processed_data_dir)
        
        print(f"Loading model: {args.model_name}")
        model, tokenizer = load_model_and_tokenizer(args.model_name, args.use_8bit)
        
        print("Setting up LoRA for parameter-efficient fine-tuning...")
        lora_model = setup_lora(model)
        
        print("Training model...")
        train_model(lora_model, tokenizer, train_dataset, val_dataset, args.output_dir)
    else:
        print("\n=== Skipping training step ===")
    
    # Step 3: Evaluate model
    if not args.skip_evaluation:
        print(f"\n=== Step 3: Evaluating model ===")
        test_file = os.path.join(args.processed_data_dir, "test.json")
        print(f"Loading test data from {test_file}...")
        test_dataset = load_test_data(test_file)
        
        print(f"Loading fine-tuned model from {args.output_dir}...")
        model, tokenizer = load_eval_model(args.output_dir, args.model_name)
        
        print("Evaluating model...")
        evaluation_results, predictions = evaluate_model(model, tokenizer, test_dataset)
        
        print("\nEvaluation Results:")
        for metric, value in evaluation_results.items():
            print(f"{metric}: {value}")
        
        print(f"\nSaving evaluation results to {args.eval_dir}...")
        save_evaluation_results(evaluation_results, predictions, test_dataset, args.eval_dir)
    else:
        print("\n=== Skipping evaluation step ===")
    
    print("\n=== Fine-tuning pipeline completed! ===")

if __name__ == "__main__":
    main()
