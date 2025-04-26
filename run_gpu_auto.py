#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Automated script to run the fine-tuning pipeline using GPU without requiring user input."""

import os
import sys
import subprocess
import torch
from huggingface_hub import login
from huggingface_hub.utils import HfHubHTTPError
from config import Config

def main():
    # Check if GPU is available and configure PyTorch accordingly
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\nGPU detected: {gpu_name} with {gpu_memory:.2f} GB memory")
        
        # Display CUDA version
        cuda_version = torch.version.cuda
        print(f"CUDA Version: {cuda_version}")
        
        # Enable GPU usage by updating config
        Config.DEVICE = "cuda"
        
        # Adjust batch sizes to better utilize GPU
        Config.BATCH_SIZE = 4
        Config.EVAL_BATCH_SIZE = 4
        Config.GRADIENT_ACCUMULATION_STEPS = 2
    else:
        print("\nWARNING: No GPU detected! Running on CPU will be very slow.")
        print("Proceeding with CPU-based training...")
        Config.DEVICE = "cpu"
        Config.BATCH_SIZE = 1
        Config.EVAL_BATCH_SIZE = 1
        Config.GRADIENT_ACCUMULATION_STEPS = 4
    
    # Try to log in to Hugging Face using token from environment variable
    # This is a silent attempt - if it fails, we'll continue anyway
    try:
        token = os.environ.get('HF_TOKEN')
        if token:
            login(token=token, add_to_git_credential=False)
            print("Successfully logged into Hugging Face!")
        else:
            print("No Hugging Face token found in environment. Some models may not be accessible.")
            print("To log in manually, please run 'huggingface-cli login' separately.")
    except Exception as e:
        print(f"Hugging Face login error: {e}")
        print("Continuing without Hugging Face login...")
    
    # Create necessary directories
    os.makedirs(Config.PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Check if data file exists
    data_file = "qa_dataset.csv"
    if not os.path.exists(data_file):
        print(f"\nERROR: Could not find {data_file}")
        return
    
    # Run the fine-tuning pipeline
    print("\nStarting fine-tuning pipeline...")
    
    # Set use_8bit flag if GPU is available for more efficient memory usage
    use_8bit_flag = "--use_8bit" if torch.cuda.is_available() else ""
    
    command = [sys.executable, "run_finetuning.py"]
    if use_8bit_flag:
        command.append(use_8bit_flag)
    
    try:
        subprocess.run(command, check=True)
        print("\nFine-tuning completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Fine-tuning failed with exit code {e.returncode}")
        return

if __name__ == "__main__":
    main()
