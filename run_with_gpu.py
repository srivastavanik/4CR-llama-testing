#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Simplified script to run the fine-tuning pipeline using GPU."""

import os
import sys
import subprocess
import torch
from getpass import getpass
from huggingface_hub import login
from config import Config

def main():
    # Check if GPU is available
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\nGPU detected: {gpu_name} with {gpu_memory:.2f} GB memory")
        
        # Display CUDA version
        cuda_version = torch.version.cuda
        print(f"CUDA Version: {cuda_version}")
    else:
        print("\nWARNING: No GPU detected! Running on CPU will be very slow.")
        user_input = input("Do you want to continue without GPU? (y/n): ").lower()
        if user_input != 'y':
            print("Exiting script.")
            return
    
    # Check and login to Hugging Face if not already logged in
    try:
        # Try to get token from environment variable first
        token = os.environ.get('HF_TOKEN')
        
        # If not in environment, prompt user
        if not token:
            print("\nHugging Face login required to access the Llama model.")
            print("Please enter your Hugging Face token:")
            token = getpass()
        
        # Login to Hugging Face
        if token:
            login(token=token)
            print("Successfully logged into Hugging Face!")
        else:
            print("No token provided. Some models may not be accessible.")
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
    print("\nStarting fine-tuning pipeline with GPU support...")
    command = [sys.executable, "run_finetuning.py", "--use_8bit"]
    
    try:
        subprocess.run(command, check=True)
        print("\nFine-tuning completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Fine-tuning failed with exit code {e.returncode}")
        return

if __name__ == "__main__":
    main()
