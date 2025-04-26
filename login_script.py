#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to log in to Hugging Face interactively."""

import os
import sys
import subprocess
from getpass import getpass
from huggingface_hub import login

def main():
    print("\n====== Hugging Face Login ======\n")
    print("This script will help you log in to Hugging Face to access the Llama-3.2-3B-Instruct model.")
    print("You'll need a Hugging Face account with access to this model.")
    print("\nPlease enter your Hugging Face token (hidden input): ")
    token = getpass()
    
    if not token.strip():
        print("No token provided. Exiting.")
        return False
    
    try:
        # Try to log in
        login(token=token)
        print("\nSuccessfully logged into Hugging Face!")
        return True
    except Exception as e:
        print(f"\nError logging in: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\nYou can now run the fine-tuning script with:")
        print("python run_finetuning.py --use_8bit")
    else:
        print("\nLogin failed. Please try again.")
