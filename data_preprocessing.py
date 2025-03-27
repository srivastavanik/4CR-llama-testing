#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data preprocessing for LLM fine-tuning."""

import os
import json
import argparse
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split
from config import Config

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from csv file with robust error handling."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    print(f"Loading data from {file_path}...")
    try:
        # First try to load with default settings
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='warn')
        print("Successfully loaded data with default settings")
        return df
    except pd.errors.ParserError as e:
        print(f"Warning: Default CSV parsing failed: {e}")
        print("Attempting alternative parsing methods...")
        
        try:
            # Try with different quotechar and escapechar
            df = pd.read_csv(
                file_path,
                encoding='utf-8',
                on_bad_lines='warn',
                quotechar='"',
                escapechar='\\'
            )
            print("Successfully loaded data with alternative settings")
            return df
        except pd.errors.ParserError as e:
            print(f"Warning: Alternative parsing failed: {e}")
            print("Attempting to read file line by line...")
            
            # As a last resort, read line by line and clean data
            lines = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        # Try to parse line as CSV
                        line = line.strip()
                        if line:  # Skip empty lines
                            lines.append(line)
                    except Exception as e:
                        print(f"Warning: Skipping malformed line: {e}")
            
            # Create DataFrame from cleaned lines
            df = pd.DataFrame([line.split(',') for line in lines])
            print(f"Successfully loaded data with {len(df)} rows")
            return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data for fine-tuning with robust error handling."""
    # Check if all required columns exist
    required_columns = ["prompt", "context", "answer"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_columns)
    
    # Clean text data
    def clean_text(text):
        if pd.isna(text):
            return ""
        try:
            text = str(text).strip()
            # Remove any control characters
            text = ''.join(c for c in text if ord(c) >= 32)
            return text
        except:
            return ""
    
    # Clean all text columns
    df[required_columns] = df[required_columns].applymap(clean_text)
    
    # Handle the context column - if it contains list-like strings, parse them
    if df["context"].dtype == object and df["context"].iloc[0].startswith("["):
        import ast
        def parse_context(ctx):
            try:
                if isinstance(ctx, str) and ctx.startswith("["):
                    return "\n\n".join(ast.literal_eval(ctx))
                return ctx
            except:
                return ctx
        df["context"] = df["context"].apply(parse_context)
    
    # Combine text and context if both exist
    if "text" in df.columns:
        df["context"] = df.apply(
            lambda row: f"{row['context']}\n\nAdditional information: {row['text']}"
            if not pd.isna(row.get('text')) else row['context'], 
            axis=1
        )
    
    # Format data for the model
    df["input_text"] = df.apply(
        lambda row: Config.PROMPT_TEMPLATE.format(
            context=row["context"],
            prompt=row["prompt"]
        ),
        axis=1
    )
    
    # Drop any rows with empty text after cleaning
    df = df[df["input_text"].str.strip().astype(bool)]
    
    return df[['input_text', 'answer', 'prompt', 'context']]

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation and test sets."""
    # First split off the test set
    train_val_df, test_df = train_test_split(
        df, test_size=Config.TRAIN_TEST_SPLIT, random_state=Config.SEED
    )
    
    # Then split the train_val into train and validation
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=Config.VALIDATION_SPLIT, 
        random_state=Config.SEED
    )
    
    return train_df, val_df, test_df

def save_data(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str):
    """Save preprocessed data to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert DataFrames to lists of dictionaries
    train_data = train_df.to_dict(orient='records')
    val_data = val_df.to_dict(orient='records')
    test_data = test_df.to_dict(orient='records')
    
    # Save to JSON files
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    with open(os.path.join(output_dir, 'validation.json'), 'w') as f:
        json.dump(val_data, f, indent=2)
    
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Saved preprocessed data to {output_dir}")
    print(f"Train set size: {len(train_data)}")
    print(f"Validation set size: {len(val_data)}")
    print(f"Test set size: {len(test_data)}")

def main():
    parser = argparse.ArgumentParser(description="Preprocess data for LLM fine-tuning")
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True, 
        help="Path to input data file (CSV or JSON)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=Config.PROCESSED_DATA_DIR,
        help="Directory to save preprocessed data"
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    df = load_data(args.input_file)
    
    # Preprocess data
    print("Preprocessing data...")
    preprocessed_df = preprocess_data(df)
    
    # Split data
    print("Splitting data into train, validation, and test sets...")
    train_df, val_df, test_df = split_data(preprocessed_df)
    
    # Save data
    save_data(train_df, val_df, test_df, args.output_dir)
    
if __name__ == "__main__":
    main()
