import pandas as pd
import argparse
import re
import os
import constants
from utils import parse_string

# Update to include the src folder path
SRC_FOLDER = "src/"

def check_file(filename):
    if not filename.lower().endswith('.csv'):
        raise ValueError("Only CSV files are allowed.")
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Filepath: {filename} invalid or not found.")

def sanity_check(test_filename, output_filename):
    check_file(test_filename)
    check_file(output_filename)
    
    try:
        test_df = pd.read_csv(test_filename)
        output_df = pd.read_csv(output_filename)
    except Exception as e:
        raise ValueError(f"Error reading the CSV files: {e}")
    
    if 'index' not in test_df.columns:
        raise ValueError("Test CSV file must contain the 'index' column.")
    
    if 'index' not in output_df.columns or 'prediction' not in output_df.columns:
        raise ValueError("Output CSV file must contain 'index' and 'prediction' columns.")
    
    missing_index = set(test_df['index']).difference(set(output_df['index']))
    if len(missing_index) != 0:
        print(f"Missing index in test file: {missing_index}")
        
    extra_index = set(output_df['index']).difference(set(test_df['index']))
    if len(extra_index) != 0:
        print(f"Extra index in output file: {extra_index}")
        
    output_df.apply(lambda x: parse_string(x['prediction']), axis=1)
    print(f"Parsing successful for file: {output_filename}")

if __name__ == "__main__":
    # Default file paths
    default_test_file = os.path.join(SRC_FOLDER, "test.csv")
    default_output_file = os.path.join(SRC_FOLDER, "test_out.csv")
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run sanity check on a CSV file.")
    parser.add_argument("--test_filename", type=str, default=default_test_file, help="The test CSV file name.")
    parser.add_argument("--output_filename", type=str, default=default_output_file, help="The output CSV file name to check.")
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Run sanity check
        sanity_check(args.test_filename, args.output_filename)
    except Exception as e:
        print('Error:', e)