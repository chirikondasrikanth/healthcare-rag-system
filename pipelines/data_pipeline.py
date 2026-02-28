import json
import pandas as pd
from datetime import datetime
import os

def load_raw_data(filepath):
    """Load raw CMS FAQ data"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} records")
    return data

def clean_data(data):
    """Clean and normalize the data"""
    df = pd.DataFrame(data)
    
    # Clean text fields
    df['question'] = df['question'].str.strip().str.lower()
    df['answer'] = df['answer'].str.strip()
    df['category'] = df['category'].str.strip()
    
    # Add metadata
    df['processed_at'] = datetime.now().isoformat()
    df['text'] = "Question: " + df['question'] + "\nAnswer: " + df['answer']
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    
    print(f"Cleaned {len(df)} records")
    return df

def save_processed_data(df, output_path):
    """Save processed data"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_json(output_path, orient='records', indent=2)
    print(f"Saved processed data to {output_path}")

def run_pipeline():
    print("Starting Data Pipeline...")
    
    # Step 1 - Load
    raw_data = load_raw_data('data/raw/cms_faq_data.json')
    
    # Step 2 - Clean
    df = clean_data(raw_data)
    
    # Step 3 - Save
    save_processed_data(df, 'data/processed/cms_faq_processed.json')
    
    print("\nPipeline Complete!")
    print(df[['id', 'category', 'word_count']].to_string())

if __name__ == "__main__":
    run_pipeline()