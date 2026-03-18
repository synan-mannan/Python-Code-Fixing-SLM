import pandas as pd
import json
import re
import os
from typing import List
import argparse
from sklearn.model_selection import train_test_split


def load_raw_data(raw_dir: str = 'dataset/raw') -> List[dict]:
    """Load all raw JSON files."""
    data = []
    for file in os.listdir(raw_dir):
        if file.endswith('.json'):
            try:
                with open(os.path.join(raw_dir, file), 'r', encoding='utf-8') as f:
                    data.extend(json.load(f))
            except Exception as e:
                print(f"Skipping {file}: {e}")
    return data


def clean_entry(entry: dict) -> dict:
    """Clean single entry."""
    # Remove HTML
    title = re.sub(r'<[^>]*>', '', entry.get('title', ''))
    explanation = re.sub(r'<[^>]*>', '', entry.get('explanation', ''))

    # Normalize code/traceback
    code = '\n'.join(entry.get('code_snippets', []))
    traceback = entry.get('error_traceback', '')

    if not traceback and not code:
        return None

    # Extract fix
    fix_pattern = r'(fix|solution|answer):?\s*(.*?)(?=\n\n|\Z)'
    fix_match = re.search(fix_pattern, explanation, re.IGNORECASE | re.DOTALL)
    fix = fix_match.group(2).strip() if fix_match else explanation[:200]

    return {
        'error_traceback': traceback[:800],
        'code': code[:800],
        'explanation': explanation[:400],
        'fix': fix[:400],
        'source': entry.get('source', 'unknown')
    }


def remove_duplicates(data: List[dict]) -> List[dict]:
    """Remove near-duplicates."""
    seen = set()
    unique = []

    for entry in data:
        key = re.sub(r'\s+', ' ', entry['error_traceback'][:200] + entry['code'][:200])
        if key not in seen:
            seen.add(key)
            unique.append(entry)

    return unique


def save_dataset(data: List[dict]):
    """Save cleaned data."""
    df = pd.DataFrame(data)
    os.makedirs('dataset', exist_ok=True)

    train, temp = train_test_split(df, test_size=0.2, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    train.to_json('dataset/train.jsonl', orient='records', lines=True)
    val.to_json('dataset/validation.jsonl', orient='records', lines=True)
    test.to_json('dataset/test.jsonl', orient='records', lines=True)

    print(f"Saved: train={len(train)}, val={len(val)}, test={len(test)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-dir', default='dataset/raw')
    args = parser.parse_args()

    raw_data = load_raw_data(args.raw_dir)

    cleaned = []
    for e in raw_data:
        c = clean_entry(e)
        if c:
            cleaned.append(c)

    unique_data = remove_duplicates(cleaned)

    print(f"Raw: {len(raw_data)}, Cleaned: {len(cleaned)}, Unique: {len(unique_data)}")

    save_dataset(unique_data)