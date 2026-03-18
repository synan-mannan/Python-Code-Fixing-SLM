import json
import os
from typing import List
import argparse

PROMPT_TEMPLATE = """### Python Error
{error_traceback}

### Code
{code}

### Task
Explain the error and suggest a fix.

### Response
{explanation}

Suggested Fix:
{fix}"""

def format_dataset(input_file: str, output_file: str):
    """Format cleaned data to training JSONL."""
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            prompt = PROMPT_TEMPLATE.format(
                error_traceback=entry['error_traceback'],
                code=entry['code'],
                explanation=entry['explanation'],
                fix=entry['fix']
            )
            data.append({
                'prompt': prompt,
                'source': entry['source']
            })
    
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Formatted {len(data)} samples to {output_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='dataset/cleaned.jsonl')
    parser.add_argument('--output-prefix', default='dataset/formatted')
    args = parser.parse_args()
    
    # Format splits
    format_dataset('dataset/train.jsonl', f'{args.output_prefix}_train.jsonl')
    format_dataset('dataset/validation.jsonl', f'{args.output_prefix}_val.jsonl')
    format_dataset('dataset/test.jsonl', f'{args.output_prefix}_test.jsonl')

if __name__ == '__main__':
    main()

