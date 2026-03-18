import requests
import json
import time
import os
from typing import List, Dict
from datetime import datetime
import argparse
import re

GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', '')

HEADERS = {
    'Accept': 'application/vnd.github.v3+json',
    **({'Authorization': f'token {GITHUB_TOKEN}'} if GITHUB_TOKEN else {})
}


def scrape_github_issues(query: str = 'python error', max_pages: int = 10) -> List[Dict]:
    """
    Scrape GitHub issues for Python errors (optimized + safe).
    """
    data = []
    url = 'https://api.github.com/search/issues'

    for page in range(1, max_pages + 1):
        params = {
            #  Better query (higher signal)
            'q': f'{query} "Traceback" language:python is:issue in:body',
            'sort': 'created',
            'order': 'desc',
            'page': page,
            'per_page': 30
        }

        print(f"Fetching GitHub page {page}...")
        response = requests.get(url, headers=HEADERS, params=params)

        #  Handle rate limit properly
        if response.status_code == 403:
            reset_time = int(response.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait_time = max(reset_time - int(time.time()), 5)
            print(f"Rate limited. Waiting {wait_time} seconds...")
            time.sleep(wait_time)
            continue

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break

        issues = response.json().get('items', [])

        if not issues:
            print("No more issues found.")
            break

        for issue in issues:
            body = issue.get('body') or ''

            #  Extract traceback
            trace = extract_traceback(body)

            #  Skip useless entries
            if not trace:
                continue

            entry = {
                'source': 'github',
                'repo': issue['repository_url'].split('/')[-1],
                'title': issue.get('title', ''),
                'code_snippets': [],
                'error_traceback': trace,
                'explanation': body[:500],
                'timestamp': issue.get('created_at', '')
            }

            data.append(entry)

        # Be nice to API
        time.sleep(1)

    return data


def extract_traceback(body: str) -> str:
    """
    Extract Python traceback or error messages from text.
    """
    if not body:
        return ''

    patterns = [
        r'(Traceback[\s\S]+?)(?=\n\n|\Z)',   # Full traceback
        r'(\w+Error: .+)',                  # Single-line errors
    ]

    matches = []
    for pattern in patterns:
        matches.extend(re.findall(pattern, body))

    return '\n'.join(matches)[:1000]


def save_raw_data(data: List[Dict], filename: str):
    os.makedirs('dataset/raw', exist_ok=True)
    path = f'dataset/raw/{filename}'

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved dataset to: {path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', default='python error')
    parser.add_argument('--pages', type=int, default=10)
    args = parser.parse_args()

    data = scrape_github_issues(args.query, args.pages)

    filename = f'github_{args.query.replace(" ", "_")}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    save_raw_data(data, filename)

    print(f"\n Scraped {len(data)} GitHub issues")