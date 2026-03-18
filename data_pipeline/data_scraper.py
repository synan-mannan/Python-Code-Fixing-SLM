import requests
import json
import time
import os
from datetime import datetime
from typing import List, Dict
import argparse
from dotenv import load_dotenv
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()
SO_API_KEY = os.getenv('SO_API_KEY', '')

BASE_URL = 'https://api.stackexchange.com/2.3'


# 🔹 Fetch answers separately (IMPORTANT)
def fetch_answers(question_id: int) -> List[Dict]:
    url = f"{BASE_URL}/questions/{question_id}/answers"

    params = {
        'order': 'desc',
        'sort': 'votes',
        'site': 'stackoverflow',
        'filter': 'withbody'
    }

    if SO_API_KEY:
        params['key'] = SO_API_KEY

    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return []
        return response.json().get('items', [])
    except Exception as e:
        print(f"Error fetching answers for {question_id}: {e}")
        return []


# 🔹 Extract code using BeautifulSoup
def extract_code_from_body(body: str) -> List[str]:
    soup = BeautifulSoup(body, "html.parser")
    codes = soup.find_all("code")
    return [c.get_text().strip() for c in codes]


# 🔹 Extract traceback (simpler + more robust)
def extract_error_traceback(body: str) -> str:
    import re

    soup = BeautifulSoup(body, "html.parser")
    text = soup.get_text()

    pattern = r'Traceback \(most recent call last\):(.+?)(?=\n\n|\Z)'
    match = re.search(pattern, text, re.DOTALL)

    return match.group(0).strip() if match else ''


# 🔹 Extract accepted answer
def extract_accepted_answer(answers: List[Dict]) -> str:
    for ans in answers:
        if ans.get('is_accepted'):
            soup = BeautifulSoup(ans.get('body', ''), "html.parser")
            return soup.get_text().strip()
    return ''


# 🔹 Main scraper
def scrape_stackoverflow(query_tags: str = 'python;error', max_pages: int = 5) -> List[Dict]:
    data = []

    session = requests.Session()

    for page in range(1, max_pages + 1):
        url = f'{BASE_URL}/search/advanced'

        params = {
            'order': 'desc',
            'sort': 'activity',
            'tagged': 'python',
            'q': 'error',
            'site': 'stackoverflow',
            'page': page,
            'pagesize': 100,
            'filter': 'withbody'
        }

        if SO_API_KEY:
            params['key'] = SO_API_KEY

        print(f"\nFetching page {page}...")

        try:
            response = session.get(url, params=params, timeout=10)

            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                break

            results = response.json()
            items = results.get('items', [])

            print(f"Items fetched: {len(items)}")

            if not items:
                break

            for item in items:
                question_id = item['question_id']

                code_snippets = extract_code_from_body(item['body'])
                traceback = extract_error_traceback(item['body'])

                # Skip low-quality entries
                if not code_snippets or not traceback:
                    continue

                # Fetch answers
                answers = fetch_answers(question_id)
                explanation = extract_accepted_answer(answers)

                entry = {
                    'source': 'stackoverflow',
                    'question_id': question_id,
                    'title': item['title'],
                    'code_snippets': code_snippets,
                    'error_traceback': traceback,
                    'explanation': explanation,
                    'timestamp': item['creation_date']
                }

                data.append(entry)

            # Rate limiting
            time.sleep(0.3)

        except Exception as e:
            print(f"Error on page {page}: {e}")
            continue

    return data


# 🔹 Save dataset
def save_raw_data(data: List[Dict], filename: str):
    os.makedirs('dataset/raw', exist_ok=True)

    path = f'dataset/raw/{filename}'

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\nSaved dataset to: {path}")


# 🔹 Entry point
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tags', default='python;error')
    parser.add_argument('--pages', type=int, default=3)

    args = parser.parse_args()

    data = scrape_stackoverflow(args.tags, args.pages)

    filename = f"so_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_raw_data(data, filename)

    print(f"\n✅ Scraped {len(data)} high-quality entries")