# Python Debug AI Agent

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

A production-grade end-to-end system for building a Domain-Specialized AI Agent that debugs Python errors using a fine-tuned Small Language Model (SLM).

## 🎯 Project Goal

**Input:** Python code + error traceback  
**Output:** Structured debugging result:

```json
{
  "error_type": "NameError",
  "explanation": "Variable referenced before assignment",
  "suggested_fix": "Initialize the variable before using it"
}
```

## 🏗️ Architecture

1. **Dataset Pipeline** - Scrapes StackOverflow/GitHub for Python errors
2. **SLM Fine-tuning** - QLoRA on TinyLlama-1.1B
3. **AI Agent** - LangChain agent with AST tools + SLM
4. **FastAPI Service** - Production API endpoint
5. **Evaluation Framework** - Metrics & benchmarking

## 📁 Project Structure

```
python-debug-ai-agent/
├── data_pipeline/          # Dataset creation
├── training/               # QLoRA fine-tuning
├── models/                 # Saved models
├── agent/                  # LangChain agent
├── evaluation/             # Metrics & eval
├── app/                    # FastAPI service
├── dataset/                # Generated data
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
cd python-debug-ai-agent
python -m venv venv
# Windows:
venv\\Scripts\\activate
# Linux/Mac:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Dataset Generation

```bash
# Add API keys to .env
python -m data_pipeline.data_scraper
python -m data_pipeline.github_issue_scraper
python data_pipeline/data_cleaner.py
python data_pipeline/dataset_formatter.py
```

### 3. Train SLM

```bash
accelerate launch training/train_slm.py
```

### 4. Run Agent API

```bash
uvicorn app.api_server:app --reload --host 0.0.0.0 --port 8000
```

### 5. Test API

```bash
curl -X POST "http://localhost:8000/debug" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "print(undefined_var)",
    "traceback": "NameError: name 'undefined_var' is not defined"
  }'
```

## Dataset Strategy

- **Sources:** StackOverflow API, GitHub Issues API
- **Size:** 3000+ samples (train/val/test split 80/10/10)
- **Format:** JSONL with prompt-response pairs

## Model Selection

- **Primary:** TinyLlama-1.1B (SLM, efficient)
- **Comparison:** Mistral-7B (larger baseline)

## Fine-tuning Approach

- **Technique:** QLoRA (4-bit quantization)
- **Hyperparams:** lr=2e-4, epochs=3, batch=8, max_seq=512

## Agent Architecture

```
User Input → Preprocessing → SLM Inference → Tool Chain → Structured Output
Tools: AST Parser, Static Analyzer
```

## Limitations

- Scraping requires API keys (StackOverflow/GitHub)
- Training needs GPU (A100 recommended)
- Fix suggestions are suggestions (human review needed)
