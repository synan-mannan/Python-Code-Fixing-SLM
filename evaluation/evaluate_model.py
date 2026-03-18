import json
import time
import statistics
from pathlib import Path
from collections import Counter
from agent.debugger_agent import PythonDebuggerAgent

def load_test_dataset(file_path: str = 'dataset/test.jsonl'):
    
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def evaluate_agent(test_data, model_path='models/slm-debugger'):
    
    agent = PythonDebuggerAgent(model_path)
    
    latencies = []
    error_types = []
    results = []
    
    print("Running evaluation...")
    for i, item in enumerate(test_data[:50]):  # Sample 50
        start = time.time()
        result = agent.debug(item['code'], item['error_traceback'])
        latency = time.time() - start
        
        latencies.append(latency)
        error_types.append(result['error_type'])
        results.append(result)
        
        print(f"Sample {i+1}: {result['error_type']} - {latency:.2f}s")
    
    # Metrics
    metrics = {
        'avg_latency_ms': statistics.mean(latencies) * 1000,
        'error_types_diversity': dict(Counter(error_types)),
        'sample_size': len(latencies),
        'max_latency_ms': max(latencies) * 1000,
        'min_latency_ms': min(latencies) * 1000,
    }
    
    print("\\n=== EVALUATION RESULTS ===")
    print("| Metric | Value |")
    print("|--------|-------|")
    print(f"| Avg Latency | {metrics['avg_latency_ms']:.2f} ms |")
    print(f"| Samples | {metrics['sample_size']} |")
    print(f"| Error Types | {len(metrics['error_types_diversity'])} |")
    
    # Save
    with open('evaluation/results.json', 'w') as f:
        json.dump({'metrics': metrics, 'results': results}, f, indent=2)
    
    return metrics

if __name__ == '__main__':
    test_data = load_test_dataset()
    evaluate_agent(test_data)

