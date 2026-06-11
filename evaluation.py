import asyncio
import aiohttp
import json
import csv
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import List, Dict, Any

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
API_URL = "http://localhost:8001/chat/stream"
TEST_DATA_PATH = Path(__file__).parent / "data" / "test_dataset.csv"
OUTPUT_LOG = Path(__file__).parent / "evaluation_results.jsonl"

# ----------------------------------------------------------------------
# Helper: Send a single query and collect the final answer
# ----------------------------------------------------------------------
async def query_llm(session: aiohttp.ClientSession, query: str) -> str:
    payload = {
        "input_string": query,
        "is_voice": False,
        "stt_confidence": 1.0,
        "library_code": None,
        "latitude": None,
        "longitude": None,
        "user_memory": {}
    }
    full_answer = []
    try:
        async with session.post(API_URL, json=payload) as resp:
            async for line in resp.content:
                line = line.decode('utf-8').strip()
                if line.startswith("event: answer"):
                    # The next line(s) contain the data
                    pass
                elif line.startswith("data: "):
                    data = line[6:]
                    if data and data != "[DONE]":
                        full_answer.append(data)
    except Exception as e:
        print(f"Error: {e}")
        return ""
    return " ".join(full_answer).strip()

# ----------------------------------------------------------------------
# Load test dataset
# ----------------------------------------------------------------------
def load_test_queries():
    queries = []
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            queries.append({
                "query": row["query"],
                "expected_answer": row["expected_answer_text"],
                "domain": row["domain"],
                "expected_context": row["expected_context_snippet"]
            })
    return queries

# ----------------------------------------------------------------------
# Simple evaluation metrics (can be extended)
# ----------------------------------------------------------------------
def compute_metrics(expected: str, actual: str) -> Dict[str, Any]:
    # 1. Exact match (case‑insensitive, stripped)
    exact_match = expected.strip().lower() == actual.strip().lower()
    
    # 2. Keyword overlap (rough indicator)
    exp_words = set(expected.lower().split())
    act_words = set(actual.lower().split())
    if not exp_words:
        precision = recall = f1 = 0.0
    else:
        intersection = exp_words.intersection(act_words)
        precision = len(intersection) / len(act_words) if act_words else 0.0
        recall = len(intersection) / len(exp_words)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# ----------------------------------------------------------------------
# Main evaluation loop
# ----------------------------------------------------------------------
async def run_evaluation():
    queries = load_test_queries()
    print(f"Loaded {len(queries)} test queries.")
    
    results = []
    async with aiohttp.ClientSession() as session:
        for item in tqdm(queries, desc="Evaluating"):
            actual_answer = await query_llm(session, item["query"])
            metrics = compute_metrics(item["expected_answer"], actual_answer)
            
            record = {
                "timestamp": datetime.now().isoformat(),
                "query": item["query"],
                "expected_answer": item["expected_answer"],
                "actual_answer": actual_answer,
                "domain": item["domain"],
                "metrics": metrics
            }
            results.append(record)
    
    # Save detailed results
    with open(OUTPUT_LOG, "w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    # Print summary
    total = len(results)
    exact_matches = sum(1 for r in results if r["metrics"]["exact_match"])
    avg_f1 = sum(r["metrics"]["f1"] for r in results) / total
    
    print("\n" + "="*50)
    print(f"Evaluation Summary (Total queries: {total})")
    print(f"Exact Match Accuracy: {exact_matches/total:.2%}")
    print(f"Average F1 (word overlap): {avg_f1:.4f}")
    print(f"Detailed results saved to {OUTPUT_LOG}")
    print("="*50)

if __name__ == "__main__":
    asyncio.run(run_evaluation())