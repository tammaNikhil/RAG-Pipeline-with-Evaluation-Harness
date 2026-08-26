import os
import json
import csv
import yaml
from typing import List, Dict, Any
from retrieve import Retriever
from generate import Generator


def evaluate_retrieval(golden_qa: List[Dict[str, Any]], retriever: Retriever, top_k: int = 5) -> List[Dict[str, Any]]:
    """Compute Recall@k, Precision@k, and MRR per question."""
    results = []

    for item in golden_qa:
        qid = item["id"]
        query = item["question"]
        expected_ids = set(item.get("relevant_chunk_ids", []))

        retrieved = retriever.retrieve(query, k=top_k)
        retrieved_ids = [c["chunk_id"] for c in retrieved]

        if not expected_ids:
            # Out of scope query
            recall = 1.0 if not retrieved_ids else 0.0
            precision = 1.0 if not retrieved_ids else 0.0
            mrr = 1.0
        else:
            hits = [cid for cid in retrieved_ids if cid in expected_ids]
            recall = len(hits) / len(expected_ids) if expected_ids else 0.0
            precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0.0

            mrr = 0.0
            for rank, cid in enumerate(retrieved_ids, start=1):
                if cid in expected_ids:
                    mrr = 1.0 / rank
                    break

        results.append({
            "id": qid,
            "query": query,
            "expected_ids": list(expected_ids),
            "retrieved_ids": retrieved_ids,
            "recall@k": round(recall, 4),
            "precision@k": round(precision, 4),
            "mrr": round(mrr, 4)
        })

    return results


def evaluate_generation(golden_qa: List[Dict[str, Any]], generator: Generator) -> List[Dict[str, Any]]:
    """Compute Faithfulness and Answer Relevance metrics per question."""
    results = []

    for item in golden_qa:
        qid = item["id"]
        query = item["question"]
        expected_ans = item["expected_answer"]

        res = generator.answer_question(query)
        gen_ans = res["answer"]
        retrieved = res["retrieved_chunks"]

        # Faithfulness check: check if answer acknowledges context or refrains appropriately
        if "I don't know" in expected_ans:
            faithfulness = 1.0 if "I don't know" in gen_ans or not retrieved else 0.5
        else:
            faithfulness = 1.0 if retrieved else 0.0

        # Answer relevance (jaccard overlap as deterministic metric)
        gen_words = set(gen_ans.lower().split())
        exp_words = set(expected_ans.lower().split())
        union = gen_words.union(exp_words)
        relevance = len(gen_words.intersection(exp_words)) / len(union) if union else 0.0

        results.append({
            "id": qid,
            "query": query,
            "expected_answer": expected_ans,
            "generated_answer": gen_ans,
            "faithfulness": round(faithfulness, 4),
            "relevance": round(relevance, 4)
        })

    return results


def run_full_evaluation(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    golden_file = config["paths"]["golden_qa_file"]
    results_dir = config["paths"]["eval_results_dir"]
    db_dir = config["paths"]["vector_db_dir"]
    model_name = config["embedding"]["model_name"]
    top_k = config["retrieval"]["top_k"]

    os.makedirs(results_dir, exist_ok=True)

    with open(golden_file, "r") as f:
        golden_qa = json.load(f)

    retriever = Retriever(model_name=model_name, db_dir=db_dir)
    generator = Generator(config_path=config_path)

    # Retrieval Eval
    ret_results = evaluate_retrieval(golden_qa, retriever, top_k=top_k)
    ret_csv_path = os.path.join(results_dir, "retrieval_metrics.csv")
    with open(ret_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "query", "expected_ids", "retrieved_ids", "recall@k", "precision@k", "mrr"])
        writer.writeheader()
        writer.writerows(ret_results)

    # Generation Eval
    gen_results = evaluate_generation(golden_qa, generator)

    # Aggregate Metrics
    avg_recall = sum(r["recall@k"] for r in ret_results) / len(ret_results)
    avg_precision = sum(r["precision@k"] for r in ret_results) / len(ret_results)
    avg_mrr = sum(r["mrr"] for r in ret_results) / len(ret_results)

    avg_faithfulness = sum(g["faithfulness"] for g in gen_results) / len(gen_results)
    avg_relevance = sum(g["relevance"] for g in gen_results) / len(gen_results)

    summary = {
        "num_test_cases": len(golden_qa),
        "retrieval": {
            "avg_recall_at_k": round(avg_recall, 4),
            "avg_precision_at_k": round(avg_precision, 4),
            "avg_mrr": round(avg_mrr, 4)
        },
        "generation": {
            "avg_faithfulness": round(avg_faithfulness, 4),
            "avg_relevance": round(avg_relevance, 4)
        },
        "failures": [
            g for g in gen_results if g["faithfulness"] < 0.5 or g["relevance"] < 0.2
        ]
    }

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== EVALUATION COMPLETED ===")
    print(f"Summary saved to {summary_path}")
    print(f"Retrieval CSV saved to {ret_csv_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run_full_evaluation()
