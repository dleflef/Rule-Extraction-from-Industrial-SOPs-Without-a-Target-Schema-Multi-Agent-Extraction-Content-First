import os
import csv
import json
import glob
from collections import Counter

# Paths based on your architecture
LAYER_2C_DIR = "outputs/layer_2c"
GT_CSV_PATH = os.path.join("layers", "extracted_seed","dataset", "kg_seeds", "ground_truth.csv")

def load_ground_truth(csv_path: str) -> list:
    """Parses the ground truth CSV into a list of expected relationship triples."""
    expected_triples = []
    if not os.path.exists(csv_path):
        print(f"ERROR: Ground truth CSV not found at {csv_path}")
        return expected_triples

    with open(csv_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            station = row.get("station", "").strip()
            sensor = row.get("sensor", "").strip()
            condition = row.get("condition", "").strip()
            action = row.get("action", "").strip()

            # 1. Structural Rule: Component monitors Sensor
            if station and sensor:
                expected_triples.append({
                    "subject": station,
                    "predicate": "monitors",
                    "object": sensor
                })
            
            # 2. Causal Rule: Condition triggers Action
            if condition and action:
                expected_triples.append({
                    "subject": condition,
                    "predicate": "triggers",
                    "object": action
                })
                
    return expected_triples

def load_extracted_triples(layer_2c_dir: str) -> list:
    """Loads all aligned triples extracted by Agent 2C."""
    extracted_triples = []
    json_files = glob.glob(os.path.join(layer_2c_dir, "*_agent2c_aligned.json"))
    
    for file_path in json_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for result in data.get("results", []):
                for relation in result.get("aligned_relations", []):
                    extracted_triples.append({
                        "subject": relation.get("subject", ""),
                        "predicate": relation.get("predicate", ""),
                        "object": relation.get("object", "")
                    })
    return extracted_triples

def calculate_similarity(str1: str, str2: str) -> float:
    """A basic similarity function to match LLM strings to CSV strings."""
    # Convert to lowercase and split into words
    words1 = set(str1.lower().replace("°c", "c").split())
    words2 = set(str2.lower().replace("°c", "c").split())
    
    if not words1 or not words2:
        return 0.0
        
    intersection = words1.intersection(words2)
    # Calculate overlap percentage based on the shorter string
    return len(intersection) / min(len(words1), len(words2))

def evaluate_pipeline():
    print(f"{'='*60}")
    print("LAYER 3: SYSTEM EVALUATION (BASELINE)")
    print(f"{'='*60}")

    expected = load_ground_truth(GT_CSV_PATH)
    extracted = load_extracted_triples(LAYER_2C_DIR)
    
    print(f"Loaded {len(expected)} expected rules from Ground Truth.")
    print(f"Loaded {len(extracted)} extracted triples from Agent 2C.\n")

    # Tally up the predicates to see what the LLM favors
    expected_predicates = dict(Counter([r["predicate"] for r in expected]))
    extracted_predicates = dict(Counter([r["predicate"] for r in extracted]))

    true_positives = []
    false_positives = []
    false_negatives = expected.copy() # Start with all expected, remove as we find them

    # Match extracted triples to expected triples
    for ext in extracted:
        match_found = False
        matched_exp = None
        match_score = 0.0

        for exp in false_negatives:
            # Must have the same predicate to even be considered
            if ext["predicate"] != exp["predicate"]:
                continue
                
            if ext["predicate"] == "monitors":
                # For structural nodes, require exact matches
                if ext["subject"] == exp["subject"] and ext["object"] == exp["object"]:
                    match_found = True
                    matched_exp = exp
                    match_score = 1.0
                    break
            
            elif ext["predicate"] == "triggers":
                # For causal rules, use semantic overlap (threshold 0.6 = 60% word overlap)
                sub_sim = calculate_similarity(ext["subject"], exp["subject"])
                obj_sim = calculate_similarity(ext["object"], exp["object"])
                
                if sub_sim >= 0.6 and obj_sim >= 0.5:
                    match_found = True
                    matched_exp = exp
                    match_score = (sub_sim + obj_sim) / 2.0
                    break

        if match_found:
            # Enrich the True Positive data with match evidence
            tp_record = ext.copy()
            tp_record["matched_ground_truth"] = matched_exp
            tp_record["average_similarity_score"] = round(match_score, 3)
            true_positives.append(tp_record)
            
            false_negatives.remove(matched_exp) # Remove from FN since we found it
        else:
            false_positives.append(ext)

    # Calculate Metrics
    TP = len(true_positives)
    FP = len(false_positives)
    FN = len(false_negatives)

    # Prevent division by zero
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    print("--- BASELINE METRICS ---")
    print(f"True Positives (TP) : {TP}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print("-" * 24)
    print(f"Precision : {precision:.3f}")
    print(f"Recall    : {recall:.3f}")
    print(f"F1 Score  : {f1_score:.3f}")
    print(f"{'='*60}")
    
    # Build a much richer JSON report
    report_payload = {
        "summary": {
            "totals": {
                "expected_ground_truth_rules": len(expected),
                "extracted_ai_triples": len(extracted)
            },
            "metrics": {
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1_score, 4)
            },
            "counts": {
                "true_positives": TP,
                "false_positives": FP,
                "false_negatives": FN
            },
            "predicate_breakdown": {
                "expected_in_ground_truth": expected_predicates,
                "extracted_by_agent_2c": extracted_predicates
            }
        },
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives
    }

    # Save the detailed evaluation for the professor
    output_report = "outputs/baseline_evaluation_report.json"
    os.makedirs(os.path.dirname(output_report), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=4)
        
    print(f"Detailed rule breakdown saved to: {output_report}")
    print("Ready for presentation!")

if __name__ == "__main__":
    evaluate_pipeline()