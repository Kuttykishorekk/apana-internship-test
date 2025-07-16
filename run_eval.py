import os
import sys
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
from langsmith import Client

# Load environment variables
load_dotenv()

# Import evaluation modules
from src.evaluate import (
    load_dataset,
    evaluate_response,
    keyword_overlap_score,
    llm_judge_score,
    bleu_score,
    rouge_score_all,
    hallucination_flag,
    print_summary
)
from src.generate import generate_response

# Initialize LangSmith client
try:
    client = Client()
    LANGSMITH_ENABLED = True
except Exception as e:
    print(f"[WARNING] LangSmith client initialization failed: {e}")
    LANGSMITH_ENABLED = False

def create_or_get_dataset(dataset_name: str):
    """Create or get existing LangSmith dataset."""
    if not LANGSMITH_ENABLED:
        return None
    
    try:
        # Try to get existing dataset first
        datasets = client.list_datasets()
        for dataset in datasets:
            if dataset.name == dataset_name:
                print(f"✅ Using existing dataset: {dataset_name}")
                return dataset
        
        # Create new dataset if doesn't exist
        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description=f"Apana LLM Evaluation Dataset - Created {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        print(f"✅ Created new dataset: {dataset_name}")
        return dataset
    except Exception as e:
        print(f"[ERROR] Dataset creation/retrieval failed: {e}")
        return None

def evaluate_single_example(prompt: str, reference: str, index: int, total: int):
    """Evaluate a single example and return all metrics."""
    print(f"\n[{index+1}/{total}] Evaluating: {prompt[:50]}...")
    
    # Generate response
    generated = generate_response(prompt)
    
    if generated.startswith("[ERROR]"):
        print(f"❌ Generation failed for example {index+1}")
        return None
    
    # Calculate all metrics
    metrics = {
        'generated_answer': generated,
        'similarity_score': evaluate_response(reference, generated),
        'keyword_overlap_score': keyword_overlap_score(reference, generated),
        'llm_judge_score': llm_judge_score(prompt, reference, generated),
        'bleu_score': bleu_score(reference, generated),
        'hallucination': hallucination_flag(prompt, reference, generated)
    }
    
    # Add ROUGE scores
    rouge_scores = rouge_score_all(reference, generated)
    metrics.update(rouge_scores)
    
    print(f"✅ Completed evaluation for example {index+1}")
    return metrics

def log_to_langsmith(dataset, prompt: str, reference: str, metrics: dict):
    """Log results to LangSmith."""
    if not LANGSMITH_ENABLED or not dataset:
        return
    
    try:
        client.create_example(
            dataset_id=dataset.id,
            inputs={"prompt": prompt},
            outputs={"response": metrics['generated_answer']},
            metadata={
                "reference_answer": reference,
                "similarity_score": metrics['similarity_score'],
                "keyword_overlap_score": metrics['keyword_overlap_score'],
                "llm_judge_score": metrics['llm_judge_score'],
                "bleu_score": metrics['bleu_score'],
                "rouge1": metrics['rouge1'],
                "rouge2": metrics['rouge2'],
                "rougeL": metrics['rougeL'],
                "hallucination": metrics['hallucination']
            }
        )
    except Exception as e:
        print(f"[WARNING] LangSmith logging failed: {e}")

def main():
    """Main evaluation pipeline."""
    print("🚀 Starting Apana LLM Evaluation System")
    print("="*50)
    
    # Load dataset
    df = load_dataset("data/eval_set.json")
    if df.empty:
        print("❌ No data loaded. Exiting.")
        return
    
    print(f"📊 Loaded {len(df)} examples for evaluation")
    
    # Setup LangSmith dataset
    dataset_name = f"apana_llm_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset = create_or_get_dataset(dataset_name)
    
    # Initialize results storage
    results = []
    
    # Process each example
    for index, row in df.iterrows():
        prompt = row["prompt"]
        reference = row["reference_answer"]
        
        metrics = evaluate_single_example(prompt, reference, index, len(df))
        
        if metrics is None:
            continue
        
        # Add to results
        result_row = {
            'prompt': prompt,
            'reference_answer': reference,
            **metrics
        }
        results.append(result_row)
        
        # Log to LangSmith
        log_to_langsmith(dataset, prompt, reference, metrics)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    if results_df.empty:
        print("❌ No successful evaluations. Exiting.")
        return
    
    # Save results
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"output/results_{timestamp}.csv"
    results_df.to_csv(output_file, index=False, encoding="utf-8")
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary
    print_summary(results_df)
    
    print("\n🎉 Evaluation completed successfully!")

if __name__ == "__main__":
    main()