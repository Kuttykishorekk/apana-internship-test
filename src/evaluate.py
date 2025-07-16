import os
from dotenv import load_dotenv
load_dotenv()

import json
import re
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

from langsmith import traceable
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# Initialize models
embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # I have researched on Quora, This Model Better for French

llm_judge = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.1,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def load_dataset(path='data/eval_set.json') -> pd.DataFrame:
    """Load evaluation dataset from JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"[ERROR] Dataset file not found: {path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return pd.DataFrame()

def evaluate_response(reference: str, generated: str) -> float:
    """Calculate semantic similarity using multilingual embeddings."""
    try:
        if not reference or not generated:
            return 0.0
        embeddings = embed_model.encode([reference, generated])
        score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return round(max(0.0, float(score)), 4)
    except Exception as e:
        print(f"[ERROR] Similarity calculation failed: {e}")
        return 0.0

def keyword_overlap_score(reference: str, generated: str) -> float:
    """Calculate keyword overlap score."""
    try:
        if not reference or not generated:
            return 0.0
        
        def tokenize(text):
            return set(re.findall(r'\b[a-zA-ZÀ-ÿ]+\b', text.lower()))
        
        ref_tokens = tokenize(reference)
        gen_tokens = tokenize(generated)
        
        if not ref_tokens:
            return 0.0
        
        intersection = len(ref_tokens.intersection(gen_tokens))
        return round(intersection / len(ref_tokens), 4)
    except Exception as e:
        print(f"[ERROR] Keyword overlap calculation failed: {e}")
        return 0.0

@traceable(name="LLM Judge Evaluation")
def llm_judge_score(prompt: str, reference: str, generated: str) -> float:
    """Score response using LLM as judge."""
    scoring_prompt = f"""
Tu es un expert en évaluation de modèles d'IA spécialisé en finance.
Évalue la réponse générée sur une échelle de 0 à 10 selon ces critères :
- Exactitude factuelle (40%)
- Pertinence par rapport à la question (30%)
- Clarté et structure (20%)
- Complétude (10%)

Question posée :
"{prompt}"

Réponse de référence :
"{reference}"

Réponse à évaluer :
"{generated}"

Donne uniquement un score numérique entre 0 et 10 (ex: 7.5) :
"""
    try:
        result = llm_judge.invoke([HumanMessage(content=scoring_prompt)]).content.strip()
        score_match = re.search(r'(\d+(?:\.\d+)?)', result)
        if score_match:
            score = float(score_match.group(1))
            return round(min(10.0, max(0.0, score)), 2)
        return 0.0
    except Exception as e:
        print(f"[ERROR] LLM judge scoring failed: {e}")
        return 0.0

def bleu_score(reference: str, generated: str) -> float:
    """Calculate BLEU score."""
    try:
        if not reference or not generated:
            return 0.0
        smoothie = SmoothingFunction().method4
        return round(sentence_bleu([reference.split()], generated.split(), smoothing_function=smoothie), 4)
    except Exception as e:
        print(f"[ERROR] BLEU calculation failed: {e}")
        return 0.0

def rouge_score_all(reference: str, generated: str) -> dict:
    """Calculate all ROUGE scores."""
    try:
        if not reference or not generated:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, generated)
        return {
            "rouge1": round(scores['rouge1'].fmeasure, 4),
            "rouge2": round(scores['rouge2'].fmeasure, 4),
            "rougeL": round(scores['rougeL'].fmeasure, 4)
        }
    except Exception as e:
        print(f"[ERROR] ROUGE calculation failed: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

@traceable(name="LLM Hallucination Detection with Reason")
def hallucination_flag(prompt: str, reference: str, generated: str) -> dict:
    """
    Detect hallucinations and provide rationale.
    Returns dict: {"flag": YES/NO, "reason": text}
    """
    scoring_prompt = f"""
Tu es un expert en détection d'hallucinations dans les réponses d'IA.
Analyse si la réponse générée contient des informations incorrectes ou inventées par rapport à la réponse de référence.

Question :
"{prompt}"

Réponse de référence :
"{reference}"

Réponse à analyser :
"{generated}"

Si la réponse contient des hallucinations, réponds par :
"YES - [raison]"

Si elle est correcte, réponds par :
"NO - [raison]"

Exemples de raison : "Taux incorrect", "Omission d'informations importantes", "Pas d'erreur détectée"
"""
    try:
        result = llm_judge.invoke([HumanMessage(content=scoring_prompt)]).content.strip()
        if result.upper().startswith("YES"):
            return {"flag": "YES", "reason": result}
        elif result.upper().startswith("NO"):
            return {"flag": "NO", "reason": result}
        else:
            return {"flag": "UNKNOWN", "reason": result}
    except Exception as e:
        print(f"[ERROR] Hallucination detection failed: {e}")
        return {"flag": "ERROR", "reason": str(e)}

def llm_self_confidence(prompt: str, generated: str) -> int:
    """
    LLM self-assesses confidence (0-100).
    """
    scoring_prompt = f"""
Sur une échelle de 0 à 100, indique ta confiance que la réponse est correcte et complète.

Question :
"{prompt}"

Réponse :
"{generated}"

Donne uniquement un nombre entier.
"""
    try:
        result = llm_judge.invoke([HumanMessage(content=scoring_prompt)]).content.strip()
        match = re.search(r'(\d+)', result)
        if match:
            return int(match.group(1))
        return 0
    except Exception as e:
        print(f"[ERROR] Self-confidence estimation failed: {e}")
        return 0

def check_regulatory_mentions(text: str) -> bool:
    """
    Check if text mentions key regulatory terms.
    """
    keywords = ["code monétaire", "AMF", "ACPR", "loi PACTE"]
    return any(k in text.lower() for k in keywords)

def print_summary(df: pd.DataFrame):
    """Print evaluation summary statistics."""
    print("\n" + "="*50)
    print("🔍 EVALUATION SUMMARY")
    print("="*50)
    
    if df.empty:
        print("No data to summarize.")
        return
    
    print(f"📊 Dataset size: {len(df)} examples")
    print(f"📈 Avg Cosine Similarity:     {df['similarity_score'].mean():.4f}")
    print(f"📈 Avg Keyword Overlap:       {df['keyword_overlap_score'].mean():.4f}")
    print(f"📈 Avg LLM-Judge Score:       {df['llm_judge_score'].mean():.2f}/10")
    print(f"📈 Avg BLEU:                  {df['bleu_score'].mean():.4f}")
    print(f"📈 Avg ROUGE-1:               {df['rouge1'].mean():.4f}")
    print(f"📈 Avg ROUGE-2:               {df['rouge2'].mean():.4f}")
    print(f"📈 Avg ROUGE-L:               {df['rougeL'].mean():.4f}")
    
    halluc_count = df[df['hallucination_flag'] == 'YES'].shape[0]
    print(f"⚠️  Hallucinations detected:  {halluc_count}/{len(df)} ({halluc_count/len(df)*100:.1f}%)")
    
    compliance_count = df[df['regulatory_compliance'] == True].shape[0]
    print(f"✅ Regulatory mentions present in {compliance_count}/{len(df)} responses.")
    
    print("="*50)
