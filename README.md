# 📄 Apana LLM Evaluation System

## 🎯 Project Overview

This project implements a complete evaluation pipeline for assessing a **French financial domain language model**, as part of the Apana technical challenge.

It includes:

- Automated response generation (Gemini)
- Rich evaluation metrics
- Regulatory compliance checks
- Interactive analysis capabilities
- Visual performance reporting

---

## 🛠️ Project Structure

```

llm-eval-apana/
├── data/
│   └── eval_set.json              ← JSON dataset containing prompts and reference answers
│
├── src/                           ← Core logic and evaluation functions
│   ├── generate.py                ← LLM response generation using Gemini API (lazy-loaded)
│   └── evaluate.py                ← All scoring functions: similarity, BLEU, ROUGE, LLM judge, etc.
│
├── tests/
│   ├── test_generate.py           ← Unit tests for LLM generation (mocked)
│   └── test_evaluate.py           ← Unit tests for metrics and scoring functions
│   └── conftest.py                ← Shared test setup: sets import path
│
├── output/                        ← Stores results from evaluations (e.g., CSVs, logs)
│   └── results.csv                ← Final scored outputs (optional)
│
├── run_eval.py                    ← Main script that:
│                                     1. Loads dataset
│                                     2. Generates LLM answers
│                                     3. Applies multiple evaluation metrics
│                                     4. Writes results to /output and prints summary
│
├── requirements.txt               ← Python dependencies (LangChain, SentenceTransformers, etc.)
│
├── .env                           ← Your local Gemini API key (not committed)
├── .env.template                  ← Template for `.env`, shared safely in repo
│
├── .gitignore                     ← Excludes .env, venv, __pycache__, etc. from commits
│
└── README.md                      ← Complete documentation of the system


````

---

## 🚀 How to Run

1️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
````

2️⃣ **Configure environment variables**

Create a `.env` file:

```
GOOGLE_API_KEY=your_gemini_api_key
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_PROJECT=apana-llm-eval
LANGCHAIN_TRACING_V2=true
```

3️⃣ **Run the evaluation**

```bash
python run_eval.py
```

---

## ✅ Implemented Features

**1. Dataset Loading**

* Loads prompts and reference answers from `data/eval_set.json`.

**2. Response Generation**

* Gemini 1.5 Flash model via LangChain.
* Custom financial system prompt for domain accuracy.

**3. Evaluation Metrics**

* Cosine similarity
* Keyword overlap
* BLEU score
* ROUGE-1 / ROUGE-2 / ROUGE-L
* LLM-Judge score (0–10 scale)
* Self-confidence estimation (0–100)
* Hallucination detection + rationale
* Regulatory compliance check (keywords: *AMF*, *ACPR*, etc.)

**4. Result Logging**

* CSV export to `/output`.
* Optional LangSmith logging for advanced review.

---

## 📊 Evaluation Summary

| Metric                    | Description                                       |
| ------------------------- | ------------------------------------------------- |
| **Cosine Similarity**     | Embedding similarity with reference               |
| **Keyword Overlap**       | Token overlap ratio                               |
| **BLEU Score**            | N-gram similarity                                 |
| **ROUGE Scores**          | Overlap of n-grams and longest common subsequence |
| **LLM-Judge Score**       | Gemini-evaluated quality score                    |
| **Self-confidence**       | Model's self-reported confidence                  |
| **Regulatory Compliance** | Presence of regulatory keywords                   |
| **Hallucination Flag**    | Whether content likely contains hallucination     |
| **Hallucination Reason**  | Explanation of hallucination or correctness       |

---
### ✅ Testing & Validation

This project includes a test suite to validate core functionality such as:

- Dataset loading and input structure
- LLM response generation behavior (mocked)
- Evaluation metrics: BLEU, ROUGE, cosine similarity, keyword overlap, compliance, hallucination, etc.

#### 🔧 Run Tests Locally

Make sure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

Then run the full test suite with:

```bash
pytest -q
```

You should see:

```
............                                                                                                                 [100%]
12 passed in X.XXs
```

#### 📂 Test Structure

- `tests/test_evaluate.py` – Unit tests for metric functions and dataset handling  
- `tests/test_generate.py` – Tests Gemini response generation logic (mocked)  
- `tests/conftest.py` – Automatically adds `src/` to `sys.path` for import resolution

#### 🧪 Test Coverage

All critical logic is covered:
- 100% test pass rate
- Validates error-handling and edge cases (empty prompts, hallucinations, API failures)

## 🖼️ Evaluation Visuals

Below are key charts generated from results:

**1️⃣ Number of Hallucinations**

![Hallucination Chart](./Charts/Figure_1.png)

**2️⃣ Self-confidence Distribution**

![Self-confidence Chart](./Charts/Figure_2.png)

**3️⃣ BLEU vs. Similarity Scatter Plot**

![BLEU vs Similarity](./Charts/Figure_3.png)

---

## 🧠 How to Interpret

* **BLEU vs Similarity** shows content fidelity.
* **Hallucinations** indicate factual inconsistencies.
* **Self-confidence** helps assess model calibration.
* **Regulatory compliance** ensures adherence to financial standards.

---

## 🌐 LangSmith Integration

If enabled in `.env`, all examples are logged to LangSmith:

* Interactive inspection
* Metadata filtering
* Custom scoring columns

Example fields logged:

* Prompt
* Generated Answer
* All metrics
* Hallucination rationale

👉 **[View Interactive LangSmith Dashboard](./Charts/langsmithchart.png)

---

## ✨ Notes & Next Steps

This system was designed for:

* Easy extension (more metrics, more prompts)
* Compatibility with other models
* Reproducible reporting

Possible enhancements:

* Larger evaluation datasets
* Adversarial prompt testing
* Integration with Streamlit dashboards

---

## 📬 Submitted By

* contact to **[contact@kishore](mailto:kishorekumarmourougane@gmail.com)**
* **Subject:** `LLM Evaluation Test - Kishorekumar Mourougane`


