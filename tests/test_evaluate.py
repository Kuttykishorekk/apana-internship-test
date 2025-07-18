import pytest
import pandas as pd
from src.evaluate import (
    load_dataset,
    evaluate_response,
    keyword_overlap_score,
    bleu_score,
    rouge_score_all,
    check_regulatory_mentions
)

def test_load_dataset_missing(tmp_path, capsys):
    df = load_dataset(str(tmp_path / "no_such_file.json"))
    captured = capsys.readouterr()
    assert "[ERROR] Dataset file not found" in captured.out
    assert isinstance(df, pd.DataFrame) and df.empty

def test_evaluate_response_empty():
    assert evaluate_response("", "") == 0.0

def test_keyword_overlap_score_perfect():
    ref = "Le taux fixe est stable"
    gen = "Le taux fixe est stable sur la durée"
    assert keyword_overlap_score(ref, gen) == pytest.approx(1.0)

def test_keyword_overlap_score_none():
    ref = "AMF ACPR"
    gen = "autre contenu"
    assert keyword_overlap_score(ref, gen) == 0.0

def test_bleu_score_no_overlap():
    ref = "a b c"
    gen = "x y z"
    assert bleu_score(ref, gen) == pytest.approx(0.0)

def test_bleu_score_some_overlap():
    ref = "le prêt immobilier"
    gen = "prêt immobilier possible"
    assert bleu_score(ref, gen) > 0.0

def test_rouge_score_all_empty():
    scores = rouge_score_all("", "")
    assert scores == {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}

def test_rouge_score_all_basic():
    ref = "budget épargne retraite"
    gen = "budget retraite épargne"
    scores = rouge_score_all(ref, gen)
    assert scores["rouge1"] > 0.0

def test_check_regulatory_mentions_true():
    assert check_regulatory_mentions("Mention de l'AMF et du code monétaire")

def test_check_regulatory_mentions_false():
    assert not check_regulatory_mentions("Pas de termes réglementaires ici")
