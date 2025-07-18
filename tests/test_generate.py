import pytest

def test_generate_success(monkeypatch):
    class DummyLLM:
        def invoke(self, messages):
            class DummyResp:
                content = "Réponse simulée."
            return DummyResp()

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        lambda *args, **kwargs: DummyLLM()
    )

    from src.generate import generate_response
    out = generate_response("Quelle est la fiscalité ?")
    assert out == "Réponse simulée."

def test_generate_error(monkeypatch):
    class FailingLLM:
        def invoke(self, messages):
            raise RuntimeError("API down")

    monkeypatch.setattr(
        "langchain_google_genai.ChatGoogleGenerativeAI",
        lambda *args, **kwargs: FailingLLM()
    )

    from src.generate import generate_response
    out = generate_response("Test d'erreur")
    assert "[ERROR]" in out
