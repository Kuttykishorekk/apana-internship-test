import pytest
from src.generate import generate_response

def test_generate_success(monkeypatch):
    class DummyResp:
        content = "Réponse simulée."

    def fake_invoke(self, messages):
        return DummyResp()

    # Patch the class (lazy-loaded inside function)
    import langchain_google_genai.chat_models
    monkeypatch.setattr(langchain_google_genai.chat_models.ChatGoogleGenerativeAI, "invoke", fake_invoke)

    out = generate_response("Quelle est la fiscalité ?")
    assert isinstance(out, str)
    assert out == "Réponse simulée."

def test_generate_error(monkeypatch):
    def fake_invoke(self, messages):
        raise RuntimeError("API down")

    import langchain_google_genai.chat_models
    monkeypatch.setattr(langchain_google_genai.chat_models.ChatGoogleGenerativeAI, "invoke", fake_invoke)

    out = generate_response("Tout plante")
    assert out.startswith("[ERROR]") and "Impossible de générer" in out
