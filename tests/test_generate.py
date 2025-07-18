import pytest
from src.generate import generate_response, llm

def test_generate_success(monkeypatch):
    class DummyResp:
        content = "Réponse simulée."

    # Note: fake_invoke must accept (self, messages)
    def fake_invoke(self, messages):
        return DummyResp()

    # Patch on the class so Pydantic allows it
    monkeypatch.setattr(llm.__class__, "invoke", fake_invoke, raising=True)

    out = generate_response("Quelle est la fiscalité ?")
    assert isinstance(out, str)
    assert out == "Réponse simulée."

def test_generate_error(monkeypatch):
    # And here too: include self
    def fake_invoke(self, messages):
        raise RuntimeError("API down")

    monkeypatch.setattr(llm.__class__, "invoke", fake_invoke, raising=True)

    out = generate_response("Tout plante")
    assert out.startswith("[ERROR]") and "Impossible de générer" in out
