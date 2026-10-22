import os

from pipeline import summarize


def test_backend_selection(monkeypatch):
    for var in ("LB_SUMMARY_BACKEND", "OLLAMA_MODEL", "LB_GGUF_PATH", "LB_GGUF_REPO"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(summarize, "_llama_cpp_installed", lambda: False)
    assert summarize.backend() == "extractive"
    monkeypatch.setattr(summarize, "_llama_cpp_installed", lambda: True)
    assert summarize.backend() == "llama_cpp"
    monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:3b")
    assert summarize.backend() == "ollama"
    monkeypatch.setenv("LB_SUMMARY_BACKEND", "extractive")
    assert summarize.backend() == "extractive"


def test_llm_failure_falls_back_to_extractive(monkeypatch):
    def boom(_):
        raise RuntimeError("no model")

    monkeypatch.setattr(summarize, "_llama_cpp", boom)
    out = summarize.summarize(TRANSCRIPT, mode="llama_cpp")
    assert out["backend"].startswith("extractive (fallback")
    assert out["summary"]


TRANSCRIPT = (
    "Speaker 1: Mir schwätzen haut iwwer de Reporting fir d'CSSF. "
    "Speaker 2: De Reporting muss bis Freideg fäerdeg sinn, dat ass wichteg. "
    "Speaker 1: Mir hunn decidéiert, datt de Pedro de Reporting iwwerhëlt. "
    "Speaker 2: Et reent haut. "
    "Speaker 1: Da si mer eis eens, Merci alleguer fir de Reporting."
)


def test_extractive_finds_actions_and_decisions():
    out = summarize.extractive(TRANSCRIPT, max_points=2)
    assert out["backend"] == "extractive"
    assert len(out["summary"]) == 2
    assert any("muss bis Freideg" in a for a in out["actions"])
    assert any("decidéiert" in d for d in out["decisions"])
    assert all("reent" not in s for s in out["summary"])


def test_extractive_handles_empty():
    assert summarize.extractive("") == {"summary": [], "decisions": [], "actions": [], "backend": "extractive"}


def test_markdown_rendering_in_french():
    md = summarize.to_markdown({"summary": ["a"], "decisions": [], "actions": ["b"]}, "fr")
    assert md.startswith("## Résumé\n- a\n")
    assert "## Décisions\n- (aucune)" in md and "## Actions\n- b" in md


def test_parse_json_tolerates_prose_around_object():
    out = summarize._parse_json('Here you go:\n{"summary": ["x"], "decisions": [], "actions": [1]}\nthanks')
    assert out == {"summary": ["x"], "decisions": [], "actions": ["1"]}
