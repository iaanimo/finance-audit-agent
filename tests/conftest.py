"""Shared fixtures for the test suite.

Tests never touch the network, the real DeepSeek API, or the real data/
directory. All LLM calls go through FakeLLM; all file writes go to tmp dirs.
"""

import sys
from pathlib import Path

# Make the project root importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest


class _Message:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Deterministic chat model: returns scripted responses per call.

    `responses` is a list of strings; the last one repeats on overflow.
    """

    def __init__(self, responses=None):
        self.responses = list(responses or ["ok"])
        self.calls = 0
        self.last_prompt = ""

    async def ainvoke(self, messages):
        self.calls += 1
        self.last_prompt = messages[-1].content if messages else ""
        content = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        return _Message(content)


@pytest.fixture
def make_llm():
    """Factory: make_llm("resp1", "resp2", ...) -> FakeLLM."""
    return lambda *responses: FakeLLM(list(responses))


@pytest.fixture
def temp_project_root(tmp_path, monkeypatch):
    """Point settings.project_root at a temp dir so tests never touch data/.

    resolve_data_path (used by file_ops and the agent's report writing) reads
    settings via tools.file_ops.get_settings, so patching that is enough.
    """
    import tools.file_ops

    class _FakeSettings:
        def __init__(self, root: Path):
            self.project_root = root

    monkeypatch.setattr("tools.file_ops.get_settings", lambda: _FakeSettings(tmp_path))
    return tmp_path
