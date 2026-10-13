from argparse import Namespace

from heteroagent_rl.evaluate import build_client
from heteroagent_rl.clients.mock import MockLLMClient


def test_build_client_mock_shortcut():
    args = Namespace(
        mock=True,
        backend="ollama",
        ollama_url="http://localhost:11434",
        num_ctx=4096,
    )
    assert isinstance(build_client(args), MockLLMClient)
