import json
from unittest.mock import patch

from heteroagent_rl.clients.ollama import OllamaClient


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return json.dumps(
            {
                "model": "qwen3:4b-instruct",
                "message": {"role": "assistant", "content": "hello"},
                "prompt_eval_count": 12,
                "eval_count": 4,
            }
        ).encode("utf-8")


def test_ollama_client_parses_response():
    client = OllamaClient(base_url="http://localhost:11434", num_ctx=4096)

    with patch("heteroagent_rl.clients.ollama.urlopen", return_value=FakeResponse()):
        result = client.generate(
            model="qwen3:4b-instruct",
            system_prompt="system",
            user_prompt="user",
        )

    assert result.text == "hello"
    assert result.model == "qwen3:4b-instruct"
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 4
