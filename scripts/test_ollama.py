import json
import urllib.request


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"


payload = {
    "model": MODEL_NAME,
    "prompt": "Explain what a university admission requirement is in one sentence.",
    "stream": False,
}


request = urllib.request.Request(
    OLLAMA_URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)


try:
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.loads(response.read().decode("utf-8"))

    print("=" * 60)
    print("OLLAMA CONNECTION TEST")
    print("=" * 60)
    print("Model:", MODEL_NAME)
    print()
    print("Generated answer:")
    print(result["response"])

except Exception as error:
    print("Could not connect to Ollama.")
    print("Error:", error)