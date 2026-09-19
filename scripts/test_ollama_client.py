from src.generation.ollama_client import OllamaClient


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - OLLAMA CLIENT TEST")
    print("=" * 60)

    client = OllamaClient()

    print(f"\nBase URL: {client.base_url}")
    print(f"Model: {client.model}")
    print(f"Temperature: {client.temperature}")

    print("\nChecking Ollama server...")

    if not client.health_check():
        raise RuntimeError("Ollama server is not reachable.")

    print("Ollama server: OK")

    print("\nGenerating test response...")

    response = client.generate(
        "Reply with exactly: OLLAMA_CLIENT_OK"
    )

    print(f"Response: {response}")

    assert response == "OLLAMA_CLIENT_OK"

    print("\nOllama client test passed.")


if __name__ == "__main__":
    main()