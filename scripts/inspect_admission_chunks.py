import json

with open(
    "data/processed/chunks.json",
    "r",
    encoding="utf-8",
) as file:
    chunks = json.load(file)

for chunk in chunks:
    if chunk["page"] in [176, 177, 178]:
        print("=" * 100)
        print(f"PAGE: {chunk['page']}")
        print(f"HEADING: {chunk.get('heading')}")
        print(f"SECTION: {chunk.get('section')}")
        print(f"CONTENT TYPE: {chunk.get('content_type')}")
        print(f"POSITION: {chunk.get('chunk_position')}")
        print("TEXT:")
        print(chunk["text"])