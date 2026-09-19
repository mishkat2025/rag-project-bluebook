import json

with open(
    "data/processed/metadata.json",
    "r",
    encoding="utf-8",
) as file:
    metadata = json.load(file)

for item in metadata:
    if item["chunk_id"] in {
        "ewu-p176-c01167",
        "ewu-p176-c01168",
        "ewu-p176-c01169",
    }:
        print("=" * 80)
        print(item)