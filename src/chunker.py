import re


def clean_text(text: str) -> str:
    """
    Clean common PDF extraction problems while preserving meaning.
    """

    text = text.replace("\x00", " ")

    # Join words broken across lines:
    # admission-
    # requirements
    text = re.sub(r"-\s*\n\s*", "", text)

    # Convert repeated whitespace and line breaks into spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 350,
    chunk_overlap: int = 60,
) -> list[str]:
    """
    Split text into overlapping word-based chunks.
    """

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    words = clean_text(text).split()

    if not words:
        return []

    chunks = []
    start = 0
    step = chunk_size - chunk_overlap

    while start < len(words):
        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        ).strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


def create_chunks(
    pages: list[dict],
    chunk_size: int = 350,
    chunk_overlap: int = 60,
) -> list[dict]:
    """
    Convert PDF pages into smaller searchable chunks.
    """

    all_chunks = []

    for page_data in pages:
        page_number = page_data["page"]
        page_text = page_data["text"]

        page_chunks = chunk_text(
            text=page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_number, chunk in enumerate(
            page_chunks,
            start=1,
        ):
            all_chunks.append(
                {
                    "chunk_id": len(all_chunks),
                    "page": page_number,
                    "chunk_number": chunk_number,
                    "text": chunk,
                }
            )

    return all_chunks