import json
from dataclasses import asdict

from src.config.settings import settings
from src.ingestion.chunker import StructureAwareChunker
from src.ingestion.metadata_builder import MetadataBuilder
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.structure_analyzer import StructureAnalyzer


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - BUILD INGESTION DATA")
    print("=" * 60)

    settings.processed_data_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # 1. Parse PDF
    # ---------------------------------------------------------

    print("\n[1/4] Parsing PDF...")

    parser = PDFParser(settings.pdf_path)
    pages = parser.parse()

    print(f"      Pages extracted: {len(pages)}")

    # ---------------------------------------------------------
    # 2. Analyze structure
    # ---------------------------------------------------------

    print("\n[2/4] Analyzing document structure...")

    analyzer = StructureAnalyzer()
    structures = analyzer.analyze(pages)

    print(f"      Pages analyzed: {len(structures)}")

    # ---------------------------------------------------------
    # 3. Create chunks
    # ---------------------------------------------------------

    print("\n[3/4] Creating chunks...")

    chunker = StructureAwareChunker(
        max_characters=1800,
        overlap_characters=250,
    )

    chunks = chunker.chunk(
        pages=pages,
        structures=structures,
    )

    print(f"      Chunks created: {len(chunks)}")

    # ---------------------------------------------------------
    # 4. Build metadata
    # ---------------------------------------------------------

    print("\n[4/4] Building metadata...")

    builder = MetadataBuilder(
        source_name=settings.pdf_path.name,
    )

    metadata_records = builder.build(chunks)

    # ---------------------------------------------------------
    # Save chunks
    # ---------------------------------------------------------

    chunk_records = [
        asdict(chunk)
        for chunk in chunks
    ]

    with settings.chunks_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            chunk_records,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ---------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------

    with settings.metadata_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata_records,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSaved:")
    print(f"  Chunks:   {settings.chunks_path}")
    print(f"  Metadata: {settings.metadata_path}")

    print("\nIngestion build completed successfully.")


if __name__ == "__main__":
    main()