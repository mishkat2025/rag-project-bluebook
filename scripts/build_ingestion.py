"""Rebuild the processed corpus from the PDF.

    .\\.venv\\Scripts\\python.exe scripts\\build_ingestion.py

Writes chunks.json (children + tables), metadata.json (index records),
parents.json (section text for parent expansion) and section_tree.json
(the outline, for inspection). Fails loudly rather than writing a broken
corpus.
"""
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.settings import settings  # noqa: E402
from src.ingestion.chunker import StructureAwareChunker  # noqa: E402
from src.ingestion.metadata_builder import MetadataBuilder  # noqa: E402
from src.ingestion.pdf_parser import PDFParser  # noqa: E402
from src.ingestion.structure_analyzer import StructureAnalyzer  # noqa: E402
from src.ingestion.validator import ChunkValidator  # noqa: E402


def write_json(path: Path, payload) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def main() -> None:
    print("=" * 60)
    print("EWU RAG - BUILD INGESTION DATA")
    print("=" * 60)

    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Parsing PDF (fonts + tables)...")
    pages = PDFParser(settings.pdf_path).parse()
    print(
        f"      pages {len(pages)}   "
        f"lines {sum(len(page.lines) for page in pages)}   "
        f"tables {sum(len(page.tables) for page in pages)}"
    )

    print("\n[2/6] Building the section tree...")
    tree = StructureAnalyzer().analyze(pages)
    levels: dict[int, int] = {}

    for node in tree.iter_nodes():
        levels[node.level] = levels.get(node.level, 0) + 1

    print(f"      body font {tree.body_size}pt   outline entries {len(tree.toc_entries)}")
    print(f"      faculties mapped {len(set(tree.faculty_of_department.values()))}")
    print(f"      nodes {len(tree.nodes)}   by level {dict(sorted(levels.items()))}")

    print("\n[3/6] Chunking (parents / children / tables)...")
    chunker = StructureAwareChunker()
    chunks, parents = chunker.chunk(tree)
    print(f"      children+tables {len(chunks)}   parent sections {len(parents)}")

    print("\n[4/6] Validating...")
    validator = ChunkValidator()
    chunks, report = validator.validate(
        chunks,
        known_parent_ids={parent.parent_id for parent in parents},
    )
    print(f"      {report.summary()}")

    if report.examples:
        print(f"      dropped examples: {report.examples[:3]}")

    print("\n[5/6] Building metadata from tree position...")
    builder = MetadataBuilder(source_name=settings.pdf_path.name)
    metadata_records = builder.build(chunks, tree)
    parent_records = builder.build_parents(parents)

    labelled = sum(1 for record in metadata_records if record["department"])
    print(
        f"      department labelled {labelled}/{len(metadata_records)} "
        f"({labelled / len(metadata_records):.1%})"
    )

    print("\n[6/6] Saving...")
    write_json(settings.chunks_path, [asdict(chunk) for chunk in chunks])
    write_json(settings.metadata_path, metadata_records)
    write_json(settings.parents_path, parent_records)
    write_json(
        settings.tree_path,
        [
            {
                "node_id": node.node_id,
                "level": node.level,
                "title": node.title,
                "section_path": node.breadcrumb_text,
                "start_page": node.page,
                "end_page": node.end_page,
                "anchored": node.anchored,
                "blocks": len(node.blocks),
            }
            for node in tree.iter_nodes()
        ],
    )

    for path in (
        settings.chunks_path,
        settings.metadata_path,
        settings.parents_path,
        settings.tree_path,
    ):
        print(f"      {path}")

    print("\nIngestion build completed successfully.")


if __name__ == "__main__":
    main()
