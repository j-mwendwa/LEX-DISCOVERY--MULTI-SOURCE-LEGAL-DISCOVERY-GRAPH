"""
scripts/ingest.py — CLI script to ingest documents into Qdrant.

Usage:
    python scripts/ingest.py --dir data/case_law
    python scripts/ingest.py --dir data/raw --collection case_law_precedents
"""
import argparse
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.core.llamaindex_setup import setup_llamaindex


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Qdrant.")
    parser.add_argument("--dir", required=True, help="Directory containing PDFs to ingest.")
    parser.add_argument(
        "--collection",
        default="case_law_precedents",
        help="Qdrant collection name (default: case_law_precedents).",
    )
    args = parser.parse_args()

    print(f"🔧 Setting up LlamaIndex...")
    setup_llamaindex(google_api_key=settings.google_api_key)

    print(f"📄 Ingesting from: {args.dir}")
    print(f"📦 Target collection: {args.collection}")

    from src.ingestion.llamaindex_pipeline import ingest_lease_pdf

    try:
        chunks = ingest_lease_pdf(
            file_path=args.dir,
            collection_name=args.collection,
            qdrant_url=settings.qdrant_url,
            qdrant_api_key=settings.qdrant_api_key,
        )
        print(f"✅ Ingestion complete. {chunks} chunks upserted into '{args.collection}'.")
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
