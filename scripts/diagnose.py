#!/usr/bin/env python3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.metadata.extractor import extract_metadata
from src.ingestion.loader import load_documents
from src.pipelines.engineered import EngineeredPipeline
from src.pipelines.baseline import BaselinePipeline

settings = load_settings()
p = Path("data/raw/reports/treasury_bulletins_parsed/transformed/treasury_bulletin_2024_03.txt")
content = p.read_text(encoding="utf-8", errors="replace")[:500]
meta = extract_metadata(p, content)
print("Metadata:", meta.to_dict())

docs = load_documents(settings)
print("Loaded docs:", len(docs))
print("Years:", sorted(set(d.metadata.year for d in docs)))

for name, pipe in [("baseline", BaselinePipeline(settings)), ("engineered", EngineeredPipeline(settings))]:
    print(f"\n{name} index count:", pipe.vector_store.count)
    if pipe.vector_store.count == 0:
        continue
    for label, kwargs in [
        ("filtered 2024 March", {"year": 2024, "month": "March", "apply_metadata_filter": True}),
        ("filtered 2024 only", {"year": 2024, "apply_metadata_filter": True}),
        ("unfiltered", {"apply_metadata_filter": False}),
    ]:
        hits = pipe.vector_store.similarity_search("total receipts March 2024", top_k=5, **kwargs)
        print(f"  {label}: {len(hits)} hits")
        for h in hits[:2]:
            print(f"    {h.metadata.get('filename')} year={h.metadata.get('year')} month={h.metadata.get('month')}")
