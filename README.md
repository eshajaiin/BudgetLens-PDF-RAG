# BudgetLens — PDF RAG challenge

A demonstration RAG system for ten US government budget appendix PDFs (2015–2024), totalling 13,742 pages in the source corpus. Actual indexed coverage depends on ingestion settings and must not be confused with downloaded page count.

## Quick start

```bash
python -m pip install -r requirements.txt
python ingest.py --pages-per-pdf 30
streamlit run app.py
```

The initial run indexes only the first 30 pages per document for a quick smoke test. Increase the coverage in successive runs (`--pages-per-pdf 100`, then `0` for all), which reuses stable IDs and persists a Chroma collection in `data/chroma`.

## Answer modes

Without `OPENAI_API_KEY`, the interface shows **extractive evidence snippets**, not generative LLM answers. For hosted synthesis allowed by Challenge 1, install `openai` and configure `OPENAI_API_KEY`, `LLM_MODEL`, and optionally `OPENAI_BASE_URL` as environment variables. Never commit credentials. A full generative requirement is not satisfied until synthesis is configured and tested.

## Extraction and provenance

PyMuPDF extracts native page text. Optional `--ocr` attempts Tesseract for near-empty scanned pages; requires local Tesseract installation (`brew install tesseract`) plus pytesseract and Pillow. Ingestion records source filename, document identifier, year and page. Chunking is deterministic with stable hashes and a 250-character overlap.

## Evaluation limitations

Similarity search and live query elapsed-time measurements are exposed. Recall@k, MRR, citation precision and hallucination rate require annotated evaluation questions and independent grading; no unmeasured claims are made. The 2–5-second latency target is not guaranteed on an Intel CPU or a remote API.

## Architecture

PyMuPDF → conditional OCR → deterministic overlapping chunks → all-MiniLM-L6-v2 embeddings → Chroma persistent HNSW cosine search → optional hosted LLM / extractive fallback → Streamlit chat and evidence panel.
