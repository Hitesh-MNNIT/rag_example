# RAG Assignment: Baseline vs. Re-Ranking

Retrieval-Augmented Generation pipeline comparing a Bi-Encoder baseline against a Cross-Encoder re-ranking stage, evaluated on the D&D 5e Basic Rules document.

## Document

**D&D 5e Basic Rules PDF:** [dnd5e_basic_rules.pdf](data/dnd5e_basic_rules.pdf)

Original source: https://media.wizards.com/2018/dnd/downloads/DnD_BasicRules_2018.pdf

## Repository Structure

```
rag_assignment.ipynb      # Main notebook (fully executed)
report/report.md          # Written report (architecture, analysis, latency, scaling)
data/
  dnd5e_basic_rules.pdf   # Source document (180 pages)
  faiss_index.bin         # FAISS IndexFlatIP (384-dim, 449 chunks)
  chunks.pkl              # Serialized chunk list
```

## Models

| Role | Model |
|------|-------|
| Bi-Encoder | `sentence-transformers/all-MiniLM-L6-v2` |
| Cross-Encoder | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | `claude-haiku-4-5-20251001` |
