# LLM4EKG: A Dual-Channel Knowledge Graph-Augmented Generation Framework for Emergency Response

LLM4EKG is a traceable reasoning and knowledge graph construction framework designed for Emergency Standard Operating Procedures (SOPs). It targets common RAG failure modes on safety-critical long texts (semantic fragmentation, entity drift, and insufficient traceability) via:

- Hybrid extraction (pattern + LLM)
- Centrality-anchored constrained clustering (alias alignment)
- Ontology-guided routing (disaster-chain / plan meta)
- Dual-layer indexing (entity + rule)

## Quick Start

### 1) Install dependencies

```bash
pip install -r requirement.txt
```

### 2) Configure environment variables

Neo4j (required for Modules 2/3):
- `NEO4J_URI` (default: `bolt://localhost:7687`)
- `NEO4J_USERNAME` (default: `neo4j`)
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE` (default: `neo4j`)

Paths (optional):
- `KG_DATA_DIR` (default: `src/kg/data`)
- `KG_OUTPUT_DIR` (default: `src/kg/outputs`)

Cloud model keys (required only if you actually call those providers):
- `DASHSCOPE_API_KEY` (Module 1, rule extraction)
- `ALIYUN_API_KEY` (Module 2, LLM extraction)
- `ZHIPU_API_KEY` (Module 2, embeddings)

### 3) Prepare input rules

- Put your Rule JSON files under `src/kg/data/` (or set `KG_DATA_DIR`).
- Default Module 2 reads `src/kg/data/*.json`.

### 4) Run the end-to-end pipeline

```bash
python src/scripts/run_pipeline.py
```

By default it runs Module 2 + Module 3. To start from Module 1 (DOCX extraction), call `run_pipeline(extract_rules=True, input_docs_dir=...)` inside `src/scripts/run_pipeline.py`.

## Project Structure

```
src/
  scripts/
    extract_rules_from_docx.py   # Module 1: DOCX -> Rule JSON
    build_disaster_chain.py      # Module 3: PlanMeta ontology + anchoring
    run_pipeline.py              # One-click pipeline (Modules 1/2/3)
  kg/
    config/                      # Config (env-based)
    pipeline/                    # Module 2 pipeline entry
    entity_extraction/
    relation_extraction/
    graph_db/
    utils/
    data/                        # Default Rule JSON input directory
    outputs/                     # Default outputs (logs/embeddings/etc.)
```

## Three Modules

### Module 1: Generative Rule Structuring (DOCX → Rule JSON)

- Script: `src/scripts/extract_rules_from_docx.py`
- Output: Rule JSON (atomic normative rules with tags and semantic elements)

### Module 2: Hybrid Robust Information Extraction (Rule JSON → KG → Neo4j)

- Entry: `src/kg/pipeline/main.py`
- Features:
  - Dual-channel extraction (regex + LLM)
  - Alias alignment via embeddings and constrained clustering
  - Bridging relations with optional NLI verification
  - Neo4j ingestion

### Module 3: Ontology-Guided Graph Construction (PlanMeta + Anchoring)

- Script: `src/scripts/build_disaster_chain.py`
- Features:
  - Build `PlanMeta` ontology tree (`SUB_PLAN_OF`)
  - Compute `specificity` weights
  - Anchor graph entities to plans (`BELONGS_TO_SOURCE`)

## Models to Prepare (Download / Configure)

This project uses cloud-hosted LLMs (no local download) and an optional local NLI model (download needed if enabled).

### Cloud Models (No Local Download)

- Module 1 (DOCX → Rule JSON)
  - Default LLM: `deepseek-r1` via DashScope OpenAI-compatible endpoint
  - Configure with: `DASHSCOPE_API_KEY`
  - Change model by editing `RuleExtractor(model=...)` in `src/scripts/extract_rules_from_docx.py`
- Module 2 (Rule JSON → KG)
  - Default LLM: `deepseek-r1-0528` (Aliyun Bailian)
    - Configure with: `ALIYUN_API_KEY`, optional `ALIYUN_MODEL`
  - Default embedding model: `embedding-3` (Zhipu)
    - Configure with: `ZHIPU_API_KEY`

### Local Model (Download Required)

- NLI / Entailment Verification (filters hallucinated bridging relations)
  - Default model: `microsoft/deberta-v3-large-mnli`
  - Download:
    - `python src/kg/scripts/download_nli_model.py`
  - Cache/local path:
    - `NLI_CACHE_DIR` / `NLI_LOCAL_DIR` (defaults under `src/kg/outputs/`)

## Pre-run Checklist

- Neo4j is running and accessible (URI/user/password/database).
- API keys are set as environment variables (never commit plaintext keys).
- Rule JSON files exist under `src/kg/data/` (or `KG_DATA_DIR` points to them).
