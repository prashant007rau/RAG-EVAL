# RAG-EVAL

A RAG evaluation framework for benchmarking retrieval, answer generation, safety, and operational behavior against transcript-based knowledge sources.

This project ingests `.vtt` transcript files, stores them in a Chroma vector database, retrieves relevant chunks, reranks them, generates grounded answers with a Groq-backed LLM, and then evaluates the whole pipeline with custom and DeepEval-style metrics.

## Why this project exists

The codebase is designed to answer questions like:

- Does the retriever surface the right evidence?
- Does the generator answer from context without hallucinating?
- Does the end-to-end RAG pipeline remain consistent over time?
- Are safety, toxicity, and latency constraints still acceptable?

It combines a practical RAG pipeline with a repeatable evaluation harness and baseline-vs-candidate comparison workflow.

## Architecture

The repository is organized around a modular evaluation stack:

- `src/retriever.py` loads transcript data, builds a Chroma store, and creates a retriever.
- `src/rerankers.py` applies reranking logic to improve retrieval quality.
- `src/generator.py` produces grounded answers from retrieved context using a Groq model.
- `src/rag_pipeline.py` orchestrates retrieve -> rerank -> generate.
- `evals/` contains per-component and full-suite evaluation modules.
- `golden/` stores golden datasets used for correctness and safety tests.
- `data/` contains the transcript documents used as the knowledge base.
- `chroma_store/` persists the vector DB used by the retriever.

## Project structure

```text
RAG-EVAL/
├── data/                           # transcript datasets (.vtt)
├── golden/                         # evaluation goldens
├── chroma_store/                   # persisted Chroma vector database
├── evals/                          # evaluation modules and harness
│   ├── compare.py
│   ├── harness.py
│   ├── metric_registry.py
│   ├── run_suit.py
│   └── ...
├── src/                            # retrieval, generation, and pipeline code
│   ├── generator.py
│   ├── rag_pipeline.py
│   ├── rerankers.py
│   └── retriever.py
├── tests/
│   └── test_eval_harness.py
├── main.py
├── pyproject.toml
├── README.md
└── .env.example                   # optional if added by local setup
```

## Tech stack

- Python 3.11+
- Chroma for vector storage
- LangChain + LangChain Chroma + Hugging Face embeddings
- Groq LLM model for generation
- DeepEval-style evaluation framework
- Pytest for validation

## Prerequisites

Before running the project:

1. Install Python 3.11+
2. Create a virtual environment
3. Set your Groq API key in an environment file

## Setup

```bash
cd /path/to/RAG-EVAL
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
```

If your environment already has a working Groq setup, the code will also pick up these values automatically from the environment.

## Run the core components

### 1. Build the retrieval store

```bash
python -m src.retriever
```

This loads transcript files from `data/`, splits them into chunks, creates embeddings, and persists them to `chroma_store/`.

### 2. Generate a grounded answer

```bash
python -m src.generator
```

This invokes the generation prompt with a small test context and prints the answer.

### 3. Run the full RAG pipeline

```bash
python -m src.rag_pipeline
```

This builds a retriever, fetches context, reranks the passages, and generates an answer for a sample query.

## Run the evaluation suite

The main evaluation workflow is run through `evals/run_suit.py`.

```bash
python -m evals.run_suit
```

Optional flags:

```bash
python -m evals.run_suit --baseline --label "initial baseline"
python -m evals.run_suit --out baselines/candidate.json
python -m evals.run_suit --quiet
python -m evals.run_suit --full
```

This writes a JSON snapshot containing benchmark metrics and metadata for the pipeline.

## Compare snapshots

The comparison tool reads two evaluation snapshots and reports PASS / REVIEW / FAIL based on configured metric rules.

```bash
python -m evals.compare
```

With custom paths:

```bash
python -m evals.compare --baseline baselines/baseline.json --candidate baselines/candidate.json --all
```

This is useful for CI-style regression checks when comparing a baseline against a candidate model or prompt change.

## Evaluation categories

The suite is designed around several metric families:

- Retriever quality
- Generator quality
- End-to-end pipeline quality
- Application quality
- Safety and toxicity checks
- Operational checks such as latency and cost

Golden datasets under `golden/` support tasks such as:

- correctness
- faithfulness
- leakage
- retriever metrics
- scope safety
- toxicity

## Testing

Run project tests with:

```bash
pytest
```

The existing test checks that the evaluation harness loads golden datasets successfully and that the core eval entry points exist.

## Notes

- The code assumes transcript data lives in `data/*.vtt`.
- The vector store is persisted in `chroma_store/` to avoid reindexing on every run.
- Prompt and model configuration is driven by environment variables, especially `GROQ_API_KEY` and `GROQ_MODEL`.
- The project is intended for benchmarking and regression testing of changes to retrieval, reranking, and generation logic.

## Typical workflow

1. Place or update transcript data in `data/`
2. Run the retriever to build or refresh the Chroma store
3. Run the full suite to capture a snapshot
4. Compare a candidate snapshot against a baseline
5. Use the results to decide whether the change is safe to promote

## License

This project does not currently declare a license in the repository metadata. If you plan to distribute or reuse it publicly, add an explicit license file and update the project metadata accordingly.
