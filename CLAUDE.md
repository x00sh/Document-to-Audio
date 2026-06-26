# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Document-to-Audio pipeline: ingest a document (PDF/DOCX), turn it into a podcast script with an LLM, fact-check the script against the source, and synthesize the final audio. Orchestrated as a LangGraph state machine. The design goal is to run entirely on free tiers / local models.

The entire project lives in [docx-to-audio.ipynb](docx-to-audio.ipynb): markdown cells holding the plan/blueprint and the flow diagram, a code cell holding the compiled LangGraph app, and a run cell. The nodes are **fully implemented** — real Gemini calls for scripting/fact-checking and local Kokoro TTS for audio. Long documents are split into chunks and processed one at a time.

## Intended Stack

- **Orchestration:** LangGraph
- **LLM (scriptwriter + fact-checker):** Gemini 1.5 Flash via Google AI Studio (`langchain-google-genai`), chosen for its large context window (1M+ tokens) so full documents fit in one prompt.
- **TTS:** Kokoro-82M (local CPU) or Edge-TTS, with `soundfile` for writing audio.
- **Parsing:** `pypdf` (and/or `python-docx`) for document ingestion.

## Setup

```bash
pip install langgraph langchain-google-genai langchain-text-splitters kokoro soundfile pypdf python-docx
```

A free Google AI Studio API key is required for the Gemini calls (read it from the environment; do not hardcode it).

## Architecture

The pipeline is a single `StateGraph` over a shared `PodcastState` (TypedDict). Every node reads and writes this state:

| Field                | Meaning                                                          |
|----------------------|------------------------------------------------------------------|
| `document_path`      | Input — path to the source `.pdf`/`.docx`                        |
| `document_name`      | Input — base name used for the output audio file                 |
| `document_text`      | Raw text extracted from the document (set by `parse_document`)   |
| `chunks`             | The document split into processable pieces (set by `chunk_document`) |
| `current_chunk_index`| Outer-loop pointer into `chunks`                                 |
| `script`             | Current chunk's draft podcast script                             |
| `feedback`           | Fact-checker's critique, fed back into the next rewrite          |
| `is_factual`         | Bool — did the current chunk's script pass the fact-check        |
| `iteration_count`    | Per-chunk rewrite count (reset between chunks; the loop cap key) |
| `audio_segments`     | Accumulated per-chunk audio arrays                              |
| `audio_path`         | Path to the final combined audio file                           |

**Nodes** (`parse_document` → `chunk_document` → [outer loop: `generate_script` → `fact_check_script` → conditional → `generate_audio`]):

- `parse_document` — extracts raw text from the document into `document_text` (reuses the `parse_document(path)` helper).
- `chunk_document` — if `len(document_text) <= CHUNK_THRESHOLD` (16k) uses a single-element `chunks` array; otherwise splits with `RecursiveCharacterTextSplitter` into ~`CHUNK_SIZE` (9k) pieces. Also initializes the outer-loop state.
- `generate_script` — Gemini writes/rewrites the script for the **current chunk** and any prior `feedback`. Scripts carry no intros/outros (each chunk is one continuous section).
- `fact_check_script` — Gemini compares `script` against the **current chunk**, writes a `feedback` critique, sets `is_factual`, and increments `iteration_count`.
- `generate_audio` — synthesizes the chunk's audio with Kokoro, appends it to `audio_segments`, writes the running concatenation to `audio_path`, advances `current_chunk_index`, and resets `iteration_count`/`feedback` for the next chunk.

**Two control loops** are the core of the design:

- **Inner (rewrite) loop** — `should_rewrite` routes after each fact-check: go to `generate_audio` if `is_factual` is true **or** `iteration_count >= 3`; otherwise loop back to `generate_script`. The `iteration_count >= 3` cap is a hard guard against infinite rewrite loops (and runaway API usage), enforced **per chunk**. **Preserve this cap** — it is the safety boundary of the graph, not an arbitrary number.
- **Outer (chunk) loop** — `has_remaining_chunks` routes after `chunk_document` and after each `generate_audio`: go to `generate_script` if `current_chunk_index < len(chunks)`, otherwise `END`.

The graph is compiled to `app`; invoke it with `{"document_path": ..., "document_name": ...}` (the loop state is initialized inside `chunk_document`). Long documents produce many super-steps, so pass a raised `config={"recursion_limit": ...}`.

## When working on the nodes

- Keep node functions pure-ish: read from `state`, return a partial dict of the fields they update (LangGraph merges it). Don't mutate `state` in place.
- The fact-check → rewrite contract depends on `feedback` and `iteration_count` being written every fact-check pass; `generate_script` must actually consume `feedback` for the loop to converge.
- The outer loop depends on `generate_audio` advancing `current_chunk_index` and resetting `iteration_count`/`feedback`; if any of those is dropped, the chunk loop will stall or leak critique between chunks.
