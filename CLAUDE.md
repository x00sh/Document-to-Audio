# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Document-to-Audio pipeline: ingest a document (PDF/DOCX), turn it into a podcast script with an LLM, fact-check the script against the source, and synthesize the final audio. Orchestrated as a LangGraph state machine. The design goal is to run entirely on free tiers / local models.

Currently the entire project lives in [main.ipynb](main.ipynb): a markdown cell holding the plan/blueprint, and one code cell holding the compiled LangGraph skeleton. The node bodies are **stubs** — they print and return hardcoded values (e.g. `"Generated script text..."`, `"output_podcast.mp3"`). The wiring (state, edges, loop control) is real and working; the LLM/TTS integrations are the work to be filled in.

## Intended Stack

- **Orchestration:** LangGraph
- **LLM (scriptwriter + fact-checker):** Gemini 1.5 Flash via Google AI Studio (`langchain-google-genai`), chosen for its large context window (1M+ tokens) so full documents fit in one prompt.
- **TTS:** Kokoro-82M (local CPU) or Edge-TTS, with `soundfile` for writing audio.
- **Parsing:** `pypdf` (and/or `python-docx`) for document ingestion.

## Setup

```bash
pip install langgraph langchain-google-genai kokoro soundfile pypdf
```

A free Google AI Studio API key is required for the Gemini calls (read it from the environment; do not hardcode it).

## Architecture

The pipeline is a single `StateGraph` over a shared `PodcastState` (TypedDict). Every node reads and writes this state:

| Field             | Meaning                                                        |
|-------------------|----------------------------------------------------------------|
| `document_text`   | Raw text extracted from the uploaded document                  |
| `script`          | Current draft of the podcast script                            |
| `feedback`        | Fact-checker's critique, fed back into the next rewrite        |
| `is_factual`      | Bool — did the script pass the fact-check                       |
| `iteration_count` | Number of rewrites so far (the loop cap key)                   |
| `audio_path`      | Path to the final synthesized audio file                       |

**Nodes** (`generate_script` → `fact_check_script` → conditional → `text_to_speech`):

- `generate_script` — Gemini writes/rewrites the script from `document_text` and any prior `feedback`.
- `fact_check_script` — Gemini compares `script` against `document_text`, writes a `feedback` critique, sets `is_factual`, and increments `iteration_count`.
- `text_to_speech` — passes the finalized `script` to the local TTS engine and writes `audio_path`.

**The control loop** is the core of the design. `should_continue` routes after each fact-check:

- Go to `text_to_speech` if `is_factual` is true **or** `iteration_count >= 3`.
- Otherwise loop back to `generate_script` for another rewrite.

The `iteration_count >= 3` cap is a hard guard against infinite rewrite loops (and runaway API usage) when the script never fully passes the fact-check. **Preserve this cap** when implementing the real nodes — it is the safety boundary of the graph, not an arbitrary number.

The graph is compiled to `app`; invoke it with an initial state that has `iteration_count: 0` and `document_text` populated by the parsing step.

## When implementing the stubs

- Keep node functions pure-ish: read from `state`, return a partial dict of the fields they update (LangGraph merges it). Don't mutate `state` in place.
- The fact-check → rewrite contract depends on `feedback` and `iteration_count` being written every fact-check pass; `generate_script` must actually consume `feedback` for the loop to converge.
