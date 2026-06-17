# Document to Audio

Ingests a PDF or DOCX, turns it into a podcast script via an LLM, fact-checks the
script against the source, and synthesizes the final audio — all on free tiers and
local models.

## Stack

- **Orchestration:** LangGraph (open-source)
- **LLM (scriptwriter + fact-checker):** Gemini 2.5 Flash via Google AI Studio —
  generous free tier, 1M+ token context window, structured output for reliable
  fact-check results
- **Text-to-Speech:** Kokoro-82M (local CPU) — lightweight open-source TTS,
  studio-quality voices, no API cost
- **Parsing:** `pypdf` / `python-docx` for document ingestion
- **Text splitting:** `langchain-text-splitters` (`RecursiveCharacterTextSplitter`)

## Setup

```bash
pip install langgraph langchain-google-genai langchain-text-splitters \
            pypdf python-docx kokoro soundfile python-dotenv
```

A free Google AI Studio API key is required. Set `GOOGLE_API_KEY` in a `.env`
file — it is loaded with `python-dotenv` and never hardcoded.

## Architecture

### Shared State (`PodcastState`)

| Field | Meaning |
|---|---|
| `document_path` | Input — path to the source `.pdf`/`.docx` |
| `document_name` | Input — base name for the output audio file |
| `document_text` | Raw text extracted from the document |
| `chunks` | Document split into processable pieces |
| `current_chunk_index` | Outer-loop pointer into `chunks` |
| `script` | Current chunk's draft podcast script |
| `feedback` | Fact-checker's critique, fed into the next rewrite |
| `is_factual` | Whether the current chunk's script passed the fact-check |
| `iteration_count` | Per-chunk rewrite count (reset between chunks; loop-cap key) |
| `audio_segments` | Accumulated per-chunk audio arrays |
| `audio_path` | Path to the final combined `.wav` file |

### Nodes

- **`parse_document`** — extracts raw text from the file into `document_text`
- **`chunk_document`** — splits documents over 16k chars into ~9k-char pieces;
  initialises outer-loop state
- **`generate_script`** — Gemini writes/rewrites the script for the current chunk,
  incorporating any `feedback`
- **`fact_check_script`** — Gemini compares `script` against the source chunk via
  structured output; sets `is_factual`, writes `feedback`, increments `iteration_count`
- **`generate_audio`** — Kokoro synthesises the chunk's audio, appends it to
  `audio_segments`, writes the running `.wav`, and advances `current_chunk_index`

### Control Loops

- **Inner (rewrite) loop** — after each fact-check, route back to `generate_script`
  unless `is_factual` is true **or** `iteration_count >= 3` (hard cap per chunk)
- **Outer (chunk) loop** — after `generate_audio`, continue to the next chunk while
  `current_chunk_index < len(chunks)`, otherwise `END`

The entire workflow is orchestrated as a LangGraph state machine, ensuring robust, multi-step processing from document ingestion to final audio output.

## 🚀 Features

*   **Document Ingestion:** Supports reading raw text from PDF and DOCX files using `pypdf` and `python-docx`.
*   **Script Generation (LLM):** Uses Gemini 2.5 Flash to transform the source material into a coherent, podcast-ready script chunk by chunk.
*   **Fact-Checking:** Implements an iterative fact-checking loop where the LLM critiques the generated script against the original document chunk, ensuring factual accuracy before proceeding.
*   **Audio Synthesis (TTS):** Synthesizes the final audio segments using local TTS models like Kokoro-82M or Edge-TTS.
*   **Orchestration:** Managed entirely by a LangGraph state machine for reliable, sequential execution of complex steps.

## 🛠️ Intended Stack & Dependencies

The project is built around Python and utilizes several key libraries:

*   **Orchestration:** `langgraph`
*   **LLM:** `langchain-google-genai` (Gemini 2.5 Flash)
*   **TTS:** `soundfile`, Kokoro-82M / Edge-TTS
*   **Parsing:** `pypdf`, `python-docx`

### Installation

To set up the environment, run the following command in your terminal:

```bash
pip install langgraph langchain-google-genai langchain-text-splitters kokoro soundfile pypdf python-docx
```

> **Note:** A free Google AI Studio API key is required for the Gemini calls. Please set this as an environment variable (e.g., `GEMINI_API_KEY`).

## ⚙️ Architecture Overview

The pipeline operates on a single shared state (`PodcastState`) managed by a LangGraph `StateGraph`. The process flows through several interconnected nodes:

1.  **`parse_document`**: Extracts raw text from the input document path into `document_text`.
2.  **`chunk_document`**: Splits the raw text into manageable chunks (default size ~9k) for processing, initializing the outer loop state.
3.  **Outer Loop (`has_remaining_chunks`)**: Iterates through all document chunks.
    *   **`generate_script`**: Writes/rewrites the script for the *current chunk*, incorporating any prior `feedback`.
    *   **`fact_check_script`**: Compares the generated script against the current chunk's source material, generating a critique (`feedback`) and setting the factual status (`is_factual`).
    *   **Conditional Routing**: Determines if rewriting is necessary or if the process can move to audio generation.
4.  **Inner Loop (Rewrite)**: If fact-checking fails, the script loops back to `generate_script` for a rewrite attempt. This loop has a hard cap of **3 iterations per chunk** to prevent infinite loops and manage API usage.
5.  **`generate_audio`**: Synthesizes the audio segment using TTS, appends it to the running audio file, and advances the outer loop pointer (`current_chunk_index`).

## 💻 Usage

The main execution logic is contained within `main.ipynb`.

To run the pipeline:
1.  Ensure your API key is set in your environment variables.
2.  Execute the notebook cell containing the compiled LangGraph application.

**Input Parameters:**
When invoking the graph, you must provide:
*   `document_path`: Path to the source `.pdf` or `.docx`.
*   `document_name`: Base name for the final output audio file.

## 💡 Development Notes (For Contributors)

*   **State Management:** All nodes should read from and write partial updates to the `PodcastState` dictionary, allowing LangGraph to merge changes correctly. Do not mutate the state in place.
*   **Loop Integrity:** The outer loop relies on `generate_audio` advancing `current_chunk_index` and resetting per-chunk variables (`iteration_count`, `feedback`). Dropping these steps will cause the chunk loop to stall or leak critique data between chunks.
*   **Scripting Contract:** The fact-check $\rightarrow$ rewrite contract is critical: `generate_script` *must* consume the `feedback` field from the previous step for the inner loop to converge correctly.

---
*This project was developed using LangGraph and Gemini 2.5 Flash.*