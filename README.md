# Document-to-Audio Pipeline

This repository implements a sophisticated pipeline designed to convert source documents (PDF/DOCX) into fully synthesized podcast audio scripts. The process involves using a Large Language Model (LLM) for script generation and fact-checking, followed by local Text-to-Speech (TTS) synthesis.

The entire workflow is orchestrated as a LangGraph state machine, ensuring robust, multi-step processing from document ingestion to final audio output.

## 🚀 Features

*   **Document Ingestion:** Supports reading raw text from PDF and DOCX files using `pypdf` and `python-docx`.
*   **Script Generation (LLM):** Uses Gemini 1.5 Flash to transform the source material into a coherent, podcast-ready script chunk by chunk.
*   **Fact-Checking:** Implements an iterative fact-checking loop where the LLM critiques the generated script against the original document chunk, ensuring factual accuracy before proceeding.
*   **Audio Synthesis (TTS):** Synthesizes the final audio segments using local TTS models like Kokoro-82M or Edge-TTS.
*   **Orchestration:** Managed entirely by a LangGraph state machine for reliable, sequential execution of complex steps.

## 🛠️ Intended Stack & Dependencies

The project is built around Python and utilizes several key libraries:

*   **Orchestration:** `langgraph`
*   **LLM:** `langchain-google-genai` (Gemini 1.5 Flash)
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
*This project was developed using LangGraph and Gemini 1.5 Flash.*